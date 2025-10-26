import torch
import pandas as pd
from datasets import Dataset as HFDataset
from transformers import (
    DistilBertTokenizerFast,
    Trainer,
    TrainingArguments,
    DataCollatorWithPadding
)
from evaluate import load

# ===== CONFIG =====
MODEL_NAME = "distilbert-base-uncased"
TEXT_COL = "text"
LABEL_COL = "age"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ===== Load full data =====
test_df = pd.read_parquet("../../../DATA/age/test.parquet")

test_df[LABEL_COL] = test_df[LABEL_COL].astype(float)

# ===== Tokenizer =====
tokenizer = DistilBertTokenizerFast.from_pretrained(MODEL_NAME)

def tokenize(batch):
    texts = [str(t) for t in batch[TEXT_COL]]
    return tokenizer(texts, padding="max_length", truncation=True, max_length=256)

# ===== HuggingFace Dataset =====
test_dataset = HFDataset.from_pandas(test_df)

# Tokenize
test_dataset = test_dataset.map(tokenize, batched=True, num_proc=8)

# Rename label column
test_dataset = test_dataset.rename_column(LABEL_COL, "labels")

# ===== Model =====
# single output for regression
from transformers import DistilBertModel
from torch import nn

class DistilBertRegression(nn.Module):
    def __init__(self, model_name):
        super().__init__()
        self.bert = DistilBertModel.from_pretrained(model_name)
        self.regressor = nn.Linear(self.bert.config.hidden_size, 1)

    def forward(self, input_ids, attention_mask, labels=None):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        hidden_state = outputs.last_hidden_state[:,0,:]  # [CLS] token
        logits = self.regressor(hidden_state).squeeze(-1)  # shape: (batch,)
        loss = None
        if labels is not None:
            loss_fn = nn.MSELoss()
            loss = loss_fn(logits, labels)
        return {"loss": loss, "logits": logits}

model = DistilBertRegression(MODEL_NAME)

from safetensors.torch import load_file
model.load_state_dict(load_file("distilBERT_age_regression_model/model.safetensors"))

model.to(DEVICE)

# ===== Metrics =====
mse = load("mse")
def compute_metrics(eval_pred):
    logits, labels = eval_pred
    preds = torch.tensor(logits)
    return mse.compute(predictions=preds, references=labels)

# ===== Data collator =====
data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

# ===== Trainer =====
trainer = Trainer(
    model=model,
    args=TrainingArguments(
        output_dir="results_loss",
        per_device_eval_batch_size=96
    ),
    eval_dataset=test_dataset,
    data_collator=data_collator,
    compute_metrics=compute_metrics,
)

# ===== Evaluate =====
results = trainer.evaluate()
print("Evaluation results:", results)
