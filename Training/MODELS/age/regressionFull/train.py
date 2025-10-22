import torch
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from datasets import Dataset as HFDataset
from transformers import (
    DistilBertTokenizerFast,
    DistilBertForSequenceClassification,
    Trainer,
    TrainingArguments,
    DataCollatorWithPadding
)
from evaluate import load

# ===== CONFIG =====
MODEL_NAME = "distilbert-base-uncased"
TEXT_COL = "text"
LABEL_COL = "age"
MAX_LEN = 256
BATCH_SIZE = 96
NUM_EPOCHS = 4
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ===== Load full data =====
train_df = pd.read_parquet("../../../DATA/age/train.parquet")
test_df = pd.read_parquet("../../../DATA/age/test.parquet")

# Convert labels to float (raw ages)
train_df[LABEL_COL] = train_df[LABEL_COL].astype(float)
test_df[LABEL_COL] = test_df[LABEL_COL].astype(float)

# See the label ranges
print("Train label range:", train_df[LABEL_COL].min(), "→", train_df[LABEL_COL].max())
print("Test label range:", test_df[LABEL_COL].min(), "→", test_df[LABEL_COL].max())

# ===== Tokenizer =====
tokenizer = DistilBertTokenizerFast.from_pretrained(MODEL_NAME)

def tokenize(batch):
    texts = [str(t) for t in batch[TEXT_COL]]
    return tokenizer(texts, padding="max_length", truncation=True, max_length=MAX_LEN)

# ===== HuggingFace Dataset =====
train_dataset = HFDataset.from_pandas(train_df).shuffle(seed=42)
test_dataset = HFDataset.from_pandas(test_df)

# Tokenize
train_dataset = train_dataset.map(tokenize, batched=True, num_proc=8)
test_dataset = test_dataset.map(tokenize, batched=True, num_proc=8)

# Rename label column
train_dataset = train_dataset.rename_column(LABEL_COL, "labels")
test_dataset = test_dataset.rename_column(LABEL_COL, "labels")

# ⚡ Set PyTorch format
train_dataset.set_format("torch", columns=["input_ids", "attention_mask", "labels"])
test_dataset.set_format("torch", columns=["input_ids", "attention_mask", "labels"])

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

# ===== Metrics =====
mse = load("mse")
def compute_metrics(eval_pred):
    logits, labels = eval_pred
    preds = torch.tensor(logits)
    return mse.compute(predictions=preds, references=labels)


# ===== Data collator =====
data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

# ===== Training args =====
training_args = TrainingArguments(
    output_dir="./results_age",
    eval_strategy="epoch",
    save_strategy="epoch",
    learning_rate=5e-5,
    per_device_train_batch_size=BATCH_SIZE,
    per_device_eval_batch_size=BATCH_SIZE,
    num_train_epochs=NUM_EPOCHS,
    weight_decay=0.01,
    logging_dir="./logs_age",
    logging_steps=5000,
    load_best_model_at_end=True,
    metric_for_best_model="mse",
    fp16=torch.cuda.is_available(),
    gradient_accumulation_steps=2,
    report_to="none",
)

# ===== Trainer =====
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=test_dataset,
    tokenizer=tokenizer,
    data_collator=data_collator,
    compute_metrics=compute_metrics,
)

# ===== Train & Evaluate =====
trainer.train()
results = trainer.evaluate()
print("Evaluation results:", results)

# ===== Save model =====
trainer.save_model("./distilBERT_age_regression_model")
