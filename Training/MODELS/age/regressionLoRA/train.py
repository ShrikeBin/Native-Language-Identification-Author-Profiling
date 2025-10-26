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
from peft import (
    LoraConfig,
    get_peft_model
)

# ===== CONFIG =====
MODEL_NAME = "distilbert-base-uncased"
TEXT_COL = "text"
LABEL_COL = "age"
MAX_LEN = 256
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ===== Load full data =====
train_df = pd.read_parquet("../../../DATA/age/train.parquet")
test_df = pd.read_parquet("../../../DATA/age/test.parquet")

# Convert labels to float for regression
train_df[LABEL_COL] = train_df[LABEL_COL].astype(float)
test_df[LABEL_COL] = test_df[LABEL_COL].astype(float)

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


# ===== Model =====
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

# ===== LoRA Config =====
config = LoraConfig(
    r=16,
    lora_alpha=32,
    target_modules=["q_lin", "k_lin", "v_lin"], # typical transformer stuff
    lora_dropout=0.1,
    modules_to_save=["regressor"],
)
model = get_peft_model(model, config)

model.print_trainable_parameters()

# ===== Train regression Head =====
for name, param in model.named_parameters():
    if "regressor" in name:
        param.requires_grad = True

model.print_trainable_parameters()

model.to(DEVICE)

# ===== Metrics =====
mse = load("mse")
def compute_metrics(eval_pred):
    logits, labels = eval_pred
    return mse.compute(predictions=logits, references=labels)

# ===== Data collator =====
data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

# ===== Training args =====
training_args = TrainingArguments(
    output_dir="./results",
    eval_strategy="epoch",
    save_strategy="epoch",
    learning_rate=1e-4, # apparently for lora you go bigger
    per_device_train_batch_size=96,
    per_device_eval_batch_size=96,
    num_train_epochs=4,
    weight_decay=0.01,
    load_best_model_at_end=True,
    fp16=torch.cuda.is_available(), # important for memory
    gradient_accumulation_steps=2,
)

# ===== Trainer =====
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=test_dataset,
    data_collator=data_collator,
    compute_metrics=compute_metrics,
)

# ===== Train & Evaluate =====
trainer.train()
results = trainer.evaluate()
print("Evaluation results:", results)

# ===== Save model =====
model.save_pretrained("./model")
