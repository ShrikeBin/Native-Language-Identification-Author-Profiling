import torch
import pandas as pd
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
LABEL_COL = "political_view"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ===== Load full data =====
train_df = pd.read_parquet("../../../DATA/political/short_train.parquet")
test_df = pd.read_parquet("../../../DATA/political/short_test.parquet")

# Labels can stay 0/1/2 but must be float for regression
label_map = {"left": 0, "center": 1, "right": 2}
train_df[LABEL_COL] = train_df[LABEL_COL].map(label_map).astype(float)
test_df[LABEL_COL] = test_df[LABEL_COL].map(label_map).astype(float)

# ===== Tokenizer =====
tokenizer = DistilBertTokenizerFast.from_pretrained(MODEL_NAME)

def tokenize(batch):
    texts = [str(t) for t in batch[TEXT_COL]]
    return tokenizer(texts, padding="max_length", truncation=True, max_length=256)

# ===== HuggingFace Dataset =====
train_dataset = HFDataset.from_pandas(train_df).shuffle(seed=42)
test_dataset = HFDataset.from_pandas(test_df)
# Tokenize
train_dataset = train_dataset.map(tokenize, batched=True, num_proc=8)
test_dataset = test_dataset.map(tokenize, batched=True, num_proc=8)
# Rename
train_dataset = train_dataset.rename_column(LABEL_COL, "labels")
test_dataset = test_dataset.rename_column(LABEL_COL, "labels")

# ===== Model =====
# single output for regression
model = DistilBertForSequenceClassification.from_pretrained(MODEL_NAME, num_labels=1)
model.to(DEVICE)

# ===== Metrics =====
mse = load("mse")
def compute_metrics(eval_pred):
    logits, labels = eval_pred
    logits = logits.squeeze(-1)
    return mse.compute(predictions=logits, references=labels)

# ===== Data collator =====
data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

# ===== Training args =====
training_args = TrainingArguments(
    output_dir="./results_political",
    learning_rate=2e-5,
    per_device_train_batch_size=96,
    per_device_eval_batch_size=96,
    gradient_accumulation_steps=2,
    num_train_epochs=2,
    weight_decay=0.01,
    eval_strategy="epoch",
    save_strategy="epoch",
    load_best_model_at_end=True,
    push_to_hub=False,
    report_to="none",
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
trainer.save_model("./distilBERT_political_regression_model")
