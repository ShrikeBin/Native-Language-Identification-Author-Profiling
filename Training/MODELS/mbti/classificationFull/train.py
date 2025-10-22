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
LABEL_COL = "type"
MAX_LEN = 256
BATCH_SIZE = 96
NUM_EPOCHS = 2
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ===== Load full data =====
train_df = pd.read_parquet("../../../DATA/mbti/train.parquet")
test_df = pd.read_parquet("../../../DATA/mbti/test.parquet")

# Convert labels to float (raw ages)
train_df[LABEL_COL] = train_df[LABEL_COL].astype(int)
test_df[LABEL_COL] = test_df[LABEL_COL].astype(int)

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
model = DistilBertForSequenceClassification.from_pretrained(
    MODEL_NAME,
    num_labels=16
)

from safetensors.torch import load_file
model.load_state_dict(load_file("./distilBERT_mbti_classification_model/model.safetensors"))

# ===== Metrics =====
accuracy = load("accuracy")
def compute_metrics(eval_pred):
    logits, labels = eval_pred
    preds = logits.argmax(axis=-1)
    return accuracy.compute(predictions=preds, references=labels)


# ===== Data collator =====
data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

# ===== Training args =====
training_args = TrainingArguments(
    output_dir="./results_mbti",
    eval_strategy="epoch",
    save_strategy="epoch",
    learning_rate=4e-5,
    per_device_train_batch_size=BATCH_SIZE,
    per_device_eval_batch_size=BATCH_SIZE,
    num_train_epochs=NUM_EPOCHS,
    weight_decay=0.01,
    logging_dir="./logs_mbti",
    logging_steps=5000,
    load_best_model_at_end=True,
    metric_for_best_model="accuracy",
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
trainer.save_model("./distilBERT_mbti_classification_model")
