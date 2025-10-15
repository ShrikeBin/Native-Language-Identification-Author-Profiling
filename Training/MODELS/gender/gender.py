import pandas as pd
import torch
from transformers import (
    DistilBertTokenizerFast,
    DistilBertForSequenceClassification,
    Trainer,
    TrainingArguments
)
from sklearn.preprocessing import LabelEncoder
from datasets import Dataset as HFDataset

# Setup GPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

# Load data
train_df = pd.read_parquet("../DATA/gender/train.parquet")
test_df = pd.read_parquet("../DATA/gender/test.parquet")

# Parquet columns
text_col = "text"
label_col = "gender"

# Encode gender numerically ( 0 | 1 )
le = LabelEncoder()
train_df[label_col] = le.fit_transform(train_df[label_col])
test_df[label_col] = le.transform(test_df[label_col])

# Tokenizer
tokenizer = DistilBertTokenizerFast.from_pretrained("distilbert-base-uncased")

def tokenize(batch):
    return tokenizer(batch[text_col], padding="max_length", truncation=True, max_length=256)

# Convert to HuggingFace Dataset format
train_dataset = HFDataset.from_pandas(train_df)
test_dataset = HFDataset.from_pandas(test_df)

train_dataset = train_dataset.map(tokenize, batched=True)
test_dataset = test_dataset.map(tokenize, batched=True)

train_dataset = train_dataset.rename_column(label_col, "labels")
test_dataset = test_dataset.rename_column(label_col, "labels")

train_dataset.set_format("torch", columns=["input_ids", "attention_mask", "labels"])
test_dataset.set_format("torch", columns=["input_ids", "attention_mask", "labels"])

# Model creation
num_labels = len(le.classes_)
model = DistilBertForSequenceClassification.from_pretrained(
    "distilbert-base-uncased",
    num_labels=num_labels
)
model.to(device)

# Training arguments
training_args = TrainingArguments(
    output_dir="./results_gender",
    evaluation_strategy="epoch",
    save_strategy="epoch",
    learning_rate=2e-5,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=16,
    num_train_epochs=3,
    weight_decay=0.01,
    logging_dir="./logs_gender",
    load_best_model_at_end=True,
    metric_for_best_model="accuracy",
    push_to_hub=False,
    fp16=torch.cuda.is_available(),
)

# Metrics
from evaluate import load
accuracy = load("accuracy")

def compute_metrics(eval_pred):
    logits, labels = eval_pred
    preds = torch.argmax(torch.tensor(logits), dim=1)
    return accuracy.compute(predictions=preds, references=labels)

# Create Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=test_dataset,
    tokenizer=tokenizer,
    compute_metrics=compute_metrics,
)

# Train
trainer.train()

# Evaluate
results = trainer.evaluate()
print("Evaluation results:", results)

# Save
trainer.save_model("./distilBERT_gender_model")
