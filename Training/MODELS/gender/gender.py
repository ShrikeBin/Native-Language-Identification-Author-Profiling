import os
import torch
from transformers import Trainer, TrainingArguments, DistilBertForSequenceClassification, DistilBertTokenizerFast, DataCollatorWithPadding
from datasets import load_from_disk
from sklearn.preprocessing import LabelEncoder
from evaluate import load

# ===== Config =====
NUM_EPOCHS = 4
BATCH_SIZE = 8
MODEL_NAME = "distilbert-base-uncased"
OUTPUT_DIR = "./results_gender"
LOG_DIR = "./logs_gender"
TOKENIZED_TRAIN_DIR = "./tokenized_train"
TOKENIZED_TEST_DIR = "./tokenized_test"

# ===== Device =====
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

# ===== Load tokenized datasets =====
if os.path.exists(TOKENIZED_TRAIN_DIR) and os.path.exists(TOKENIZED_TEST_DIR):
    print("Loading tokenized datasets from disk...")
    train_dataset = load_from_disk(TOKENIZED_TRAIN_DIR)
    test_dataset = load_from_disk(TOKENIZED_TEST_DIR)
    print("Datasets loaded.")
else:
    raise FileNotFoundError("Tokenized datasets not found. Run tokenization script first.")

# ===== Model =====
# Infer number of labels from dataset
num_labels = len(set(train_dataset['labels']))
model = DistilBertForSequenceClassification.from_pretrained(MODEL_NAME, num_labels=num_labels)
model.to(device)

# ===== Metrics =====
accuracy = load("accuracy")

def compute_metrics(eval_pred):
    logits, labels = eval_pred
    preds = torch.argmax(torch.tensor(logits), dim=1)
    return accuracy.compute(predictions=preds, references=labels)

# ===== Data Collator =====
tokenizer = DistilBertTokenizerFast.from_pretrained(MODEL_NAME)
data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

# ===== Training Arguments =====
training_args = TrainingArguments(
    output_dir=OUTPUT_DIR,
    eval_strategy="epoch",
    save_strategy="epoch",
    learning_rate=2e-5,
    per_device_train_batch_size=BATCH_SIZE,
    per_device_eval_batch_size=BATCH_SIZE,
    num_train_epochs=NUM_EPOCHS,
    weight_decay=0.01,
    logging_dir=LOG_DIR,
    logging_steps=50,
    load_best_model_at_end=True,
    metric_for_best_model="accuracy",
    push_to_hub=False,
    fp16=torch.cuda.is_available(),
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

# ===== Train =====
trainer.train()

# ===== Evaluate =====
results = trainer.evaluate()
print("Evaluation results:", results)

# ===== Save =====
trainer.save_model("./distilBERT_gender_model")
