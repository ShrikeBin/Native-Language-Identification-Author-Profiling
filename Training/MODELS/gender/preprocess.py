import os
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from datasets import Dataset as HFDataset
from transformers import DistilBertTokenizerFast

# ===== CONFIG =====
TEXT_COL = "text"
LABEL_COL = "gender"
MAX_LEN = 256
MODEL_NAME = "distilbert-base-uncased"
NUM_PROC = 8  # adjust to your CPU cores
TOKENIZED_TRAIN_DIR = "./tokenized_train"
TOKENIZED_TEST_DIR = "./tokenized_test"

# ===== Load data =====
train_df = pd.read_parquet("../../DATA/gender/train.parquet")
test_df = pd.read_parquet("../../DATA/gender/test.parquet")

# Encode labels
le = LabelEncoder()
train_df[LABEL_COL] = le.fit_transform(train_df[LABEL_COL])
test_df[LABEL_COL] = le.transform(test_df[LABEL_COL])

# ===== Tokenizer =====
tokenizer = DistilBertTokenizerFast.from_pretrained(MODEL_NAME)

def tokenize(batch):
    texts = [str(t) for t in batch[TEXT_COL]]  # safe conversion
    return tokenizer(texts, padding="max_length", truncation=True, max_length=MAX_LEN)

# ===== Convert to HuggingFace dataset =====
train_dataset = HFDataset.from_pandas(train_df).shuffle(seed=42)
test_dataset = HFDataset.from_pandas(test_df)

# ===== Tokenize =====
print("Tokenizing training dataset...")
train_dataset = train_dataset.map(tokenize, batched=True, num_proc=NUM_PROC)

print("Tokenizing test dataset...")
test_dataset = test_dataset.map(tokenize, batched=True, num_proc=NUM_PROC)

# ===== Rename label column =====
train_dataset = train_dataset.rename_column(LABEL_COL, "labels")
test_dataset = test_dataset.rename_column(LABEL_COL, "labels")

# ===== Set PyTorch format =====
train_dataset.set_format("torch", columns=["input_ids", "attention_mask", "labels"])
test_dataset.set_format("torch", columns=["input_ids", "attention_mask", "labels"])

# ===== Save tokenized datasets =====
train_dataset.save_to_disk(TOKENIZED_TRAIN_DIR)
test_dataset.save_to_disk(TOKENIZED_TEST_DIR)

print("Tokenization complete. Datasets saved to disk.")
