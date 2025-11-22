import pandas as pd
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
import numpy as np
from sklearn.metrics import accuracy_score
from tqdm import tqdm

df = pd.read_parquet("test_gender.parquet")
texts = df["text"].tolist()
true_labels = df["gender"].tolist()

model_name = "malcolm/REA_GenderIdentification_v1"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSequenceClassification.from_pretrained(model_name)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)
model.eval()

def predict_batch(texts_batch):
    inputs = tokenizer(texts_batch, return_tensors="pt", padding=True, truncation=True)
    inputs = {k: v.to(device) for k, v in inputs.items()}
    with torch.no_grad():
        logits = model(**inputs).logits
    probs = torch.softmax(logits, dim=-1)
    return probs.cpu().numpy()

batch_size = 96
all_probs = []

for i in tqdm(range(0, len(texts), batch_size), desc="Evaluating"):
    batch_texts = texts[i:i+batch_size]
    batch_probs = predict_batch(batch_texts)
    all_probs.append(batch_probs)

all_probs = np.vstack(all_probs)
pred_indices = np.argmax(all_probs, axis=1)

class_map = {0: "female", 1: "male"}
pred_labels = [class_map[i] for i in pred_indices]

df["pred_label"] = pred_labels
df["pred_prob"] = all_probs.max(axis=1)

acc = accuracy_score(true_labels, pred_labels)
print(f"Accuracy: {acc:.4f}")
