# predict_gender.py
import torch
from transformers import DistilBertTokenizerFast, DistilBertForSequenceClassification
from sklearn.preprocessing import LabelEncoder

# ===== Config =====
MODEL_PATH = "./distilBERT_gender_model"
INPUT_FILE = "file.txt"
MAX_LEN = 256
BATCH_SIZE = 32  # adjust if GPU memory is tight

# ===== Load model and tokenizer =====
tokenizer = DistilBertTokenizerFast.from_pretrained(MODEL_PATH)
model = DistilBertForSequenceClassification.from_pretrained(MODEL_PATH)
model.eval()

# ===== Label mapping (replace with your training labels) =====
import numpy as np

le = LabelEncoder()
le.classes_ = np.array(["female", "male"])  # must be np.array, not list


# ===== Read input text =====
with open(INPUT_FILE, "r", encoding="utf-8") as f:
    texts = [f.read()]

# ===== Batch inference =====
pred_labels = []
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

for i in range(0, len(texts), BATCH_SIZE):
    batch_texts = texts[i:i+BATCH_SIZE]
    inputs = tokenizer(
        batch_texts,
        padding="max_length",
        truncation=True,
        max_length=MAX_LEN,
        return_tensors="pt"
    )
    inputs = {k: v.to(device) for k, v in inputs.items()}

    with torch.no_grad():
        outputs = model(**inputs)
        logits = outputs.logits
        preds = torch.argmax(logits, dim=1)
        pred_labels.extend(le.inverse_transform(preds.cpu().numpy()))

# ===== Print predictions =====
for text, label in zip(texts, pred_labels):
    print(f"{label}: {text}")
