import torch


# === Config ===
EVAL_FILE = "./to_eval"
MODEL_NAME = "distilbert-base-uncased"
AGE_PATH =  "../MODELS/age/regressionFull/distilBERT_age_regression_model/model.safetensors"
GENDER_PATH = "../MODELS/gender/regressionFull/distilBERT_gender_regression_model/model.safetensors"
POLITICAL_PATH = "../MODELS/political/regressionFullShort/distilBERT_political_regression_model/model.safetensors"
MBTI_PATH = "../MODELS/mbti/classificationFull/distilBERT_mbti_classification_model/model.safetensors"
LANGUAGE_PATH = "../MODELS/language/classificationFull/distilBERT_language_classification_model/model.safetensors"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# === Tokenizer ===
from transformers import DistilBertTokenizerFast

tokenizer = DistilBertTokenizerFast.from_pretrained(MODEL_NAME)


# === Generic Functions ===
from safetensors.torch import load_file

def load_model(model, path):
    model.load_state_dict(load_file(path))
    model.to(DEVICE)
    model.eval()

def get_predictions(model, texts):
    inputs = tokenizer(
        texts,
        return_tensors="pt",
        truncation=True,
        padding=True,
        max_length=256
    )
    inputs = {k: v.to(DEVICE) for k, v in inputs.items()}

    with torch.no_grad():
        preds = model(**inputs)["logits"].cpu()
    return preds

def clear_model(model):
    del model
    torch.cuda.empty_cache()


# === Regression ===
from torch import nn
from transformers import DistilBertModel

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

def run_regression(path, texts):
    model = DistilBertRegression(MODEL_NAME)
    load_model(model, path)
    preds = get_predictions(model, texts).tolist()
    clear_model(model)
    return preds


# === Classification ===
from transformers import DistilBertForSequenceClassification

def run_classification(path, texts, n, k):
    model = DistilBertForSequenceClassification.from_pretrained(MODEL_NAME, num_labels=n)
    load_model(model, path)
    preds = get_predictions(model, texts)
    probs = torch.softmax(preds, dim=-1)
    probs, preds = probs.topk(k=k, dim=-1)
    preds = [[int(x) for x in row] for row in preds.tolist()]
    probs = [[float(x) for x in row] for row in probs.tolist()]
    clear_model(model)
    return preds, probs


# === Run Models ===
with open(EVAL_FILE, "r") as f:
    texts = f.read().splitlines()

predictions = {}

# regression
models = {
    "age": AGE_PATH,
    "gender": GENDER_PATH,
    "political": POLITICAL_PATH
}

for name, path in models.items():
    predictions[name] = run_regression(path, texts)

# classification
mbti_labels = {0: "ISTJ", 1: "ISFJ", 2: "INFJ", 3: "INTJ", 4: "ISTP", 5: "ISFP", 6: "INFP", 7: "INTP", 8: "ESTP", 9: "ESFP", 10: "ENFP", 11: "ENTP", 12: "ESTJ", 13: "ESFJ", 14: "ENFJ", 15: "ENTJ"}
predictions["mbti"] = run_classification(MBTI_PATH, texts, 16, 3)

import json
with open("../DATA/language/label_to_num.json") as f:
    lang_labels = json.load(f)
lang_labels = {v : k for k, v in lang_labels.items()}
predictions["language"] = run_classification(LANGUAGE_PATH, texts, 20, 3)

# === Present Results ===
for i in range(len(texts)):
    age, gender, political = [predictions[k][i] for k in ["age", "gender", "political"]]
    mbti = predictions["mbti"][0][i]
    prob_mbti = predictions["mbti"][1][i]
    lang = predictions["language"][0][i]
    prob_lang = predictions["language"][1][i]

    print(f"Text: {texts[i]}")
    
    # age
    print(f"Age: {round(age)}")

    # gender
    print(f"Gender: {({0: "female", 1: "male"}[round(gender)])} ({gender})")

    # political
    print(f"Political: {({0: "left", 1: "center", 2: "right"}[round(political)])} ({political})")

    # mbti
    print(f"MBTI: {mbti_labels[mbti[0]]} ({prob_mbti[0]}) or {mbti_labels[mbti[1]]} ({prob_mbti[1]}) or {mbti_labels[mbti[2]]} ({prob_mbti[2]})")

    # language
    print(f"Native Language: {lang_labels[lang[0]]} ({prob_lang[0]}) or {lang_labels[lang[1]]} ({prob_lang[1]}) or {lang_labels[lang[2]]} ({prob_lang[2]})")

    print("=" * 40)
    