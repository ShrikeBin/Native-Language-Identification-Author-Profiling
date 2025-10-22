import torch
from torch import nn
from safetensors.torch import load_file
from transformers import (
    DistilBertModel,
    DistilBertTokenizerFast
)

# === Change if need be ===
TENSORS_PATH = "../MODELS/political/regressionFullShort/distilBERT_political_regression_model/model.safetensors"
MODEL_NAME = "distilbert-base-uncased"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# === Same tokenizer ===
tokenizer = DistilBertTokenizerFast.from_pretrained(MODEL_NAME)

# === Same head ===
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

# === Load Model and weights ===
model = DistilBertRegression(MODEL_NAME)
state_dict = load_file(TENSORS_PATH)
model.load_state_dict(state_dict)
model.eval() # set into evaluation mode
model.to(DEVICE) # load into gpu

# === Handle input stream line by line ===
# can be read from a pipe instead, keeping the model loaded
print(f"=== Reading Model: {TENSORS_PATH} ===")
with open("to_eval", "r") as f:
    for line in f:
        text = line.strip()
        if not text:
            continue # skip empty lines

        # tokenize single line
        inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=256)
        inputs = {k: v.to(DEVICE) for k, v in inputs.items()}

        # === Forward pass ===
        with torch.no_grad():
            pred = model(**inputs)["logits"].item() # logits is one elem. list, because regression
        
        print(f"Predicted: {pred:.2f}, Text: {text}") # round() if you want an integer
