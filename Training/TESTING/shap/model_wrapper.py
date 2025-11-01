import torch
import shap
import numpy as np
from Training.TESTING.shap.regression_head import (
    DistilBertRegression,
    RoBertRegression
)
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification
)
from peft import PeftModel

# ===== Device =====
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ===== Train Model Maps =====
model_maps = {
    'Full': "distilbert-base-uncased",
    'LoRA': "distilbert-base-uncased",
    'Baseline': "distilbert-base-uncased",
    'RoBERT': "roberta-base",
    'RoBERTLarge': "roberta-large",
    'MpNet': "sentence-transformers/all-mpnet-base-v2",
}

# ===== Trait Label Maps =====
label_maps = {
    'gender': {0: "female", 1: "male"},
    'political': {0: "left", 1: "center", 2: "right"},
    'mbti': {0: "ISTJ", 1: "ISFJ", 2: "INFJ", 3: "INTJ", 4: "ISTP", 5: "ISFP", 6: "INFP", 7: "INTP", 8: "ESTP", 9: "ESFP", 10: "ENFP", 11: "ENTP", 12: "ESTJ", 13: "ESFJ", 14: "ENFJ", 15: "ENTJ"},
    'language': {0: "English", 1: "German", 2: "Nordic", 3: "French", 4: "Italian", 5: "Portuguese", 6: "Spanish", 7: "Russian", 8: "Polish", 9: "Other Slavic", 10: "Turkic", 11: "Chinese", 12: "Vietnamese", 13: "Koreanic", 14: "Japonic", 15: "Tai", 16: "Indonesian", 17: "Uralic", 18: "Arabic", 19: "Indo-Iranian"},
}

# ===== Model Wrapper =====
from safetensors.torch import load_file
class Model:
    def __init__(self, trait_name, head_type, train):

        # === Basic Config ===
        self.name = f"{trait_name} ({train})"
        self.type = head_type
        self.tokenizer = AutoTokenizer.from_pretrained(model_maps[train])
        self.label_map = label_maps.get(trait_name, None)

        # === Model Base ===
        path = f"Training/MODELS/{trait_name}/{head_type}{train}/model"
        match head_type:
            # Transformers Classifiers
            case 'classification' | 'classreg':
                self.model = AutoModelForSequenceClassification.from_pretrained(path)
            # Custom Regression Head
            case 'regression':
                match train:
                    case 'RoBERT':
                        self.model = RoBertRegression(model_maps[train])
                    case _:
                        self.model = DistilBertRegression(model_maps[train])
                # Load State Dict or LoRA adapters by hand
                match train:
                    case 'LoRA':
                        self.model = PeftModel.from_pretrained(self.model, path)
                        self.model.merge_adapter()
                    case _:
                        self.model.load_state_dict(load_file(f"{path}/model.safetensors"))
            case _:
                raise KeyError(f"Unknown head type: {head_type}")

        # === Prepare Model for Inference ===
        self.model.to(DEVICE)
        self.model.eval()

        # === Shap Explainer ===
        self.explainer = shap.Explainer(
            self.predict,
            shap.maskers.Text(self.tokenizer),
            output_names=(list(self.label_map.values()) if head_type == 'classification' else None)
        )

        print(f"Initialized model {self.name}")

    def predict(self, text):

        # === Preprocess Input ===
        text = [str(t) for t in text]
        inputs = self.tokenizer(text, return_tensors="pt", padding=True, truncation=True, max_length=256)
        inputs = {k: v.to(DEVICE) for k, v in inputs.items()}

        # === Inference ===
        with torch.no_grad():
            output = self.model(**inputs)["logits"].cpu()

        # === Customized Output ===
        match self.type:
            case 'classification':
                output = torch.softmax(output, dim=-1)
            case 'classreg':
                output = output.squeeze(-1)
            
        return output.numpy()

    def prediction_string(self, text):

        # === String Base ===
        prediction_string = f"{self.name}: "

        # === Inference ===
        pred = self.predict([text])[0]

        # === Customized String ===
        match self.type:
            case 'classification':
                labels = np.argsort(pred)[::-1]
                probs = pred[labels]
                last_index = np.searchsorted(np.cumsum(probs), 0.5) + 1
                for i in range(last_index):
                    prediction_string += f"{self.label_map[labels[i]]} ({100 * probs[i]:.2f}%) "
            case 'regression' | 'classreg':
                if self.label_map == None:
                    prediction_string += f"{pred:.2f}"
                else:
                    prediction_string += f"{self.label_map[round(pred)]} ({pred:.2f})"
        return prediction_string
    
    def explain(self, text):

        # === Inference ===
        pred = self.predict([text])[0]
        shap_values = self.explainer([text])

        # === Customized Explanation ===
        match self.type:
            case 'classification':
                pred = np.argmax(pred)
                return shap_values.base_values[0][pred], shap_values.values[0][:,pred], shap_values.data[0]
            case _:
                return shap_values.base_values[0], shap_values.values[0], shap_values.data[0]

# ===== Load Models =====
def load_models():
    models = [
        Model('gender', 'regression', 'Full'),
        Model('gender', 'regression', 'RoBERT'),
    ]
    return models
    