import torch
import shap
import numpy as np
from regression_head import DistilBertRegression
from transformers import (
    DistilBertForSequenceClassification,
    DistilBertTokenizerFast
)
from peft import PeftModel

# ===== Config =====
MODEL_NAME = "distilbert-base-uncased"
AGE_PATH = "../../MODELS/age/regressionFull/distilBERT_age_regression_model/model.safetensors"
AGE_PATH2 = "../../MODELS/age/regressionFull/results_age/checkpoint-13479/model.safetensors"
AGE_LORA = "../../MODELS/age/regressionLoRA/model"
GENDER_PATH = "../../MODELS/gender/binaryFull/distilBERT_gender_model/model.safetensors"
GENDER_PATH2 = "../../MODELS/gender/regressionFull/distilBERT_gender_regression_model/model.safetensors"
GENDER_PATH3 = "../../MODELS/gender/regressionFull/results_gender/checkpoint-9396/model.safetensors"
POLITICAL_PATH = "../../MODELS/political/classregFull/distilBERT_political_regression_model/model.safetensors"
POLITICAL_PATH2 = "../../MODELS/political/regressionFullShort/distilBERT_political_regression_model/model.safetensors"
POLITICAL_PATH3 = "../../MODELS/political/regressionFullShort/results_political/checkpoint-2598/model.safetensors"
MBTI_PATH = "../../MODELS/mbti/classificationFull/distilBERT_mbti_classification_model/model.safetensors"
LANGUAGE_FULL = "../../MODELS/language/classificationFull/distilBERT_language_classification_model/model.safetensors"
LANGUAGE_LORA = "../../MODELS/language/classificationLoRA/model"
LANGUAGE_BASELINE = "../../MODELS/language/classificationBaseline/model/model.safetensors"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ===== Tokenizer =====
tokenizer = DistilBertTokenizerFast.from_pretrained(MODEL_NAME)

# ===== Model Wrapper =====
from safetensors.torch import load_file
class Model:
    def __init__(self, name, type, path, load=None, label_map=None):
        self.name = name
        self.type = type
        if type == 'regression':
            self.model = DistilBertRegression(MODEL_NAME)
        elif type == 'classification':
            self.model = DistilBertForSequenceClassification.from_pretrained(
                MODEL_NAME,
                num_labels = len(label_map) # expect map for classifiers
            )
        elif type == 'class-reg':
            self.model = DistilBertForSequenceClassification.from_pretrained(
                MODEL_NAME,
                num_labels=1
            )
        self.label_map = label_map
        self.explainer = shap.Explainer(
            self.predict,
            shap.maskers.Text(tokenizer),
            output_names=(list(label_map.values()) if type == 'classification' else None)
        )

        if load == 'lora':
            self.model = PeftModel.from_pretrained(self.model, path)
            self.model.merge_adapter()
        else:
            self.model.load_state_dict(load_file(path))

        self.model.to(DEVICE)
        self.model.eval()
        print(f"Initialized model {name}")

    def predict(self, text):
        text = [str(t) for t in text]
        inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True, max_length=256)
        inputs = {k: v.to(DEVICE) for k, v in inputs.items()}
        with torch.no_grad():
            output = self.model(**inputs)["logits"].cpu()
        if self.type == 'classification':
            output = torch.softmax(output, dim=-1)
        elif self.type == 'class-reg':
            output = output.squeeze(-1)
        return output.numpy()

    def prediction_string(self, text):
        prediction_string = f"{self.name}: "
        pred = self.predict([text])[0]
        if self.type == 'classification':
            labels = np.argsort(pred)[::-1]
            probs = pred[labels]
            last_index = np.searchsorted(np.cumsum(probs), 0.5) + 1
            for i in range(last_index):
                prediction_string += f"{self.label_map[labels[i]]} ({100 * probs[i]:.2f}%) "
        elif self.type == 'regression' or self.type == 'class-reg':
            if self.label_map == None:
                prediction_string += f"{pred:.2f}"
            else:
                prediction_string += f"{self.label_map[round(pred)]} ({pred:.2f})"
        return prediction_string
    
    def explain(self, text):
        pred = self.predict([text])[0]
        shap_values = self.explainer([text])
        if self.type == 'classification':
            pred = np.argmax(pred)
            return shap_values.base_values[0][pred], shap_values.values[0][:,pred], shap_values.data[0]
        elif self.type == 'regression' or self.type == 'class-reg':
            return shap_values.base_values[0], shap_values.values[0], shap_values.data[0]

# ===== Load Models =====
def load_models():
    models = [
        # Model("age", 'regression', AGE_PATH),
        # Model("age", 'regression', AGE_PATH2),
        # Model("age (LoRA)", 'regression', AGE_LORA, load='lora')
        # Model("gender (c)", 'classification', GENDER_PATH, label_map={0: "female", 1: "male"}),
        # Model("gender (r better)", 'regression', GENDER_PATH3, label_map={0: "female", 1: "male"}),
        # Model("gender (r)", 'regression', GENDER_PATH2, label_map={0: "female", 1: "male"})
        # Model("political (c-r)", 'class-reg', POLITICAL_PATH, label_map={0: "left", 1: "center", 2: "right"}),
        # Model("political (r)", 'regression', POLITICAL_PATH2, label_map={0: "left", 1: "center", 2: "right"}),
        # Model("political (r better)", 'regression', POLITICAL_PATH3, label_map={0: "left", 1: "center", 2: "right"})
        # Model("mbti", 'classification', MBTI_PATH, label_map={0: "ISTJ", 1: "ISFJ", 2: "INFJ", 3: "INTJ", 4: "ISTP", 5: "ISFP", 6: "INFP", 7: "INTP", 8: "ESTP", 9: "ESFP", 10: "ENFP", 11: "ENTP", 12: "ESTJ", 13: "ESFJ", 14: "ENFJ", 15: "ENTJ"}),
        Model("lang (full)", 'classification', LANGUAGE_FULL, label_map={0: "English", 1: "German", 2: "Nordic", 3: "French", 4: "Italian", 5: "Portuguese", 6: "Spanish", 7: "Russian", 8: "Polish", 9: "Other Slavic", 10: "Turkic", 11: "Chinese", 12: "Vietnamese", 13: "Koreanic", 14: "Japonic", 15: "Tai", 16: "Indonesian", 17: "Uralic", 18: "Arabic", 19: "Indo-Iranian"}),
        Model("lang (lora)", 'classification', LANGUAGE_LORA, load='lora', label_map={0: "English", 1: "German", 2: "Nordic", 3: "French", 4: "Italian", 5: "Portuguese", 6: "Spanish", 7: "Russian", 8: "Polish", 9: "Other Slavic", 10: "Turkic", 11: "Chinese", 12: "Vietnamese", 13: "Koreanic", 14: "Japonic", 15: "Tai", 16: "Indonesian", 17: "Uralic", 18: "Arabic", 19: "Indo-Iranian"}),
        Model("lang (baseline)", 'classification', LANGUAGE_BASELINE, label_map={0: "English", 1: "German", 2: "Nordic", 3: "French", 4: "Italian", 5: "Portuguese", 6: "Spanish", 7: "Russian", 8: "Polish", 9: "Other Slavic", 10: "Turkic", 11: "Chinese", 12: "Vietnamese", 13: "Koreanic", 14: "Japonic", 15: "Tai", 16: "Indonesian", 17: "Uralic", 18: "Arabic", 19: "Indo-Iranian"}),
    ]
    return models
    