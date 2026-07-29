import torch
from transformers import AutoTokenizer
from src.model import FinancialSentimentModel
from src.config import DEVICE, MAX_LENGTH, MODEL_SAVE_DIR, MODEL_NAME



class InferenceService:
    LABELS = {0: "negative", 1: "positive", 2: "neutral"}

    def __init__(self):
        self.device = DEVICE
        self.tokenizer = AutoTokenizer.from_pretrained (MODEL_NAME)
        self.model = FinancialSentimentModel().load_checkpoint()
        self.model.to(self.device)
        self.model.eval()


    def process (self, text: str):
        encoding = self.tokenizer (text, truncation=True, padding="max_length", max_length=MAX_LENGTH, return_tensors="pt")
        return {k: v.to(self.device) for k,v in encoding.items()}
    

    def predict (self, text: str):
        inputs = self.process (text)
        with torch.no_grad():
            outputs = self.model(**inputs)
            probabilities = torch.softmax (outputs.logits, dim=1)
            confidence, prediction = torch.max (probabilities, dim=1)
        prediction = prediction.item()
        probs = {self.LABELS[i]: round(probabilities[0][i].item(), 4,) for i in range(3)}

        return {
            "sentiment": self.LABELS[prediction],
            "confidence": round (confidence.item(), 4),
            "probabilities": probs
            }

        