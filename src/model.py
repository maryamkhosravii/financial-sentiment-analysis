from transformers import AutoConfig, AutoModelForSequenceClassification
from src.config import MODEL_NAME, NUM_LABELS, MODEL_SAVE_DIR, MODEL_SAVE_NAME, DEVICE
import torch



class FinancialSentimentModel:

    def __init__(self):
        self.config = AutoConfig.from_pretrained (pretrained_model_name_or_path=MODEL_NAME, num_labels=NUM_LABELS)
        self.model = AutoModelForSequenceClassification.from_pretrained (pretrained_model_name_or_path=MODEL_NAME, num_labels=NUM_LABELS)

    def get_model (self):
        return self.model
    
    def print_model_summary (self):
        print (self.model)

    def load_checkpoint (self):
        checkpoint = torch.load (MODEL_SAVE_DIR/MODEL_SAVE_NAME, map_location=DEVICE)
        self.model.load_state_dict (checkpoint["model_state_dict"])
        self.model.to(DEVICE)
        self.model.eval()
        return self.model




""" #Test-----------------------------
def main():
    model_builder = FinancialSentimentModel()
    model_builder.get_model()
    model_builder.print_model_summary()

    if __name__ == "__main__":
        main()
"""