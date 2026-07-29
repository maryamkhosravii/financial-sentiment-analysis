import torch
from pathlib import Path
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report, confusion_matrix
from config import DEVICE, MODEL_SAVE_DIR, MODEL_SAVE_NAME
from data_module import DatasetModule
from model import FinancialSentimentModel



def load_model():
    model = FinancialSentimentModel().get_model()
    checkpoint = torch.load (MODEL_SAVE_DIR / MODEL_SAVE_NAME, map_location=DEVICE)
    model.load_state_dict (checkpoint["model_state_dict"])
    model.to(DEVICE)
    model.eval()
    return model


def evaluate (model, test_loader, device):
    predictions = []
    labels = []

    with torch.no_grad():
        for batch in test_loader:
            batch = {key: value.to(device) for key, value in batch.items()}

            outputs = model (**batch)
            logits = outputs.logits
            preds = torch.argmax (logits, dim=1)
            predictions.extend (preds.cpu().numpy())
            labels.extend (batch["labels"].cpu().numpy())

    accuracy = accuracy_score (labels, predictions)
    precision = precision_score (labels, predictions, average="macro")
    recall = recall_score (labels, predictions, average="macro")
    f1 = f1_score (labels, predictions, average="macro")

    report = classification_report (labels, predictions)
    cm = confusion_matrix (labels, predictions)

    return {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "classification_report": report,
        "confusion_matrix": cm
    }



def print_metrics (metrics):
    print ("=" * 60)
    print (f"Accuracy: {metrics['accuracy']: .4f}")
    print (f"Recall: {metrics['recall']: .4f}")
    print (f"F1: {metrics['f1']: .4f}")
    print ()
    print (metrics['classification_report'])
    print ()
    print (metrics['confusion_matrix'])



def main():

    data_module = DatasetModule(dataset_path=Path(__file__).parent.parent/"data"/"tokenized")
    data_module.load_data()
    data_module.setup()
    test_loader = data_module.test_dataloader()

    model = load_model()

    metrics = evaluate (model, test_loader, DEVICE)

    print_metrics (metrics)


if __name__ == "__main__":
    main()

