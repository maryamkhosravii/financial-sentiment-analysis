from pathlib import Path
import random
import numpy as np
import torch
from torch.optim import AdamW
from transformers import get_linear_schedule_with_warmup
from config import RANDOM_SEED, DEVICE, BATCH_SIZE, WARMUP_RATIO, WEIGHT_DECAY, LEARNING_RATE, NUM_EPOCHS, MODEL_SAVE_DIR, MODEL_SAVE_NAME
from data_module import DatasetModule
from model import FinancialSentimentModel
from tqdm.auto import tqdm
from sklearn.metrics import accuracy_score, f1_score



def set_seed (seed: int):
    random.seed (seed)
    np.random.seed (seed)
    torch.manual_seed (seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed (seed)
        torch.cuda.manual_seed_all (seed)



def load_data ():
    data_module = DatasetModule (dataset_path=Path(__file__).parent.parent/"data"/"tokenized")

    data_module.load_data()
    data_module.setup()

    train_loader = data_module.train_dataloader()
    val_loader = data_module.val_dataloader()
    test_loader = data_module.test_dataloader()

    return train_loader, val_loader, test_loader



def load_model ():
    model_builder = FinancialSentimentModel()
    model = model_builder.get_model()
    model.to(DEVICE)
    return model



def build_optimizer (model):
    optimizer = AdamW (
        params=model.parameters(),
        lr = LEARNING_RATE,
        weight_decay=WEIGHT_DECAY
    )
    return optimizer



def build_scheduler (optimizer, train_loader,):

    total_training_steps = (len(train_loader)*NUM_EPOCHS)
    warmup_steps = int(total_training_steps*WARMUP_RATIO)

    scheduler = get_linear_schedule_with_warmup (optimizer=optimizer, num_warmup_steps=warmup_steps, num_training_steps=total_training_steps)

    return scheduler



def train_one_epoch (model, train_loader, optimizer, scheduler, device):

    model.train()

    total_loss = 0

    progress_bar = tqdm (train_loader, desc="Training")
    for batch in progress_bar:
        batch = {key: value.to(device) for key, value in batch.items()}

        optimizer.zero_grad()
        outputs = model (**batch)
        loss = outputs.loss
        logits = outputs.logits
        loss.backward()
        torch.nn.utils.clip_grad_norm_ (model.parameters(), max_norm=1.0)
        optimizer.step()
        scheduler.step()
        total_loss += loss.item()
        progress_bar.set_postfix (loss=loss.item())

    average_loss = total_loss / len(train_loader)
    return average_loss



def validate (model, val_loader, device):

    model.eval()
    total_loss = 0
    predictions = []
    labels = []

    with torch.no_grad():
        for batch in tqdm (val_loader, desc="Validation"):
            batch = {key: value.to(device) for key, value in batch.items()}

            outputs = model (**batch)
            loss = outputs.loss
            total_loss += loss.item()
            logits = outputs.logits

            preds = torch.argmax (logits, dim=1)
            predictions.extend (preds.cpu().numpy())

            labels.extend (batch["labels"].cpu().numpy())

    accuracy = accuracy_score (labels, predictions)
    f1 = f1_score (labels, predictions, average="macro")
    average_loss = total_loss / len(val_loader)
    metrics = {"loss": average_loss, "accuracy": accuracy, "f1": f1}
    return metrics



def save_checkpoint (model, optimizer, epoch, metrics):

    MODEL_SAVE_DIR.mkdir (parents=True, exist_ok=True)

    checkpoint = {
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "validation_loss": metrics['loss'],
        "accuracy": metrics ["f1"],
    }
    save_path = MODEL_SAVE_DIR / MODEL_SAVE_NAME

    torch.save (checkpoint, save_path)
    print (f"Model saved to: {save_path}")



def train():
    set_seed (RANDOM_SEED)
    train_loader, val_loader, test_loader = load_data()
    model = load_model()
    optimizer = build_optimizer(model)
    scheduler = build_scheduler (optimizer, train_loader)

    best_val_loss = float("inf")

    for epoch in range (NUM_EPOCHS):
        print ("=" * 60)
        print (f"Epoch {epoch + 1} / {NUM_EPOCHS}")

        train_loss = train_one_epoch (model=model, train_loader=train_loader, optimizer=optimizer, scheduler=scheduler, device=DEVICE)

        metrics = validate (model=model, val_loader=val_loader, device=DEVICE)

        print (f"Train Loss: {train_loss: .4f}")
        print (f"Validation Loss: {metrics['loss']: .4f}")
        print (f"Accuracy: {metrics['accuracy']: .4f}")
        print (f"Macro F1: {metrics['f1']: .4f}")

        if metrics["loss"] < best_val_loss:
            best_val_loss = metrics['loss']
            save_checkpoint (model=model, optimizer=optimizer, epoch=epoch+1, metrics=metrics)

    print ("=" * 60)
    print ("Training Finished.")






def main():
    train()

if __name__ == "__main__":
    main()



