from pathlib import Path
import torch
from torch.utils.data import DataLoader
from datasets import load_from_disk
from config import BATCH_SIZE, NUM_WORKERS


class DatasetModule:
    def __init__ (self, dataset_path):

        self.dataset_path = dataset_path
        self.batch_size = BATCH_SIZE
        self.num_workers = NUM_WORKERS
        self.dataset = None


    def load_data (self):
        """Load tokenized dataset from disk."""
        self.dataset = load_from_disk (self.dataset_path)


    def setup (self):
        """Prepare dataset for PyTorch."""
        self.dataset = self.dataset.rename_column ("target", "labels")
        self.dataset = self.dataset.remove_columns (["label", "sentence"])

        columns = ["input_ids", "attention_mask", "labels",]
        if "token_type_ids" in self.dataset["train"].column_names:
            columns.insert (2, "token_type_ids")

        self.dataset.set_format (type="torch", columns=columns)



    def train_dataloader (self):
        return DataLoader (self.dataset["train"], batch_size=self.batch_size, shuffle=True, num_workers=self.num_workers, pin_memory=torch.cuda.is_available())
    
    def val_dataloader (self):
        return DataLoader (self.dataset["validation"], batch_size=self.batch_size, shuffle=False, num_workers=self.num_workers, pin_memory=torch.cuda.is_available())
    
    def test_dataloader (self):
        return DataLoader (self.dataset["test"], batch_size=self.batch_size, shuffle=False, num_workers=self.num_workers, pin_memory=torch.cuda.is_available())

