from pathlib import Path
from datasets import load_dataset
from transformers import AutoTokenizer
from config import MODEL_NAME, MAX_LENGTH



class TextTokenizer:
    def __init__(self, model_name: str = MODEL_NAME):
        self.tokenizer = AutoTokenizer.from_pretrained (model_name)

    def tokenize (self, examples):
        return self.tokenizer (examples["sentence"], truncation=True, padding="max_length", max_length=MAX_LENGTH)



def load_splits (data_dir: Path):
    dataset = load_dataset (
        "csv",
        data_files= {
            "train": str (data_dir/"train_df.csv"),
            "validation": str (data_dir/"val_df.csv"),
            "test": str (data_dir/"test_df.csv")
        },
    )
    return dataset


def tokenize_dataset (dataset, tokenizer):
    tokenized_dataset = dataset.map (tokenizer.tokenize, batched=True, desc="Tokenizing Dataset",)
    return tokenized_dataset


def main():
    data_dir = Path (__file__).parent.parent/"data"/"processed"
    output_dir = Path (__file__).parent.parent/"data"/"tokenized"
    tokenizer = TextTokenizer()

    dataset = load_splits (data_dir)
    tokenized_dataset = tokenize_dataset (dataset, tokenizer)

    tokenized_dataset.save_to_disk (output_dir)

    print (f"Saved to {output_dir}")
    


if __name__ == "__main__":
    main()
