import pandas as pd
from pathlib import Path



def load_data (data_path):
    """Load dataset from csv file."""
    df = pd.read_csv (data_path)
    return df

def remove_missing_values (df):
    """Remove rows containing missing values."""
    missing_len = df.isnull().sum().sum()
    df = df.dropna ()
    print (f"{missing_len} missing values detected and removed.")
    return df

def remove_duplicate (df):
    """Remove duplicate rows."""
    dup_len = df.duplicated().sum()
    df = df.drop_duplicates()
    print (f"{dup_len} duplicated samples detected and removed.")
    return df

def remove_empty_sentences (df):
    """Remove rows with empty text."""
    df = df [df["sentence"].str.strip() !=""]
    return df

def clean_whitespaces (df):
    """Remove extra whitespaces."""
    df["sentence"] = df["sentence"].str.strip()
    df["sentence"] = df["sentence"].str.replace (r"\s+"," ", regex=True)
    return df

def encode_labels (df):
    """Convert sentiment labels to integers."""
    target_map = {"negative": 0, "positive": 1, "neutral": 2}
    df["target"] = df["label"].map(target_map)
    return df

def save_data (df, output_path):
    """Saved processed dataset."""
    df.to_csv (output_path, index=False)
    return "clean dataset has been saved."



def main():

    print ("Starting Preprocessing...")
    data_path = Path(__file__).parent.parent/"data"/"raw"/"financial_phrasebank.csv"
    output_path = Path(__file__).parent.parent/"data"/"processed"/"financial_phrasebank_clean.csv"

    df = load_data (data_path)
    df = remove_duplicate (df)
    df = remove_missing_values (df)
    df = remove_empty_sentences (df)
    df = clean_whitespaces (df)
    df = encode_labels (df)

    save_data (df, output_path)

    print ("Preprocessing completed successfully.")
    print (df.head())
    print (f"Dataset shape: {df.shape}")


if __name__ == "__main__":
    main ()









