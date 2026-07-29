from sklearn.model_selection import train_test_split
from pathlib import Path
import pandas as pd
from config import RANDOM_SEED


def split_train_test (df):
    train_df_, test_df = train_test_split (df, test_size=0.15, stratify=df["target"], random_state=RANDOM_SEED, shuffle=True)
    print (f"Train/Test split completed.")
    return train_df_, test_df


def train_validation_split (train_df_):
    train_df, val_df = train_test_split (train_df_, test_size=0.20, stratify=train_df_["target"], random_state=RANDOM_SEED, shuffle=True)
    print (f"Train/Validation completed.")
    return train_df, val_df


def main():

    DATA_PATH = Path (__file__).parent.parent/"data"/"processed"/"financial_phrasebank_clean.csv"
    df = pd.read_csv (DATA_PATH)

    TRAIN_OUTPUT_PATH = Path (__file__).parent.parent/"data"/"processed"/"train_df.csv"
    VAL_OUTPUT_PATH = Path (__file__).parent.parent/"data"/"processed"/"val_df.csv"
    TEST_OUTPUT_PATH = Path (__file__).parent.parent/"data"/"processed"/"test_df.csv"


    train_df_ , test_df = split_train_test (df)
    train_df, val_df = train_validation_split (train_df_)

    train_df.to_csv (TRAIN_OUTPUT_PATH, index=False)
    val_df.to_csv (VAL_OUTPUT_PATH, index=False)
    test_df.to_csv (TEST_OUTPUT_PATH, index=False)

    print (f"Data Splitting Completed.")
    print (f"Train: {len(train_df)}")
    print (f"Validation: {len(val_df)}")
    print (f"Test: {len(test_df)}")



if __name__ == "__main__":
        main()


