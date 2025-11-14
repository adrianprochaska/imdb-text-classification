import os
import pandas as pd


def collect_imdb_data():
    """Collect IMDB dataset from Hugging Face and save locally."""
    os.makedirs("data/raw", exist_ok=True)

    paths = {
        "train": "hf://datasets/stanfordnlp/imdb/plain_text/train-00000-of-00001.parquet",
        "test": "hf://datasets/stanfordnlp/imdb/plain_text/test-00000-of-00001.parquet",
    }

    for split, src in paths.items():
        df = pd.read_parquet(src)
        df.to_parquet(f"data/raw/{split}.parquet")
        print(f"Saved {split} to data/raw/{split}.parquet")


if __name__ == "__main__":
    collect_imdb_data()
    # Optionally, you can call a function to save locally if needed
