# imdb-text-classification

**Goal:** This project aims at predicting sentiments based on IMDb movie reviews.

## Structure
```
imdb-text-classification/
├─ data/
│  ├─ raw/
│  └─ processed/
├─ notebooks/
│  ├─ 00_pytorch_basics.ipynb
│  └─ 01_imdb_baseline.ipynb
├─ src/
│  ├─ data.py
│  ├─ models.py
│  └─ train.py
├─ data_collector.py
├─ setup.py
├─ requirements.txt
├─ README.md
└─ .gitignore
```
## How to use this repo
1. install `requirements.txt`  
2. run `setup.py`  
3. Get the data: `python data_collector.py`

## Results (update)

| Model                                   | Accuracy | F1-score | Notes |
|----------------------------------------:|:--------:|:--------:|:--------:|
| Baseline: Logistic Regression (TF-IDF)           | 0.89     | 0.89       | max_feature=50k, bigrams, min_df=2 |
| Baseline II: 5-Fold CV Logistic Regression (TF-IDF)           | 0.90     | 0.90       | max_feature=50k, bigrams, min_df=2 |^
| Baseline III: PyTorch embedding-mean           | 0.89     | 0.89       | max_len=512, train_emb_dim=64 |
| to be updated...                   | …        | …        |