import pandas as pd
import re
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem.porter import PorterStemmer
from nltk.stem.wordnet import WordNetLemmatizer
from sklearn.feature_extraction.text import CountVectorizer

splits = {
    "train": "plain_text/train-00000-of-00001.parquet",
    "test": "plain_text/test-00000-of-00001.parquet",
    "unsupervised": "plain_text/unsupervised-00000-of-00001.parquet",
}
df = pd.read_parquet("hf://datasets/stanfordnlp/imdb/" + splits["train"])

# data cleaning
# remove html tags
df["text"] = df["text"].apply(lambda x: re.sub("<.*?>", "", x))

# convert to lowercase
df["text"] = df["text"].apply(lambda x: x.lower())

# replace regular sentece signs with space
df["text"] = df["text"].apply(lambda x: re.sub(r"[.,;:!?\"']", " ", x))

# remove special characters
df["text"] = df["text"].apply(lambda x: re.sub(r"[^a-zA-Z0-9 ]", "", x))

# remove multiple spaces
df["text"] = df["text"].apply(lambda x: re.sub(r"\s+", " ", x).strip())

# tokenization
# Split text into words using NLTK
df["text"] = df["text"].apply(word_tokenize)

# remove stop words
stop_words = set(stopwords.words("english"))
df["text"] = df["text"].apply(
    lambda x: [word for word in x if word not in (stop_words)]
)

# stemming
# Reduce words to their stems
df["text"] = df["text"].apply(lambda x: [PorterStemmer().stem(w) for w in x])

# lemmatization
# Reduce nouns to their root form
df["text"] = df["text"].apply(lambda x: [WordNetLemmatizer().lemmatize(w) for w in x])
df["text"] = df["text"].apply(
    lambda x: [WordNetLemmatizer().lemmatize(w, pos="v") for w in x]
)

vectorizer = CountVectorizer(ngram_range=(1, 1))
X = vectorizer.fit_transform(df["text"])
y = df["label"]


import torch

X_tensor = torch.tensor(X.toarray(), dtype=torch.bool)

# # tokenize the text
# df['text'] = df['text'].apply(word_tokenize)
# print(df.head())
