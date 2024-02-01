
import os
import sys
import pickle
import json
import logging
import argparse

import numpy as np
import pandas as pd
import re
import nltk
import matplotlib.pyplot as plt
from nltk.tokenize import word_tokenize
from nltk.probability import FreqDist
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer, PorterStemmer
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from nltk.tag import pos_tag
from sklearn.svm import LinearSVC
from sklearn.metrics import accuracy_score, f1_score
from collections import defaultdict
import time
from collections import Counter

# Download necessary NLTK resources
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('averaged_perceptron_tagger')

df_train = pd.read_csv('../data/raw/train.csv')
df_test = pd.read_csv('../data/raw/test.csv')

# Adds the root directory to system path
ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.dirname(ROOT_DIR))

CONF_FILE = "settings.json"

from utils import get_project_dir, configure_logging

# Loads configuration settings from JSON
with open(CONF_FILE, "r") as file:
    conf = json.load(file)

# Defines paths
DATA_DIR = get_project_dir(conf['general']['data_dir'])
MODEL_DIR = get_project_dir(conf['general']['models_dir'])
TRAIN_PATH = os.path.join(DATA_DIR, conf['train']['table_name'])

# Initializes parser for command line arguments
parser = argparse.ArgumentParser()
parser.add_argument("--train_file", 
                    help="Specify inference data file", 
                    default=conf['train']['table_name'])
parser.add_argument("--model_path", 
                    help="Specify the path for the output model")

class TextPreprocessor():
    def __init__(self, use_lemmatization=True, vectorization_type=None):
        self.use_lemmatization = use_lemmatization
        self.vectorization_type = vectorization_type
        self.stop_words = set(stopwords.words('english')) - {'not'}
        self.lemmatizer = WordNetLemmatizer() if use_lemmatization else None
        self.stemmer = PorterStemmer() if not use_lemmatization else None
        self.vectorizer = None
        self.rare_words = None
        # Define a dictionary for contraction expansions
        self.contraction_mapping = {
            r"\bdidn't\b": "did not", r"\bdon't\b": "do not",
            r"\bwasn't\b": "was not", r"\bisn't\b": "is not",
            r"\bweren't\b": "were not", r"\bare't\b": "are not",
            r"\bwouldn't\b": "would not", r"\bwon't\b": "will not",
            r"\bcouldn't\b": "could not", r"\bcan't\b": "can not",
            r"\bain't\b": "am not", r"\bdoesn't\b": "does not",
            r"\bshouldn't\b": "should not", r"\bhadn't\b": "had not",
            r"\bhaven't\b": "have not", r"\bhasn't\b": "has not",
            r"\bmustn't\b": "must not"
        }

    def preprocess(self, data, fit_vectorizer=False):
        if 'sentiment' in data.columns:
            y = data['sentiment'].map({'negative': 0, 'positive': 1})
        else:
            y = None
        
        if self.rare_words is None:
            self._calculate_rare_words(data['review'])

        X_cleaned = data['review'].apply(self._clean_text)

        if self.vectorization_type and fit_vectorizer:
            self.vectorizer = self._get_vectorizer()
            X_vectorized = self.vectorizer.fit_transform(X_cleaned)
        elif self.vectorization_type:
            X_vectorized = self.vectorizer.transform(X_cleaned)
        else:
            X_vectorized = X_cleaned

        return X_vectorized, y, self.vectorizer if fit_vectorizer else None

    def _get_vectorizer(self):
        if self.vectorization_type.lower() == 'ngrams':
            return CountVectorizer(ngram_range=(1, 3), stop_words=list(self.stop_words))
        elif self.vectorization_type.lower() == 'tf-idf':
            return TfidfVectorizer(stop_words=list(self.stop_words))
        else:
            raise ValueError("Invalid vectorization type specified.")

    def _initial_preprocess(self, text):
        # Expand contractions, case-insensitive
        for contraction, expanded in self.contraction_mapping.items():
            text = re.sub(contraction, expanded, text, flags=re.IGNORECASE)
        text = re.sub(r'http\S+|www\S+|https\S+', ' ', text, flags=re.MULTILINE)
        text = re.sub(r'<.*?>', ' ', text)
        text = re.sub(r"(n't|'d|'ll|'m|'re|'s|'ve|')", '', text, flags=re.IGNORECASE)
        tokens = word_tokenize(text)
        tokens = [word for word, pos in pos_tag(tokens) if pos not in ['NNP', 'NNPS']]
        tokens = [re.sub(r'\W+', ' ', word) for word in tokens if not word.isnumeric()]
        return [word.lower() for word in tokens]

    def _calculate_rare_words(self, reviews):
        all_words = [word for review in reviews for word in self._initial_preprocess(review)]
        word_counts = Counter(all_words)
        self.rare_words = {word for word, count in word_counts.items() if count == 1}

    def _clean_text(self, text):
        tokens = self._initial_preprocess(text)
        tokens = [word for word in tokens if word not in self.rare_words and word not in self.stop_words and len(word) > 2]

        if self.use_lemmatization:
            tokens = [self.lemmatizer.lemmatize(word) for word in tokens]
        else:
            tokens = [self.stemmer.stem(word) for word in tokens]

        return ' '.join(tokens)
    

class Training():
    """
    Manages the training process including running training, evaluating 
    the model, and saving the trained model.
    """
    def __init__(self) -> None:
        self.model = LinearSVC(max_iter=5000, C=0.01, random_state=conf['general']['random_state'])
    
    def train(self, X_train: pd.DataFrame, y_train: pd.DataFrame) -> None:
        logging.info("Training the model...")
        self.model.fit(X_train, y_train)
