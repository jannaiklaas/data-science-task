"""
Script to preprocess text data.
"""
# Import libraries
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer, PorterStemmer
from nltk.tokenize import word_tokenize
from nltk.tag import pos_tag
from collections import Counter
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer

# Download necessary NLTK resources
nltk.download('punkt', quiet=True)
nltk.download('stopwords', quiet=True)
nltk.download('wordnet', quiet=True)
nltk.download('averaged_perceptron_tagger', quiet=True)

class TextPreprocessor:
    """
    Preprocesses text data for sentiment analysis. 
    """
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
        """Cleans, tokenizes, and vectorizes text data."""
        cleaned_data = data.copy()
        if 'sentiment' in data.columns:
            cleaned_data['sentiment'] = data['sentiment'].map({'negative': 0, 'positive': 1})
            y = cleaned_data['sentiment']
        
        else:
            y = None

        if self.rare_words is None:
            self._calculate_rare_words(data['review'])

        cleaned_data['review'] = data['review'].apply(self._clean_text)

        if self.vectorization_type and fit_vectorizer:
            self.vectorizer = self._get_vectorizer()
            X_vectorized = self.vectorizer.fit_transform(cleaned_data['review'])
        elif self.vectorization_type:
            X_vectorized = self.vectorizer.transform(cleaned_data['review'])
        else:
            X_vectorized = cleaned_data['review']

        return X_vectorized, y, cleaned_data

    def _get_vectorizer(self):
        if self.vectorization_type.lower() == 'ngrams':
            return CountVectorizer(ngram_range=(1, 3), stop_words=list(self.stop_words))
        elif self.vectorization_type.lower() == 'tf-idf':
            return TfidfVectorizer(stop_words=list(self.stop_words))
        else:
            raise ValueError("Invalid vectorization type specified.")

    def _initial_preprocess(self, text):
        """Filters through data to obtain sensical words."""
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
        """Calculates and eliminates rare words from the corpus."""
        all_words = [word for review in reviews for word in self._initial_preprocess(review)]
        word_counts = Counter(all_words)
        self.rare_words = {word for word, count in word_counts.items() if count == 1}

    def _clean_text(self, text):
        """Outputs lemmatized or stemmed tokens."""
        tokens = self._initial_preprocess(text)
        tokens = [word for word in tokens if word not in self.rare_words and word not in self.stop_words and len(word) > 2]

        if self.use_lemmatization:
            tokens = [self.lemmatizer.lemmatize(word) for word in tokens]
        else:
            tokens = [self.stemmer.stem(word) for word in tokens]

        return ' '.join(tokens)