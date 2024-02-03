"""
This script prepares the data, runs the training, and saves the model.
"""
import os
import sys
import time
import pickle
import logging
import pandas as pd
from scipy.sparse import csr_matrix, save_npz
from sklearn.model_selection import train_test_split
from sklearn.svm import LinearSVC
from sklearn.metrics import accuracy_score, f1_score, precision_score, \
    recall_score, roc_auc_score

# Define directories
ROOT_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(ROOT_DIR)  
DATA_DIR = os.path.join(ROOT_DIR, 'data')
PROCESSED_TRAIN_DIR = os.path.join(DATA_DIR, 'processed', 'train')
MODEL_DIR = os.path.join(ROOT_DIR, 'outputs', 'models')
PROCESSOR_DIR = os.path.join(ROOT_DIR, 'outputs', 'processors')
RAW_TRAIN_DATA_PATH = os.path.join(DATA_DIR, 'raw', 'train', 'train.csv')

from src.text_processor import TextPreprocessor

class DataProcessor():
    def __init__(self) -> None:
        self.processor = TextPreprocessor(use_lemmatization=True, vectorization_type="ngrams")

    def prepare_data(self) -> tuple:
        """Takes a single file path and outputs preprocessed datasets ready for modeling"""
        logging.info("Preparing data for training...")
        df = self.data_extraction(RAW_TRAIN_DATA_PATH)
        logging.info(f"Removing {df.duplicated().sum()} duplicates...")
        df.drop_duplicates(inplace=True)
        logging.info(f"Train dataset contains {df.shape[1]} columns and {df.shape[0]} rows")
        train, test = self.data_split(df)
        logging.info(f"Model to be trained with {train.shape[0]} " 
                     f"reviews and validated with {test.shape[0]} reviews.")
        X_train, y_train, train_processed = self.processor.preprocess(train, fit_vectorizer=True)
        X_test, y_test, test_processed = self.processor.preprocess(test, fit_vectorizer=False)
        logging.info(f"Preprocessing: lemmatization with {self.processor.vectorization_type} vectorization")
        logging.info(f"Processed train data shape: {X_train.shape}")
        logging.info(f"Processed test data shape: {X_test.shape}")
        self.save_processor("processor_1.pkl")
        self.save_dataset(train_processed, PROCESSED_TRAIN_DIR, 'train_processed')
        self.save_dataset(test_processed, PROCESSED_TRAIN_DIR, 'validation_processed')
        return X_train, y_train, X_test, y_test

    def data_extraction(self, path: str) -> pd.DataFrame:
        """Loads a .csv file and converts it DataFrame."""
        if not os.path.isfile(path):
            raise FileNotFoundError("The specified dataset does not exist at "
                                    f"{path}. Check the file path.")
        try:
            logging.info(f"Loading dataset from {path}...")
            df = pd.read_csv(path)
            logging.info("Dataset loaded. It contains "
                         f"{df.shape[1]} columns and {df.shape[0]} rows.")
            return df
        except Exception as e:
            logging.error(f"An error occurred while loading data from {path}: {e}")
            sys.exit(1)

    def data_split(self, df: pd.DataFrame) -> tuple:
        """Splits data into test and train."""
        logging.info("Splitting data into training and test sets...")
        train, test = train_test_split(df, test_size=0.2, stratify=df['sentiment'],
                                random_state=42)
        return train, test
    
    def save_processor(self, processor_name):
        logging.info("Saving the preprocessor configuraiton...")
        if not os.path.exists(PROCESSOR_DIR):
            os.makedirs(PROCESSOR_DIR)
        path = os.path.join(PROCESSOR_DIR, processor_name) 
        with open(path, 'wb') as f:
            pickle.dump(self.processor, f)
        logging.info(f"Preprocessor saved to {path}")
    
    @staticmethod
    def save_dataset(data, dir, filename: str, y=None):
        """Stores preprocessed datasets."""
        logging.info("Storing preprocessed data...")
        if not os.path.exists(dir):
            os.makedirs(dir)
        if isinstance(data, csr_matrix):
            path = os.path.join(dir, "review_" + filename + ".npz")
            save_npz(path, data)
            if y:
                path = os.path.join(dir, "sentiment_" + filename + ".csv")
                y.to_csv(path, index=False)
        else:
            path = os.path.join(dir, filename + ".csv")
            data.to_csv(path, index=False)
        logging.info(f"Data saved to {path}")

    
class Model():
    def __init__(self) -> None:
        self.model = LinearSVC(random_state=42, C=0.01, max_iter=5000)

    def run_training(self, X_train, y_train, X_test, y_test) -> None:
        logging.info("Running training...")
        start_time = time.time()
        self.train(X_train, y_train)
        end_time = time.time()
        logging.info(f"Training completed in {end_time - start_time:.2f} seconds.")
        logging.info("Testing model on validation set:")
        self.evaluate(self.model, X_test, y_test)
        self.save()
    
    def train(self, X_train: pd.DataFrame, y_train: pd.DataFrame) -> None:
        logging.info("Training the model...")
        self.model.fit(X_train, y_train)

    @staticmethod
    def evaluate(model: LinearSVC, X_test: pd.DataFrame, y_test: pd.DataFrame) -> float:
        logging.info("Calculating performance metrics...")
        # Calculate metrics
        y_pred = model.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred)
        recall = recall_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)
        auc_roc = roc_auc_score(y_test, model.decision_function(X_test))
        metrics = {
            'Accuracy': accuracy,
            'Precision': precision,
            'Recall': recall,
            'F1 Score': f1,
            'AUC ROC': auc_roc
        }
        logging.info("Performance metrics:\n"
                     f"Accuracy: {metrics['Accuracy']: .4f}\n"
                     f"Precision: {metrics['Precision']: .4f}\n"
                     f"Recall: {metrics['Recall']: .4f}\n"
                     f"F1 score: {metrics['F1 Score']: .4f}\n"
                     f"AUC ROC: {metrics['AUC ROC']: .4f}\n")
        return metrics

    def save(self) -> None:
        """Saves the trained model to the specified path."""
        logging.info("Saving the model...")
        if not os.path.exists(MODEL_DIR):
            os.makedirs(MODEL_DIR)
        # Use the model name from settings.json
        path = os.path.join(MODEL_DIR, 'model_1.pkl') 
        with open(path, 'wb') as f:
            pickle.dump(self.model, f)
        logging.info(f"Model saved to {path}")

def main():
    logging.basicConfig(level=logging.INFO, 
                        format='%(asctime)s - %(levelname)s - %(message)s')

    data_proc = DataProcessor()
    model = Model()

    X_train, y_train, X_test, y_test = data_proc.prepare_data()
    model.run_training(X_train, y_train, X_test, y_test)


if __name__ == "__main__":
    main()