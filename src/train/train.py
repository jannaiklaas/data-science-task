"""
This script prepares the data, runs the training, and saves the model.
"""
import os
import sys
import time
import pickle
import logging
import numpy as np
import pandas as pd
import seaborn as sns
from datetime import datetime
import matplotlib.pyplot as plt
from scipy.sparse import csr_matrix, save_npz
from sklearn.model_selection import train_test_split
from sklearn.svm import LinearSVC
from sklearn.metrics import accuracy_score, f1_score, precision_score, \
    recall_score, roc_auc_score, confusion_matrix

# Define directories
ROOT_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(ROOT_DIR) 

# Directories for raw and processed train data
DATA_DIR = os.path.join(ROOT_DIR, 'data')
RAW_TRAIN_DATA_PATH = os.path.join(DATA_DIR, 'raw', 'train', 'train.csv')
PROCESSED_TRAIN_DIR = os.path.join(DATA_DIR, 'processed', 'train')

# Directories for models, processors, validation metrics and figures
OUTPUT_DIR = os.path.join(ROOT_DIR, 'outputs')
MODEL_DIR = os.path.join(OUTPUT_DIR, 'models')
PROCESSOR_DIR = os.path.join(OUTPUT_DIR, 'processors')
FIG_DIR = os.path.join(OUTPUT_DIR, 'figures')
PREDICTIONS_DIR = os.path.join(OUTPUT_DIR, 'predictions')

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
        return X_train, y_train, X_test, y_test, self.processor.vectorizer

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

    def run_training(self, X_train, y_train, X_test, y_test, vectorizer) -> None:
        logging.info("Running training...")
        start_time = time.time()
        self.model.fit(X_train, y_train)
        end_time = time.time()
        logging.info(f"Training completed in {end_time - start_time:.2f} seconds.")
        logging.info("Testing model on validation set:")
        self.evaluate(self.model, X_test, y_test, status="validation")
        self.plot_feature_importances(vectorizer=vectorizer, top_n=50, fig_name='feature_importance')
        self.save()

    @staticmethod
    def evaluate(model: LinearSVC, X_test: pd.DataFrame, y_test: pd.DataFrame, status: str, model_name = "model_1") -> float:
        logging.info("Calculating performance metrics...")
        # Calculate metrics
        y_pred = model.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred)
        recall = recall_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)
        auc_roc = roc_auc_score(y_test, model.decision_function(X_test))
        metrics = {
            'Time': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            'Model': model_name,
            'Status': status,
            'Accuracy': accuracy,
            'Precision': precision,
            'Recall': recall,
            'F1 Score': f1,
            'AUC ROC': auc_roc
        }
        logging.info("Performance metrics:\n"
                     f"Model name: {model_name}\n"
                     f"Status: {status}\n"
                     f"Accuracy: {metrics['Accuracy']: .4f}\n"
                     f"Precision: {metrics['Precision']: .4f}\n"
                     f"Recall: {metrics['Recall']: .4f}\n"
                     f"F1 score: {metrics['F1 Score']: .4f}\n"
                     f"AUC ROC: {metrics['AUC ROC']: .4f}\n")
        
        logging.info("Saving performance metrics...")
        if not os.path.exists(PREDICTIONS_DIR):
            os.makedirs(PREDICTIONS_DIR)
        path = os.path.join(PREDICTIONS_DIR, 'metrics.txt')
        metrics_str = "\n".join([f"{key}: {value}" for key, value in metrics.items()])
        with open(path, "a") as file:
            file.write(metrics_str + "\n\n")
        logging.info(f"Metrics saved to {path}")
        Model.plot_confusion_matrix(y_test, y_pred, status, model_name)

    def plot_feature_importances(self, vectorizer, top_n=20, fig_name='feature_importance_plot', model_name='model_1'):
        """Plot and save feature importances."""
        logging.info("Plotting feature importances...")
        
        # Get coefficients and feature names
        coefs = self.model.coef_.ravel()
        feature_names = vectorizer.get_feature_names_out()
        
        # Get the indices of the top n positive and negative coefficients
        top_positive_indices = np.argsort(coefs)[-top_n:][::-1]  # Most positive at the top
        top_negative_indices = np.argsort(coefs)[:top_n]  # Most negative at the bottom
        
        # Combine positive and negative indices, with positive first for top of plot
        top_indices = np.hstack([top_positive_indices, top_negative_indices[::-1]])
        top_features = feature_names[top_indices]
        top_coefs = coefs[top_indices]
        
        # Create the plot
        plt.figure(figsize=(10, top_n/2))  # Adjust height as needed
        colors = ['green' if c > 0 else 'red' for c in top_coefs]
        plt.barh(np.arange(top_n * 2), top_coefs, color=colors)
        plt.yticks(np.arange(top_n * 2), top_features)
        plt.xlabel('Coefficient Value')
        plt.title(f'Top {top_n} Positive and Negative Feature Importances in {model_name}')
        plt.axvline(x=0, color='k', linestyle='--')  # Add a vertical line at x=0
        # Invert y-axis to have the most important feature at the top
        plt.gca().invert_yaxis()
        # Save the plot
        FIG_DIR = os.path.join('outputs', 'figures')  # Define your FIG_DIR
        if not os.path.exists(FIG_DIR):
            os.makedirs(FIG_DIR)
        fig_path = os.path.join(FIG_DIR, f"{fig_name}.png")
        plt.savefig(fig_path, bbox_inches='tight')  # Use bbox_inches='tight' to fit the plot
        logging.info(f"Feature importance plot saved to {fig_path}")
        plt.close()

    @staticmethod
    def plot_confusion_matrix(y_true, y_pred, status, model_name):
        # Calculate confusion matrix
        cm = confusion_matrix(y_true, y_pred)
        fig, ax = plt.subplots(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='plasma', ax=ax)
        ax.set_title(f'Confusion Matrix: {status}')
        ax.set_xlabel('Predicted Labels')
        ax.set_ylabel('True Labels')
        ax.xaxis.set_ticklabels(['Negative', 'Positive'])
        ax.yaxis.set_ticklabels(['Negative', 'Positive'])
        FIG_DIR = os.path.join('outputs', 'figures')  # Define your FIG_DIR
        if not os.path.exists(FIG_DIR):
            os.makedirs(FIG_DIR)
        fig_path = os.path.join(FIG_DIR, f"{model_name}_{status}_confusion_matrix.png")
        plt.savefig(fig_path, bbox_inches='tight')
        logging.info(f"Confusion matrix plot saved to {fig_path}")
        plt.close(fig)

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

    X_train, y_train, X_test, y_test, vectorizer = data_proc.prepare_data()
    model.run_training(X_train, y_train, X_test, y_test, vectorizer)


if __name__ == "__main__":
    main()