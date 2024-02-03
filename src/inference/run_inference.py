import os
import sys
import time
import pickle
import logging
import pandas as pd
from sklearn.svm import LinearSVC


# Define directories
ROOT_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(ROOT_DIR)  
DATA_DIR = os.path.join(ROOT_DIR, 'data')
PROCESSED_INFERENCE_DIR = os.path.join(DATA_DIR, 'processed', 'inference')
MODEL_DIR = os.path.join(ROOT_DIR, 'outputs', 'models')
PROCESSOR_DIR = os.path.join(ROOT_DIR, 'outputs', 'processors')
RESULTS_DIR = os.path.join(ROOT_DIR, 'outputs', 'predictions')
RAW_TEST_DATA_PATH = os.path.join(DATA_DIR, 'raw', 'inference', 'test.csv')
PREDICTIONS_DIR = os.path.join(ROOT_DIR, 'outputs', 'predictions')

from src.text_processor import TextPreprocessor
from src.train.train import DataProcessor, Model

def run_inference(path):
    infer_raw = DataProcessor().data_extraction(path)
    processor = get_processor('processor_1.pkl')
    model = get_model('model_1.pkl')
    logging.info("Preprocessing inference data...")
    X_infer, y_infer, infer_processed = processor.preprocess(infer_raw, fit_vectorizer=False)
    DataProcessor().save_dataset(infer_processed, PROCESSED_INFERENCE_DIR, 'test_processed')
    predictions = predict_sentiment(model, X_infer, infer_raw)
    metrics = Model().evaluate(model, X_infer, y_infer)
    store_predictions(predictions)
    store_metrics(metrics)

def get_model(model_name: str = 'model_1.pkl') -> LinearSVC:
    """Loads and returns a trained model."""
    path = os.path.join(MODEL_DIR, model_name)
    if not os.path.isfile(path):
        raise FileNotFoundError(f"No trained model found at {path}." 
                                "Please train the model first.")
    try:
        with open(path, 'rb') as f:
            model = pickle.load(f)
            logging.info(f"Model loaded from {path}")
            return model
    except Exception as e:
        logging.error(f"An error occurred while loading the model from {path}: {e}")
        sys.exit(1)

def get_processor(processor_name: str = 'processor_1.pkl') -> TextPreprocessor:
    path = os.path.join(PROCESSOR_DIR, processor_name)
    if not os.path.isfile(path):
        raise FileNotFoundError(f"No preprocessor found at {path}." 
                                "Please fit the preprocessor first.")
    try:
        with open(path, 'rb') as f:
            processor = pickle.load(f)
            logging.info(f"Preprocessor loaded from {path}")
            return processor
    except Exception as e:
        logging.error(f"An error occurred while loading the preprocessor from {path}: {e}")
        sys.exit(1)

def predict_sentiment(model: LinearSVC, X_infer, infer_data: pd.DataFrame):
    """Predict results and join it with the inderence dataframe."""
    logging.info("Running inference...")
    start_time = time.time()
    predictions = model.predict(X_infer)
    end_time = time.time()
    sentiment_map = {0: "negative", 1: "positive"}
    infer_data['predictions'] = [sentiment_map[pred] for pred in predictions]
    logging.info(f"Inference completed in {end_time - start_time} seconds.")
    return infer_data

def store_predictions(predictions: pd.DataFrame) -> None:
    """Store the prediction in 'predictions' directory."""
    logging.info("Saving predictions...")
    if not os.path.exists(PREDICTIONS_DIR):
        os.makedirs(PREDICTIONS_DIR)
    path = os.path.join(PREDICTIONS_DIR, 'predictions.csv')
    predictions.to_csv(path, index=False)
    logging.info(f"Predictions saved to {path}")

def store_metrics(metrics: dict) -> None:
    """Store the inference performance metrics in a .txt file."""
    logging.info("Saving performance metrics...")
    if not os.path.exists(PREDICTIONS_DIR):
        os.makedirs(PREDICTIONS_DIR)
    path = os.path.join(PREDICTIONS_DIR, 'metrics.txt')
    metrics_str = "\n".join([f"{key}: {value}" for key, value in metrics.items()])
    with open(path, "w") as file:
        file.write(metrics_str)
    logging.info(f"Metrics saved to {path}")

def main():
    logging.basicConfig(level=logging.INFO, 
                        format='%(asctime)s - %(levelname)s - %(message)s')
    
    run_inference(RAW_TEST_DATA_PATH)


# Executing the script
if __name__ == "__main__":
    main()