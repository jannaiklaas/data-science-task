"""
This script imports raw data and saves it locally as .csv files.
It allows the user to specify local file paths or download the files.
"""
import requests
import os
import logging
import sys
import shutil
import argparse

# Define directories
ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(ROOT_DIR, 'data')
RAW_TRAIN_DIR = os.path.join(DATA_DIR, 'raw', 'train')
RAW_INFERENCE_DIR = os.path.join(DATA_DIR, 'raw', 'inference')

# Create logger
logger = logging.getLogger(__name__)

# URL for datasets
URL_TRAIN = "https://raw.githubusercontent.com/jannaiklaas/datasets/main/movie-reviews/train.csv"
URL_TEST = "https://raw.githubusercontent.com/jannaiklaas/datasets/main/movie-reviews/test.csv"

def setup_directories():
    """
    Create required directories if they don't exist.
    """
    for directory in [RAW_TRAIN_DIR, RAW_INFERENCE_DIR]:
        if not os.path.exists(directory):
            os.makedirs(directory)
            logger.info(f"Created directory: {directory}")

def download_file(url, filename):
    """
    Download file from a given URL and save it to a specified filename.
    """
    try:
        response = requests.get(url, stream=True)
        response.raise_for_status()  # Raise an exception for HTTP errors
        with open(filename, 'wb') as file:
            for chunk in response.iter_content(chunk_size=8192):
                file.write(chunk)
        logger.info(f"Downloaded and saved file: {filename}")
    except requests.exceptions.RequestException as e:
        logger.error(f"Error downloading {url}: {e}")
        sys.exit(1)

def copy_file(src, dest):
    """
    Copy a file from src to dest.
    """
    try:
        shutil.copy(src, dest)
        logger.info(f"Copied file from {src} to {dest}")
    except IOError as e:
        logger.error(f"Error copying {src} to {dest}: {e}")
        sys.exit(1)

def main(local_train_path=None, local_test_path=None):
    # Configure logging
    logging.basicConfig(level=logging.INFO, 
                        format='%(asctime)s - %(levelname)s - %(message)s')

    # Setup directories
    setup_directories()

    # Download or copy datasets
    train_dest = os.path.join(RAW_TRAIN_DIR, 'train.csv')
    test_dest = os.path.join(RAW_INFERENCE_DIR, 'test.csv')

    if local_train_path:
        copy_file(local_train_path, train_dest)
    else:
        download_file(URL_TRAIN, train_dest)

    if local_test_path:
        copy_file(local_test_path, test_dest)
    else:
        download_file(URL_TEST, test_dest)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Data Loader Script')
    parser.add_argument('--local_train_path', type=str, help='Local path to the train dataset')
    parser.add_argument('--local_test_path', type=str, help='Local path to the test dataset')
    args = parser.parse_args()

    main(local_train_path=args.local_train_path, local_test_path=args.local_test_path)
