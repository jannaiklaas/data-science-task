# Sentiment Analysis Project with Movie Reviews
The purpose of this project is to predict the sentiment of a movie review by training a model, validating it and deploying through containerization.

# Contents
[Part I: Data Science](#part-i-data-science)
* [EDA Findings](#eda-findings)
* [Feature Preprocessing](#feature-preprocessing)
* [Modeling](#modeling)
* [Comprehensive Evaluation](#comprehensive-evaluation)
* [Potential Value for Business](#potential-value-for-business)

[Part II: Machine Learning Engineering](#part-ii-machine-learning-engineering)
* [TL;DR: Execution Instructions](#tldr-execution-instructions)

# Part I: Data Science
In this part, different preprocessing approaches were explored and several models were tried on the train dataset ("train.csv"). Detailed methods and findings are described in the project's notebook ("Iklaas_J_Final_Project_DS23.ipynb"). The ultimate goal of this part was to determine some of the best algorithms to achieve a validation accuracy of at least 0.85. 

As it was noted at the beginning of the notebook, some terms are used interchangeably in this project: "words"/"tokens"/"features", "reviews"/"documents", "dataset"/"corpus" and "target"/"sentiment".
## EDA Findings
The raw train dataset initially contained 40,000 reviews, equally split between the two sentiment labels - positive and negative. There were only two columns in the dataset: "review" and "sentiment". The initial inspection of the dataset revealed the presence of special characters, contractions, and proper nouns, all of which were removed from the dataset in the preprocessing stage. 

No missing values were identified. However, the dataset contained 272 duplicates, which were immediately dropped. Since this was comparatively a small fraction of the data, this removal was unlikely to disrupt the balance between the target classes - even if all duplicates were from the same class.

Upon closer inspection of the reviews, the following descriptive statistics was obtained:
* The total number of unique tokens: 172,303
* Smallest number of tokens in a document: 9
* Largest number of tokens in a document: 2,911
* Average number of tokens in a document: 279.82
* Median number of tokens in a document: 209
* Number of tokens that appeared only once in the entire corpus: 93,648
* Most common tokens in either sentiment label were special characters, HTML tags, articles, prepositions, pronouns, and other stop words. All of these were removed in the preprocessing stage.
* Least common tokens in either sentiment label mostly contained special characters (like periods and hyphens) and/or were capitalized. 
* There were a few words that were misspelled, meaning that the dataset (or at least the rare words) should have been run through spell-checking to reduce the volume of unique features and improve data quality. However, due to the large number of tokens, spell-checker's high computational cost and constrained hardware and time, spell-checking was not implemented in this project.

## Feature Preprocessing

The following feature preprocessing steps were taken:
* Splitting train dataset into train and validation subsets (test_size=0.2)
* Splitting each subset into X (reviews) and y (sentiment). The latter was numerically encoded (1 for positive, 0 for negative)
* Expansion of contractions (e. g. "didn't" became "did not"). This was necessary to preserve negation in the sentence after removal of stop words.
* Removal of URLs
* Removal of HTML tags
* Removal of contractions ('ve, 'd, 'll, 'm, etc.)
* Tokenization
* Removal of proper nouns
* Removal of special characters
* Removal of numbers
* Conversion to lower case
* Removal of stop words from the English NLTK library, excluding "not". The word "not" was preserved in the corpus to preserve the so that "I didn't enjoy the movie" would not become "enjoy movie".
* Removal of tokens with length of 2 or less characters
* Removal of rare words (that appear only once in the entire corpus). It might have made more sense to spell check those rare words first, but it would have taken several hours to do that every time dataset is prepared since we have tens of thousands of rare words. Additionally, rare words were only calculated for the train subset, i.e. the rare words that were removed from the validation subset were deemed rare only for the train corpus.
* Stemming or lemmatization
* Vectorization: n-grams or TF-IDF. For n-grams, the range was set to include unigrams, bigrams, and trigrams. Vectorizer was fit on the train subset and used to transform the validation subset.

Following the creation of a single preprocessing pipeline (which can be found in the "text_processor.py" script), four sets of train-validation data were created with varying parameters:
* Lemmatization with n-grams vectorization
* Lemmatization with TF-IDF vectorization
* Stemming with n-grams vectorization
* Stemming with TF-IDF vectorization

After preprocessing (but before vectorization), the number of unique tokens was significantly brought down. Now the smallest document contained only one token, and the longest document had 969 tokens. Both negative and positive labels had "not", "movie", "film", "one" and "like" as the most common features. Some words that would typically be considered as stop words ("would", "could", "one", "get", "even", etc.) were still present in the corpus and were some of the most common for both labels. In the notebook's Appendix section I tried removing words like "movie" and "film", and I also tried removing stop words from SpaCy library on top of the NLTK stop words. However, those extra cleaning steps did not increase the model's accuracy, therefore I decided to keep the original preprocessing steps unchanged.

### Stemming vs. Lemmatization
Here are some things that differentiated stemming from lemmatization

* Stemming took slightly longer to run preprocessing: 351.44 seconds for stemming and 324.47 for lemmatization
* Less unique tokens in the stemmed data: 20,762 tokens for stemming and 30,853 for lemmatization
* As expected, stemming resulted in creation of non-existent words like "charact" for "character", "realli" for "really", "peopl" for "people", etc.
* Most common words for the two techniques were mostly the same. However, some tokens, despite preserving their spelling, became more or less common for either technique. For example, word "great" appeared more frequently in the positive lemmatized reviews than in the positive stemmed reviews.


## Modeling
Modeling was performed with four traditional machine learning algorithms:
* LinearSVC. This one is a lighter version of the SVC algorithm. It takes less time to train than the original version with linear kernel - which can be somewhat problematic with a large dataset like the one in this project.
* Logistic regression. Ideally, I would have implemented logistic regression with a deep learning model- which would be a long and heavy task to perform for my machine or Docker.
* Random Forest, which I chose in lieu of a Decision Tree since the latter tend to overfit more
* CatBoost. I chose this one because it works well with categorical features and does not require much hyperparameter tuning.

I chose these algorithms because in the past I had seen them provide great results in other classification tasks -- better than KNN, Naive Bayes, or decision trees.  

All four models were trained on four datasets, outputting 16 baseline models, where only random state was set (for Logistic Regression and LinearSVM `max_iter` was increased to 5000 to ensure complete convergence). 

The problem of performing hyperparameter tuning on all four models was in the training time. A single training took anywhere between a few seconds to an hour. Performing hyperparameter tuning for the decision tree model on the "best" dataset (lemmatized with n-grams vectorization) using grid search, cross-validation, two hyperparameter, each with only 2 possible values would have taken at least 2.5 hours (assuming I won't do anything else on my machine in the meantime). Therefore, hyperparameters were tuned only for the fastest models (which were LinearSVC and LogisticRegression) and only with the "best" dataset. 

The LinearSVC model was tuned for the regularization parameter `C` (values given were 0.001, 0.01, 0.1, and 1) and the loss function type (`hinge` or `squared_hinge`). Grid search was performed with cross-validation over 5 folds. The highest accuracy was obtained at `C = 0.01, loss = "squared_hinge"`. It took about a couple of minutes to perform hyperparameter tuning for this model.

The Logistic Regression model was also tuned for the regularization parameter (values given were 1, 10, and 100) and for the solver (`saga` or `lbfgs`). Grid search was also performed with cross-validation over 5 folds. The highest accuracy was obtained at `C = 100, solver = "saga"`. The tuning took almost an hour for this model, despite it having 10 less fits than the LinearSVC tuning.

## Comprehensive Evaluation




## Potential Value for Business

# Part II: Machine Learning Engineering
## TL;DR: Execution Instructions

0. Ensure you have [Python](https://www.python.org/downloads/) (preferably version 3.10.12), [git](https://git-scm.com/book/en/v2/Getting-Started-Installing-Git), and [Docker Desktop](https://www.docker.com/products/docker-desktop/) installed on your machine (in case you don't, click the corresponding hyperlinks). If you know how to clone a repository and have already done so with this project, jump to Step 6. 
1. Choose a local destination for the project, e.g. a new folder on your Desktop
2. Open Terminal (if you are on Mac) or Powershell (if you are on Windows) at the chosen destination
3. Copy the repository's URL from GitHub
4. In the Terminal/Powershell, type `git clone`, paste the URL you just copied and hit enter

5. Now you should see the "data-science-task" folder in your local destination. If it is there, you may close the Terminal/Powershell.

6. Open a new Terminal/Powershell at "data-science-task" folder

7. Download the raw data by running this code:
```bash
python3 src/data_loader.py
```
* You will see logs directly in your Terminal/Powershell. Once the data is dowloaded, make sure the "data" directory is created inside the "data-science-task" folder. Click on "data" folder -> "raw" -> "train" or "inference" and check if "train.csv" and "test.csv" exist at the respective locations. Go to the next step only after ensuring both datasets exist. 

8. Build the training Docker image. Paste this code and hit enter:
```bash
docker build -f ./src/train/Dockerfile -t training_image .
```
9. Run the training container as follows:
```bash
docker run -v $(pwd)/outputs:/app/outputs -v $(pwd)/data:/app/data training_image
```
Note: If you are using Windows Powershell, the code above may not work. In that case, try running this:
```bash
docker run -v ${PWD}/outputs:/app/outputs -v ${PWD}/data:/app/data training_image
```

10. You will see the training logs directly in the Terminal/Powershell. Once the training is complete, model's validation metrics will be displayed in the terminal, which you can also check in the "metrics.txt" file. The file will be located in the newly created "outputs" directory, inside "predictions" folder. The "outputs" directory will also contain "models", "figures" and "processors" folders. Each of these folders should have relevant files stored. For example, the "figures" folder will have two .png files: one with a feature importance plot and one with a validaiton confusion matrix.

11. Build the inference Docker image. Paste this code into the Terminal/Powershell and hit enter:
```bash
docker build -f ./src/inference/Dockerfile --build-arg model_name=model_1.pkl --build-arg processor_name=processor_1.pkl -t inference_image .
```
12. Run the inference container as follows:
```bash
docker run -v $(pwd)/outputs:/app/outputs -v $(pwd)/data:/app/data inference_image
```
Note: If you are using Windows Powershell, the code above may not work. In that case, try running this:
```bash
docker run -v ${PWD}/outputs:/app/outputs -v ${PWD}/data:/app/data inference_image
```
13. After the container finishes running, you will see the inference metrics in the Terminal/Powershell, which will be automatically added to the "metrics.txt" file. In the local "data-science-task" folder, you should also see the newly created "predictions.csv" file inside "predictions" folder in "outputs" directory.  

## Project Structure
This project has a modular structure, where each folder serves a specific purpose. Folders "data" and "outputs" are not included in this repository as they are created during training and inference.

```
/data-science-task/
├── notebooks
│   └── Iklaas_J_Final_Project_DS23.ipynb              
├── src                       # All necesary scripts and dockerfiles
│   ├── inference
│   │   ├── Dockerfile
│   │   ├── run_inference.py
│   │   └── __init__.py
│   ├── train
│   │   ├── Dockerfile
│   │   ├── train.py
│   │   └── __init__.py
│   ├── data_loader.py
│   ├── text_processor.py
│   └── __init__.py
├── .gitignore                # File that filters out all data and outputs
├── README.md                        
└── requirements.txt          # File with all the necessary libraries and their versions
```
After running "data_loader.py", "data" folder should appear in the project's directory:
```
/data-science-task/
├── data
│   └── raw
│   │   ├── inference
│   │   │   └── test.csv
│   │   └── train
│   │       └── train.csv
├── notebooks
│   └── Iklaas_J_Final_Project_DS23.ipynb              
<...>                      
└── requirements.txt     
```
After running "train.py" either locally or in Docker, processed data will be added to "data" directory, and "outputs" folder will also appear:
```
/data-science-task/
├── data
│   └── raw
│   │   ├── inference
│   │   │   └── test.csv
│   │   └── train
│   │       └── train.csv
│   └── processed
│       └── train
│           ├── train_processed.csv
│           └── validation_processed.csv
<...>
├── outputs
│   ├── figures
│   │   ├── feature_importance.png
│   │   └── model_1_validation_confusion_matrix.png
│   ├── models
│   │   └── model_1.pkl
│   ├── predictions
│   │   └── metrics.txt
│   └── processors
│       └── processor_1.pkl
<...>                      
└── requirements.txt      
```
Finally, after running inference, the project directory will look as follows:
```
/data-science-task/
├── data
│   <...>
│   └── processed
│       ├──inference
│       │   └── test_processed.csv
│       └── train
│           ├── train_processed.csv
│           └── validation_processed.csv
<...>
├── outputs
│   ├── figures
│   │   ├── feature_importance.png
│   │   ├── model_1_inference_confusion_matrix.png
│   │   └── model_1_validation_confusion_matrix.png
│   ├── models
│   │   └── model_1.pkl
│   ├── predictions
│   │   └── metrics.txt         # inference metrics will be added to the file
│   └── processors
│       └── processor_1.pkl
<...>                      
└── requirements.txt      
```

## Prerequisites
This project requires Docker and git. An IDE (like VScode) is recommended for running or examining Python scripts. 

When running scripts outside Docker, ensure your Python environment matches the project's requirements. If using Conda, install libraries separately. For non-Conda environments, use `pip install -r "requirements.txt"`. 

This project uses Python 3.10.12. Different Python versions might require adjusted library versions. Consider aligning your Python version with this project's for compatibility.

## Data
The data used is the [Iris dataset](https://en.wikipedia.org/wiki/Iris_flower_data_set). It contains 150 observations, 4 features and 3 labels (each having 50 entries). The data is imported and processed using the "load_data.py" script, which splits it into training and inference sets with stratification. Each set is then scaled using MinMaxScaler (fit only on training data to avoid data leakage) and saved to a .csv file. The training set has 135 rows, and the inference set contains only features (without labels) and has 15 rows. The training set is further split into train and test sets in train.py script. Before running data through training and inference, each set is converted into a Torch DataLoader with a batch size of 1 (since the datasets are small).


## Training 

The initial training data ("iris_train_data.csv") is split into train and test sets with the test size being 0.2 (can be modified in "settings.json"). Once the training is complete, the model is evaluated on the test set and the classification report is displayed.

The recommended way to run the training process is in a Docker container. First, build the training image. Clone the repository to a local destination of your choice, open Terminal (on Mac) or Powershell (on Windows) at your local "Homework8" folder, and run the following code:

```bash
docker build -f ./training/Dockerfile --build-arg settings_name=settings.json -t training_image .
```

It may take around 10 minutes to build the image, particularly because the code needs to install the packages from "requirements.txt". Once the image is built, you will see it in the "Images" tab in Docker Desktop. Then, you may run the image by pressing the run button in the application (the trained model will then be saved in the container only), but I recommend running it with this code in your Terminal:

```bash
docker run -v $(pwd)/models:/app/models training_image
```
If you are using Powershell and facing an error running the code above, try running this:
```bash
docker run -v ${PWD}/models:/app/models training_image
```
Running the container this way will create the "models" folder in the local working directory (unless it exists already) and save the trained model to it. You will need this saved model to build the inference image later. Once the container is done running, you will also see the classification report on the test set.

Alternatively, you may run the "train.py" script in your IDE. Yet again, you may run into issues when installing the required packages depending on your interpreter version.

## Inference

Before building the inference image, make sure a trained model exists locally. Go to your "Homework8" folder, find the newly created "models" folder and check if "trained_model.pth" exists in it. If the file is there, you can build the inference image by running this code in the Terminal/Powershell:
```bash
docker build -f ./inference/Dockerfile --build-arg model_name=trained_model.pth --build-arg settings_name=settings.json -t inference_image .
```

Similarly, if you want to save the inference results locally, you can do that by running the container with this code in your Terminal:
```bash
docker run -v $(pwd)/results:/app/results inference_image
```
or with this one if you are using Windows Powershell:
```bash
docker run -v ${PWD}/results:/app/results inference_image
```
This will create a "results" folder in your "Homework8" folder and add the inference results to it in a .csv file. If you don't want to save the results locally, you can just run the container by pressing the run button or typing `docker run inference_image` in your Terminal/Powershell. Regardless of whether you want to save the file or not, you will be able to see the inference results in your Terminal/Powershell once the container has finished running.

Alternatively, you may run the "run.py" script in your IDE. Yet again, you may run into issues when installing the required packages depending on your interpreter version.

Note that if you run the inference process again (whether through Docker or locally), it will overwrite the previous ouput in the "inference_results.csv" file.

