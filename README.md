# Sentiment Analysis Project with Movie Reviews
This project is focused on developing and deploying a machine learning model capable of predicting the sentiment of movie reviews. The process encompasses training, validating, and ultimately deploying the model through containerization for practical application.

# Contents
[Part I: Data Science](#part-i-data-science)
* [EDA Findings](#eda-findings)
* [Feature Preprocessing](#feature-preprocessing)
* [Modeling](#modeling)
* [Potential Value for Business](#potential-value-for-business)

[Part II: Machine Learning Engineering](#part-ii-machine-learning-engineering)
* [TL;DR: Execution Instructions](#tldr-execution-instructions)
* [Project Structure](#project-structure)
* [Prerequisites](#prerequisites)
* [Data](#data)
* [Training](#training)
* [Inference](#inference)

# Part I: Data Science
In this part, different preprocessing approaches were explored and several models were tried on the train dataset ("train.csv"). Detailed methods and findings are described in the project's notebook ("Iklaas_J_Final_Project_DS23.ipynb"). The ultimate goal of this part was to determine the most optimal algorithm to achieve a validation accuracy of at least 0.85. 

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
* There were a few words that were misspelled, meaning that the dataset (or at least the rare words) should have been run through spell-checking to reduce the volume of unique features and improve data quality. However, given the extensive volume of tokens in the dataset, combined with the substantial computational demand of spell-checking and limitations in hardware and time resources, spell-checking was not performed for this project.

## Feature Preprocessing

The raw train data underwent the following preprocessing steps:

* Split into train and validation subsets using `test_size = 0.2`.
* Separate into X (reviews) and y (sentiment), with y numerically encoded (1 for positive, 0 for negative).
* Expand negating contractions (e.g., "didn't" to "did not") to maintain negation after stop word removal.
* Remove of URLs, HTML tags, and contractions ('ve, 'd, 'll, 'm , etc.).
* Tokenize the text.
* Exclude proper nouns, special characters, and numbers.
* Convert all text to lowercase.
* Remove stop words, except for "not", using the English NLTK library. The word "not" is preserved in the corpus since it may be pointing at a negative sentiment (e.g. so that in a review saying "I did not enjoy this movie", the words "not", "enjoy" and "movie" would remain after removal of stop words).
* Exclude words with less than three characters.
* Eliminate rare words that appear only once in the training corpus. These words will be identified based on the train subset and removed from both subsets. It may make more sense to spell check those rare words first, but it will take several hours to do that since we have tens of thousands of rare words. 
* Apply stemming or lemmatization.
* Vectorize the text using either n-grams (including unigrams, bigrams, and trigrams) or TF-IDF. The vectorizer will be fitted on the train subset and applied to both subsets.

Eventually, four sets of train-test data were prepared with varying parameters:
* Lemmatization with n-grams vectorization
* Lemmatization with TF-IDF vectorization
* Stemming with n-grams vectorization
* Stemming with TF-IDF vectorization

**General remarks:**
* The preprocessing significantly reduced the number of unique tokens, resulting in documents ranging from one to 969 tokens.
* Common features like "not", "movie", "film", "one", and "like" were prevalent in both negative and positive labels.
* Words typically regarded as stop words ("would", "could", "one", "get", "even") remained prominent in the corpus. These words are not included in the NLTK library.
* In the notebook's Appendix section, I attempted to improve the performance of the "best" model by adding extra preprocessing steps. In particular, I tried excluding common domain-specific words such as "movie" and "film" and expanding the stop word list using the SpaCy library. Neither of these steps improved the accuracy of the model, therefore these steps were not eventually added to the preprocessing pipeline.

**Comparison of lemmatized and stemmed datasets:**
* The stemmed dataset had fewer features than the lemmatized one: 22,395 compared to 33,349. 
* Stemming required more processing time than lemmatization for the training data: 351.44 seconds compared to 324.47 prior to vectorization.
* The top 30 common words were similar between the datasets, with minor differences in order for words like "watch", "life", and "way".
* Stemming resulted in the creation of non-standard words (e.g., "stori", "charact", "realli"), which is a known characteristic of this technique.

**Comparison of the vectorization techniques:**
* Vectorizing with n-grams took slightly longer than using TF-IDF. 
* The use of n-grams vectorization, ranging from unigrams to trigrams, significantly increased the dimensionality of the datasets. The lemmatized and stemmed datasets now contained over 4 million columns each. In contrast, TF-IDF vectorization did not alter the size of either the lemmatized or stemmed datasets.
* Both the train and validation (test) sets aligned in terms of column numbers. This alignment was likely due to vectorization standardizing the features, ensuring that the features used in training are consistently applied to the test set.
* The features vectorized with n-grams now included word combinations and phrases such as "best movie ever" and "worst film ever", highlighting the way n-grams capture more nuanced expressions of sentiment.
* The features vectorized with TF-IDF were assigned scores ranging from 0 to 1, reflecting their relative importance. This scaling effect was evident in the altered y-axis scale of the 'Top 30 Most Common/Important Features' plots, where the scores indicated the aggregated weighted significance of each feature within the dataset.

Below is the vectorization results summary:
```
Lemmatization with n-grams vectorization: Time taken - 380.23 seconds, Train set shape - (31782, 4575415), Test set shape - (7946, 4575415)
Stemming with n-grams vectorization:: Time taken - 414.48 seconds, Train set shape - (31782, 4335322), Test set shape - (7946, 4335322)
Lemmatization with TF-IDF vectorization:: Time taken - 369.51 seconds, Train set shape - (31782, 30703), Test set shape - (7946, 30703)
Stemming with TF-IDF vectorization:: Time taken - 401.97 seconds, Train set shape - (31782, 20616), Test set shape - (7946, 20616)

```


## Modeling
Modeling was performed with four traditional machine learning algorithms:
* LinearSVC, a streamlined version of the SVC algorithm. It offers faster training times compared to the original SVC with a linear kernel, which is advantageous for large datasets like ours.
* Logistic Regression. While a deep learning approach would be ideal for logistic regression, it demands extensive computational resources, which are beyond the current project's capacity.
* Random Forest, selected over a Decision Tree to mitigate the risk of overfitting.
* CatBoost, known for its effectiveness with categorical features and minimal hyperparameter tuning requirements.

These algorithms were chosen based on their proven performance in previous classification tasks, where they outperformed alternatives like KNN, Naive Bayes, or Decision Trees.  

Each of the four models was trained on four datasets, resulting in 16 baseline models. This approach allowed for a comprehensive comparison of lemmatization, stemming, and both vectorization techniques. In these baseline models, only the random state was set, with `max_iter` increased to 5000 for Logistic Regression and LinearSVC to ensure convergence. 

**Modeling Summary and Model Choice**:

* A consistent trend observed throughout this analysis was the superior performance of lemmatization over stemming, and n-grams vectorization over TF-IDF, although at the expense of increased training time. Consequently, the lemmatized dataset with n-grams vectorization emerged as the preferred choice.
* Due to extended training durations, tree-based models such as Random Forest and CatBoost were not subjected to hyperparameter tuning.
* Among the four machine learning algorithms evaluated, Logistic Regression and Linear SVC not only yielded the best results but also demonstrated the quickest training times.
* The tuned LinearSVC model, in particular, achieved the highest test (validation) accuracy at 0.8928, along with impressive metrics in other areas. Its training efficiency — being ten times faster than Logistic Regression on a lemmatized n-grams vectorized dataset — further backed its position as the optimal model for this project.
* Feature importance analysis revealed that negative adjectives were highly influential in sentiment classification, especially in tree-based models. 

Below are the results from the LinearSVC models:

*Baseline results:*
```
Lemmatization with n-grams vectorization: Training Time: 6.71 seconds, Train Accuracy: 1.0000, Test Accuracy: 0.8903, Test F1 Score: 0.8920
Stemming with n-grams vectorization: Training Time: 7.46 seconds, Train Accuracy: 1.0000, Test Accuracy: 0.8875, Test F1 Score: 0.8892
Lemmatization with TF-IDF vectorization: Training Time: 0.22 seconds, Train Accuracy: 0.9729, Test Accuracy: 0.8806, Test F1 Score: 0.8821
Stemming with TF-IDF vectorization: Training Time: 0.22 seconds, Train Accuracy: 0.9597, Test Accuracy: 0.8799, Test F1 Score: 0.8818
```
Hyperparameters tuning was conducted on the lemmatized, vectorized with n-grams dataset. *These are the test (validation) metrics of the model that was eventually used in the MLE part of this project:*
```
Best parameters: {'C': 0.01, 'loss': 'squared_hinge'}
Test Accuracy: 0.8928, Test F1 Score: 0.8946
Test Precision: 0.8824, Test Recall: 0.9072
Test AUC ROC score: 0.9549
```

### Results from Other Models
#### Logistic Regression
*Baseline results:*
```
Lemmatization with n-grams vectorization: Training Time: 76.66 seconds, Train Accuracy: 1.0000, Test Accuracy: 0.8916, Test F1 Score: 0.8933
Stemming with n-grams vectorization: Training Time: 75.28 seconds, Train Accuracy: 1.0000, Test Accuracy: 0.8911, Test F1 Score: 0.8929
Lemmatization with TF-IDF vectorization: Training Time: 0.52 seconds, Train Accuracy: 0.9224, Test Accuracy: 0.8864, Test F1 Score: 0.8880
Stemming with TF-IDF vectorization: Training Time: 0.32 seconds, Train Accuracy: 0.9146, Test Accuracy: 0.8862, Test F1 Score: 0.8878
```
Hyperparameters tuning was conducted on the lemmatized, vectorized with n-grams dataset:
```
Best parameters: {'C': 100, 'solver': 'saga'}
Test Accuracy: 0.8916, Test F1 Score: 0.8932
Test Precision: 0.8828, Test Recall: 0.9039
Test AUC ROC score: 0.9544
```
#### Random Forest
*Baseline results:*
```
Lemmatization with n-grams vectorization: Training Time: 636.27 seconds, Train Accuracy: 1.0000, Test Accuracy: 0.8574, Test F1 Score: 0.8608
Stemming with n-grams vectorization: Training Time: 598.90 seconds, Train Accuracy: 1.0000, Test Accuracy: 0.8541, Test F1 Score: 0.8610
Lemmatization with TF-IDF vectorization: Training Time: 17.48 seconds, Train Accuracy: 1.0000, Test Accuracy: 0.8529, Test F1 Score: 0.8543
Stemming with TF-IDF vectorization: Training Time: 16.61 seconds, Train Accuracy: 1.0000, Test Accuracy: 0.8460, Test F1 Score: 0.8473
```
#### CatBoost
*Baseline results:*
```
Lemmatization with n-grams vectorization: Training Time: 2317.67 seconds, Train Accuracy: 0.9105, Test Accuracy: 0.8724, Test F1 Score: 0.8753
Stemming with n-grams vectorization: Training Time: 2223.59 seconds, Train Accuracy: 0.9105, Test Accuracy: 0.8714, Test F1 Score: 0.8745
Lemmatization with TF-IDF vectorization: Training Time: 106.22 seconds, Train Accuracy: 0.9218, Test Accuracy: 0.8674, Test F1 Score: 0.8700
Stemming with TF-IDF vectorization: Training Time: 87.43 seconds, Train Accuracy: 0.9207, Test Accuracy: 0.8637, Test F1 Score: 0.8668
```
#### Further Steps
For future endeavors to enhance accuracy, I recommend exploring additional preprocessing steps like spell-checking, experimenting with different libraries, and considering advanced deep learning methods such as RNNs and transformers. This study, constrained by time and hardware limitations, focused solely on traditional machine learning models. 


## Potential Value for Business
The potential application and value for businesses in predicting the sentiment of movie reviews through binary classification include:
1. *Market Analysis and Product Development*: Film production companies and distributors can use sentiment analysis to gauge public reception of their movies. Understanding viewer sentiments can guide future film projects, marketing strategies, and content creation to align with audience preferences. For instance, if a particular genre or theme consistently receives positive sentiment, studios might focus more on such themes.
2. *Targeted Marketing and Advertising*: By analyzing sentiments in movie reviews, marketers can create more targeted and effective advertising campaigns. For example, if a movie receives overwhelmingly positive reviews, marketing efforts can leverage this positivity with campaigns like product placement. Conversely, if reviews are mixed, marketing can address perceived shortcomings or target a more specific audience segment.
3. *Customer Feedback Analysis*: This project can serve as a framework for broader customer feedback analysis tools. Companies across various sectors can adapt the model to understand customer sentiment about their products or services, not just movies.
4. *Competitive Analysis*: By comparing sentiment analysis results of different movies, studios and streaming platforms can gain insights into competitive strengths and weaknesses, influencing strategic decisions and content acquisition choices.
5. *Personalized Recommendations*: For streaming services and recommendation platforms, sentiment analysis can enhance personalized recommendation engines. Understanding which movies are well-received helps in curating more accurate and appealing content suggestions for users.
6. *Enhancing User Experience on Platforms*: For online movie platforms and forums, IMDb or Rotten Tomatoes, implementing sentiment analysis can improve user experience. It can be used to highlight the most liked or disliked aspects of a movie, helping users make informed viewing choices.
7. *Investment and Stock Market Decisions*: Investor decisions in the entertainment industry can be influenced by public sentiment towards movies. Positive sentiment can indicate a good investment opportunity in a particular studio or film franchise.

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

This is the original structure of the repository:
```
/data-science-task/
├── notebooks                 # Notebooks containing the Data Science part
│   └── Iklaas_J_Final_Project_DS23.ipynb              
├── src                       # All necesary scripts and Dockerfiles
│   ├── inference             # Scripts and Dockerfiles used for inference
│   │   ├── Dockerfile
│   │   ├── run_inference.py
│   │   └── __init__.py
│   ├── train                 # Scripts and Dockerfiles used for training
│   │   ├── Dockerfile
│   │   ├── train.py
│   │   └── __init__.py
│   ├── data_loader.py        # Script to download raw data
│   ├── text_processor.py     # Classes and methods to preprocess text
│   └── __init__.py
├── .gitignore                # File that filters out all data and outputs
├── README.md                        
└── requirements.txt          # File with all the necessary libraries and their versions
```
<a id="data-creation"></a> 
After running "data_loader.py", "data" folder should appear in the project's directory :
```
/data-science-task/
├── data
│   └── raw                   # Raw data directory
│   │   ├── inference
│   │   │   └── test.csv
│   │   └── train
│   │       └── train.csv
├── notebooks
│   └── Iklaas_J_Final_Project_DS23.ipynb              
<...>                      
└── requirements.txt     
```
<a id="post-train"></a>
After running "train.py" either locally or in Docker, processed data will be added to "data" directory, and "outputs" folder will also appear:
```
/data-science-task/
├── data
│   └── raw
│   │   ├── inference
│   │   │   └── test.csv
│   │   └── train
│   │       └── train.csv
│   └── processed             # Processed data directory
│       └── train
│           ├── train_processed.csv
│           └── validation_processed.csv
<...>
├── outputs                   # Directory training and inference outputs
│   ├── figures               # Directory for validation and inference plots
│   │   ├── feature_importance.png
│   │   └── model_1_validation_confusion_matrix.png
│   ├── models                # Directory for trained models
│   │   └── model_1.pkl
│   ├── predictions           # Directory for inference predictions
│   │   └── metrics.txt       # File with validation and inference metrics
│   └── processors            # Directory for fit text processors
│       └── processor_1.pkl
<...>                      
└── requirements.txt      
```
<a id="post-inference"></a>
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
│   │   ├── metrics.txt         # Inference metrics will be added to this file
│   │   └── predictions.csv     # Original inference with predictions added
│   └── processors
│       └── processor_1.pkl
<...>                      
└── requirements.txt      
```

## Prerequisites
This project requires installed [Python](https://www.python.org/downloads/) (preferably version 3.10.12), [git](https://git-scm.com/book/en/v2/Getting-Started-Installing-Git), and [Docker Desktop](https://www.docker.com/products/docker-desktop/). An IDE (like [VScode](https://code.visualstudio.com/)) is recommended for running or examining Python scripts. Alternatively, they can be run in Terminal/Powershell. 

When running scripts outside Docker, ensure your Python environment matches the project's requirements. If using Conda, install libraries separately. For non-Conda environments, use `pip install -r "requirements.txt"`. 

This project uses Python 3.10.12. Different Python versions might require adjusted library versions. Consider aligning your Python version with this project's for compatibility.

### Forking and Cloning from GitHub
To start using this project, you first need to create a copy on your own GitHub account by 'forking' it. On the main page of the `data-science-task` project, click on the 'Fork' button at the top right corner. This will create a copy of the project under your own account. You can then 'clone' it to your local machine for personal use. To do this, click the 'Code' button on your forked repository, copy the provided link, and use the `git clone` command in your terminal followed by the copied link. This will create a local copy of the repository on your machine, and you're ready to start!

### Setting Up Development Environment
Next, you need to set up a suitable Integrated Development Environment (IDE). Visual Studio Code (VSCode) is a great tool for this. You can download it from the official website (https://code.visualstudio.com/Download). After installing VSCode, open it and navigate to the `File` menu and click `Add Folder to Workspace`. Navigate to the directory where you cloned the forked repository and add it. VSCode supports a wide range of programming languages with features like syntax highlighting, code completion, and debugging configurations. You can now edit the files, navigate through your project, and start contributing to `data-science-task`. For running scripts, open a new terminal in VSCode by selecting `Terminal -> New Terminal`. Now you can execute your Python scripts directly in the terminal.

### Installing Docker Desktop
Installing Docker Desktop is a straightforward process. Head over to the Docker official website's download page ([Docker Download Page](https://www.docker.com/products/docker-desktop)), and select the version for your operating system - Docker Desktop is available for both Windows and Mac. After downloading the installer, run it, and follow the on-screen instructions. 

Once the installation is completed, you can open Docker Desktop to confirm it's running correctly. It will typically show up in your applications or programs list. After launching, Docker Desktop will be idle until you run Docker commands. This application effectively wraps the Docker command line and simplifies many operations for you, making it easier to manage containers, images, and networks directly from your desktop. 

Keep in mind that Docker requires you to have virtualization enabled in your system's BIOS settings. If you encounter issues, please verify your virtualization settings, or refer to Docker's installation troubleshooting guide. Now you're prepared to work with Dockerized applications!

## Data
The raw data is not included with the project by default. It should downloaded from an [online source](https://github.com/jannaiklaas/datasets/tree/main/movie-reviews). Once you have the remote repository cloned to your local destination, you can run the following code in the Terminal/Powershell at the local "data-science-task" repository's location:

```bash
python3 src/data_loader.py
```

Alternatively, you can provide path to the raw data if it is already on your local machine:

```bash
python3 src/data_loader.py --local_train_path /path_to_local_train.csv --local_test_path /path_to_local_test.csv
```

Replace `path_to_local_train.csv` and `/path_to_local_test.csv` with actual paths on your local machine where your raw datasets are located.

Running the "data_loader.py" script will create "data" folder at the local "data-science-task" location. Inside you should see the "train.csv" and "test.csv" files, as shown in the [project structure](#data-creation).

## Training 

The recommended way to run the training process is in a Docker container. First, build the training image. In the Terminal/Powershell at your local "data-science-task" folder run the following code:

```bash
docker build -f ./src/train/Dockerfile -t training_image .
```
Once the image is built, you will see it in the "Images" tab in Docker Desktop. Then, you may run the image by pressing the run button in the application (the outputs will then be saved in the container only), but I recommend running it with this code in your Terminal:

```bash
docker run -v $(pwd)/outputs:/app/outputs -v $(pwd)/data:/app/data training_image
```
If you are using Powershell and facing an error running the code above, try running this:
```bash
docker run -v ${PWD}/outputs:/app/outputs -v ${PWD}/data:/app/data training_image
```
Running the container this way will create the "outputs" folder in the local working directory (unless it exists already) and save the trained model, fit preprocessor, test (validation) metrics and plots to it. It will also save preprocessed data to the "data" folder, as shown in the [project structure](#post-train). The saved trained model and preprocessor will be used to build the inference image later. Once the container is done running, you will also see the performance metrics on the test set in the coding environement.

Alternatively, you may run the "train.py" script in your IDE as follows:

```bash
python3 src/train/train.py
```

Yet again, if you choose to run it outside Docker, you may encounter issues when installing the required packages depending on your interpreter version.

## Inference

Before building the inference image, make sure a trained model exists locally. Go to your "data-science-task" folder -> "outputs" -> "models" and check if "model_1.pkl" exists in it. Also check if "processor_1.pkl" exists at "data-science-task" -> "outputs" -> "processors". If both files exist, you can build the inference image by running this code in the Terminal/Powershell:
```bash
docker build -f ./src/inference/Dockerfile --build-arg model_name=model_1.pkl --build-arg processor_name=processor_1.pkl -t inference_image .
```

Similarly, if you want to save the inference results locally, you can do that by running the container with this code in your Terminal:
```bash
docker run -v $(pwd)/outputs:/app/outputs -v $(pwd)/data:/app/data inference_image
```
or with this one if you are using Windows Powershell:
```bash
docker run -v ${PWD}/outputs:/app/outputs -v ${PWD}/data:/app/data inference_image
```
Once the container is done running, the inference predictions will be saved to "predictions.csv" at "outputs"/"predictions". The inference metrics will be displayed in the Terminal/Powershell and also added to "metrics.txt" file (which should already contain validation metrics). To see the complete list of outputs and newly created files, check the [project structure](#post-inference).

Alternatively, you may run the "run_inference.py" script in your IDE as follows:

```bash
python3 src/inference/run_inference.py
```

Yet again, if you choose to run it outside Docker, you may encounter issues when installing the required packages depending on your interpreter version.

