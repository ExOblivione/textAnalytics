# Welcome!

This is the documentation for the project which aims to perform a comprehensive data analysis, using the dataset available from [Kaggle](https://www.kaggle.com/c/jigsaw-toxic-comment-classification-challenge/data), to identify instances of toxic content within various comments on Wikipedia. This task is important for promoting safety and a more inclusive online space for the users.


# Content

- asset_count.py -> Different functions for calculating rows per labels and their values.
- data_exploration.py -> A wrapper for calling functions that explain the training dataset.
- feature_extraction.py -> Used for splitting the data and tokenize for training and testing of the models. Can be personalized to use specific vectorizers.
- modelling.py -> Functions for training and evaluating the models.
- starter.py -> The main function. This can be personalized to perform only data exploration or model trainig with specific models.
- text_preprocessing.py -> Functions for tokenization and storing a clean dataset used in later steps.

# Setup environment

1. Install miniconda, Azure CLI.
2. Create environment. `conda env create -f environment.yml`
3. Select interpreter in VS Code to be the newly created environment
4. In terminal, run `conda activate nlp-aimsc`
5. Set up starter.py with the necessary model name you want to run.
6. Set up feature_extraction.py, splitting() function with the specific vectorizer you want to use.
7. In terminal, `run python starter.py`
