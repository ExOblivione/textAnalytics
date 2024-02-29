# Welcome!

This is the documentation for the project which aims to perform a comprehensive data analysis, using the dataset available from Kaggle, to identify instances of toxic content within various comments on Wikipedia. This task is important for promoting safety and a more inclusive online space for the users.

Author: Eva Pardi (E.Pardi@liverpool.ac.uk)

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


# References not cited in-line

Toxic Comment Classification Challenge | Kaggle (2017). https://www.kaggle.com/c/jigsaw-toxic-comment-classification-challenge/data.

Kovács, G., Alonso, P. and Saini, R. (2021) ‘Challenges of Hate Speech Detection in Social Media: Data Scarcity, and Leveraging External Resources’, SN Computer Science, 2(2). Available at: https://doi.org/10.1007/s42979-021-00457-3.

George, J.A. (2022) 'An Introduction to Multi-Label Text Classification - Analytics Vidhya - Medium,' Medium, 30 March. https://medium.com/analytics-vidhya/an-introduction-to-multi-label-text-classification-b1bcb7c7364c.

Galke, L. et al. (2022) ‘Are We Really Making Much Progress in Text Classification? A Com-parative Review’. Available at: http://arxiv.org/abs/2204.03954.

Morales-Hernández, R.C., Juagüey, J.G. and Becerra-Alonso, D. (2022) ‘A Comparison of Multi-Label Text Classification Models in Research Articles Labeled with Sustainable De-velopment Goals’, IEEE Access, 10, pp. 123534–123548. Available at: https://doi.org/10.1109/ACCESS.2022.3223094.