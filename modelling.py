# Multi-label text classification
# Wrap model into Binary Relevance, Classifier Chains, or Label Powerset.
# Ref.: https://pypi.org/project/scikit-multilearn/
#       https://medium.com/analytics-vidhya/an-introduction-to-multi-label-text-classification-b1bcb7c7364c
#       https://towardsdatascience.com/multi-label-text-classification-with-scikit-learn-30714b7819c5
from skmultilearn.problem_transform import BinaryRelevance
from skmultilearn.problem_transform import ClassifierChain
from skmultilearn.problem_transform import LabelPowerset

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import GridSearchCV

from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score

from feature_extraction import splitting

def log_reg(data):
    """
    This function performs multi-label text classification using a Logistic Regression model.
    Ref.: https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LogisticRegression.html
    """
    # Split the data into training and testing sets
    trainX, trainY, testX, testY = splitting(data)

    # Create a ClassifierChain model using LogisticRegression as the base classifier.
    classifier = ClassifierChain(
            LogisticRegression(
                multi_class='multinomial',  # Specify the multiclass strategy ('multinomial' for cross-entropy loss)
                penalty='l2',  # Regularization penalty
                class_weight='balanced',  # Handle class imbalance by adjusting class weights.
                solver='lbfgs',  # Choose the optimization solver ('lbfgs' for limited-memory)
                tol=1e-5,  # Tolerance for stopping criteria during optimization
                C=0.1,  # Inverse of regularization strength
                max_iter=500,  # Maximum number of iterations for optimization
                verbose=1  # Print progress messages during training
            )
        )

    # Fit the model on the training data
    classifier.fit(trainX, trainY)
    # Predict the labels for the test data
    predicted = classifier.predict(testX)
    # Evaluate the model's performance and return the evaluation metrics
    return model_eval(testY, predicted)

def rand_forest(data):
    """
    This function performs multi-label text classification using a Random Forest model.
    Ref.: https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestClassifier.html
    """
    # Split the data into training and testing sets
    trainX, trainY, testX, testY = splitting(data)
    
    # Create a BinaryRelevance model using RandomForestClassifier as the base classifier.
    classifier = BinaryRelevance(
        RandomForestClassifier(
            class_weight='balanced',  # Adjust class weights to handle class imbalance
            n_estimators=600,  # Number of decision trees in the random forest
            max_depth=50,  # Maximum depth of each decision tree
            min_samples_split=10,  # Minimum number of samples required to split an internal node
            min_samples_leaf=5,  # Minimum number of samples required to be at a leaf node
            verbose=1  # Print progress messages during training
        )
    )
    
    # Fit the model on the training data
    classifier.fit(trainX, trainY)
    
    # Predict the labels for the test data
    predicted = classifier.predict(testX)
    
    # Evaluate the model's performance and return the evaluation metrics
    return model_eval(testY, predicted)

def mlp_nn(data):
    """
    This function performs multi-label text classification using a Multi-Layer Perceptron model.
    Ref.: https://scikit-learn.org/stable/modules/generated/sklearn.neural_network.MLPClassifier.html
    """
    # Split the data into training and testing sets
    trainX, trainY, testX, testY = splitting(data)
        
    # Initialize a wrapper with a Multi-Layer Perceptron (MLP) Classifier
    # Wrapper can be set to BinaryRelevance, ClassifierChain, or LabelPowerset
    classifier = LabelPowerset(
        MLPClassifier(
            hidden_layer_sizes=(150, 150, 100),  # Specify the architecture of hidden layers
            activation='relu',  # Activation function for hidden layers
            solver='adam',  # Optimization algorithm ('adam' for larger datasets, 'lbfgs' for smaller datasets)
            alpha=0.1,  # Regularization strength (higher values increase regularization)
            learning_rate_init=0.000001,  # Initial learning rate for weight updates (small value for stability)
            max_iter=1500,  # Maximum number of iterations for training
            verbose=1  # Print progress message during training
        )
    )

    # Fit the model on the training data
    classifier.fit(trainX, trainY)
    # Predict the labels for the test data
    predicted = classifier.predict(testX)
    
    # Evaluate the model's performance and return the evaluation metrics
    return model_eval(testY, predicted)

def model_eval(testY, predicted):
    """
    Calculate evaluation metrics for a certain model:
        - accuracy
        - precision
        - recall
        - F1-score
        - ROC/AUC
    """
    acc = accuracy_score(testY, predicted)

    # Calculate the precision, recall, and F1-score of the model's predictions
    # 'micro' calculates metrics globally by counting the total true positives, false negatives and false positives
    prec = precision_score(testY, predicted, average='micro', zero_division='warn')
    recall = recall_score(testY, predicted, average='micro', zero_division='warn')
    f1 = f1_score(testY, predicted, average='micro')

    # Calculate the Area Under the Receiver Operating Characteristic Curve (ROC AUC) from prediction scores
    # 'ovr' stands for One-vs-Rest
    auc = roc_auc_score(testY, predicted.toarray(), average='micro', multi_class='ovr')

    # Return all the calculated metrics
    return acc, prec, recall, f1, auc