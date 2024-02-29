import pandas as pd
from data_exploration import explore
from modelling import log_reg, rand_forest, mlp_nn

if __name__ == "__main__":

    trainDF = pd.read_csv("./source/train.csv").drop(columns=["id"])
    labels = ['toxic', 'severe_toxic', 
              'obscene', 'threat', 'insult', 
              'identity_hate']

    explore(trainDF, labels)

    acc, prec, recall, f1, auc = log_reg(trainDF)
    
    print("Accuracy:", acc)
    print("Precision:", prec)
    print("Recall:", recall)
    print("F1 score:", f1)
    print("AUC:", auc)

    acc, prec, recall, f1, auc = rand_forest(trainDF)
    
    print("Accuracy:", acc)
    print("Precision:", prec)
    print("Recall:", recall)
    print("F1 score:", f1)
    print("AUC:", auc)

    acc, prec, recall, f1, auc = mlp_nn(trainDF)
    
    print("Accuracy:", acc)
    print("Precision:", prec)
    print("Recall:", recall)
    print("F1 score:", f1)
    print("AUC:", auc)