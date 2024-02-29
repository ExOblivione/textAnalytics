import numpy as np
import pandas as pd

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from gensim.models import Word2Vec

from sklearn.model_selection import train_test_split
from text_preprocessing import spacy_tokenizer

def featurization_TFIDF(train, test):
    """
    Convert a collection of raw documents to a matrix of TF-IDF features.
        - The ngram_range parameter defines the range of n-grams to consider, 
            in this case, it uses unigrams, bigrams, and trigrams as well.
        - The min_df parameter sets the minimum document frequency for a term to be included in the vocabulary,
            which is configured to be 5.
    
    After fitting, the resulting vector contains the learned vocabulary and its TF-IDF weights.
    Finally, it transforms the training and test features using the vocabulary learned during the training.

    Ref.: https://scikit-learn.org/stable/modules/generated/sklearn.feature_extraction.text.TfidfVectorizer.html
    """

    # Initialize a TfidfVectorizer object with a specific tokenizer, minimum document frequency, and ngram range
    tfidf_vector = TfidfVectorizer(tokenizer=spacy_tokenizer, min_df=5, ngram_range=(1,3))
    # Fit the TfidfVectorizer to the training data
    tfidf_vector.fit(train)

    # feature_names_tfidf contains the names of the features (terms) in the TF-IDF representation.
    # These correspond to the columns in the sparse matrix.
    feature_names_tfidf = tfidf_vector.get_feature_names_out()

    # Convert the feature names to a DataFrame
    feature_tfidfDF = pd.DataFrame(feature_names_tfidf)
    # Save the feature names to a CSV file
    feature_tfidfDF.to_csv("./processed/feature_names_tfidf.csv", index=False,header=False)
    
    # Transform training and test features with the learned vocabulary, and return them for training
    X_train = tfidf_vector.transform(train)
    X_test = tfidf_vector.transform(test)
    return X_train, X_test

def featurization_BoW(train, test):
    """
    Convert a collection of text documents to a matrix of token counts.
    Ref.: https://scikit-learn.org/stable/modules/generated/sklearn.feature_extraction.text.CountVectorizer.html
    """
    # Initialize a CountVectorizer object with a specific tokenizer and ngram range
    bow_vector = CountVectorizer(tokenizer=spacy_tokenizer, ngram_range=(1,3))
    # Fit the CountVectorizer to the training data and transform it into a matrix of token counts
    X_train = bow_vector.fit_transform(train)
    # Transform the test data into a matrix of token counts using the fitted CountVectorizer
    X_test = bow_vector.transform(test)

    # Get the feature names from the CountVectorizer
    feature_names_bow = bow_vector.get_feature_names_out()
    # Convert the feature names to a DataFrame
    feature_bowDF = pd.DataFrame(feature_names_bow)
    # Save the feature names to a CSV file
    feature_bowDF.to_csv("./processed/feature_names_bow.csv", index=False,header=False)
    
    # Return the transformed training and test data
    return X_train, X_test

def featurization_WE(train, test):
    """
    It creates dense vector representations for words based on their context in a large text.
    The two main architectures in Word2Vec are:
        - Skip-gram: Predicts context words given a target word.
        - Continuous Bag of Words (CBOW): Predicts a target word based on its context.
    Ref.: https://www.tensorflow.org/text/tutorials/word2vec
    """
    # Initialize a Word2Vec model with specific parameters
    we_vector = Word2Vec(train, min_count=1, vector_size=100, window=5, sg=0, epochs=50, seed=42)
    # Generate sentence vectors for each sentence in the training data
    sentence_vectors = [sentence_vector(sentence, we_vector) for sentence in train]
    # Convert the sentence vectors to a DataFrame
    feature_weDF = pd.DataFrame(sentence_vectors)
    # Save the sentence vectors to a CSV file
    feature_weDF.to_csv("./processed/sentence_vectors_we.csv", index=False,header=False)
    
    # Transform the train and test data into vectors using the fitted Word2Vec model
    X_train = np.array([words_to_vector(doc, we_vector) for doc in train])
    X_test = np.array([words_to_vector(doc, we_vector) for doc in test])

    # Return the transformed training and test data        
    return X_train, X_test

def sentence_vector(sentence, model):
    """
    This function is used to convert a sentence into a vector representation by averaging the Word2Vec vectors
    of all the words in the sentence that are present in the given Word2Vec model's vocabulary.

    Ref.: https://radimrehurek.com/gensim/models/word2vec.html
    """
    # For each word in the sentence, if the word is in the model's vocabulary,
    # get its vector representation and store it in the 'vectors' list
    vectors = [model.wv[word] for word in sentence if word in model.wv]
    
    if vectors:
        # If the 'vectors' list is not empty, return the average of all vectors.
        # This will create a vector that represents the sentence.
        return sum(vectors) / len(vectors)
    else:
        # If the 'vectors' list is empty, return None.
        return None

# Define a function to transform a list of words into a vector
def words_to_vector(words, model):
    # Initialize an empty list to store the word vectors
    vectors = []
    # Loop through each word in the list
    for word in words:
        # Try to get the vector for the word from the model
        try:
            vector = model.wv.get_vector(word)
            # Append the vector to the list
            vectors.append(vector)
        # If the word is not in the model vocabulary, skip it
        except KeyError:
            continue
    # If the list is not empty, return the mean of the vectors
    if vectors:
        return np.mean(vectors, axis=0)
    # Otherwise, return a zero vector
    else:
        return np.zeros(model.vector_size)
    
def splitting(data):
    """
    This function is used to split the given data into training and testing sets,
    transform the input data into a numerical representation using the specified vectorizer.

    The shapes of the transformed data and labels are printed for debugging purposes.
    """

    # Split the data into training and testing sets. The 'comment_text' column is used as the input features (X),
    # and all other columns (from index 1 onwards) are used as the output labels (Y).
    # The test_size parameter is set to 0.3, meaning that 30% of the data will be used for the test set, and the rest for the training set.
    X_train, X_test, trainY, testY = train_test_split(data["comment_text"], data[data.columns[1:]], test_size=0.3)

    # Transform the training and test input data into a numerical representation using a vectorizer
    trainX, testX = featurization_TFIDF(X_train, X_test)
    # Notes:
    #   - featurization_BoW for LogReg
    #   - featurization_TFIDF for RandForest, NN

    # Print the shapes of the transformed training and test data and labels
    print(trainX.shape)
    print(trainY.shape)
    print(testX.shape)
    print(testY.shape)

    # Return the transformed training and test data and labels
    return trainX, trainY, testX, testY