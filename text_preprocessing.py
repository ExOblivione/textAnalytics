import spacy
import string
import re

# Load the English language model, spacy stopwords, and punctuations
nlp = spacy.load("en_core_web_sm")
spacy_stopwords = spacy.lang.en.stop_words.STOP_WORDS
punctuations = string.punctuation

def spacy_tokenizer(text):
    """ 
    This function takes a string of text as input and returns a list of lemmatized tokens (words),
    where each token is in lower case, non-alphabetic characters and stop words are removed.
    """
    # Remove non-alphabetic characters from the text using a regular expression
    cleaned_text = re.sub(r"[^a-zA-Z\s]", "", text)
    # For each token in the processed text, check if it is an alphabetic word and not a stop word
    # If it is, lemmatize the token and convert it to lower case, then add it to the list of tokens
    tokens = [token.lemma_.lower() for token in nlp(cleaned_text) if token.is_alpha and not token.is_stop]
    # Return the list of tokens
    return tokens

def clean_text(text):
    doc = nlp(text)
    # Lemmatize each token in the processed text, convert it to lower case, and remove leading/trailing whitespace
    # If the lemma of the token is "-PRON-", use the lower case form of the token instead
    cleaned_tokens = [ token.lemma_.lower().strip() if token.lemma_ != "-PRON-" else token.lower_ for token in doc ]
    # Remove stop words and punctuation from the list of cleaned tokens
    cleaned_tokens = [ word for word in cleaned_tokens if word not in spacy_stopwords and word not in punctuations ]
    # Join the cleaned tokens into a single string with spaces in between
    cleaned_text = ' '.join(cleaned_tokens)
    # Return the cleaned text
    return cleaned_text

def store(df):
    # Select the 'comment_text' column from the DataFrame 'df'
    col2tokenize = df['comment_text']
    
    # Apply the clean_text function to each value in the 'comment_text' column
    # Store the results in a new column 'cleaned_text' in the DataFrame
    df['cleaned_text'] = col2tokenize.apply(clean_text)
    
    # Write the DataFrame to a CSV file
    df.to_csv("./processed/processed_train.csv", sep=",", index=False)
    
    # Return the DataFrame
    return df