from collections import Counter
from nltk import ngrams
import spacy

# Import the english language library for spacy
nlp = spacy.load("en_core_web_sm")

def asset_count(data, labels):
    """
    This function takes a DataFrame data and a list of labels as input.
    It counts the number of sentences and words in the comments of each row for each label.
    The counts are then printed out.
    """

    # Initialize two dictionaries to hold the counts of sentences and words for each label
    sentence_counts = {label: 0 for label in labels}
    word_counts = {label: 0 for label in labels}
    
    # Iterate over each row in the data
    for index, row in data.iterrows():
        # Get the comment text from the current row
        comment = row['comment_text']
        # Iterate over each label
        for label in labels:
            # Use the nlp function to process the comment text
                doc = nlp(comment)
                # Increment the sentence count for this label by the number of sentences in the comment
                sentence_counts[label] += len(list(doc.sents))
                # Increment the word count for this label by the number of words in the comment
                word_counts[label] += len(doc)

    # Print the sentence and word counts for each label
    print(f"Number of sentences: {sentence_counts}")
    print(f"Number of words: {word_counts}")

def important_ngrams(data, labels, n):
    """
    This function important_ngrams takes a DataFrame data, a list of labels, and an integer n as input.
    It finds the most common n-grams in the sentences, then prints the 5 most common n-grams for each label.
    """
    # Iterate over each label
    for label in labels:
        # Select the sentences from the rows where the current label is present
        sentences = data[data[label] == 1]["cleaned_text"]
        all_ngrams = []
        # Iterate over each sentence
        for sentence in sentences:
            # Use the nlp function to process the sentence
            doc = nlp(sentence)
            # Extract the words from the sentence, convert to lower case, and filter out stop words
            words = [token.text.lower() for token in doc if not token.is_stop]
            # Generate n-grams from the list of words
            n_grams = ngrams(words, n)
            # Add the generated n-grams to the list of all n-grams
            all_ngrams.extend(n_grams)
        # Count the occurrences of each n-gram
        ngram_counts = Counter(all_ngrams)
        # Get the 5 most common n-grams
        most_common_ngrams = ngram_counts.most_common(5)
        print(f"\nMost common {n}-grams for label '{label}':")
        # Print each of the most common n-grams and their counts
        for ngram, count in most_common_ngrams:
            print(f"{' '.join(ngram)}: {count}")