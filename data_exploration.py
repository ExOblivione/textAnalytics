import pandas as pd

from asset_count import asset_count, important_ngrams
from text_preprocessing import store

def explore(df, labels):
    """
    The function performs detailed data analysis together with observation counting.
    """
    
    print("The shape of the training data:")
    print(df.shape)
    print("\nData types of each columns:")
    print(df.info())

    print("\nThe dataset holds missing values: ")
    print(df.isnull().values.any())
    print("\nThe dataset holds duplicates: ")
    print(df.duplicated().values.any())
    print(df.describe())

    k = (
    pd.DataFrame(  # Create a new DataFrame
        df[labels]  # Select the label columns in the dataframe
            .melt(var_name='column', value_name='value')  # Reshape the DataFrame (labels, and values in two columns)
            .value_counts()  # Count the number of occurrences
        )
        .sort_values(by=['column'])  # Sort the DataFrame by the label column
        .rename(columns={0: 'count'})  # Rename the column that contains the counts
    )

    print("\n")
    print(k)

    asset_count(df, labels)
    important_ngrams(store(df), labels=labels, n=1)
    important_ngrams(store(df), labels=labels, n=2)
    important_ngrams(store(df), labels=labels, n=3)