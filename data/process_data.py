import sys
import pandas as pd
import numpy as np
import sqlalchemy
from sqlalchemy import create_engine


def load_data(messages_filepath, categories_filepath):
    """Method to load 2 csv files and create 1 dataframe to be returned
    Args:
        message_filepath : path to the file containing messages
        categories_filepath : path to the file containing categories where
                                each of messages were classified
    Returns:
        df : merged dataframe containg messages and the labels
    """

    ### read in 2 files and merge them together
    messages = pd.read_csv(messages_filepath)
    categories = pd.read_csv(categories_filepath)
    df = pd.merge(messages, categories, on="id")

    return df



def clean_data(df):
    """Method to clean the dataframe and return it
    Args :
        df : dataframe to be cleaned
    Returns :
        df : cleaned dataframe
    """

    # extract a dataframe of the 36 individual category columns
    categories = df['categories'].str.split(';', expand=True)
    # select the first row of the categories dataframe
    row = categories.iloc[0]
    # use this row to extract a list of new column names for categories.
    category_colnames = row.apply(lambda x: x[:-2])
    # rename the columns of `categories`
    categories.columns = category_colnames

    # Convert category values to just numbers 0 or 1.
    for column in categories:
        # set each value to be the last character of the string
        categories[column] = categories[column].astype(str).str[-1]
        # convert column from string to numeric
        categories[column] = categories[column].astype(int)
    # drop the original categories column from `df`
    df = df.drop(['categories'], axis=1)
    # concatenate the original dataframe with the new `categories` dataframe
    df = pd.concat([df, categories], axis=1, join='inner')

    # Remove duplicates
    df = df.drop_duplicates()
    # Remove two columns
    df.drop(columns=['related','child_alone'],inplace=True)
    return df


def save_data(df, database_filename):
    '''
    Save a pandas DataFrame into a sqlite database.
    input:
        df: The pandas DataFrame object to be saved.
        database_filename: the name of the sqlite database file.
    '''
    engine = create_engine('sqlite:///{}.db'.format(database_filename))
    df.to_sql('ResponseCategory', engine, index=False)


def main():
    if len(sys.argv) == 4:

        messages_filepath, categories_filepath, database_filepath = sys.argv[1:]

        print('Loading data...\n    MESSAGES: {}\n    CATEGORIES: {}'
              .format(messages_filepath, categories_filepath))
        df = load_data(messages_filepath, categories_filepath)

        print('Cleaning data...')
        df = clean_data(df)

        print('Saving data...\n    DATABASE: {}'.format(database_filepath))
        save_data(df, database_filepath)

        print('Cleaned data saved to database!')

    else:
        print('Please provide the filepaths of the messages and categories '\
              'datasets as the first and second argument respectively, as '\
              'well as the filepath of the database to save the cleaned data '\
              'to as the third argument. \n\nExample: python process_data.py '\
              'disaster_messages.csv disaster_categories.csv '\
              'DisasterResponse.db')


if __name__ == '__main__':
    main()
