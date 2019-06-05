import sys
import pandas as pd
import numpy as np
import pickle
from sqlalchemy import create_engine

import re

import nltk
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords
nltk.download(['punkt', 'wordnet','stopwords'])

from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.multioutput import MultiOutputClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB, MultinomialNB
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import classification_report, accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import f1_score
from sklearn.metrics import precision_recall_fscore_support


def load_data(database_filepath):
    # load data from database
    engine = create_engine("sqlite:///{}".format(database_filepath))
    df = pd.read_sql_table('ResponseCategory', engine)
    X = df["message"] # features
    Y = df.drop(axis=1, labels=["id", "message", "original", "genre"]) # labels
    category_names = Y.columns
    return X, Y, category_names



def tokenize(text):
    '''
    process text data
    tokenization, lemmatization, remove stopwords
    '''
    text = re.sub(r"[^a-zA-Z0-9]", " ", text)
    tokens = word_tokenize(text)

    lemmatizer = WordNetLemmatizer()
    clean_tokens = []

    for token in tokens:
        clean_tok = lemmatizer.lemmatize(token).lower().strip()

        # remove stop words
        if clean_tok not in set(stopwords.words("english")):
            clean_tokens.append(clean_tok)

    return clean_tokens


def build_model():
    pipeline = Pipeline([("vect",CountVectorizer(tokenizer=tokenize)),
                     ("tfidf",TfidfTransformer()),
                     ("clf", MultiOutputClassifier(LogisticRegression()))])

    parameters = {
    "clf__estimator__C" : [0.1,1,10]
    }
    model = GridSearchCV(pipeline, param_grid=parameters, cv=3)
    return model

def metrics(Y_test, Y_pred):
    name = [];
    precision=[];
    recall=[];
    fscore=[];
    accuracy=[];
    for i in range(Y_test.shape[1]):
        name.append(Y_test.columns.tolist()[i])
        precision.append(precision_recall_fscore_support(np.array(Y_test)[:, i], np.array(Y_pred)[:, i], average='weighted')[0])
        recall.append(precision_recall_fscore_support(np.array(Y_test)[:, i], np.array(Y_pred)[:, i], average='weighted')[1])
        fscore.append(precision_recall_fscore_support(np.array(Y_test)[:, i], np.array(Y_pred)[:, i], average='weighted')[2])
        accuracy.append(accuracy_score(np.array(Y_test)[:, i], np.array(Y_pred)[:, i]))

    metrics_df = pd.DataFrame(data = {'Precision':precision,'Recall':recall,'F1-Score':fscore, 'Accuracy':accuracy}, index = name)#, index = name)#, columns = ['Precision', 'Recall', 'F1','accuracy'])
    return metrics_df

def evaluate_model(model, X_test, Y_test, category_names):
    """
    Shows model's performance on test data
    Args:
    model: trained model
    X_test: Test features
    Y_test: Test targets
    category_names: Target labels
    """

    # predict
    Y_pred = model.predict(X_test)

    # print classification report
    metrics_df = metrics(Y_test, Y_pred)
    print(metrics_df)
    print("F1 score mean : ", metrics_df['F1-Score'].mean())
    #plt.figure(figsize=[12,4])
    #metrics_df['F1-Score'].sort_values(ascending=False).plot("bar", colormap="Blues_r")


def save_model(model, model_filepath):
    pickle.dump(model, open(model_filepath, 'wb'))


def main():
    if len(sys.argv) == 3:
        database_filepath, model_filepath = sys.argv[1:]
        print('Loading data...\n    DATABASE: {}'.format(database_filepath))
        X, Y, category_names = load_data(database_filepath)
        X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2)

        print('Building model...')
        model = build_model()

        print('Training model...')
        model.fit(X_train, Y_train)

        print('Evaluating model...')
        evaluate_model(model, X_test, Y_test, category_names)

        print('Saving model...\n    MODEL: {}'.format(model_filepath))
        save_model(model, model_filepath)

        print('Trained model saved!')

    else:
        print('Please provide the filepath of the disaster messages database '\
              'as the first argument and the filepath of the pickle file to '\
              'save the model to as the second argument. \n\nExample: python '\
              'train_classifier.py ../data/DisasterResponse.db classifier.pkl')


if __name__ == '__main__':
    main()
