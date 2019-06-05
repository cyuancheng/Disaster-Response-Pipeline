import json
import plotly
import pandas as pd
import re


import re

import nltk
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords
nltk.download(['punkt', 'wordnet','stopwords'])

from flask import Flask
from flask import render_template, request, jsonify
from plotly.graph_objs import Bar
from sklearn.externals import joblib
from sqlalchemy import create_engine


app = Flask(__name__)


def tokenize(text):
     #Normalize
    text = re.sub(r"[^a-zA-Z0-9]", " ", text.lower())

    #Tokenize
    words = nltk.word_tokenize(text)

    #Remove stopwords
    words= [w for w in words if w not in stopwords.words("english")]
    # Reduce words to their root form
    lemmatizer = WordNetLemmatizer()

    clean_tokens = [lemmatizer.lemmatize(lemmatizer.lemmatize(word)) for word in words]

    return clean_tokens

# load data
engine = create_engine('sqlite:///../data/DisasterResponse.db')
df = pd.read_sql_table('ResponseCategory', engine)

# load model
model = joblib.load("../model/classifier.pkl")


# index webpage displays cool visuals and receives user input text for model
@app.route('/')
@app.route('/index')
def index():

    # extract data needed for visuals
    # TODO: Below is an example - modify to extract data for your own visuals
    genre_counts = df.groupby('genre').count()['message']
    genre_names = list(genre_counts.index)

    # Calculate proportion of each category with label = 1

    cat_props = df.drop(['id', 'message', 'original', 'genre'], axis = 1).sum()/len(df)
    cat_props = cat_props.sort_values(ascending = False)
    cat_names = list(cat_props.index)

    # Get occurence of each type
    df_copy = df.copy()
    counts_percentage = df_copy.drop(['id', 'message', 'original', 'genre'], axis = 1).sum()/len(df)
    counts_percentage = counts_percentage.sort_values(ascending = False)
    df_copy_cols = df_copy.columns
    #counts = []
    #for col in df_copy:
#        counts.append((df_copy[col]==1).sum())

    # Get occurence of each type in percentage
#    counts_percentage = []
#    for col in df_copy:
#        counts_percentage.append((df_copy[col]==1).sum()/df.size * 100)


    # create visuals
    # TODO: Below is an example - modify to create your own visuals
    graphs = [
        {
            'data': [
                Bar(
                    x=genre_names,
                    y=genre_counts
                )
            ],

            'layout': {
                'title': 'Distribution of Message Genres',
                'yaxis': {
                    'title': "Count"
                },
                'xaxis': {
                    'title': "Genre"
                }
            }
        },
    #    {
        #    'data': [
    #            Bar(
    #                x=df_copy_cols,
    #                y=counts
    #            )
    #        ],
        #    'layout': {
    #            'title': 'Distribution of Tags',
#                'yaxis': {
#                    'title': "Count"
#                },
#                'xaxis': {
#                    'title': "Tags"
#                }
#            }
    #    },
            {
            'data': [
                Bar(
                    x=df_copy_cols,
                    y=counts_percentage
                )
            ],

            'layout': {
                'title': 'Distribution of Tags',
                'yaxis': {
                    'title': "%"
                },
                'xaxis': {
                    'title': "Tags"
                }
            }
        }
    ]

    # encode plotly graphs in JSON
    ids = ["graph-{}".format(i) for i, _ in enumerate(graphs)]
    graphJSON = json.dumps(graphs, cls=plotly.utils.PlotlyJSONEncoder)

    # render web page with plotly graphs
    return render_template('master.html', ids=ids, graphJSON=graphJSON)

# web page that handles user query and displays model results
@app.route('/go')
def go():
    # save user input in query
    query = request.args.get('query', '')

    # use model to predict classification for query
    classification_labels = model.predict([query])[0]
    classification_results = dict(zip(df.columns[4:], classification_labels))

    # This will render the go.html Please see that file.
    return render_template(
        'go.html',
        query=query,
        classification_result=classification_results
    )


def main():
    app.run(host='0.0.0.0', port=3001, debug=True)


if __name__ == '__main__':
    main()
