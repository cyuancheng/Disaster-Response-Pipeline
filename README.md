# Disaster Response Pipeline

### Chiyuan Cheng 06/04/2019

## Table of Contents

- [Project Overview](#overview)
  - [ETL Pipeline](#etl_pipeline)
  - [ML Pipeline](#ml_pipeline)
  - [Flask Web App](#flask)
- [Project Descriptions](#run)
  - [Data Cleaning](#cleaning)
  - [Training Classifier](#training)
  - [Starting the Web App](#starting)
- [Conclusion](#conclusion)
- [Files](#files)
- [Software Requirements](#sw)
- [Credits and Acknowledgements](#credits)


<a id='overview'></a>

## 1. Project Overview

In this project, I used data engineering skills to: 

-  analyze real natural disaster data from <a href="https://www.figure-eight.com/" target="_blank">Figure Eight</a> , 
-  create pipelines for ETL and machine learning model that classifying disaster messages sent during a natural disaster, and 
-  create a web app where a relief agent can input a new message and get classification results in several categories in a real time.  

The goal is to direct each emergency message to the appropriate disaster relief agency who can provide immediate assistance.


### 1.1. ETL Pipeline

ETL pipeline In *data/process_data.py* to:

- Load the `messages` and `categories` dataset
- Merge the two datasets
- Clean the data
- Store it in a **SQLite database**

<a id='ml_pipeline'></a>

### 1.2. ML Pipeline

ML pipeline in *models/train_classifier.py* to:

- Load data from the **SQLite database**
- Split the data into training and testing sets
- Build a text processing and optimized machine learning pipeline
- Train and tunes a model using GridSearchCV
- Output result on the test set
- Export the final model as a pickle file

<a id='flask'></a>

### 1.3. Flask Web App

<a id='eg'></a>

Running [run.py](#com) **from app directory** will start the web app where users can enter their message queries during a natural disaster, e.g. *"Is the Hurricane over or is it not over"*.

This app will classify the text message into categories, so that appropriate relief agency can be reached out for help.

<a id='run'></a>

## 2. Project Description

Three steps for this project:

<a id='cleaning'></a>

### 3.1. Data Processing (ETL)

**Go to the project directory** and the run the following command:

```bat
python data/process_data.py data/disaster_messages.csv data/disaster_categories.csv data/DisasterResponse.db
```

The first two arguments are input data and the third argument is the SQLite Database where the cleaned data is saved. The ETL pipeline is wrote in *process_data.py*.


<a id='training'></a>

### 2.2. Train Classifier (ML)

After the data cleaning process, run this command **from the project directory**:

```bat
python models/train_classifier.py data/DisasterResponse.db models/classifier.pkl
```

This will use cleaned data to train the model, optimize the hyperparameters of the model with grid search and saved the model to a pickle file (*classifer.pkl*).


<a id='starting'></a>

### 2.3. Web app

Now we use the Flask web app to predict the classification from the input messages.

**Go the app directory** and run the following command:

<a id='com'></a>

```bat
python run.py
```

The screen shots of the web app are below:

**_Screenshot 1_**

![master](app/Disasters.png)



**_Screenshot 2_**

![results](app/Disasters_1.png)

<a id='conclusion'></a>

## 3. Conclusion

Some information about training data set as seen on the main page of the web app.


As we can see the data is highly imbalanced. Though the accuracy metric is [high](#acc) (you will see the exact value after the model is trained by grid search, it is ~0.94), it has a poor value for recall (~0.6). So, take appropriate measures when using this model for decision-making process at a larger scale or in a production environment.

<a id='files'></a>

## 4. Files

<pre>
.
├── app
│   ├── run.py------------------------# flask file to run app
│   ├── imag_webapp_1		# screenshot of web app
│   ├── imag_webapp_2 		# screenshot of web app
│   └── templates
│       ├── go.html-------------------# classification result page of web app
│       └── master.html---------------# main page of web app
├── data
│   ├── DisasterResponse.db-----------# database to save cleaned data
│   ├── disaster_categories.csv-------# raw data to process
│   ├── disaster_messages.csv---------# raw data to process
│   └── process_data.py---------------# perform ETL pipline
├── models
│   ├── train_classifier.py-----------# perform classification pipeline
│   └── classifier.pkl		-----------# classifier result
├── notebook
│   ├── ETL Pipeline Preparation.ipynb----------# Jupyter notebook for ETL 
│   └── ML Pipeline Preparation.ipynb-	-----------# Jupyter notebook for ML

</pre>

<a id='sw'></a>

## 5. Software Requirements

This project uses **Python 3.6.6** and the necessary libraries are mentioned in _requirements.txt_.
The standard libraries which are not mentioned in _requirements.txt_ are _collections_, _json_, _operator_, _pickle_, _pprint_, _re_, _sys_, _time_ and _warnings_.

<a id='credits'></a>

## 6. Credits and Acknowledgements

Thanks <a href="https://www.udacity.com" target="_blank">Udacity</a> for letting me use their logo as favicon for this web app.

Another <a href="https://medium.com/udacity/three-awesome-projects-from-udacitys-data-scientist-program-609ff0949bed" target="_blank">blog post</a> was a great motivation to improve my documentation. This post discusses some of the cool projects from <a href="https://in.udacity.com/course/data-scientist-nanodegree--nd025" target="_blank">Data Scientist Nanodegree</a> students. This really shows how far we can go if we apply the concepts learned beyond the classroom content to build something that inspire others.