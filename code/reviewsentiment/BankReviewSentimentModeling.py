# This code snippet will load the referenced package and return a DataFrame.
# If the code is run in a PySpark environemnt, then the code will return a
# Spark DataFrame. If not, the code will return a Pandas DataFrame. You can
# copy this code snippet to another code file as needed.    C:\Users\zhouf\.amlenvrc.cmd

# Import python module

import pickle
import sys
import os

import csv
import pandas as pd
import matplotlib
import string
import pylab as pl

from sklearn.feature_extraction.text import TfidfVectorizer
from nltk.stem.porter import PorterStemmer

from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2
from sklearn.cross_validation import train_test_split
from sklearn import grid_search
from sklearn.linear_model import LogisticRegression
from sklearn import metrics

from azureml.logging import get_azureml_logger
from azureml.dataprep.package import run

# initialize the logger

run_logger = get_azureml_logger() 

# Create the outputs folder

os.makedirs('./outputs', exist_ok=True)

print('Python version: {}'.format(sys.version))
print()

# Step 1 - Data Preparation

# Load the bank review dataset

text_df = pd.read_csv('BankReviewTrainingSample.csv')
text_df.head()

text_df.columns = ['label_column', 'text_column'] #rename column names
label_column = text_df["label_column"]
text_df["label_column"] = label_column.map(lambda label: 1 if label == 4 or label == 5 else 0) #replace label 4 to label 1
text_df = text_df.dropna()

text_df.head()

text_df["label_column"].hist()

# Step 2 - Text Preprocessing

stop_words_df = pd.read_csv('StopWords.csv')
stop_words = set(stop_words_df["Col1"].tolist())
for item in string.ascii_lowercase: #load stop words
    if item != "i":
        stop_words.add(item)

from nltk.tokenize import RegexpTokenizer
text_column = []
for line in text_df.text_column:
    value = " ".join(item.lower()
                     for item in RegexpTokenizer(r'\w+').tokenize(line)
                     if item.lower() not in stop_words)
    text_column.append(value)
text_df.text_column = text_column

text_df.head()

# Step 3 - unigrams TF-IDF feature extraction

stemmer = PorterStemmer()
text_list = text_df["text_column"].tolist()

# Tokenize the sentences in text_list and remove morphological affixes from words.

def stem_tokens(tokens, stemmer_model):
    '''
    :param tokens: tokenized word list
    :param stemmer: remove stemmer
    :return:  tokenized and stemmed words
    '''
    return [stemmer_model.stem(original_word) for original_word in tokens]

def tokenize(text):
    '''
    :param text: raw test
    :return: tokenized and stemmed words
    '''
    tokens = text.strip().split(" ")
    return stem_tokens(tokens, stemmer)

# Initialize the TfidfVectorizer to compute tf-idf for each word

tfidf = TfidfVectorizer(tokenizer=tokenize, stop_words='english', max_df=160000,
                        min_df=9, norm="l2", use_idf=True)
tfs = tfidf.fit_transform(text_list)
print(tfs)
tfs

# Perform Feature Selection using Chi-squared Test

labels = text_df.label_column.tolist()  # Load labels
# Perform chi-squared test to select top 30 features
selected_features = SelectKBest(chi2, k=30).fit_transform(tfs, labels)
selected_features

# Step 4 - Train and Evaluate Models with Different Top K Words

# Select top features

def select_top_features(chi_square, top_k_features, tf_idf_features, label_list):
    '''
    :param chi2: chi squared feature selctor
    :param top_k: number of top features
    :param tfs: tf-idf feature list
    :param labels: labels of the feature list
    :return: feature list after chi squared test
    '''
    return SelectKBest(chi_square, k=top_k_features).fit_transform(tf_idf_features, label_list)

# Split Data for training and testing

def split_data(selected_features_list, label_list):
    '''
    :param selected_features: feature list after feature selection
    :param labels: labels of the feature list
    :return: 80% training set and 20% test set
    '''
    return train_test_split(selected_features_list, label_list, test_size=0.2, random_state=12345)

# Initialize Sweep Parameter

def make_sweep_parameter_dict():
    '''
    :return: parameters of logistic regression model to tune
    '''
    return {"penalty": ("l1", "l2"), "C": [0.1, 1, 10]}

# Sweep Parameters with Logistic Regression using Cross Validation

def sweep_logistic_regression_cross_val(train_feature, train_label):
    '''
    :param train_x: feature of training set
    :param train_y: label of training set
    :return: trained logistic regression model
    '''
    model = LogisticRegression()
    classifier = grid_search.GridSearchCV(model, make_sweep_parameter_dict())
    classifier.fit(train_feature, train_label)
    return classifier

# Validate Model using Testing Data

def prediction(classifier, test_feature):
    '''
    :param clf: the trained classifier
    :param test_x: feature of test set
    :return: score probability of test set
    '''
    return [classifier.predict_proba(feature)[0][1] for feature in test_feature]

# Compute AUC

def calc_auc(test_label, predict_probability_list):
    '''
    :param test_y: label of test set
    :param predict: score probability of test set
    :return: falst positive list, true positive list,, AUC
    '''
    fpr_list, tpr_list = metrics.roc_curve(test_label,
                                           predict_probability_list, pos_label=1)[:-1]
    auc = metrics.auc(fpr_list, tpr_list)
    return fpr_list, tpr_list, auc

# Put it all together
# We select different top k words list based on Chi-squared test, and then a 3-fold cross validation is used to find the optimal parameters for the logistic regression algorithm based on the top k words. We trained the logistic regression algorithm with the optimal parameters, test its performance on test set and plot the ROC curve for each top k words list

top_k_list = [5, 10, 20, 30]  #Select top words to see performance
pl.clf()
for top_k in top_k_list:
    selected_features = select_top_features(chi2, top_k, tfs, labels)
    train_x, test_x, train_y, test_y = split_data(selected_features, labels)
    clf = sweep_logistic_regression_cross_val(train_x, train_y)
    print ("Export the model to model_"+str(top_k)+".pkl")
    f = open('./outputs/model_'+str(top_k)+'.pkl', 'wb')
    pickle.dump(clf, f)
    f.close()
    print("Import the model from model_"+str(top_k)+".pkl")
    f2 = open('./outputs/model_'+str(top_k)+'.pkl', 'rb')
    clf2 = pickle.load(f2)
    predict = prediction(clf, test_x)
    fpr, tpr, roc_auc = calc_auc(test_y, predict)
    pl.plot(fpr, tpr, label='ROC curve (area = %0.2f) with Top %d words' % (roc_auc, top_k))
pl.plot([0, 1], [0, 1], 'k--')
pl.xlim([0.0, 1.0])
pl.ylim([0.0, 1.0])
pl.xlabel('False Positive Rate')
pl.ylabel('True Positive Rate')
pl.title('Sentiment Analysis')
pl.legend(loc="lower right", prop={'size':8})
pl.show()  
