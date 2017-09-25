# This code snippet will load the referenced package and return a DataFrame.
# If the code is run in a PySpark environemnt, then the code will return a
# Spark DataFrame. If not, the code will return a Pandas DataFrame. You can
# copy this code snippet to another code file as needed.    C:\Users\zhouf\.amlenvrc.cmd

# Import python module

import pickle
import sys
import os

import dataprep
from dataprep.Package import Package

import pandas as pd
import numpy as np
import csv

from sklearn.preprocessing import LabelEncoder

from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier

from sklearn import grid_search
from sklearn import metrics

from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.cross_validation import train_test_split

from azureml.logging import get_azureml_logger
from azureml.dataprep.package import run

# initialize the logger

run_logger = get_azureml_logger() 

# Step 1 - Data Preparation

# Create the outputs folder

os.makedirs('./outputs', exist_ok=True)

print('Python version: {}'.format(sys.version))
print()

# Load the bank dataset

df = pd.read_csv('BankMarketCampaignTrainingSample.csv')

# Step 2 - Feature Engineering

# Encode columns

columns_to_encode = list(df.select_dtypes(include=['category','object']))
for column_to_encode in columns_to_encode:
    g = df.groupby(column_to_encode)
for column_to_encode in columns_to_encode:
    dummies = pd.get_dummies(df[column_to_encode])
    one_hot_col_names = []
    for col_name in list(dummies.columns):
        one_hot_col_names.append(column_to_encode + '_' + col_name)
    dummies.columns = one_hot_col_names
    df = df.drop(column_to_encode, axis=1)
    df = df.join(dummies)
    
# Keep only one response variable

df = df.drop('y_no', axis=1)

# Step 3 - Model Training and Evaluation

# Split Data for Training and Testing

train, test = train_test_split(df, test_size = 0.2, random_state=0)

# Specify the values of label and features in training and testing datasets.

train_y = train['y_yes'].values

train_x = train.drop('y_yes', axis=1)
train_x = train_x.values

test_y = test['y_yes'].values
test_x = test.drop('y_yes', axis=1)

# Initialize Sweep Parameter 

# Logistic Regression

def make_sweep_parameter_lr_dict():
    '''
    :return: parameters of logistic regression model to tune
    '''
    return {"penalty": ("l1", "l2"), "C": [0.1, 1, 10]}

# Support Vector Machine

def make_sweep_parameter_svm_dict():
    '''
    :return: parameters of support vector machine model to tune
    '''
    return {"gamma": [0.0001, 0.01, 1, 100], "C": [1]}

# Decision Tree

def make_sweep_parameter_dt_dict():
    '''
    :return: parameters of decision tree model to tune
    '''
    return{'max_depth': [4, 8, 16, 32]}

# Sweep Parameters with Each Classifier using Cross Validation

def sweep_classifier_cross_val(train_feature, train_label, model_name, parameter_dict):
    '''
    :param train_feature: feature of training set
    :param train_label: label of training set
    :param model_name: model
    :param parameter_dict: parameters of the model to tune
    :return: trained model
    '''
    classifier = grid_search.GridSearchCV(model_name, parameter_dict)
    classifier.fit(train_feature, train_label)
    return classifier

# Validate Model using Testing Data

def prediction(classifier, test_feature):
    '''
    :param classifier: the trained classifier
    :param test_feature: feature of test set
    :return: scored label of test set
    '''
    predicted_label = classifier.predict(test_feature)
    return predicted_label

# Compute Metrics

def calc_metrics(test_label, predicted_label):
    '''
    :param test_label: label of test set
    :param predicted_label: scored label of test set
    :return: evaluation metrics
    '''
    cm = confusion_matrix(test_label, predicted_label)
    acc = accuracy_score(test_label, predicted_label)
    pre = precision_score(test_label, predicted_label)
    rec = recall_score(test_label, predicted_label)
    f1 = f1_score(test_label, predicted_label)
    return cm, acc, pre, rec, f1

# Put it all together

# Train and evaluate logistic regression

model_name = LogisticRegression()
parameter_dict = make_sweep_parameter_lr_dict()
lrf = sweep_classifier_cross_val(train_feature=train_x, 
                                 train_label=train_y, 
                                 model_name=model_name, 
                                 parameter_dict=parameter_dict)

print('Logistic Regression Classifier:')
print(lrf)

predicted = prediction(lrf, test_x)

cm, acc, pre, rec, f1 = calc_metrics(test_y, predicted)

print('Confusion Matrix:')
print(cm)
print('Accuracy: {:.2f}'.format(acc))
print('Precision: {:.2f}'.format(pre))
print('Recall: {:.2f}'.format(rec))
print('F1: {:.2f}'.format(f1))

# Log lr model accuracy

run_logger.log("Logistic Regression Accuracy", acc)

# Train and evaluate support vector machine

model_name = SVC(kernel="rbf")
parameter_dict = make_sweep_parameter_svm_dict()
svf = sweep_classifier_cross_val(train_feature=train_x, 
                                 train_label=train_y, 
                                 model_name=model_name, 
                                 parameter_dict=parameter_dict)

print('Support Vector Machine Classifier:')
print(svf)

predicted = prediction(svf, test_x)

cm, acc, pre, rec, f1 = calc_metrics(test_y, predicted)

print('Confusion Matrix:')
print(cm)
print('Accuracy: {:.2f}'.format(acc))
print('Precision: {:.2f}'.format(pre))
print('Recall: {:.2f}'.format(rec))
print('F1: {:.2f}'.format(f1))

# Log sv model accuracy

run_logger.log("SVM Accuracy", acc)

# Train and evaluate support vector machine

model_name = DecisionTreeClassifier(min_samples_split=20, random_state=0)
parameter_dict = make_sweep_parameter_dt_dict()
dtf = sweep_classifier_cross_val(train_feature=train_x, 
                                 train_label=train_y, 
                                 model_name=model_name, 
                                 parameter_dict=parameter_dict)

print('Decision Tree Classifier:')
print(dtf)

predicted = prediction(dtf, test_x)

cm, acc, pre, rec, f1 = calc_metrics(test_y, predicted)

print('Confusion Matrix:')
print(cm)
print('Accuracy: {:.2f}'.format(acc))
print('Precision: {:.2f}'.format(pre))
print('Recall: {:.2f}'.format(rec))
print('F1: {:.2f}'.format(f1))

# Log dt model accuracy

run_logger.log("Decision Tree Accuracy", acc)

print("")
print("==========================================")
print("Serialize and deserialize using the outputs folder.")
print("")

# Serialize the model on disk in the special 'outputs' folder

print ("Export the lr model to lr.pkl")
f = open('./outputs/lr.pkl', 'wb')
pickle.dump(lrf, f)
f.close()

print ("Export the svf model to sv.pkl")
f = open('./outputs/sv.pkl', 'wb')
pickle.dump(svf, f)
f.close()

print ("Export the dtf model to dt.pkl")
f = open('./outputs/dt.pkl', 'wb')
pickle.dump(dtf, f)
f.close()

# Load the model back from the 'outputs' folder into memory

print("Import the model from lr.pkl")
f2 = open('./outputs/lr.pkl', 'rb')
lrf2 = pickle.load(f2)

print("Import the model from sv.pkl")
f2 = open('./outputs/sv.pkl', 'rb')
svf2 = pickle.load(f2)

print("Import the model from dt.pkl")
f2 = open('./outputs/dt.pkl', 'rb')
dtf2 = pickle.load(f2)




