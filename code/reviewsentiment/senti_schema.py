# This script generates the scoring and schema files
# necessary to operationalize the Review Sentiment sample
# Init and run functions

from azureml.api.schema.dataTypes import DataTypes
from azureml.api.schema.sampleDefinition import SampleDefinition
from azureml.api.realtime.services import generate_schema

import pandas as pd
import os
import string

from nltk.tokenize import RegexpTokenizer
from sklearn.feature_extraction.text import TfidfVectorizer
from nltk.stem.porter import PorterStemmer

from sklearn.linear_model import LogisticRegression
from sklearn import metrics

# Prepare the web service definition by authoring
# init() and run() functions. Test the functions
# before deploying the web service.

def init():
    from sklearn.externals import joblib

    # load the model file
    global model
    model = joblib.load('model_30.pkl')

# run takes an input dataframe and performs sentiment prediction
def run(input_df):
    import json
    
    input_df.columns = ['input_column'] 
    
    stop_words_df = pd.read_csv('StopWords.csv')
    stop_words = set(stop_words_df["Col1"].tolist())
    for item in string.ascii_lowercase: #load stop words
        if item != "i":
            stop_words.add(item)

    input_column = []
    for line in input_df.input_column:
        value = " ".join(item.lower()
                         for item in RegexpTokenizer(r'\w+').tokenize(line)
                         if item.lower() not in stop_words)
        input_column.append(value)
    input_df.input_column = input_column

    stemmer = PorterStemmer()
    input_list = input_df["input_column"].tolist()

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
                            min_df=1, norm="l2", use_idf=True)
    tfs = tfidf.fit_transform(input_list)
    
    pred = model.predict(tfs[0, :30])
    return json.dumps(str(pred[0]))
    #return pred[0]
print('executed')

def main():
  from azureml.api.schema.dataTypes import DataTypes
  from azureml.api.schema.sampleDefinition import SampleDefinition
  from azureml.api.realtime.services import generate_schema
  import pandas as pd

  # create the outputs folder
  os.makedirs('./outputs', exist_ok=True)

  df1 = pd.DataFrame(data=[["I absolutely love my bank. There's a reason this bank's customer base is so strong--their customer service actually acts like people and not robots. I love that anytime my card is swiped, I'm instantly notified. And the built in budgeting app is something that really makes life easier. The biggest setback is not being able to deposit cash (you have to get a money order), and if you have another, non-simple bank account, transferring money between accounts can take a few days, which frankly isn't acceptable with most ACH taking a business day or less. Overall, it's a great bank, and I would recommend it to anyone."]], columns=['review'])

  # Turn on data collection debug mode to view output in stdout
  os.environ["AML_MODEL_DC_DEBUG"] = 'true'

  # Test the output of the functions
  init()
  input1 = pd.DataFrame(data=[["I absolutely love my bank. There's a reason this bank's customer base is so strong--their customer service actually acts like people and not robots. I love that anytime my card is swiped, I'm instantly notified. And the built in budgeting app is something that really  makes life easier. The biggest setback is not being able to deposit cash (you have to get a money order), and if you have another, non-simple bank account, transferring money between accounts can take a few days, which frankly isn't acceptable with most ACH taking a business day or less. Overall, it's a great bank, and I would recommend it to anyone."]], columns=['review'])

  print("Result: " + run(input1))
  
  inputs = {"input_df": SampleDefinition(DataTypes.PANDAS, df1)}
  
  #Genereate the schema
  #generate_schema(run_func=run, inputs=inputs, filepath='./outputs/senti_schema2.json')
  generate_schema(run_func=run, inputs=inputs, filepath='senti_service_schema.json')
  print("Schema generated")

if __name__ == '__main__':
    #init()
    #input1 = pd.DataFrame(data=[["I absolutely love my bank. There's a reason this bank's customer base is so strong--their customer service actually acts like people and not robots. I love that anytime my card is swiped, I'm instantly notified. And the built in budgeting app is something that really  makes life easier. The biggest setback is not being able to deposit cash (you have to get a money order), and if you have another, non-simple bank account, transferring money between accounts can take a few days, which frankly isn't acceptable with most ACH taking a business day or less. Overall, it's a great bank, and I would recommend it to anyone."]], columns=['review'])
    #res = run(input1)
    #print(res)
    main()
