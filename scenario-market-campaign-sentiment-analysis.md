# Market Campaign Prediction with Sentiment Analysis using Azure Machine Learning

## Link of the Gallery GitHub Repository

Following is the link to the public GitHub repository where all the codes are hosted:

[https://github.com/Azure/MachineLearningSamples-MarketCampaign](https://github.com/Azure/MachineLearningSamples-MarketCampaign)

## Prerequisites

* An [Azure account](https://azure.microsoft.com/en-us/free/) (free trials are available)

* An installed copy of [Azure Machine Learning Workbench](./overview-what-is-azure-ml) following the [quick start installation guide](./quick-start-installation) to install the program and create a workspace

* For operationalization, it is best if you have Docker engine installed and running locally. If not, you can use the cluster option but be aware that running an Azure Container Service (ACS) can be expensive.

* This Solution assumes that you are running Azure ML Workbench on Windows 10 with Docker engine locally installed. If you are using macOS the instruction is largely the same.

## Introduction

In business, companies are commonly recruiting new customers through market campaign. As a result, marketing executives often find themselves trying to predict the likelihood of customer purchase and finding the necessary actions to maximize the purchase rate.

The aim of this solution is to demonstrate predictive market analytics using AML Workbench. This solution provides an easy to use template to develop market campaign predictive data pipelines for retailers. The template can be used with different datasets and different definitions of success of market campaign. The aim of this tutorial is to:

1. Understand AML Workbench's Data Preparation tools to ingest and pre-process customer relationship data for market campaign prediction and customer review data for sentiment analysis.

2. Perform feature transformation to handle noisy heterogeneous market data.

3. Perform Unigrams TF-IDF feature extraction to convert unstructured text review data.

4. Train and validate various machine learning models (such as Logistic Regression, Support Vector Machine, Decision Tree) with hyper-parameter sweeping for predicting the success of market campaign, as well as predicting the sentiment score of customer review.

5. Model operationalization.

## Use Case Overview

A company, such as retail bank, wants to do market campaign prediction. The task is to build a pipeline that automatically analyze the bank market dataset, to predict the success of telemarketing calls for selling bank long-term deposits. The aim is to provide market intelligence for the bank and better target valuable customers and hence reduce marketing cost. Moreover, the bank may also want to analyze customer feedback in order to provide additional insight to enhance market campaign prediction. This requires a pipeline that automatically analyzes customer feedback messages, to provide the overall sentiment for the bank, thus helping the bank gain extra information from social media to optimize their market strategy.

Some of the factors contributing to bank market campaign include:

* Customer personal and financial situation
* Mode of market campaign
* Customer feedback from social media
* Offers from other banks 

In this solution, we will use a concrete example of building a predictive market campaign model for retail banks.

## Data Description

Two datasets are used in this solution, bank market data and bank review data.

The bank market data is from the UCI machine learning library, called BankMarketCampaignTrainingSample.csv. This dataset consists of heterogeneous noisy data (numerical/categorical variables) from Portuguese banking institution. Its variables capture customer demographic, bank account information, history telemarketing activity record. 

The bank review data is from the credit karma website, called BankReviewTrainingSample.csv. This dataset contains unstructured text data. It has two variables, representing sentiment score and customer review, respectively. 

## Scenario Structure

The folder structure is arranged as follows:

_Data_: Contains the dataset used in the solution.

_Code_: Contains all the code related to market campaign prediction with sentiment analysis using AMLWorkbench.

_Docs_: Contains end-to-end tutorial in the forms of jupyter notebook and markdown.

### Part 1 - Market Campaign Prediction

| Folder | Sub-Folder | Related Files |
|--------|------------|---------------|
| data   | NA       | 'BankMarketCampaignTrainingSample.csv' |
| code   | marketcampaign | 'BankMarketCampaignModeling.py', 'BankMarketCampaignModelingDocker.py' |
|        | marketcampaign | 'market_schema_gen.py', 'market_score.py', 'market_service_schema.json', 'dt.pkl' |
| docs   | media    | images  |
|        | NA       | [tutorial-market-campaign.md](docs/tutorial-market-campaign.md) |
|        |          | 'BankMarketCampaignNoteBook.ipynb', 'BankMarketCampaignOperationalization.ipynb' |

### Part 2 - Review Sentiment Analysis

| Folder | Sub-Folder | Related Files |
|--------|------------|---------------|
| data   | NA       | 'BankReviewTrainingSample.csv' |
| code   | reviewsentiment | 'BankReviewSentimentModeling.py', 'BankReviewSentimentModelingDocker.py' |
|        | reviewsentiment | 'senti_schema_gen.py', 'senti_score.py', 'senti_service_schema.json', 'model_30.pkl', 'stopwords.pkl' |
| docs   | media    | images|
|        | NA       | [tutorial-review-sentiment.md](docs/tutorial-review-sentiment.md)|
|        |          | 'BankReviewSentimentNoteBook.ipynb', 'BankReviewSentimentOperationalization.ipynb' |

Practice the end-to-end tutorial by following the [tutorial-market-campaign](docs/tutorial-market-campaign.md) and [tutorial-review-sentiment](docs/tutorial-review-sentiment.md)

## Conclusion

This scenario gives an overview of how to perform market campaign prediction with sentiment analysis using AMLWorkbench's Data Preparation tools, perform feature engineering to handle noisy heterogeneous data and unstructured text data, as well as operationalize.

## References

S. Moro, P. Cortez and P. Rita. A Data-Driven Approach to Predict the Success of Bank Telemarketing. Decision Support Systems, Elsevier, 62:22-31, June 2014.


