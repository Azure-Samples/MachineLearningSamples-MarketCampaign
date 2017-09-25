# Market Campaign Prediction using Azure Machine Learning

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

1. Understand AML Workbench's Data Preparation tools to ingest and pre-process customer relationship data for market campaign analysis.

2. Perform feature transformation to handle noisy heterogeneous market data.

3. Integrate third-party libraries (such as scikit-learn and azureml) to develop Logistic Regression, Support Vector Machine and Tree-based classifiers for predicting the success of market campaign.

4. Deploy model.

## Use Case Overview

A company, such as retail bank, wants to do market campaign prediction. The task is to build a pipeline that automatically analyze the bank market dataset, to predict the success of telemarketing calls for selling bank long-term deposits. The aim is to provide market intelligence for the bank and better target valuable customers and hence reduce marketing cost.

Some of the factors contributing to bank market campaign include:

* Customer personal and financial situation
* Mode of market campaign
* Customer feedback from social media
* Offers from other banks 

In this solution, we will use a concrete example of building a predictive market campaign model for retail banks.

## Data Description

The dataset used to ingest is from the UCI machine learning library. It is called BankMarketCampaignTrainingSample.csv and is located in the Data folder. The dataset consists of heterogeneous noisy data (numerical/categorical variables) from Portuguese banking institution and is anonymized.

The variables capture customer demographic, bank account information, history telemarketing activity record. 

## Scenario Structure

The folder structure is arranged as follows:

- Code: Contains all the code related to market campaign prediction using AMLWorkbench.  
- Data: Contains the dataset used in the solution  
- Docs: Contains end-to-end tutorial in the forms of jupyter notebook and markdown.

Practice the tutorial by following the [tutorial-market-campaign](docs/tutorial-market-campaign.md) and [tutorial-review-sentiment](docs/tutorial-review-sentiment.md)

## Conclusion

This scenario gives an overview of how to perform market campaign prediction using AMLWorkbench's Data Preparation tools, perform feature engineering to handle noisy heterogeneous data and operationalize.

## References

S. Moro, P. Cortez and P. Rita. A Data-Driven Approach to Predict the Success of Bank Telemarketing. Decision Support Systems, Elsevier, 62:22-31, June 2014.

## Contact

Please feel free to contact Fang Zhou (zhouf@microsoft.com) with any question or comment.

## Contributing

This project welcomes contributions and suggestions.  Most contributions require you to agree to a
Contributor License Agreement (CLA) declaring that you have the right to, and actually do, grant us
the rights to use your contribution. For details, visit https://cla.microsoft.com.

When you submit a pull request, a CLA-bot will automatically determine whether you need to provide
a CLA and decorate the PR appropriately (e.g., label, comment). Simply follow the instructions
provided by the bot. You will only need to do this once across all repos using our CLA.

This project has adopted the [Microsoft Open Source Code of Conduct](https://opensource.microsoft.com/codeofconduct/).
For more information see the [Code of Conduct FAQ](https://opensource.microsoft.com/codeofconduct/faq/) or
contact [opencode@microsoft.com](mailto:opencode@microsoft.com) with any additional questions or comments.


