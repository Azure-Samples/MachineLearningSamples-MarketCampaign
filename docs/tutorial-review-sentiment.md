# Tutorial: Review Sentiment Analysis

In this tutorial, we show you the basics of Azure ML preview features by creating a data preparation package, building a model and operationalizing it as a real-time web service. To make things simple, we use the timeless [customer feedback dataset](https://www.creditkarma.com/reviews/banking/single/id/Simple#single-review-listingPaper). 

## Step 1. Launch Azure ML Workbench
Follow the [installation guide](https://github.com/Azure/ViennaDocs/blob/master/Documentation/Installation.md) to install Azure ML Workbench desktop application, which also includes command-line interface (CLI). Launch the Azure ML Workbench desktop app and log in if needed.

## Step 2. Create a new project
Click on _File_ --> _New Project_ (or click on the "+" sign in the project list pane). You can also create a new Workspace first from this drop down menu.

![new ws](media/tutorial-review-sentiment/new_ws.png)

Fill in the project name (this tutorial assumes you use `ReviewSentiment`). Choose the directory the project is going to be created in (this tutorial assumes you choose `C:\Users\dsvmadmin\Documents\AzureML\ReviewSentiment`). Enter an optional description. Choose a Workspace (this tutorial uses `zfviennaws01`). 

![New Project](media/tutorial-review-sentiment/new_project.png)
>Optionally, you can fill in the Git repo field with an existing empty (with no master branch) Git repo on VSTS. 

Click on _Create_ button to create the project. The project is now created and opened.

## Step 3. Create a Data Preparation package

Under Data Explorer view, click on "+" to add a new data source. This launches the _Add Data Source_ wizard. 

![data add](media/tutorial-review-sentiment/data_add.png)

Select the _Excel(s)/Directory_ option, and choose the `BankReviewTrainingSample.csv` local file. Accept all the default settings for each screen and finally click on _Finish_. 

![select bank](media/tutorial-review-sentiment/select_review_excel.png)

>Make sure you select the `BankReviewTrainingSample.xlsx` file from within the current project directory for this exercise, otherwise latter steps may fail. 

This creates an `BankReviewTrainingSample-1.dsource` file (because the sample project already comes with an `BankReviewTrainingSample.dsource` file) and opens it in the _Data_ view. 

![bank data view](media/tutorial-review-sentiment/review_data_view.png)

Now click on the _Metrics_ button. Observe the histograms and a complete set of statistics that are calculated for you for each column. You can also switch over to the _Data_ view to see the data itself. 

![bank data view](media/tutorial-review-sentiment/review_metrics_view.png)

Now click on the _Prepare_ button next to the _Metrics_ button, and this creates a new file named `BankReviewTrainingSample-1.dprep`. Again, this is because the sample project already comes with an `BankReviewTrainingSample.dprep` file. It opens in Data Prep editor and shows you the data flow to process the data.

Now close the DataPrep editor. Don't worry, it is auto-saved. Right click on the `BankReviewTrainingSample.dprep` file, and choose _Generate Data Access Code File_. 

![generate code](media/tutorial-review-sentiment/generate_access_code.png)

This creates an `BankReviewTrainingSample.py` file with following two lines of code prepopulated (along with some comments):

```python
# Use the Azure Machine Learning data preparation package
from azureml.dataprep import package

# Use the Azure Machine Learning data collector to log various metrics
from azureml.logging import get_azureml_logger
logger = get_azureml_logger()

# This call will load the referenced package and return a DataFrame.
# If run in a PySpark environment, this call returns a
# Spark DataFrame. If not, it will return a Pandas DataFrame.
df = package.run('BankReviewTrainingSample.dprep', dataflow_idx=0)

# Remove this line and add code that uses the DataFrame
df.head(10)
```
This code snippet shows how you can invoke the data wrangling logic you have created as a Data Prep package. Depending on the context in which this code runs, `df` can be a Python Pandas DataFrame if executed in Python runtime, or a Spark DataFrame if executed in a Spark context. 

## Step 4. View Python Code in `BankReviewSentimentModeling.py` File

Now, open the `BankReviewSentimentModeling.py` file.

![open file](media/tutorial-review-sentiment/open_review_sentiment_modeling.png)

Observe that this script does the following tasks:
1. Load data from local to generate a [Pandas](http://pandas.pydata.org/) data frame.
2. Do text pre-processing and cleaning, such as replacing special characters and punctuation marks with spaces, normalizing case, removing duplicate characters, removing user-defined or built-in stop-words, and word stemming.
3. Use [scikit-learn](http://scikit-learn.org/stable/index.html) to perform feature engineering, by converting variable-length unstructured text data into equal-length numeric feature vectors using [TfidfVectorizer](http://scikit-learn.org/stable/modules/generated/sklearn.feature_extraction.text.TfidfVectorizer.html) and selecting features using [SelectKBest](http://scikit-learn.org/stable/modules/generated/sklearn.feature_selection.SelectKBest.html) using Chi-squared test.
4. Train and validate logistic regression model using [scikit-learn](http://scikit-learn.org/stable/index.html) with parameter sweeping.
5. Serialize the models using [pickle](https://docs.python.org/2/library/pickle.html) into a file in a special `outputs` folder, loads it and de-serializes it back into memory

## Step 5. Run the Python Script on the Local Computer from Command-line

Launch the command-line window by clicking on _File_ --> _Open Command-Line Interface_, noice that you are automatically placed in the project folder. In this example, the project is located in `C:\Users\dsvmadmin\Documents\AzureML\ReviewSentiment`.

>Important: You must use the command-line window opened from Workbench to accomplish the following steps.

```batch
REM login using aka.ms/devicelogin site.
az login

REM list all Azure subscriptions you have access to. 
az account list -o table

REM set the current Azure subscription to the one you want o use.
az set account -s <subscriptionId>

REM verify your current subscription is set correctly
az account show
```

Once you are authenticated, type the following commands in the terminal window. 
```batch
REM Kick off an execution of the BankMarketCampaignModeling.py file against local compute context
az ml experiment submit -c local .\BankReviewSentimentModeling.py
```
This command executes the *BankReviewSentimentModeling.py* file locally. After the execution finishes, you should see the output in the CLI window. 

![execute local](media/tutorial-review-sentiment/execute_local.png)

## Step 6. Run the Python Script in a Docker Container 

### Run in a local Docker container

If you have a Docker engine running locally, in the command line window, repeat the same command. Except this time, let's change the run configuration from _local_ to _docker_:

```batch
REM execute against a local Docker container with Python context
az ml experiment submit -c docker .\BankReviewSentimentModeling.py
```
This command pulls down a base Docker image, lays a conda environment on that base image based on the _conda_dependencies.yml_ file in your_aml_config_ directory, and then starts a Docker container. It then executes your script. You should see some Docker image construction messages in the CLI window. And in the end, you should see the exact same output as step 5. You can find the _docker.runconfig_ file and _docker.compute_ file under _aml_config_ folder and examine the content to understand how they control the execution behavior. 

### Run in a Docker container on a remote Linux machine

To execute your script in a Docker container on a remote Linux machine, you need to have SSH access (using username and password) to that remote machine, and that remote machine must have the Docker engine installed. The easiest way to obtain such a Linux machine is to create a [Ubuntu-based Data Science Virtual Machine (DSVM)](https://azuremarketplace.microsoft.com/en-us/marketplace/apps/microsoft-ads.linux-data-science-vm-ubuntu) on Azure. 

Generate the _myvm.compute_ and _myvm.runconfig_ files by running the following command:
```batch
REM create myvm compute target
az ml computetarget attach --name myvm --address 52.187.129.184 --username ldsvmadmin --password <password> --type remotedocker
```
Edit the generated _myvm.runconfig_ file under _aml_config_ and change the Framework from default PySpark to Python:
```yaml
"Framework": "Python"
```
Prepare the myvm compute context
```batch
REM prepare myvm compute context
az ml experiment prepare -c myvm
```
![execute remotedocker](media/tutorial-review-sentiment/execute_remotedocker.png)

Now issue the same command as you did before in the CLI window, except this time we will target _myvm_:
```batch
REM execute in remote Docker container
az ml experiment submit -c myvm .\BankReviewSentimentModeling.py
```
When the command is executed, the exact same thing happens as Step 6a except it happens on that remote machine. You should observe the exact same output information in the CLI window.

## Step 7. Explore Run History

After you run the BankReviewSentimentModeling.py script a few times in the CLI window, go back to the Azure ML Workbench desktop app.

Now click on the Run History icon. You should see BankReviewSentimentModeling.py listed as an item in the run history list. Click on it to see the run history dashboard for this particular script, which includes some graphs depicting metrics recorded in each run, along with the list of runs showing basic information including as created date, status, and duration. 

You can click on an individual run, and explore the details recorded for that run. In this example, you can see the models generated in the outputs folder and the ROC Curve plot rendered in the images section.

![Run History](media/tutorial-review-sentiment/run_history.png)

If the run is still underway, you will see execution messages streaming into the log file window that are opened in the run details page. 

![Run History](media/tutorial-review-sentiment/streaming_log.png)

## Step 8. Obtain the Pickled Model

In the BankReviewSentimentModeling.py script, we serialize the logistic regression model using the popular object serialization package -- pickle, into files named _model_top_k.pkl_ on disk. Here is the code snippet.

```python
print ("Export the model to model_"+str(top_k)+".pkl")
f = open('./outputs/model_'+str(top_k)+'.pkl', 'wb')
pickle.dump(clf, f)
f.close()
```

When you executed the *BankReviewSentimentModeling.py* script using the *az ml experiment submit* command, the model was written to the *outputs* folder with the name *model_top_k.pkl*. This folder lives in the compute target, not your local project folder. You can find it in the run history detail page and download this binary file by clicking on the download button next to the file name. 

Now, download the optimal model file _model_30.pkl_ and save it to the root of your  project folder. You will need it in the later steps.

## Step 9. Prepare for Operationalization Locally using a DSVM on Azure

Local mode deployments run in Docker containers on your local computer, whether that is your desktop or a Linux VM running on Azure. You can use local mode for development and testing. The Docker engine must be running locally to complete the operationalization steps as shown in the following steps.

Let's prepare the operationalization environment. 

Launch a Data Science Virtual Machine (Ubuntu) from portal.azure.com as shown below. Follow the steps to create the virtual machine on selection and ssh into the machine.

![DataScienceVirtualMachine](media/tutorial-review-sentiment/data_science_virtual_machine.png)

Pip is a better alternative to Easy Install for installing Python packages. To install pip on ubuntu run the bellow command:

```
sudo apt-get install python-pip
```

Only users with sudo access will be able to run docker commands. Optionally, add non-sudo access to the Docker socket by adding your user to the docker group.

```
sudo usermod -a - G docker $(whoami)
```

If you encounter "locale.Error: unsupported locale setting" error, perform the below export:

```
export LC_ALL=C
```

Update pip to use the latest:

```
pip install --upgrade pip
```

Update azure to the latest:

```
pip install --upgrade azure
```
Install azure-cli and azure-cli-ml using pip:

```
pip install azure-cli
pip install azure-cli-ml
```
In addition, change python default version and run the following commands. 

Create a bash_aliases file

```gedit ~/.bash_aliases```

Open your ~/.bash_aliases file and add the following and save it to home directory

```alias python=python3```

Source the ~/.bash_aliases file

```source ~/.bash_aliases```

Setup azure ml environment

```
az ml env setup -n <environment name> -g <resource group> -l <location>
az ml env set -g <resource group> -n <environment name>
```
To verify that you have properly configured your operationalization environment for local web service deployment, enter the following command:

```batch
az ml env local
```

## Step 10. Create a Realtime Web Service

### Schema and Score

To deploy the web service, you must have a model, a scoring script, and optionally a schema for the web service input data. The scoring script loads the model_30.pkl file from the current folder and uses it to produce a new predicted class. The input to the model is review text.

To generate the scoring and schema files, execute the senti_schema.py file that comes with the sample project in the AMLWorkbench CLI command prompt using Python interpreter directly.

```
python senti_schema.py
```

This will create senti_service_schema.json (this file contains the schema of the web service input)

```
Upload the below files to the vm (you could use scp to perform the upload):
conda_dependencies.yml
model_30.pkl
senti_service_schema.json
senti_schema.py
StopWords.csv
```

Edit the conda_dependencies.yml to contain only the following dependencies:

```
dependencies:
  - pip:
    # This is the operationalization API for Azure Machine Learning. Details:
    # https://github.com/Azure/Machine-Learning-Operationalization
    - azure-ml-api-sdk
    - matplotlib
    - sklearn
    - nltk
```

### Model Management

The real-time web service requires a modelmanagement account. This can be created using the following commands:

```
az group create -l <location> -n <name>
az ml account modelmanagement create -l <location> -g <resource group> -n <account name>
az ml account modelmanagement set -n <account name> -g <resource group>
```
The following command creates an image which can be used in any environment.

```
az ml image create -n zfviennagrp -v -c conda_dependencies.yml -m model_30.pkl -s senti_service_schema.json -f senti_schema.py -r python
```
![Image Create](media/tutorial-review-sentiment/image_create.png)

You will find the image id displayed when you create the image. Use the image id in the next command to specify the image to use. 

```
az ml image usage -i 9bebf880-dc0d-4b2c-9e00-f19f8e09102a
```
In some cases, you may have more than one image and to list them, you can run ```az ml image list```

Don't forget to copy the 'StopWords.csv' file from the VM to Docker container:
```
docker cp StopWords.csv mycontainer:/StopWords.csv
```

Ensure local is used as the deployment environment:
```
az ml env local
```
In local mode, the CLI creates locally running web services for development and testing.
Change to root:
```
sudo -i
```

### Real-time Web Service

Create a realtime service by running the below command using the image-id. In the following command, we create a realtime service called sentiservice.

```
az ml service create realtime -n sentiservice --image-id 9bebf880-dc0d-4b2c-9e00-f19f8e09102a 
```
An example of a successful run of az ml service create looks as follows. In addition, you can also type docker ps to view the container.

![Docker Ps](media/tutorial-review-sentiment/service_create.png)

Run the service (sentiservice) created using az ml service run. Note the review text created and passed to call the web service.

```
az ml service run realtime -i sentiservice -d "{\"input_df\": [{\"review\": \"I absolutely love my bank. There's a reason this bank's customer base is so strong--their customer service actually acts like people and not robots. I love that anytime my card is swiped, I'm instantly notified. And the built in budgeting app is something that really makes life easier. The biggest setback is not being able to deposit cash (you have to get a money order), and if you have another, non-simple bank account, transferring money between accounts can take a few days, which frankly isn't acceptable with most ACH taking a business day or less. Overall, it's a great bank, and I would recommend it to anyone.\"}]}"
```

![Sentiservice](media/tutorial-review-sentiment/service_run.png)

## Congratulations!
Great job! You have successfully run a training script in various compute environments, created a model, serialized the model, and operationalized the model through a Docker-based web service. 