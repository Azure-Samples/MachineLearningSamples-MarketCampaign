# This code snippet will load the referenced package and return a DataFrame.
# If the code is run in a PySpark environment, then the code will return a
# Spark DataFrame. If not, the code will return a Pandas DataFrame. You can
# copy this code snippet to another code file as needed.
from azureml.dataprep.package import run


# Use this DataFrame for further processing
df = run('BankMarketCampaignTrainingSample.dprep', dataflow_idx=0)
