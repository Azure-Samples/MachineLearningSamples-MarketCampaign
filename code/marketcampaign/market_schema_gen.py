# This script generates the scoring and schema files
# necessary to operaitonalize the Market Campaign prediction sample
# Init and run functions

from azureml.api.schema.dataTypes import DataTypes
from azureml.api.schema.sampleDefinition import SampleDefinition
from azureml.api.realtime.services import generate_schema
import pandas as pd

# Prepare the web service definition by authoring
# init() and run() functions. Test the fucntions
# before deploying the web service.

def init():
    from sklearn.externals import joblib

    # load the model file
    global model
    model = joblib.load('dt.pkl')

def run(input_df):
    import json
    df = df1.append(input_df, ignore_index=True)
    columns_to_encode = list(df.select_dtypes(include=['category','object']))
    for column_to_encode in columns_to_encode:
        dummies = pd.get_dummies(df[column_to_encode])
        one_hot_col_names = []
        for col_name in list(dummies.columns):
            one_hot_col_names.append(column_to_encode + '_' + col_name)
        dummies.columns = one_hot_col_names
        df = df.drop(column_to_encode, axis=1)
        df = df.join(dummies)
    pred = model.predict(df)
    return json.dumps(str(pred[12]))
    #return pred[12]
print('executed')

df1 = pd.DataFrame(data=[[30,'admin.','divorced','unknown','yes',1787,'no','no','telephone',19,'oct',79,1,-1,0,'unknown'],[33,'blue-collar','married','secondary','no',4789,'yes','yes','cellular',11,'may',220,1,339,4,'success'],[35,'entrepreneur','single','tertiary','no',1350,'yes','no','cellular',16,'apr',185,1,330,1,'failure'],[30,'housemaid','married','tertiary','no',1476,'yes','yes','unknown',3,'jun',199,4,-1,0,'unknown'],[59,'management','married','secondary','no',0,'yes','no','unknown',5,'jan',226,1,-1,0,'unknown'],[35,'retired','single','tertiary','no',747,'no','no','cellular',23,'feb',141,2,176,3,'failure'],[36,'self-employed','married','tertiary','no',307,'yes','no','cellular',14,'mar',341,1,330,2,'other'],[39,'services','married','secondary','no',147,'yes','no','cellular',6,'jul',151,2,-1,0,'unknown'],[41,'student','married','tertiary','no',221,'yes','no','unknown',14,'aug',57,2,-1,0,'unknown'],[43,'technician','married','primary','no',-88,'yes','yes','cellular',17,'sep',313,1,147,2,'failure'],[39,'unemployed','married','secondary','no',9374,'yes','no','unknown',20,'nov',273,1,-1,0,'unknown'],[43,'unknown','married','secondary','no',264,'yes','no','cellular',17,'dec',113,2,-1,0,'unknown']], columns=['age','job','marital','education','default','balance','housing','loan','contact','day','month','duration','campaign','pdays','previous','poutcome'])

df1.dtypes
df1

df = pd.DataFrame([[30,'unemployed','married','primary','no',1787,'no','no','cellular',19,'oct',79,1,-1,0,'unknown']], columns=['age','job','marital','education','default','balance','housing','loan','contact','day','month','duration','campaign','pdays','previous','poutcome'])
df.dtypes
df

init()
input1 = pd.DataFrame([[30,'unemployed','married','primary','no',1787,'no','no','cellular',19,'oct',79,1,-1,0,'unknown']], columns=['age','job','marital','education','default','balance','housing','loan','contact','day','month','duration','campaign','pdays','previous','poutcome'])

run(input1)

inputs = {"input_df": SampleDefinition(DataTypes.PANDAS, df)}

# The prepare statement writes the scoring file (main.py) and
# the schema file (market_service_schema.json) the the output folder.

generate_schema(run_func=run, inputs=inputs, filepath='market_service_schema.json')
print("Schema generated")