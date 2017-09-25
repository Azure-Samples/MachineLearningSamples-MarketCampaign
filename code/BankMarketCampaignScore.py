import pandas as pd
    
def init():
    import numpy
    import scipy
    from sklearn.tree import DecisionTreeClassifier

    global model
    import pickle
    f = open('./dt.pkl', 'rb')
    model = pickle.load(f)
    f.close()

def run(inputString):
    import json
    import numpy
    try:
        input_list = json.loads(inputString)
    except ValueError:
        return "bad input: expecting a JSON encoded list of lists."
    input_df = pd.DataFrame(input_list, columns=['age','job','marital','education','default','balance','housing','loan','contact','day','month','duration','campaign','pdays','previous','poutcome'])
    if (input_df.shape != (1, 16)):
        return 'bad input: expecting a JSON encoded list of lists of shape (1,16).'
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
    return str(pred[12])

df1 = pd.DataFrame(data=[[30,'admin.','divorced','unknown','yes',1787,'no','no','telephone',19,'oct',79,1,-1,0,'unknown'],[33,'blue-collar','married','secondary','no',4789,'yes','yes','cellular',11,'may',220,1,339,4,'success'],[35,'entrepreneur','single','tertiary','no',1350,'yes','no','cellular',16,'apr',185,1,330,1,'failure'],[30,'housemaid','married','tertiary','no',1476,'yes','yes','unknown',3,'jun',199,4,-1,0,'unknown'],[59,'management','married','secondary','no',0,'yes','no','unknown',5,'jan',226,1,-1,0,'unknown'],[35,'retired','single','tertiary','no',747,'no','no','cellular',23,'feb',141,2,176,3,'failure'],[36,'self-employed','married','tertiary','no',307,'yes','no','cellular',14,'mar',341,1,330,2,'other'],[39,'services','married','secondary','no',147,'yes','no','cellular',6,'jul',151,2,-1,0,'unknown'],[41,'student','married','tertiary','no',221,'yes','no','unknown',14,'aug',57,2,-1,0,'unknown'],[43,'technician','married','primary','no',-88,'yes','yes','cellular',17,'sep',313,1,147,2,'failure'],[39,'unemployed','married','secondary','no',9374,'yes','no','unknown',20,'nov',273,1,-1,0,'unknown'],[43,'unknown','married','secondary','no',264,'yes','no','cellular',17,'dec',113,2,-1,0,'unknown']], columns=['age','job','marital','education','default','balance','housing','loan','contact','day','month','duration','campaign','pdays','previous','poutcome'])

if __name__ == '__main__':
    import json
    init()
    print (run(json.dumps([[30,'unemployed','married','primary','no',1787,'no','no','cellular',19,'oct',79,1,-1,0,'unknown']])))