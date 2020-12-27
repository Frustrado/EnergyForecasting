from hdfs import InsecureClient
from pyspark.sql import SparkSession
import pandas as pd
from pymongo import MongoClient
from Functions import initial_run, add_model, convert_models_toDataframe, prepare_data, get_best_model, find_max
from Models import configuration, models
import warnings
warnings.filterwarnings('ignore')

import pickle
from joblib import dump, load

client = MongoClient('mongodb://localhost:27017/')

coll = client.mgr.test1
df = None
for post in coll.find():
    if df is not None:
        df = df.append(pd.DataFrame(post).T.iloc[1:])
    else:
        df = pd.DataFrame(post).T.iloc[1:]

data = df.values
data = prepare_data(data)

df, list_of_model_and_configs = initial_run(data, configuration(), models())

# df, list_of_model_and_configs = add_model(data, df, SVC(), {'kernel': ['linear'],
#                                                             'C': [0.025]}, list_of_model_and_configs)
print(df)
best_model = get_best_model(list_of_model_and_configs, find_max(df, 'as'))

# testDataFrame = convert_models_toDataframe(list_of_model_and_configs)

# spark = SparkSession \
#     .builder \
#     .appName("myApp") \
#     .config("spark.mongodb.input.uri", "mongodb://localhost:27017/mgr.test1") \
#     .config("spark.mongodb.output.uri", "mongodb://localhost:27017/mgr.test1") \
#     .config('spark.jars.packages', 'org.mongodb.spark:mongo-spark-connector_2.11:2.3.2') \
#     .getOrCreate()

# df = spark.read.format("com.mongodb.spark.sql.DefaultSource").load()

client_hdfs = InsecureClient('http://localhost:9870', user='hadoop')

with client_hdfs.write('/home/hadoop/hdfs/bestmodeltest1.pkl') as writer:
    dump(pickle.dumps(best_model), writer)