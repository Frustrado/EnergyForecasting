from hdfs import InsecureClient
from pyspark.sql import SparkSession
import pandas as pd
from pymongo import MongoClient
from Functions import prepare_data, run, get_model, get_min_model

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

train_X, train_y, test_X, test_y, scaler = prepare_data()

models, df_results = run(train_X, train_y, test_X, test_y, scaler)

# df, list_of_model_and_configs = add_model(data, df, SVC(), {'kernel': ['linear'],
#                                                             'C': [0.025]}, list_of_model_and_configs)
print(df_results)
best_model = get_model(models, get_min_model(df_results).values[0])
print(best_model.get_config())

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
#
with client_hdfs.write('/home/hadoop/hdfs/best.h5') as writer:
    best_model.save('best_model.h5')
    # dump(pickle.dumps(best_model), writer)