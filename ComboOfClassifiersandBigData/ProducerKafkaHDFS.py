import pandas as pd
from hdfs import InsecureClient
from pyspark.sql import SparkSession
from kafka import KafkaProducer
import csv
import json
from time import sleep
from json import dumps
import datetime

client_hdfs = InsecureClient('http://localhost:9870', user='hadoop')

# # Writing Dataframe to hdfs
# df  = pd.read_csv('german.uci.csv')
# with client_hdfs.write('/home/hadoop/hdfs/data2.csv', encoding='utf-8') as writer:
#     df.to_csv(writer)

spark = SparkSession \
    .builder \
    .master("local") \
    .appName("Protob Conversion to Parquet") \
    .config("spark.some.config.option", "some-value") \
    .getOrCreate()
df_load = spark.read.csv('hdfs://0.0.0.0:9000///home/hadoop/hdfs/data2.csv', header=True).toPandas()#.set_index("_c0")

producer = KafkaProducer(bootstrap_servers=['localhost:9092'],
                         value_serializer=lambda x:
                         dumps(x).encode('utf-8'))

# df_load.iloc[:,1] = pd.to_datetime(df_load.iloc[:,1].replace("T"," ").replace("Z",""))
# df_load.iloc[:, 2:] = df_load.iloc[:, 2:].astype('float')

# starttime= df_load.iloc[0,1]
# endtime = starttime+datetime.timedelta(minutes=60)
# mask = (df_load['date'] >= starttime) & (df_load['date'] < endtime)
# end = df_load.iloc[-1, 1]
# while endtime < end:
#     df_res = df_load[mask].astype("str").to_dict()
#     starttime = endtime
#     endtime = starttime+datetime.timedelta(minutes=60)
#     mask = (df_load['date'] >= starttime) & (df_load['date'] < endtime)
#     producer.send('mgr', value=df_res)
#     sleep(3)
i = 0
length_df = len(df_load)
while i+60 < length_df:
    if i+60 < length_df:
        df_res = df_load[i:i + 60].T.to_dict()
    else:
        df_res = df_load[i:].T.to_dict()
    i += 60
    producer.send('mgred1', value=df_res)
    sleep(1)

