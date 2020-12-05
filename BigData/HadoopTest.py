import pandas as pd
from hdfs import InsecureClient
from pyspark.sql import SparkSession
from kafka import KafkaProducer
import csv
import json
from time import sleep
from json import dumps

# client_hdfs = InsecureClient('http://localhost:9870', user='hadoop')
#
# # Creating a simple Pandas DataFrame
# liste_hello = ['hello1', 'hello2']
# liste_world = ['world1', 'world2']
# df = pd.DataFrame(data={'hello': liste_hello, 'world': liste_world})
#
# # Writing Dataframe to hdfs
# with client_hdfs.write('/home/hadoop/hdfs/helloworld3.csv', encoding='utf-8') as writer:
#     df.to_csv(writer)


spark = SparkSession \
    .builder \
    .master("local") \
    .appName("Protob Conversion to Parquet") \
    .config("spark.some.config.option", "some-value") \
    .getOrCreate()
df_load = spark.read.csv('hdfs://0.0.0.0:9000///home/hadoop/hdfs/helloworld3.csv', header=True).toPandas().set_index("_c0")
print(df_load)

producer = KafkaProducer(bootstrap_servers=['localhost:9092'],
                         value_serializer=lambda x:
                         dumps(x).encode('utf-8'))

# with open("hdfs://0.0.0.0:9000///home/hadoop/hdfs/helloworld3.csv") as file:
#     reader = csv.reader(file)

df_load = df_load.to_dict()
print(df_load)

producer.send('numtest', value= df_load)
sleep(3)
