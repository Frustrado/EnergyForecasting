from hdfs import InsecureClient
from pyspark.sql import SparkSession
from json import dump
import json

client_hdfs = InsecureClient('http://localhost:9870', user='hadoop')

# spark = SparkSession \
#     .builder \
#     .master("local") \
#     .appName("Protob Conversion to Parquet") \
#     .config("spark.some.config.option", "some-value") \
#     .getOrCreate()

spark = SparkSession \
    .builder \
    .appName("myApp") \
    .config("spark.mongodb.input.uri", "mongodb://localhost:27017/numtest.kafka") \
    .config("spark.mongodb.output.uri", "mongodb://localhost:27017/numtest.kafka") \
    .config('spark.jars.packages', 'org.mongodb.spark:mongo-spark-connector_2.11:2.3.2') \
    .getOrCreate()

mongo = spark.read.format("com.mongodb.spark.sql.DefaultSource").option("spark.mongodb.input.uri",'mongodb://localhost:27017/numtest.kafka').load()
results = mongo.toJSON().map(lambda j: json.loads(j)).collect()
mongo.show()
with client_hdfs.write('/home/hadoop/hdfs/helloworld2.json', encoding='utf-8') as writer:
    dump(results,writer)