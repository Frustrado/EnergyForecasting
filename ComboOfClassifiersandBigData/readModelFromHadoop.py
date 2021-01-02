from hdfs import InsecureClient
from joblib import load
from io import BytesIO
client_hdfs = InsecureClient('http://localhost:9870', user='hadoop')
path = '/home/hadoop/hdfs/test/xyBernoulliNB()-1.json'

with client_hdfs.read(path) as reader:
    model = load(BytesIO(reader.read()))
print(model)
