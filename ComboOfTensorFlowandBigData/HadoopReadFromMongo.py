from hdfs import InsecureClient
from pyspark.sql import SparkSession
import pandas as pd
from pymongo import MongoClient
from Functions import prepare_data, run, get_model, get_min_model
from tensorflow.keras.models import save_model
import warnings
warnings.filterwarnings('ignore')

import pickle
# from joblib import dump, load
#
##########################   READ STREAMED DATA FROM MONGO   ###############################3333
# client = MongoClient('mongodb://localhost:27017/')
#
# coll = client.mgr.test1
# df = None
# for post in coll.find():
#     if df is not None:
#         df = df.append(pd.DataFrame(post).T.iloc[1:])
#     else:
#         df = pd.DataFrame(post).T.iloc[1:]


#############################   TEST DATA   #############################################
train_X, train_y, test_X, test_y, scaler = prepare_data()

models, df_results = run(train_X, train_y, test_X, test_y, scaler)


# df, list_of_model_and_configs = add_model(data, df, SVC(), {'kernel': ['linear'],
#                                                             'C': [0.025]}, list_of_model_and_configs)
print(df_results)
df_results.to_csv('result10epochsroof.csv')
# best_model = get_model(models, get_min_model(df_results).values[0])
# print(best_model.get_config())

# testDataFrame = convert_models_toDataframe(list_of_model_and_configs)

######################   HADOOP   #############################

client_hdfs = InsecureClient('http://localhost:9870', user='hadoop')
#tf -3 epoki, tf50epochs - 10 epok   ,  df = pd.read_csv("databank/data_industrial_tensor_grid_train.csv")
#tf10epochsfacade - 10epok,       df = pd.read_csv("databank/data_industrial_tensor_pv_facade_train.csv")
#tf10epochsroof - 10epok     df = pd.read_csv("databank/data_industrial_tensor_pv_roof_train.csv")
for model in models:
    path = '/home/hadoop/hdfs/test/tf10epochsroof' + str(model._name)
    pathLoc = '/home/max/Desktop/EnergyForecasting/ComboOfTensorFlowandBigData/models/' + str(model._name)
    model.save(pathLoc,save_format='tf')
    client_hdfs.write(path,pathLoc)






# client_hdfs = InsecureClient('http://localhost:9870', user='hadoop')
# #
# client_hdfs.write('/home/hadoop/hdfs/bestxd.h5', '/home/max/Desktop/EnergyForecasting/ComboOfTensorFlowandBigData/best_model.h5')
# # with client_hdfs.write('/home/hadoop/hdfs/best.h5') as writer:
# #     best_model.save('/home/hadoop/hdfs/best_model.h5')
#     # dump(pickle.dumps(best_model), writer)