from hdfs import InsecureClient
import pandas as pd
from pymongo import MongoClient
from Functions import initial_run, add_model, convert_models_toDataframe, prepare_data, get_best_model, find_max
from Models import configuration, models
import warnings
warnings.filterwarnings('ignore')
from sklearn.preprocessing import MinMaxScaler
import pickle
from joblib import dump, load
from sklearn.preprocessing import StandardScaler


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
df = pd.read_csv("databank/data_industrial_scikitlearn_grid_train.csv",header=0)
df_test = pd.read_csv("databank/data_industrial_scikitlearn_grid_test.csv",header=0)

print(df.head())
data = df.values
data_test = df_test.values

X, y = data[:10000, 2:], data[:10000, 1]
# sc = StandardScaler()
sc = scaler = MinMaxScaler(feature_range=(0, 1))
sc.fit(X)
# dump(sc, 'Scalers/scalerRoof.gz')
# sc1 = load('Scalers/scaler.gz')

data = prepare_data(data[:10000,:],sc)
data_test = prepare_data(data_test[:1000,:],sc)

df, list_of_model_and_configs = initial_run(data, data_test, configuration(), models())

# df, list_of_model_and_configs = add_model(data, df, SVC(), {'kernel': ['linear'],
#                                                             'C': [0.025]}, list_of_model_and_configs)
print(df)
df.to_csv('Results/xd.csv')

for key, model in list_of_model_and_configs.items():
    key = key.split('(')[0]
    path = 'Models/roofDP' + str(key) + '.pkl'
    dump(model, path)
        # dump(pickle.dumps(model), writer)



# best_model = get_best_model(list_of_model_and_configs, find_max(df, 'as'))

# testDataFrame = convert_models_toDataframe(list_of_model_and_configs)


######################   HADOOP   #############################

# client_hdfs = InsecureClient('http://localhost:9870', user='hadoop')
# #
# for key, model in list_of_model_and_configs.items():
#     path = '/home/hadoop/hdfs/finalModels/roofDP' + str(key) + '.pkl'
#     with client_hdfs.write(path) as writer:
#         dump(model, writer)
#         # dump(pickle.dumps(model), writer)


# defaultparam
#defaultparamfacade
#defaultparamroof