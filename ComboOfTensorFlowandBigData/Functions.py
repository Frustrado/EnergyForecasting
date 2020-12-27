import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from numpy import concatenate
from sklearn.metrics import explained_variance_score,max_error,mean_absolute_error,mean_squared_error,mean_squared_log_error,median_absolute_error,r2_score

from Configuration import configuration, default_config
from Models import lstm_model, rnn_model, cnn_model, mlp_model, conv2d_model
from datetime import datetime
from math import sqrt

def prepare_data():
    df = pd.read_csv("german.uci.csv")
    # data = pd.read_csv("dfValidation.csv")


    # data = pd.read_csv("city_day.csv")
    # data['Date'] = data['Date'].apply(pd.to_datetime)
    # data.set_index('Date', inplace=True)
    #
    # df = data.loc[data['City'] == 'Delhi']
    # df.isnull().sum()
    #
    # df = df.drop(columns=['City', 'AQI_Bucket', 'Xylene'])
    #
    # df = df.fillna(df.mean())



    # df = data.iloc[:1000, 1:]



    dataset = df.values
    scaler = MinMaxScaler(feature_range=(0, 1))
    dataset = scaler.fit_transform(dataset)
    train_size = int(len(dataset) * 0.80)
    test_size = len(dataset) - train_size
    train, test = dataset[0:train_size, :], dataset[train_size:len(dataset), :]
    # train_X, train_y = train[:, 1:], train[:, 0]
    # test_X, test_y = test[:, 1:], test[:, 0]

    train_X, train_y = train[:, 0:24], train[:, 24]
    test_X, test_y = test[:, 0:24], test[:, 24]
    # train_X, train_y = train[:, :-1], train[:, -1]
    print(train_X)
    print(train_y)
    # test_X, test_y = test[:, :-1], test[:, -1]
    train_X = train_X.reshape((train_X.shape[0], 1, train_X.shape[1]))
    test_X = test_X.reshape((test_X.shape[0], 1, test_X.shape[1]))
    return train_X, train_y, test_X, test_y, scaler


def convert_data(train_X,test_X,train_y,test_y,train_predict,test_predict,scaler):
    train_X = train_X.reshape((train_X.shape[0], train_X.shape[2]))
    test_X = test_X.reshape((test_X.shape[0], test_X.shape[2]))
    # print(train_predict.shape)
    # print(train_X.shape)
    inv_train_predict = concatenate((train_predict, train_X), axis=1)
    inv_test_predict = concatenate((test_predict, test_X), axis=1)

    #transforming to original scale
    inv_train_predict = scaler.inverse_transform(inv_train_predict)
    inv_test_predict = scaler.inverse_transform(inv_test_predict)

    #predicted values on training data
    inv_train_predict = inv_train_predict[:,0]
    inv_train_predict

    #predicted values on testing data
    inv_test_predict = inv_test_predict[:,0]
    inv_test_predict

    #scaling back the original train labels
    train_y = train_y.reshape((len(train_y), 1))
    inv_train_y = concatenate((train_y, train_X), axis=1)
    inv_train_y = scaler.inverse_transform(inv_train_y)
    inv_train_y = inv_train_y[:,0]

    #scaling back the original test labels
    test_y = test_y.reshape((len(test_y), 1))
    inv_test_y = concatenate((test_y, test_X), axis=1)
    inv_test_y = scaler.inverse_transform(inv_test_y)
    inv_test_y = inv_test_y[:,0]
    return inv_test_y, inv_test_predict

def measure_error(actual, predicted):
    return {'EVC': explained_variance_score(actual, predicted),
            'ME': max_error(actual, predicted),
            'MAE': mean_absolute_error(actual, predicted),
            'MSE': mean_squared_error(actual, predicted),
            'RMSE': sqrt(mean_squared_error(actual, predicted)),
            'MSLE': mean_squared_log_error(actual, predicted),
            'MEDAE': median_absolute_error(actual, predicted),
            'R2': r2_score(actual, predicted)}

def predictions(train_X, train_y, test_X, test_y,scaler,model,cfg):
    if (model._name.split("-")[0] == 'conv2d'):
        train_X = train_X.reshape(train_X.shape[0],train_X.shape[1], 1,train_X.shape[2], 1)
        test_X = test_X.reshape(test_X.shape[0],test_X.shape[1],1, test_X.shape[2], 1)
    # print(train_X)
    # print(train_y)
    model.fit(train_X, train_y,epochs=cfg.get('n_epochs'),batch_size=cfg.get('n_batch'),validation_split=0.1,verbose=1,shuffle=False)
    train_predict = model.predict(train_X)
    test_predict = model.predict(test_X)

    if(model._name.split("-")[0] == 'conv2d'):
        train_X = train_X.reshape(train_X.shape[0],train_X.shape[1],train_X.shape[3])
        test_X = test_X.reshape(test_X.shape[0], test_X.shape[1], test_X.shape[3])

    #reshape for multilayer perceptron
    if(model._name.split("-")[0]=='mlp'):
        train_predict = train_predict.reshape(train_predict.shape[0],train_predict.shape[1])
        test_predict = test_predict.reshape(test_predict.shape[0],test_predict.shape[1])
    # if(model._name.split("-")[0]=='conv2d'):
    #     train_predict = train_predict.reshape(train_predict.shape[0], train_predict.shape[1],1,1)
    #     test_predict = test_predict.reshape(test_predict.shape[0], test_predict.shape[1],1,1)

    inv_test_y, inv_test_predict = convert_data(train_X,test_X,train_y,test_y,train_predict,test_predict,scaler)
    measure = measure_error(inv_test_y, inv_test_predict)
    measure['Model']=model.name
    return measure


def run(train_X, train_y, test_X, test_y, scaler):
    models_list = []
    models_list.append(lstm_model)
    # models_list.append(rnn_model)
    # models_list.append(cnn_model)
    # models_list.append(mlp_model)
    # models_list.append(conv2d_model)

    config_list = []
    config_list.append(configuration('lstm', default_config()))
    # config_list.append(configuration('rnn', default_config()))
    # config_list.append(configuration('cnn', default_config()))
    # config_list.append(configuration('mlp', default_config()))
    # config_list.append(configuration('conv2d', default_config()))

    array_of_results = []
    results = {}
    array_of_models = []
    i = 0
    for def_model, def_cfg in zip(models_list, config_list):
        for cfg in def_cfg:
            model = def_model(cfg,train_X.shape[1], train_X.shape[2])
            print(model)
            model._name = str(def_model.__name__.split("_")[0] + "-" + str(i))
            i += 1
            array_of_models.append(model)
            array_of_results.append(predictions(train_X, train_y, test_X,test_y, scaler, model, cfg))
    return array_of_models, pd.DataFrame(array_of_results)

def get_min_model(result_df):
    print(result_df.loc[result_df['RMSE'] == result_df['RMSE'].min()])
    return result_df.loc[result_df['RMSE'] == result_df['RMSE'].min()]['Model']

def get_model(list_of_models, name_of_model):
    for m in list_of_models:
        if m.get_config()['name']==name_of_model:
            return m

def get_config(list_of_models):
    list_of_configs = {}
    for m in list_of_models:
        list_of_configs[m.get_config()['name']] = m.get_config()['layers']
    return list_of_configs

def convert_config_to_datframes(config_json):

    index = -1
    tmp = []
    list_of_final_dataframes = []
    dfJson1 =None
    for i in config_json:
        if dfJson1 is not None:
            list_of_final_dataframes.append(dfJson1)
        dfJson1 = pd.DataFrame(index=[],columns=['name'])
        for j in config_json[i]:
            index += 1
            dfJson1.loc[index,'date']=datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            dfJson1.loc[index,'class_name']=j['class_name']
            dfJson1.loc[index,'name']=i
            for k in j['config']:
                dfJson1.loc[index,'cfg-'+str(k)]=str(j['config'][k])
    list_of_final_dataframes.append(dfJson1)
    return list_of_final_dataframes


def run_model(train_X, train_y, test_X,test_y, model_name, cfg,list_models, df_results, scaler):
    array_of_results = []
    i = int(model_name.split("-")[1])
    model = eval(model_name.split("-")[0]+"_model")(cfg,train_X.shape[1], train_X.shape[2])
    model._name=str(model_name.split("-")[0]+"-"+str(i+1))
    result = predictions(train_X, train_y, test_X,test_y, scaler,model,cfg)
    result_x = pd.DataFrame.from_dict([result])
    list_models.append(model)
    return pd.concat([result_x,df_results]), list_models