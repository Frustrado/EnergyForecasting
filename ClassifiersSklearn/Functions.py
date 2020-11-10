import json

from sklearn.model_selection import GridSearchCV, train_test_split
import pandas as pd
from datetime import datetime

from sklearn.metrics import accuracy_score, f1_score, auc, average_precision_score, roc_auc_score,roc_curve
from sklearn.preprocessing import StandardScaler


def gini_normalized(y_actual, y_pred):
    gini = lambda a, p: 2 * roc_auc_score(a, p) - 1
    return gini(y_actual, y_pred) / gini(y_actual, y_actual)

def measure_error(actual, predicted):
    return {'as': [accuracy_score(actual, predicted)],
#             'auc':[auc(actual, predicted)],
            'apc':[average_precision_score(actual, predicted)],
            'f1': [f1_score(actual, predicted)],
            'roc_auc': [roc_auc_score(actual, predicted)],
            'roc_cur': [roc_curve(actual, predicted)],#moze sie przez to wywalic
            'gini': [gini_normalized(actual, predicted)]}



def prepare_data(data):
    X, y = data[:, :14], data[:, 14]
    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0, stratify=y)

    sc = StandardScaler()
    sc.fit(X_train)
    X_train_std = sc.transform(X_train)
    X_test_std = sc.transform(X_test)

    return [X_train_std,X_test_std,y_train, y_test]

def initial_run(data, config, models):
    df = pd.DataFrame()
    list_of_models = {}

    X_train_std,X_test_std,y_train,y_test=data

    for c, m in zip(config, models):
        clf = GridSearchCV(m, c)
        clf.fit(X_train_std, y_train)
        #         print(clf.estimator)

        if (df.size < 1):
            df = pd.DataFrame(measure_error(y_test, clf.predict(X_test_std)))
            df['model'] = str(m) + "-" + str(1)
            list_of_models[str(m) + "-" + str(1)] = clf
            df['date'] = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        else:
            df1 = pd.DataFrame(measure_error(y_test, clf.predict(X_test_std)))
            df1['model'] = str(m) + "-" + str(1)
            list_of_models[str(m) + "-" + str(1)] = clf
            df1['date'] = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            df = pd.concat([df, df1])
    df.reset_index(inplace=True)
    df.drop('index', inplace=True, axis=1)
    return df, list_of_models


def add_model(data, dataframe, model, config, list_of_models):
    X_train_std, X_test_std, y_train, y_test = data
    clf = GridSearchCV(model, config)
    clf.fit(X_train_std, y_train)
    #     list_of_models.append(clf)

    df = pd.DataFrame(measure_error(y_test, clf.predict(X_test_std)))
    df['model'] = str(clf.estimator) + "-" + str(int(
        dataframe[dataframe['model'].str.contains(str(clf.estimator))]['model'].tail(1).values[0].split('-')[1]) + 1)
    list_of_models[str(clf.estimator) + "-" + str(int(
        dataframe[dataframe['model'].str.contains(str(clf.estimator))]['model'].tail(1).values[0].split('-')[
            1]) + 1)] = clf
    df['date'] = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    dataframe = pd.concat([dataframe, df])
    dataframe.reset_index(inplace=True)
    dataframe.drop('index', inplace=True, axis=1)
    return dataframe, list_of_models

def find_max(df,stat):
    return df.loc[df[stat].idxmax()]['model']

def get_best_model(models,key):
    return models.get(key)


def convert_models_toDataframe(models):
    list_of_dfs = {}
    for k, v in models.items():
        key = k.split('-')[0]
        if key in list_of_dfs:
            if not isinstance(list_of_dfs[key], list):
                list_of_dfs[key] = [list_of_dfs[key]]
            list_of_dfs[key].append({k: v})
        else:
            list_of_dfs[key] = ({k: v})

    final_list = []
    for k, v in list_of_dfs.items():
        df = pd.DataFrame()
        if isinstance(v, list):
            for i in v:
                if (df.size < 1):
                    df = pd.DataFrame((i.get(list(i.keys())[0])).param_grid)
                    df['model'] = list(i.keys())[0]
                else:
                    df2 = pd.DataFrame((i.get(list(i.keys())[0])).param_grid)
                    df2['model'] = str(list(i.keys())[0])
                    df = pd.concat([df, df2])
        else:
            df = pd.DataFrame((v.get(list(v.keys())[0])).param_grid)
            df['model'] = list(v.keys())[0]
        final_list.append(df)

    return final_list

def convert_configs_to_json(configs):
    list_of_json = []
    for i in configs:
        dict = {}
        dict['estimator'] = str((configs.get(i)).estimator)
        dict['param_grid'] = (configs.get(i)).param_grid
        json_object = json.dumps(dict, indent = 4)
        list_of_json.append(json_object)
    return list_of_json