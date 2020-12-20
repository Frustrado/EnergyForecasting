
from sklearn.svm import SVC
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import SGDClassifier
from sklearn.naive_bayes import BernoulliNB
from sklearn.neural_network import MLPClassifier
from sklearn.svm import LinearSVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import StackingClassifier
import numpy as np
import itertools



def configuration():
    array_of_configs = []

    svc = {'base_estimator__C': [0.1, 1, 10],
           'base_estimator__kernel': ['linear', 'rbf', 'poly'],
           'base_estimator__degree': [0, 1, 2, 3, 4, 5, 6],
           'base_estimator__gamma': [0.1, 1, 10, 100],
           'base_estimator__coef0': [0.0],
           'base_estimator__shrinking': [True],
           'base_estimator__probability': [False],
           'base_estimator__tol': [0.001],
           'base_estimator__cache_size': [200],
           'base_estimator__class_weight': [None],
           'base_estimator__verbose': [False],
           'base_estimator__max_iter': [-1],
           'base_estimator__decision_function_shape': ['ovr'],
           'base_estimator__break_ties': [False],
           'base_estimator__random_state': [None]}

    xgbost = {'base_estimator__loss': ['deviance', 'exponential'],
              'base_estimator__learning_rate': [0.01, 0.1],
              'base_estimator__n_estimators': [10],
              'base_estimator__subsample': [0.5, 0.8, 1.0],
              'base_estimator__criterion': ["friedman_mse",  "mae"],
              'base_estimator__min_samples_split': np.linspace(0.1, 0.5, 3),
              'base_estimator__min_samples_leaf': np.linspace(0.1, 0.5, 3),
              'base_estimator__min_weight_fraction_leaf': [0.0],
              'base_estimator__max_depth': [3,5,8],
              'base_estimator__min_impurity_decrease': [0.0],
              'base_estimator__min_impurity_split': [None],
              'base_estimator__init': [None],
              'base_estimator__random_state': [None],
              'base_estimator__max_features': ["log2","sqrt"],
              'base_estimator__verbose': [0],
              'base_estimator__max_leaf_nodes': [None],
              'base_estimator__warm_start': [False],
              'base_estimator__presort': ['deprecated'],
              'base_estimator__validation_fraction': [0.1],
              'base_estimator__n_iter_no_change': [None],
              'base_estimator__tol': [0.0001],
              'base_estimator__ccp_alpha': [0.0]}

    rf = {'base_estimator__n_estimators': [100,200],
          'base_estimator__criterion': ['gini'],
          'base_estimator__max_depth': [10,30, None],
          'base_estimator__min_samples_split': [2, 5, 10],
          'base_estimator__min_samples_leaf': [1, 2, 4],
          'base_estimator__min_weight_fraction_leaf': [0.0],
          'base_estimator__max_features': ['auto', 'sqrt'],
          'base_estimator__max_leaf_nodes': [None],
          'base_estimator__min_impurity_decrease': [0.0],
          'base_estimator__min_impurity_split': [None],
          'base_estimator__bootstrap': [True, False],
          'base_estimator__oob_score': [False],
          'base_estimator__n_jobs': [None],
          'base_estimator__random_state': [None],
          'base_estimator__verbose': [0],
          'base_estimator__warm_start': [False],
          'base_estimator__class_weight': [None],
          'base_estimator__ccp_alpha': [0.0],
          'base_estimator__max_samples': [None]}

    sgd = {'base_estimator__loss': ["hinge", "log", "squared_hinge", "modified_huber"],
           'base_estimator__penalty': ["l2", "l1", "none"],
           'base_estimator__alpha': [0.0001, 0.001, 0.01, 0.1],
           'base_estimator__l1_ratio': [0, 0.05, 0.1,0.5, 0.8, 1],
           'base_estimator__fit_intercept': [True],
           'base_estimator__max_iter': [1000],
           'base_estimator__tol': [0.001],
           'base_estimator__shuffle': [True],
           'base_estimator__verbose': [0],
           'base_estimator__epsilon': [0.1],
           'base_estimator__n_jobs': [None],
           'base_estimator__random_state': [None],
           'base_estimator__learning_rate': ['optimal'],
           'base_estimator__eta0': [0.0],
           'base_estimator__power_t': [0.5],
           'base_estimator__early_stopping': [False],
           'base_estimator__validation_fraction': [0.1],
           'base_estimator__n_iter_no_change': [5],
           'base_estimator__class_weight': [None],
           'base_estimator__warm_start': [False],
           'base_estimator__average': [False]}

    bnb = {'base_estimator__alpha': [0.0, 0.0001, 0.001, 0.01, 0.1, 0.5, 1.0, 2.0, 10.0],
           'base_estimator__binarize': [0.0],
           'base_estimator__fit_prior': [True],
           'base_estimator__class_prior': [None]}

    mlp = {'base_estimator__hidden_layer_sizes': [(100,)],#list(itertools.permutations([50,100,150],2)) + list(itertools.permutations([50,100,150],3)) + [50,100,150],
           'base_estimator__activation': ['relu'],#, 'relu'],
           'base_estimator__solver': ['adam'],
           'base_estimator__alpha': [0.0001],
           'base_estimator__batch_size': ['auto'],
           'base_estimator__learning_rate': ['constant'],#, 'adaptive', 'invscaling'],
           'base_estimator__learning_rate_init': [0.001],
           'base_estimator__power_t': [0.5],
           'base_estimator__max_iter': [200],
           'base_estimator__shuffle': [True],
           'base_estimator__random_state': [None],
           'base_estimator__tol': [1e-5],
           'base_estimator__verbose': [False],
           'base_estimator__warm_start': [False],
           'base_estimator__momentum': [0.9],
           'base_estimator__nesterovs_momentum': [True],
           'base_estimator__early_stopping': [False],
           'base_estimator__validation_fraction': [0.1],
           'base_estimator__beta_1': [0.9],
           'base_estimator__beta_2': [0.999],
           'base_estimator__epsilon': [1e-8],
           'base_estimator__n_iter_no_change': [10],
           'base_estimator__max_fun': [15000]}

    lsvc = {'base_estimator__penalty': ['l2'],
            'base_estimator__loss': ['hinge'],
            'base_estimator__dual': [True],
            'base_estimator__tol': [0.0001],
            'base_estimator__C': [0.01, 0.1, 1.0, 10.0],
            'base_estimator__multi_class': ['ovr','crammer_singer'],
            'base_estimator__fit_intercept': [True],
            'base_estimator__intercept_scaling': [1],
            'base_estimator__class_weight': [None],
            'base_estimator__verbose': [0],
            'base_estimator__random_state': [None],
            'base_estimator__max_iter': [10000]}

    sc = {}

    # array_of_configs.append(svc)
    # array_of_configs.append(xgbost)
    # array_of_configs.append(rf)
    # array_of_configs.append(sgd)
    # array_of_configs.append(bnb)
    # array_of_configs.append(mlp)
    array_of_configs.append(lsvc)
    #     array_of_configs.append(sc)

    return array_of_configs


def models():
    array_of_configs = initialConfig()



    array_of_models = []
    # array_of_models.append(SVC().set_params(**array_of_configs[0]))
    array_of_models.append(GradientBoostingClassifier().set_params(**array_of_configs[0]))
    # array_of_models.append(RandomForestClassifier().set_params(**array_of_configs[2]))
    # array_of_models.append(SGDClassifier().set_params(**array_of_configs[3]))
    # array_of_models.append(BernoulliNB().set_params(**array_of_configs[4]))
    # array_of_models.append(MLPClassifier().set_params(**array_of_configs[0]))
    # array_of_models.append(LinearSVC().set_params(**array_of_configs[0]))
    #     array_of_models.append(StackingClassifier().set_params(**array_of_configs[7]))

    return array_of_models



def initialConfig():

    array_of_configs = []

    svc = {'C': 1.0,
           'kernel': 'rbf',
           'degree': 3,
           'gamma': 'scale',
           'coef0': 0.0,
           'shrinking': True,
           'probability': False,
           'tol': 0.001,
           'cache_size': 200,
           'class_weight': None,
           'verbose': False,
           'max_iter': -1,
           'decision_function_shape': 'ovr',
           'break_ties': False,
           'random_state': None}

    xgbost = {'loss': 'deviance',
              'learning_rate': 0.1,
              'n_estimators': 100,
              'subsample': 1.0,
              'criterion': 'friedman_mse',
              'min_samples_split': 2,
              'min_samples_leaf': 1,
              'min_weight_fraction_leaf': 0.0,
              'max_depth': 3,
              'min_impurity_decrease': 0.0,
              'min_impurity_split': None,
              'init': None,
              'random_state': None,
              'max_features': None,
              'verbose': 0,
              'max_leaf_nodes': None,
              'warm_start': False,
              'presort': 'deprecated',
              'validation_fraction': 0.1,
              'n_iter_no_change': None,
              'tol': 0.0001,
              'ccp_alpha': 0.0}

    rf = {'n_estimators': 100,
          'criterion': 'gini',
          'max_depth': None,
          'min_samples_split': 2,
          'min_samples_leaf': 1,
          'min_weight_fraction_leaf': 0.0,
          'max_features': 'auto',
          'max_leaf_nodes': None,
          'min_impurity_decrease': 0.0,
          'min_impurity_split': None,
          'bootstrap': True,
          'oob_score': False,
          'n_jobs': None,
          'random_state': None,
          'verbose': 0,
          'warm_start': False,
          'class_weight': None,
          'ccp_alpha': 0.0,
          'max_samples': None}

    sgd = {'loss': 'hinge',
           'penalty': 'l2',
           'alpha': 0.0001,
           'l1_ratio': 0.15,
           'fit_intercept': True,
           'max_iter': 1000,
           'tol': 0.001,
           'shuffle': True,
           'verbose': 0,
           'epsilon': 0.1,
           'n_jobs': None,
           'random_state': None,
           'learning_rate': 'optimal',
           'eta0': 0.0,
           'power_t': 0.5,
           'early_stopping': False,
           'validation_fraction': 0.1,
           'n_iter_no_change': 5,
           'class_weight': None,
           'warm_start': False,
           'average': False}

    bnb = {'alpha': 1.0,
           'binarize': 0.0,
           'fit_prior': True,
           'class_prior': None}

    mlp = {'hidden_layer_sizes': (100,),
           'activation': 'relu',
           'solver': 'adam',
           'alpha': 0.0001,
           'batch_size': 'auto',
           'learning_rate': 'constant',
           'learning_rate_init': 0.001,
           'power_t': 0.5,
           'max_iter': 200,
           'shuffle': True,
           'random_state': None,
           'tol': 0.0001,
           'verbose': False,
           'warm_start': False,
           'momentum': 0.9,
           'nesterovs_momentum': True,
           'early_stopping': False,
           'validation_fraction': 0.1,
           'beta_1': 0.9,
           'beta_2': 0.999,
           'epsilon': 1e-08,
           'n_iter_no_change': 10,
           'max_fun': 15000}

    lsvc = {'penalty': 'l2',
            'loss': 'squared_hinge',
            'dual': True,
            'tol': 0.0001,
            'C': 1.0,
            'multi_class': 'ovr',
            'fit_intercept': True,
            'intercept_scaling': 1,
            'class_weight': None,
            'verbose': 0,
            'random_state': None,
            'max_iter': 1000}

    # array_of_configs.append(svc)
    array_of_configs.append(xgbost)
    # array_of_configs.append(rf)
    # array_of_configs.append(sgd)
    # array_of_configs.append(bnb)
    # array_of_configs.append(mlp)
    # array_of_configs.append(lsvc)

    return array_of_configs