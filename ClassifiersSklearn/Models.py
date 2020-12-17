
from sklearn.svm import SVC
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import SGDClassifier
from sklearn.naive_bayes import BernoulliNB
from sklearn.neural_network import MLPClassifier
from sklearn.svm import LinearSVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import StackingClassifier




def configuration():
    array_of_configs = []

    svc = {'C': [1.0],
           'kernel': ['rbf'],
           'degree': [3],
           'gamma': ['scale'],
           'coef0': [0.0],
           'shrinking': [True],
           'probability': [False],
           'tol': [0.001],
           'cache_size': [200],
           'class_weight': [None],
           'verbose': [False],
           'max_iter': [-1],
           'decision_function_shape': ['ovr'],
           'break_ties': [False],
           'random_state': [None]}

    xgbost = {'loss': ['deviance'],
              'learning_rate': [0.1],
              'n_estimators': [100],
              'subsample': [1.0],
              'criterion': ['friedman_mse'],
              'min_samples_split': [2],
              'min_samples_leaf': [1],
              'min_weight_fraction_leaf': [0.0],
              'max_depth': [3],
              'min_impurity_decrease': [0.0],
              'min_impurity_split': [None],
              'init': [None],
              'random_state': [None],
              'max_features': [None],
              'verbose': [0],
              'max_leaf_nodes': [None],
              'warm_start': [False],
              'presort': ['deprecated'],
              'validation_fraction': [0.1],
              'n_iter_no_change': [None],
              'tol': [0.0001],
              'ccp_alpha': [0.0]}

    rf = {'n_estimators': [100],
          'criterion': ['gini'],
          'max_depth': [None],
          'min_samples_split': [2],
          'min_samples_leaf': [1],
          'min_weight_fraction_leaf': [0.0],
          'max_features': ['auto'],
          'max_leaf_nodes': [None],
          'min_impurity_decrease': [0.0],
          'min_impurity_split': [None],
          'bootstrap': [True],
          'oob_score': [False],
          'n_jobs': [None],
          'random_state': [None],
          'verbose': [0],
          'warm_start': [False],
          'class_weight': [None],
          'ccp_alpha': [0.0],
          'max_samples': [None]}

    sgd = {'loss': ['hinge'],
           'penalty': ['l2'],
           'alpha': [0.0001],
           'l1_ratio': [0.15],
           'fit_intercept': [True],
           'max_iter': [1000],
           'tol': [0.001],
           'shuffle': [True],
           'verbose': [0],
           'epsilon': [0.1],
           'n_jobs': [None],
           'random_state': [None],
           'learning_rate': ['optimal'],
           'eta0': [0.0],
           'power_t': [0.5],
           'early_stopping': [False],
           'validation_fraction': [0.1],
           'n_iter_no_change': [5],
           'class_weight': [None],
           'warm_start': [False],
           'average': [False]}

    bnb = {'alpha': [1.0],
           'binarize': [0.0],
           'fit_prior': [True],
           'class_prior': [None]}

    mlp = {'hidden_layer_sizes': [(100,)],
           'activation': ['relu'],
           'solver': ['adam'],
           'alpha': [0.0001],
           'batch_size': ['auto'],
           'learning_rate': ['constant'],
           'learning_rate_init': [0.001],
           'power_t': [0.5],
           'max_iter': [200],
           'shuffle': [True],
           'random_state': [None],
           'tol': [0.0001],
           'verbose': [False],
           'warm_start': [False],
           'momentum': [0.9],
           'nesterovs_momentum': [True],
           'early_stopping': [False],
           'validation_fraction': [0.1],
           'beta_1': [0.9],
           'beta_2': [0.999],
           'epsilon': [1e-08],
           'n_iter_no_change': [10],
           'max_fun': [15000]}

    lsvc = {'penalty': ['l2'],
            'loss': ['squared_hinge'],
            'dual': [True],
            'tol': [0.0001],
            'C': [1.0],
            'multi_class': ['ovr'],
            'fit_intercept': [True],
            'intercept_scaling': [1],
            'class_weight': [None],
            'verbose': [0],
            'random_state': [None],
            'max_iter': [1000]}

    sc = {}

    array_of_configs.append(svc)
    array_of_configs.append(xgbost)
    # array_of_configs.append(rf)
    # array_of_configs.append(sgd)
    # array_of_configs.append(bnb)
    # array_of_configs.append(mlp)
    # array_of_configs.append(lsvc)
    #     array_of_configs.append(sc)

    return array_of_configs


def models():
    array_of_models = []
    array_of_models.append(SVC())
    # array_of_models.append(GradientBoostingClassifier())
    # array_of_models.append(RandomForestClassifier())
    # array_of_models.append(SGDClassifier())
    # array_of_models.append(BernoulliNB())
    # array_of_models.append(MLPClassifier())
    # array_of_models.append(LinearSVC())
    #     array_of_models.append(StackingClassifier())

    return array_of_models