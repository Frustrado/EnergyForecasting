import pandas as pd
import numpy as np

from Functions import initial_run, add_model, convert_models_toDataframe, prepare_data
from Models import configuration, models
from sklearn.svm import SVC

import warnings
warnings.filterwarnings('ignore')

series = pd.read_csv('wine.data', header=None)
series['value'] = np.random.randint(0, 2, series.shape[0])
data = series.values
data = prepare_data(data)

df, list_of_model_and_configs = initial_run(data, configuration(), models())

df, list_of_model_and_configs = add_model(data, df, SVC(), {'kernel': ['linear'],
                                                            'C': [0.025]}, list_of_model_and_configs)

testDataFrame = convert_models_toDataframe(list_of_model_and_configs)

print(testDataFrame)
