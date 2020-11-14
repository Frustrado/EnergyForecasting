from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM,SimpleRNN
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Dropout
from tensorflow import keras


def lstm_model(cfg, shapeX, shapeY):
    model = Sequential()
    model.add(LSTM(50, input_shape=(shapeX, shapeY)))
    model.add(Dropout(0.2))
    model.add(Dense(1))
    model.compile(loss='mae', optimizer='adam')
    return model

def rnn_model(cfg,shapeX, shapeY):
    model_rnn = Sequential()
    model_rnn.add(SimpleRNN(units=50,return_sequences=True, input_shape=(shapeX, shapeY)))
    model_rnn.add(Dropout(0.2))
    model_rnn.add(SimpleRNN(units=50))
    model_rnn.add(Dropout(0.2))
    model_rnn.add(Dense(units = 1))
    model_rnn.compile( loss='mean_squared_error',optimizer=keras.optimizers.Adam(0.001))
    return model_rnn
