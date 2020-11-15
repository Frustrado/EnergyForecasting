from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM,SimpleRNN, Conv1D, MaxPooling1D, Flatten,\
    ConvLSTM2D, LeakyReLU, BatchNormalization
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

def cnn_model(cfg, shapeX, shapeY):
    model = Sequential()
    model.add(Conv1D(filters=64, kernel_size=1, activation='relu', input_shape=(shapeX, shapeY)))
    model.add(MaxPooling1D(pool_size=1))
    model.add(Flatten())
    model.add(Dense(50, activation='relu'))
    model.add(Dense(1))
    model.compile(optimizer='adam', loss='mse')
    return model

def mlp_model(cfg, shapeX, shapeY):
    model = Sequential()
    model.add(Dense(100, activation='relu', input_dim = shapeY))
    model.add(Dense(1))
    model.compile(loss='mse', optimizer=keras.optimizers.Adam(0.001))
    return model

## https://github.com/deKeijzer/Multivariate-time-series-models-in-Keras/blob/master/notebooks/ConvLSTM2D.ipynb
def conv2d_model(cfg,shapeX,shapeY):
    model = Sequential()
    model.add(ConvLSTM2D(
        filters=8,
        kernel_size=(10, 10),
        input_shape=(1,shapeX, shapeY,1),
        padding='same',
        return_sequences=True))
    model.add(LeakyReLU())
    model.add(BatchNormalization())

    model.add(ConvLSTM2D(
        filters=8,
        kernel_size=(5, 5),
        padding='same',
        return_sequences=False))
    model.add(LeakyReLU())
    model.add(BatchNormalization())
    model.add(Dropout(0.2))

    model.add(Flatten())

    model.add(Dense(32, kernel_initializer='TruncatedNormal'))
    model.add(LeakyReLU())
    model.add(Dropout(0.2))

    model.add(Dense(16, kernel_initializer='TruncatedNormal'))
    model.add(LeakyReLU())
    model.add(Dropout(0.2))

    model.add(Dense(1))

    model.compile(loss=keras.losses.categorical_crossentropy,
                  optimizer=keras.optimizers.Adam(),
                  metrics=['accuracy'])
    return model