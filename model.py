# model.py

import numpy as np
import pandas as pd
from keras.models import Sequential
from keras.layers import Dense, Dropout, LSTM
from sklearn.preprocessing import MinMaxScaler

scaler = MinMaxScaler(feature_range=(0,1))

def train_lstm_model(data_train):
    data_train_scale = scaler.fit_transform(data_train)

    x = []
    y = []

    for i in range(100, data_train_scale.shape[0]):
        x.append(data_train_scale[i-100:i])
        y.append(data_train_scale[i, 0])

    x, y = np.array(x), np.array(y)

    model = Sequential()
    model.add(LSTM(units=50, activation='relu', return_sequences=True, input_shape=(x.shape[1], 1)))
    model.add(Dropout(0.2))
    model.add(LSTM(units=60, activation='relu', return_sequences=True))
    model.add(Dropout(0.3))
    model.add(LSTM(units=80, activation='relu', return_sequences=True))
    model.add(Dropout(0.4))
    model.add(LSTM(units=120, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(units=1))

    model.compile(optimizer='adam', loss='mean_squared_error')
    model.fit(x, y, epochs=50, batch_size=32, verbose=1)

    return model, scaler


def predict_prices(model, data_test):
    data_test_scale = scaler.transform(data_test)

    x = []
    for i in range(100, data_test_scale.shape[0]):
        x.append(data_test_scale[i-100:i])

    x = np.array(x)
    predictions = model.predict(x)

    return predictions
