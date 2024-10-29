import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from tensorflow.keras.models import load_model
from sklearn.preprocessing import MinMaxScaler
from .base_model import BaseModel

class LSTMModel(BaseModel):
    def __init__(self, input_shape, units=50, epochs=100, batch_size=32):
        super().__init__("LSTM")
        self.input_shape = input_shape
        self.units = units
        self.epochs = epochs
        self.batch_size = batch_size
        self.scaler = MinMaxScaler()

    def preprocess_data(self, data):
        scaled_data = self.scaler.fit_transform(data)
        X, y = [], []
        for i in range(self.input_shape[0], len(scaled_data)):
            X.append(scaled_data[i-self.input_shape[0]:i, 0])
            y.append(scaled_data[i, 0])
        return np.array(X), np.array(y)

    def train(self, X, y):
        self.model = Sequential([
            LSTM(units=self.units, return_sequences=True, input_shape=self.input_shape),
            LSTM(units=self.units),
            Dense(1)
        ])
        self.model.compile(optimizer='adam', loss='mean_squared_error')
        self.model.fit(X, y, epochs=self.epochs, batch_size=self.batch_size, verbose=1)

    def predict(self, X):
        return self.model.predict(X)

    def save_model(self, filepath):
        self.model.save(filepath)

    def load_model(self, filepath):
        self.model = load_model(filepath)