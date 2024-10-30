import pandas as pd
from src.models.lstm_model import LSTMModel
from src.strategies.ml_strategies import MLStrategy

# Load data
data = pd.read_csv('data/processed/indices/NIFTY_50_processed.csv')
X = data[['Open', 'High', 'Low', 'Close', 'Volume']].values

# Initialize the model and strategy
lstm_model = LSTMModel(input_shape=(X.shape[1], 1))
lstm_model.load('models/saved_models/lstm_model.h5')
strategy = MLStrategy(lstm_model)

# Execute trades
for i in range(len(X)):
    signal = strategy.generate_signal(X[i].reshape(1, -1))
    print(f"Trade Signal at step {i}: {signal}")