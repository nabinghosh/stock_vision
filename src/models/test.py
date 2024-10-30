import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
import tensorflow as tf
from tensorflow import keras
import glob
import logging

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Print TensorFlow and Keras versions for debugging
logger.info(f"TensorFlow version: {tf.__version__}")
logger.info(f"Keras version: {tf.keras.__version__}")

def prepare_data(file_path, sequence_length=60):
    df = pd.read_csv(file_path)
    
    features = ['Open', 'High', 'Low', 'Close', 'Volume', 'SMA_10', 'SMA_50', 'MACD']
    data = df[features].fillna(method='ffill')
    
    scaler = MinMaxScaler()
    scaled_data = scaler.fit_transform(data)
    
    X, y = [], []
    for i in range(len(scaled_data) - sequence_length):
        X.append(scaled_data[i:(i + sequence_length)])
        y.append(scaled_data[i + sequence_length])
    
    X, y = np.array(X), np.array(y)
    
    train_size = int(len(X) * 0.8)
    X_test, y_test = X[train_size:], y[train_size:]
    
    return X_test, y_test, scaler, df.index[train_size + sequence_length:]

def plot_predictions(actual, predicted, dates, title, save_path):
    plt.figure(figsize=(15, 6))
    plt.plot(dates, actual, label='Actual')
    plt.plot(dates, predicted, label='Predicted')
    plt.title(title)
    plt.xlabel('Date')
    plt.ylabel('Close Price')
    plt.legend()
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()

def test_model(model_path, data_path, output_path):
    try:
        # Load the model with custom_objects
        custom_objects = {
            'mse': keras.losses.MeanSquaredError,
            'mean_squared_error': keras.losses.MeanSquaredError
        }
        model = keras.models.load_model(model_path, custom_objects=custom_objects, compile=False)
        logger.info(f"Loaded model from {model_path}")

        # Recompile the model
        model.compile(optimizer='adam', loss='mean_squared_error')

        # Prepare test data
        X_test, y_test, scaler, test_dates = prepare_data(data_path)
        logger.info(f"Prepared test data from {data_path}")

        # Make predictions
        predictions = model.predict(X_test)
        logger.info("Made predictions on test data")

        # Inverse transform the predictions and actual values
        predictions_unscaled = scaler.inverse_transform(predictions)
        y_test_unscaled = scaler.inverse_transform(y_test)

        # Extract close prices
        actual_close = y_test_unscaled[:, 3]  # Assuming Close is the 4th column
        predicted_close = predictions_unscaled[:, 3]

        # Plot predictions vs actual
        plot_title = f"Stock Price Prediction - {os.path.basename(data_path).split('_')[0]}"
        plot_save_path = os.path.join(output_path, f"{os.path.basename(data_path).split('_')[0]}_test_predictions.png")
        plot_predictions(actual_close, predicted_close, test_dates, plot_title, plot_save_path)
        logger.info(f"Saved prediction plot to {plot_save_path}")

        # Calculate and log performance metrics
        mse = np.mean((actual_close - predicted_close)**2)
        mae = np.mean(np.abs(actual_close - predicted_close))
        logger.info(f"Mean Squared Error: {mse}")
        logger.info(f"Mean Absolute Error: {mae}")

    except Exception as e:
        logger.error(f"Error testing model: {str(e)}")
        logger.exception("Exception details:")

def main():
    base_path = r"D:\wwwch\Documents\GitHub\stock_vision"
    model_path = os.path.join(base_path, "model")
    data_path = os.path.join(base_path, "data", "processed")
    output_path = os.path.join(base_path, "test_results")

    os.makedirs(output_path, exist_ok=True)

    # Test stock models
    stock_models = glob.glob(os.path.join(model_path, "stocks", "*_model.h5"))
    for model in stock_models:
        stock_name = os.path.basename(model).replace("_model.h5", "")
        data_file = os.path.join(data_path, "stocks", f"{stock_name}_processed.csv")
        if os.path.exists(data_file):
            logger.info(f"Testing model for stock: {stock_name}")
            test_model(model, data_file, output_path)
        else:
            logger.warning(f"Data file not found for stock: {stock_name}")

    # Test index models
    index_models = glob.glob(os.path.join(model_path, "indices", "*_model.h5"))
    for model in index_models:
        index_name = os.path.basename(model).replace("_model.h5", "")
        data_file = os.path.join(data_path, "indices", f"{index_name}_processed.csv")
        if os.path.exists(data_file):
            logger.info(f"Testing model for index: {index_name}")
            test_model(model, data_file, output_path)
        else:
            logger.warning(f"Data file not found for index: {index_name}")

if __name__ == "__main__":
    main()