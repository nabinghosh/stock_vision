import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
import tensorflow as tf
import glob
import logging
from concurrent.futures import ThreadPoolExecutor

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Function to create sequences for LSTM
def create_sequences(data, seq_length):
    X, y = [], []
    for i in range(len(data) - seq_length):
        X.append(data[i:(i + seq_length)])
        y.append(data[i + seq_length])
    return np.array(X), np.array(y)

# Function to load and preprocess data
def prepare_data(file_path, sequence_length=60):
    df = pd.read_csv(file_path)
    
    # Select features for training
    features = ['Open', 'High', 'Low', 'Close', 'Volume', 'SMA_10', 'SMA_50', 'MACD']
    data = df[features].fillna(method='ffill')
    
    # Scale the data
    scaler = MinMaxScaler()
    scaled_data = scaler.fit_transform(data)
    
    # Create sequences
    X, y = create_sequences(scaled_data, sequence_length)
    
    # Split data into train and test sets
    train_size = int(len(X) * 0.8)
    X_train, X_test = X[:train_size], X[train_size:]
    y_train, y_test = y[:train_size], y[train_size:]
    
    return X_train, X_test, y_train, y_test, scaler

# Function to create and train model
def create_model(input_shape):
    model = tf.keras.Sequential([
        tf.keras.layers.LSTM(50, return_sequences=True, input_shape=input_shape),
        tf.keras.layers.Dropout(0.2),
        tf.keras.layers.LSTM(50, return_sequences=False),
        tf.keras.layers.Dropout(0.2),
        tf.keras.layers.Dense(25),
        tf.keras.layers.Dense(8)  # Output shape matches number of features
    ])
    
    model.compile(optimizer='adam', loss='mse')
    return model

# Function to plot results
def plot_predictions(actual, predicted, title, save_path):
    plt.figure(figsize=(15, 6))
    plt.plot(actual, label='Actual')
    plt.plot(predicted, label='Predicted')
    plt.title(title)
    plt.legend()
    plt.savefig(save_path)
    plt.close()

# Function to process a single file
def process_file(file, model_save_path):
    stock_name = os.path.basename(file).replace('_processed.csv', '')
    logger.info(f"Processing stock: {stock_name}")
    
    # Prepare data
    X_train, X_test, y_train, y_test, scaler = prepare_data(file)
    
    # Create and train model
    model = create_model(input_shape=(X_train.shape[1], X_train.shape[2]))
    history = model.fit(
        X_train, y_train,
        epochs=50,
        batch_size=32,
        validation_split=0.1,
        verbose=1
    )
    
    # Save model
    model_path = os.path.join(model_save_path, 'stocks', f"{stock_name}_model.h5")
    model.save(model_path)
    logger.info(f"Model saved to {model_path}")
    
    # Make predictions
    predictions = model.predict(X_test)
    
    # Plot training history
    plt.figure(figsize=(15, 6))
    plt.plot(history.history['loss'], label='Training Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.title(f'{stock_name} - Training History')
    plt.legend()
    plt.savefig(os.path.join(model_save_path, 'stocks', f"{stock_name}_training_history.png"))
    plt.close()
    
    # Plot predictions vs actual
    plot_predictions(y_test[:, 3], predictions[:, 3], f'{stock_name} - Close Price Predictions', 
                     os.path.join(model_save_path, 'stocks', f"{stock_name}_predictions.png"))

def run(base_path, model_save_path):
    """
        Main function to run the stock prediction model training pipeline
        
        Args:
            base_path (str): Base path containing stocks and indices directories
            model_save_path (str): Base path for saving trained models
        """
    try:
            # Create directories for models if they don't exist
            os.makedirs(os.path.join(model_save_path, 'stocks'), exist_ok=True)
            os.makedirs(os.path.join(model_save_path, 'indices'), exist_ok=True)
            
            # Process stocks in parallel
            stock_files = glob.glob(os.path.join(base_path, 'stocks', '*.csv'))
            logger.info(f"Found {len(stock_files)} stock files to process")
            
            with ThreadPoolExecutor() as executor:
                executor.map(lambda file: process_file(file, model_save_path), stock_files)
            
            # Process indices in parallel
            indices_files = glob.glob(os.path.join(base_path, 'indices', '*.csv'))
            logger.info(f"Found {len(indices_files)} index files to process")
            
            with ThreadPoolExecutor() as executor:
                executor.map(lambda file: process_file(file, model_save_path), indices_files)
            
    except Exception as e:
            logger.error(f"Error in main execution: {str(e)}")
            raise

def retrain_model(model_path, new_data_path):
    """
    Retrain a model with new data
    
    Args:
        model_path (str): Path to the existing model
        new_data_path (str): Path to the new data file
    """
    try:
        # Load existing model
        model = tf.keras.models.load_model(model_path)
        
        # Prepare new data
        X_train, X_test, y_train, y_test, scaler = prepare_data(new_data_path)
        
        # Retrain model
        history = model.fit(
            X_train, y_train,
            epochs=50,
            batch_size=32,
            validation_split=0.1,
            verbose=1
        )
        
        # Save retrained model
        model.save(model_path)
        
        return model, history
    
    except Exception as e:
        logger.error(f"Error retraining model: {str(e)}")
        raise

def main():
    """
    Main execution function
    """
    try:
        # Define paths
        base_path = r"D:\wwwch\Documents\GitHub\stock_vision\data\processed"
        model_save_path = r"D:\wwwch\Documents\GitHub\stock_vision\model"
        
        # Run the pipeline
        logger.info("Starting model training pipeline...")
        run(base_path, model_save_path)
        logger.info("Model training pipeline completed successfully")
        
    except Exception as e:
        logger.error(f"Fatal error in main execution: {str(e)}")
        raise

if __name__ == "__main__":
    main()




















# import os
# import numpy as np
# import pandas as pd
# import matplotlib.pyplot as plt
# from sklearn.preprocessing import MinMaxScaler
# import tensorflow as tf
# import glob
# import logging

# # Function to create sequences for LSTM
# def create_sequences(data, seq_length):
#     X, y = [], []
#     for i in range(len(data) - seq_length):
#         X.append(data[i:(i + seq_length)])
#         y.append(data[i + seq_length])
#     return np.array(X), np.array(y)

# # Function to load and preprocess data
# def prepare_data(file_path, sequence_length=60):
#     df = pd.read_csv(file_path)
    
#     # Select features for training
#     features = ['Open', 'High', 'Low', 'Close', 'Volume', 'SMA_10', 'SMA_50', 'MACD']
#     data = df[features].fillna(method='ffill')
    
#     # Scale the data
#     scaler = MinMaxScaler()
#     scaled_data = scaler.fit_transform(data)
    
#     # Create sequences
#     X, y = create_sequences(scaled_data, sequence_length)
    
#     # Split data into train and test sets
#     train_size = int(len(X) * 0.8)
#     X_train, X_test = X[:train_size], X[train_size:]
#     y_train, y_test = y[:train_size], y[train_size:]
    
#     return X_train, X_test, y_train, y_test, scaler

# # Function to create and train model
# def create_model(input_shape):
#     model = tf.keras.Sequential([
#         tf.keras.layers.LSTM(50, return_sequences=True, input_shape=input_shape),
#         tf.keras.layers.Dropout(0.2),
#         tf.keras.layers.LSTM(50, return_sequences=False),
#         tf.keras.layers.Dropout(0.2),
#         tf.keras.layers.Dense(25),
#         tf.keras.layers.Dense(8)  # Output shape matches number of features
#     ])
    
#     model.compile(optimizer='adam', loss='mse')
#     return model

# # Function to plot results
# def plot_predictions(actual, predicted, title):
#     plt.figure(figsize=(15, 6))
#     plt.plot(actual, label='Actual')
#     plt.plot(predicted, label='Predicted')
#     plt.title(title)
#     plt.legend()
#     plt.show()

# # Create directories for models if they don't exist
# os.makedirs('model/stocks', exist_ok=True)
# os.makedirs('model/indices', exist_ok=True)

# # # Process stocks
# # stock_files = glob.glob('D:/wwwch/Documents/GitHub/stock_vision/data/processed/stocks/*.csv')
# # for file in stock_files:
# #     stock_name = os.path.basename(file).replace('_processed.csv', '')
# #     print(f"Processing stock: {stock_name}")
    
# #     # Prepare data
# #     X_train, X_test, y_train, y_test, scaler = prepare_data(file)
    
# #     # Create and train model
# #     model = create_model(input_shape=(X_train.shape[1], X_train.shape[2]))
# #     history = model.fit(
# #         X_train, y_train,
# #         epochs=50,
# #         batch_size=32,
# #         validation_split=0.1,
# #         verbose=1
# #     )
    
# #     # Save model
# #     model.save(f'model/stocks/{stock_name}_model.h5')
    
# #     # Make predictions
# #     predictions = model.predict(X_test)
    
# #     # Plot training history
# #     plt.figure(figsize=(15, 6))
# #     plt.plot(history.history['loss'], label='Training Loss')
# #     plt.plot(history.history['val_loss'], label='Validation Loss')
# #     plt.title(f'{stock_name} - Training History')
# #     plt.legend()
# #     plt.show()
    
# #     # Plot predictions vs actual
# #     plot_predictions(y_test[:, 3], predictions[:, 3], f'{stock_name} - Close Price Predictions')

# # # Process indices
# # indices_files = glob.glob('D:/wwwch/Documents/GitHub/stock_vision/data/processed/indices/*.csv')
# # for file in indices_files:
# #     index_name = os.path.basename(file).replace('_processed.csv', '')
# #     print(f"Processing index: {index_name}")
    
# #     # Prepare data
# #     X_train, X_test, y_train, y_test, scaler = prepare_data(file)
    
# #     # Create and train model
# #     model = create_model(input_shape=(X_train.shape[1], X_train.shape[2]))
# #     history = model.fit(
# #         X_train, y_train,
# #         epochs=50,
# #         batch_size=32,
# #         validation_split=0.1,
# #         verbose=1
# #     )
    
# #     # Save model
# #     model.save(f'model/indices/{index_name}_model.h5')
    
# #     # Make predictions
# #     predictions = model.predict(X_test)
    
# #     # Plot training history
# #     plt.figure(figsize=( 15, 6))
# #     plt.plot(history.history['loss'], label='Training Loss')
# #     plt.plot(history.history['val_loss'], label='Validation Loss')
# #     plt.title(f'{index_name} - Training History')
# #     plt.legend()
# #     plt.show()
    
# #     # Plot predictions vs actual
# #     plot_predictions(y_test[:, 3], predictions[:, 3], f'{index_name} - Close Price Predictions')

# # # Function to retrain model
# # def retrain_model(model_path, new_data_path):
# #     # Load existing model
# #     model = tf.keras.models.load_model(model_path)
    
# #     # Prepare new data
# #     X_train, X_test, y_train, y_test, scaler = prepare_data(new_data_path)
    
# #     # Retrain model
# #     history = model.fit(
# #         X_train, y_train,
# #         epochs=50,
# #         batch_size=32,
# #         validation_split=0.1,
# #         verbose=1
# #     )
    
# #     # Save retrained model
# #     model.save(model_path)
    
# #     return model, history



# # Set up logging
# logging.basicConfig(
#     level=logging.INFO,
#     format='%(asctime)s - %(levelname)s - %(message)s'
# )
# logger = logging.getLogger(__name__)

# # [Previous functions remain the same: create_sequences, prepare_data, create_model, plot_predictions]

# def run(base_path, model_save_path):
#     """
#     Main function to run the stock prediction model training pipeline
    
#     Args:
#         base_path (str): Base path containing stocks and indices directories
#         model_save_path (str): Base path for saving trained models
#     """
#     try:
#         # Create directories for models if they don't exist
#         os.makedirs(os.path.join(model_save_path, 'stocks'), exist_ok=True)
#         os.makedirs(os.path.join(model_save_path, 'indices'), exist_ok=True)
        
#         # Process stocks
#         stock_files = glob.glob(os.path.join(base_path, 'stocks', '*.csv'))
#         logger.info(f"Found {len(stock_files)} stock files to process")
        
#         for file in stock_files:
#             stock_name = os.path.basename(file).replace('_processed.csv', '')
#             logger.info(f"Processing stock: {stock_name}")
            
#             try:
#                 # Prepare data
#                 X_train, X_test, y_train, y_test, scaler = prepare_data(file)
                
#                 # Create and train model
#                 model = create_model(input_shape=(X_train.shape[1], X_train.shape[2]))
#                 history = model.fit(
#                     X_train, y_train,
#                     epochs=50,
#                     batch_size=32,
#                     validation_split=0.1,
#                     verbose=1
#                 )
                
#                 # Save model
#                 model_path = os.path.join(model_save_path, 'stocks', f"{stock_name}_model.h5")
#                 model.save(model_path)
#                 logger.info(f"Model saved to {model_path}")
                
#                 # Make predictions
#                 predictions = model.predict(X_test)
                
#                 # Plot training history
#                 plt.figure(figsize=(15, 6))
#                 plt.plot(history.history['loss'], label='Training Loss')
#                 plt.plot(history.history['val_loss'], label='Validation Loss')
#                 plt.title(f'{stock_name} - Training History')
#                 plt.legend()
#                 plt.savefig(os.path.join(model_save_path, 'stocks', f"{stock_name}_training_history.png"))
#                 plt.close()
                
#                 # Plot predictions vs actual
#                 plot_predictions(y_test[:, 3], predictions[:, 3], f'{stock_name} - Close Price Predictions')
#                 plt.savefig(os.path.join(model_save_path, 'stocks', f"{stock_name}_predictions.png"))
#                 plt.close()
                
#             except Exception as e:
#                 logger.error(f"Error processing stock {stock_name}: {str(e)}")
#                 continue
        
#         # Process indices
#         indices_files = glob.glob(os.path.join(base_path, 'indices', '*.csv'))
#         logger.info(f"Found {len(indices_files)} index files to process")
        
#         for file in indices_files:
#             index_name = os.path.basename(file).replace('_processed.csv', '')
#             logger.info(f"Processing index: {index_name}")
            
#             try:
#                 # Prepare data
#                 X_train, X_test, y_train, y_test, scaler = prepare_data(file)
                
#                 # Create and train model
#                 model = create_model(input_shape=(X_train.shape[1], X_train.shape[2]))
#                 history = model.fit(
#                     X_train, y_train,
#                     epochs=50,
#                     batch_size=32,
#                     validation_split=0.1,
#                     verbose=1
#                 )
                
#                 # Save model
#                 model_path = os.path.join(model_save_path, 'indices', f"{index_name}_model.h5")
#                 model.save(model_path)
#                 logger.info(f"Model saved to {model_path}")
                
#                 # Make predictions
#                 predictions = model.predict(X_test)
                
#                 # Plot training history
#                 plt.figure(figsize=(15, 6))
#                 plt.plot(history.history['loss'], label='Training Loss')
#                 plt.plot(history.history['val_loss'], label='Validation Loss')
#                 plt.title(f'{index_name} - Training History')
#                 plt.legend()
#                 plt.savefig(os.path.join(model_save_path, 'indices', f"{index_name}_training_history.png"))
#                 plt.close()
                
#                 # Plot predictions vs actual
#                 plot_predictions(y_test[:, 3], predictions[:, 3], f'{index_name} - Close Price Predictions')
#                 plt.savefig(os.path.join(model_save_path, 'indices', f"{index_name}_predictions.png"))
#                 plt.close()
                
#             except Exception as e:
#                 logger.error(f"Error processing index {index_name}: {str(e)}")
#                 continue
                
#     except Exception as e:
#         logger.error(f"Error in main execution: {str(e)}")
#         raise

# def main():
#     """
#     Main execution function
#     """
#     try:
#         # Define paths
#         base_path = r"D:\wwwch\Documents\GitHub\stock_vision\data\processed"
#         model_save_path = r"D:\wwwch\Documents\GitHub\stock_vision\model"
        
#         # Run the pipeline
#         logger.info("Starting model training pipeline...")
#         run(base_path, model_save_path)
#         logger.info("Model training pipeline completed successfully")
        
#     except Exception as e:
#         logger.error(f"Fatal error in main execution: {str(e)}")
#         raise

# if __name__ == "__main__":
#     main()