import pandas as pd
import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import logging

logger = logging.getLogger(__name__)

class BaseModel:
    def __init__(self, model_name):
        self.model_name = model_name
        self.model = None

    def load_data(self, filepath):
        try:
            data = pd.read_csv(filepath, index_col=0, parse_dates=True)
            logger.info(f"Loaded data for {self.model_name}: {data.shape}")
            return data
        except Exception as e:
            logger.error(f"Error loading data for {self.model_name}: {e}")
            return pd.DataFrame()

    def preprocess_data(self, data):
        """Preprocess the data for model training and prediction."""
        raise NotImplementedError("Preprocess method must be implemented by subclasses.")

    def train(self, X, y):
        """Train the model on the provided data."""
        raise NotImplementedError("Train method must be implemented by subclasses.")

    def predict(self, X):
        """Make predictions using the trained model."""
        raise NotImplementedError("Predict method must be implemented by subclasses.")

    def evaluate(self, y_true, y_pred):
        """Evaluate the model's performance."""
        mse = mean_squared_error(y_true, y_pred)
        mae = mean_absolute_error(y_true, y_pred)
        r2 = r2_score(y_true, y_pred)
        logger.info(f"{self.model_name} - MSE: {mse}, MAE: {mae}, R2: {r2}")
        return {'mse': mse, 'mae': mae, 'r2': r2}

    def save_model(self, filepath):
        """Save the trained model to a file."""
        raise NotImplementedError("Save model method must be implemented by subclasses.")

    def load_model(self, filepath):
        """Load a model from a file."""
        raise NotImplementedError("Load model method must be implemented by subclasses.")