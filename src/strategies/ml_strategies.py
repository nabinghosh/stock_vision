import pandas as pd
import numpy as np
from typing import Optional
from .base_strategy import BaseStrategy
from ..models.lstm_model import LSTMModel
from ..models.random_forest_model import RandomForestModel
from ..models.xgboost_model import XGBoostModel

class MLStrategy(BaseStrategy):
    def __init__(self, model, threshold: float = 0.0):
        super().__init__(f'{model.model_name}_Strategy')
        self.model = model
        self.threshold = threshold
        
    def generate_signals(self, data: pd.DataFrame) -> pd.Series:
        # Preprocess data for the model
        X = self.model.preprocess_data(data)
        
        # Generate predictions
        predictions = self.model.predict(X)
        
        # Convert predictions to signals
        signals = pd.Series(0, index=data.index)
        signals[predictions > self.threshold] = 1  # Buy signal
        signals[predictions < -self.threshold] = -1  # Sell signal
        
        return signals

class EnsembleStrategy(BaseStrategy):
    def __init__(self, models: list, weights: Optional[list] = None):
        super().__init__('Ensemble_Strategy')
        self.models = models
        self.weights = weights if weights else [1/len(models)] * len(models)
        
    def generate_signals(self, data: pd.DataFrame) -> pd.Series:
        all_predictions = []
        
        # Get predictions from all models
        for model in self.models:
            X = model.preprocess_data(data)
            predictions = model.predict(X)
            all_predictions.append(predictions)
            
        # Combine predictions using weights
        weighted_predictions = np.average(all_predictions, axis=0, weights=self.weights)
        
        # Convert to signals
        signals = pd.Series(0, index=data.index)
        signals[weighted_predictions > 0] = 1  # Buy signal
        signals[weighted_predictions < 0] = -1  # Sell signal
        
        return signals

class AdaptiveMLStrategy(BaseStrategy):
    def __init__(self, models: list, 
                 lookback_period: int = 20,
                 update_frequency: int = 5):
         super().__init__('AdaptiveML_Strategy')
         self.models = models
         self.lookback_period = lookback_period
         self.update_frequency = update_frequency
         self.model_weights = [1/len(models)] * len(models)
         self.model_performances = [0] * len(models)
        
    def generate_signals(self, data: pd.DataFrame) -> pd.Series:
        # Update model weights based on recent performance
        if len(data) % self.update_frequency == 0:
            self._update_model_weights(data)
            
        all_predictions = []
        
        # Get predictions from all models
        for model in self.models:
            X = model.preprocess_data(data)
            predictions = model.predict(X)
            all_predictions.append(predictions)
            
        # Combine predictions using weights
        weighted_predictions = np.average(all_predictions, axis=0, weights=self.model_weights)
        
        # Convert to signals
        signals = pd.Series(0, index=data.index)
        signals[weighted_predictions > 0] = 1  # Buy signal
        signals[weighted_predictions < 0] = -1  # Sell signal
        
        return signals
        
    def _update_model_weights(self, data: pd.DataFrame):
        # Calculate recent performance for each model
        for i, model in enumerate(self.models):
            X = model.preprocess_data(data.iloc[-self.lookback_period:])
            predictions = model.predict(X)
            self.model_performances[i] = self._calculate_performance(predictions, data.iloc[-self.lookback_period:])
            
        # Update model weights based on performance
        self.model_weights = [perf / sum(self.model_performances) 
                             for perf in self.model_performances]
        
    def _calculate_performance(self, predictions: np.ndarray, data: pd.DataFrame) -> float:
        # Calculate performance metric (e.g., accuracy, profit)
        pass