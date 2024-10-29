from abc import ABC, abstractmethod
import pandas as pd
import numpy as np
from typing import Dict, Optional
import logging

logger = logging.getLogger(__name__)

class BaseStrategy(ABC):
    def __init__(self, name: str):
        self.name = name
        self.positions = {}
        self.portfolio_value = []
        self.trades = []
        
    @abstractmethod
    def generate_signals(self, data: pd.DataFrame) -> pd.Series:
        """Generate trading signals (-1: sell, 0: hold, 1: buy)"""
        pass
    
    def calculate_position_size(self, signal: float, 
                              available_capital: float,
                              current_price: float,
                              risk_per_trade: float = 0.02) -> float:
        """Calculate position size based on risk management rules"""
        if signal == 0:
            return 0
            
        position_size = (available_capital * risk_per_trade) / current_price
        return position_size if signal == 1 else -position_size
        
    def execute_trade(self, signal: float, 
                     price: float, 
                     timestamp: pd.Timestamp,
                     quantity: float):
        """Record trade execution"""
        if quantity != 0:
            self.trades.append({
                'timestamp': timestamp,
                'signal': signal,
                'price': price,
                'quantity': quantity
            })
            logger.info(f"Executed trade: {signal} {quantity} units at {price}")
            
    def update_portfolio(self, current_prices: pd.Series):
        """Update portfolio value based on current positions and prices"""
        total_value = sum(pos * current_prices[symbol] 
                         for symbol, pos in self.positions.items())
        self.portfolio_value.append(total_value)
        
    def get_performance_metrics(self) -> Dict:
        """Calculate strategy performance metrics"""
        returns = pd.Series(self.portfolio_value).pct_change().dropna()
        
        metrics = {
            'total_return': (self.portfolio_value[-1] / self.portfolio_value[0]) - 1,
            'sharpe_ratio': returns.mean() / returns.std() * np.sqrt(252),
            'max_drawdown': self._calculate_max_drawdown(),
            'num_trades': len(self.trades)
        }
        return metrics
        
    def _calculate_max_drawdown(self) -> float:
        """Calculate maximum drawdown"""
        portfolio_series = pd.Series(self.portfolio_value)
        rolling_max = portfolio_series.expanding().max()
        drawdowns = portfolio_series / rolling_max - 1
        return drawdowns.min()