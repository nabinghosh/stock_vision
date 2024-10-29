import pandas as pd
import numpy as np
from typing import Optional
from .base_strategy import BaseStrategy
import tulipy as ti

class RSIStrategy(BaseStrategy):
    def __init__(self, 
                 period: int = 14,
                 overbought: float = 70,
                 oversold: float = 30):
        super().__init__('RSI')
        self.period = period
        self.overbought = overbought
        self.oversold = oversold
        
    def generate_signals(self, data: pd.DataFrame) -> pd.Series:
        rsi = ti.rsi(data['close'].values, timeperiod=self.period)
        
        signals = pd.Series(0, index=data.index)
        signals[rsi < self.oversold] = 1  # Buy signal
        signals[rsi > self.overbought] = -1  # Sell signal
        
        return signals

class MACDStrategy(BaseStrategy):
    def __init__(self, 
                 fast_period: int = 12,
                 slow_period: int = 26,
                 signal_period: int = 9):
        super().__init__('MACD')
        self.fast_period = fast_period
        self.slow_period = slow_period
        self.signal_period = signal_period
        
    def generate_signals(self, data: pd.DataFrame) -> pd.Series:
        macd, signal, _ = ti.macd(data['close'].values,
                                    fastperiod=self.fast_period,
                                    slowperiod=self.slow_period,
                                    signalperiod=self.signal_period)
        
        signals = pd.Series(0, index=data.index)
        signals[macd > signal] = 1  # Buy signal
        signals[macd < signal] = -1  # Sell signal
        
        return signals

class BollingerBandsStrategy(BaseStrategy):
    def __init__(self, 
                 period: int = 20,
                 num_std: float = 2.0):
        super().__init__('BollingerBands')
        self.period = period
        self.num_std = num_std
        
    def generate_signals(self, data: pd.DataFrame) -> pd.Series:
        upper, middle, lower = ti.bbands(data['close'].values,
                                          timeperiod=self.period,
                                          nbdevup=self.num_std,
                                          nbdevdn=self.num_std)
        
        signals = pd.Series(0, index=data.index)
        signals[data['close'] < lower] = 1  # Buy signal
        signals[data['close'] > upper] = -1  # Sell signal
        
        return signals