# src/market_analysis/mean_reversion.py

from typing import Dict, Optional, Union
import pandas as pd
import numpy as np
import talib
from .base import MarketAnalyzer, AnalysisConfig
from dataclasses import dataclass

@dataclass
class MeanReversionMetrics:
   zscore: float
   distance_to_mean: float
   mean_level: float
   is_oversold: bool
   is_overbought: bool
   reversion_probability: float

class MeanReversionAnalyzer(MarketAnalyzer):
   """Analyzes mean reversion tendencies in price action"""

   async def analyze(
       self,
       data: pd.DataFrame,
       additional_metrics: Optional[Dict] = None
   ) -> Dict[str, Union[float, str, pd.Series]]:
       if not self._validate_input(data):
           return {}

       try:
           # Calculate rolling mean and standard deviation
           rolling_mean = data['close'].rolling(
               window=self.config.volatility_window
           ).mean()
           rolling_std = data['close'].rolling(
               window=self.config.volatility_window
           ).std()

           # Calculate z-score
           zscore = (data['close'] - rolling_mean) / rolling_std

           # RSI for overbought/oversold
           rsi = pd.Series(
               talib.RSI(data['close'].values),
               index=data.index
           )

           # Calculate mean reversion probability
           reversion_prob = self._calculate_reversion_probability(
               zscore.iloc[-1],
               rsi.iloc[-1]
           )

           metrics = MeanReversionMetrics(
               zscore=zscore.iloc[-1],
               distance_to_mean=data['close'].iloc[-1] - rolling_mean.iloc[-1],
               mean_level=rolling_mean.iloc[-1],
               is_oversold=rsi.iloc[-1] < 30,
               is_overbought=rsi.iloc[-1] > 70,
               reversion_probability=reversion_prob
           )

           return {
               'metrics': metrics,
               'zscore_series': zscore,
               'mean_series': rolling_mean,
               'rsi_series': rsi
           }

       except Exception as e:
           self.logger.error(f"Mean reversion analysis failed: {str(e)}")
           return {}

   def _calculate_reversion_probability(
       self,
       zscore: float,
       rsi: float
   ) -> float:
       """Calculate probability of mean reversion based on z-score and RSI"""
       # Base probability on z-score (higher abs value = higher prob)
       zscore_prob = min(abs(zscore) / 3.0, 1.0)

       # Adjust based on RSI extremes
       rsi_prob = 0.0
       if rsi < 30:
           rsi_prob = (30 - rsi) / 30
       elif rsi > 70:
           rsi_prob = (rsi - 70) / 30

       # Combine probabilities
       return min(0.95, (zscore_prob + rsi_prob) / 2)
