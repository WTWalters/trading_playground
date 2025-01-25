# src/market_analysis/volatility.py

from typing import Dict, Optional, Union
import pandas as pd
import numpy as np
import talib
from .base import MarketAnalyzer, VolatilityMetrics, AnalysisConfig

class VolatilityAnalyzer(MarketAnalyzer):
   """Analyzes market volatility using multiple metrics"""

   async def analyze(
       self,
       data: pd.DataFrame,
       additional_metrics: Optional[Dict] = None
   ) -> Dict[str, Union[float, str, pd.Series]]:
       """Calculate volatility metrics"""
       if not self._validate_input(data):
           return {}

       try:
           # Historical volatility (close-to-close)
           returns = np.log(data['close'] / data['close'].shift(1))
           hist_vol = returns.rolling(self.config.volatility_window).std() * np.sqrt(252)

           # ATR and Normalized ATR
           atr = pd.Series(talib.ATR(
               data['high'].values,
               data['low'].values,
               data['close'].values,
               timeperiod=self.config.volatility_window
           ))
           norm_atr = atr / data['close']

           # Z-score of current volatility
           current_vol = hist_vol.iloc[-1]
           vol_mean = hist_vol.mean()
           vol_std = hist_vol.std()
           zscore = (current_vol - vol_mean) / vol_std

           # Determine volatility regime
           if zscore > self.config.outlier_std_threshold:
               regime = "high_volatility"
           elif zscore < -self.config.outlier_std_threshold:
               regime = "low_volatility"
           else:
               regime = "normal_volatility"

           metrics = VolatilityMetrics(
               historical_volatility=current_vol,
               normalized_atr=norm_atr.iloc[-1],
               volatility_regime=regime,
               zscore=zscore
           )

           return {
               'metrics': metrics,
               'historical_volatility': hist_vol,
               'normalized_atr': norm_atr,
               'volatility_zscore': zscore
           }

       except Exception as e:
           self.logger.error(f"Volatility analysis failed: {str(e)}")
           return {}</parameter>
