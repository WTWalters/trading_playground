# src/market_analysis/trend.py

from typing import Dict, Optional, Union
import pandas as pd
import numpy as np
import talib
from .base import MarketAnalyzer, MarketRegime, AnalysisConfig

class TrendAnalyzer(MarketAnalyzer):
   """Analyzes market trends using multiple indicators"""

   async def analyze(
       self,
       data: pd.DataFrame,
       additional_metrics: Optional[Dict] = None
   ) -> Dict[str, Union[float, str, pd.Series]]:
       """Calculate trend metrics"""
       if not self._validate_input(data):
           return {}

       try:
           # Calculate EMAs for trend direction
           ema_short = pd.Series(
               talib.EMA(data['close'].values, timeperiod=20),
               index=data.index
           )
           ema_long = pd.Series(
               talib.EMA(data['close'].values, timeperiod=50),
               index=data.index
           )

           # ADX for trend strength
           adx = pd.Series(
               talib.ADX(
                   data['high'].values,
                   data['low'].values,
                   data['close'].values,
                   timeperiod=14
               ),
               index=data.index
           )

           # Calculate slopes for trend momentum
           slope_window = self.config.volatility_window
           price_slope = (
               data['close'].diff(slope_window) /
               data['close'].shift(slope_window)
           )

           # Determine current trend regime
           current_slope = price_slope.iloc[-1]
           current_adx = adx.iloc[-1]

           if current_adx > 25:  # Strong trend
               if current_slope > self.config.trend_strength_threshold:
                   regime = MarketRegime.TRENDING_UP
               elif current_slope < -self.config.trend_strength_threshold:
                   regime = MarketRegime.TRENDING_DOWN
               else:
                   regime = MarketRegime.RANGING
           else:  # Weak trend
               regime = MarketRegime.RANGING

           # Get volatility metrics if available
           if additional_metrics and 'volatility_analysis' in additional_metrics:
               if additional_metrics['volatility_analysis']['metrics'].volatility_regime == 'high_volatility':
                   regime = MarketRegime.VOLATILE

           return {
               'regime': regime,
               'adx': adx,
               'price_slope': price_slope,
               'ema_short': ema_short,
               'ema_long': ema_long,
               'trend_strength': current_adx,
               'trend_momentum': current_slope
           }

       except Exception as e:
           self.logger.error(f"Trend analysis failed: {str(e)}")
           return {}</parameter>
