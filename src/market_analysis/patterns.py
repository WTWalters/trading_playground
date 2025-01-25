# src/market_analysis/patterns.py

from typing import Dict, Optional, Union, List
import pandas as pd
import numpy as np
import talib
from .base import MarketAnalyzer, AnalysisConfig

class PatternAnalyzer(MarketAnalyzer):
   """Detects candlestick and chart patterns"""

   def __init__(self, config: AnalysisConfig):
       super().__init__(config)
       self.pattern_functions = {
           'DOJI': talib.CDLDOJI,
           'ENGULFING': talib.CDLENGULFING,
           'HAMMER': talib.CDLHAMMER,
           'SHOOTING_STAR': talib.CDLSHOOTINGSTAR,
           'MORNING_STAR': talib.CDLMORNINGSTAR,
           'EVENING_STAR': talib.CDLEVENINGSTAR,
           'THREE_WHITE_SOLDIERS': talib.CDL3WHITESOLDIERS,
           'THREE_BLACK_CROWS': talib.CDL3BLACKCROWS
       }

   async def analyze(
       self,
       data: pd.DataFrame,
       additional_metrics: Optional[Dict] = None
   ) -> Dict[str, Union[float, str, pd.Series, List[Dict]]]:
       """Detect chart patterns"""
       if not self._validate_input(data):
           return {}

       try:
           patterns = {}
           recent_patterns = []
           lookback = 5  # Check last 5 bars for recent patterns

           for name, func in self.pattern_functions.items():
               pattern = pd.Series(
                   func(
                       data['open'].values,
                       data['high'].values,
                       data['low'].values,
                       data['close'].values
                   ),
                   index=data.index
               )

               patterns[name] = pattern

               # Check for recent pattern occurrences
               recent_signals = pattern.tail(lookback)
               if (recent_signals != 0).any():
                   for idx, value in recent_signals.items():
                       if value != 0:
                           recent_patterns.append({
                               'pattern': name,
                               'date': idx,
                               'signal': 'bullish' if value > 0 else 'bearish'
                           })

           # Calculate success rate if we have trend data
           success_rates = {}
           if additional_metrics and 'trend_analysis' in additional_metrics:
               for name, pattern in patterns.items():
                   success_rates[name] = self._calculate_pattern_success(
                       pattern,
                       data['close'],
                       lookforward=10  # 10-bar forward returns
                   )

           return {
               'patterns': patterns,
               'recent_patterns': recent_patterns,
               'success_rates': success_rates
           }

       except Exception as e:
           self.logger.error(f"Pattern detection failed: {str(e)}")
           return {}

   def _calculate_pattern_success(
       self,
       pattern: pd.Series,
       prices: pd.Series,
       lookforward: int = 10
   ) -> Dict[str, float]:
       """Calculate pattern success rates"""
       try:
           bullish_signals = pattern > 0
           bearish_signals = pattern < 0

           # Calculate forward returns
           forward_returns = prices.shift(-lookforward) / prices - 1

           # Calculate success rates
           bullish_success = (
               (forward_returns[bullish_signals] > 0).sum() /
               bullish_signals.sum() if bullish_signals.any() else 0
           )
           bearish_success = (
               (forward_returns[bearish_signals] < 0).sum() /
               bearish_signals.sum() if bearish_signals.any() else 0
           )

           return {
               'bullish_rate': bullish_success,
               'bearish_rate': bearish_success,
               'total_signals': (bullish_signals | bearish_signals).sum()
           }

       except Exception as e:
           self.logger.error(f"Success rate calculation failed: {str(e)}")
           return {
               'bullish_rate': 0,
               'bearish_rate': 0,
               'total_signals': 0
           }
