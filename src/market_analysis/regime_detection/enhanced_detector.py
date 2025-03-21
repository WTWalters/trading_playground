"""
Enhanced Market Regime Detection Module

This module extends the base RegimeDetector with additional capabilities for:
1. Incorporating VIX and other macro market indicators
2. Implementing timeframe-based regime detection (intraday vs multi-day)
3. Adding "Jones' macro lens" to better identify market turning points
4. Providing more granular transition probability analysis

It integrates with the existing RegimeDetector to provide a more robust market regime
classification system.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Any, Union
from dataclasses import dataclass, field

from src.market_analysis.regime_detection.detector import RegimeDetector, RegimeType, RegimeDetectionResult

# Additional regime types for macro conditions
from enum import Enum

class MacroRegimeType(Enum):
    """Macro condition regime types."""
    # Default value
    UNDEFINED = "undefined"
    
    # Market Sentiment Regimes
    RISK_SEEKING = "risk_seeking"
    RISK_NEUTRAL = "risk_neutral"
    RISK_AVERSE = "risk_averse"
    
    # Economic Cycle Regimes
    EXPANSION = "expansion"
    PEAK = "peak"
    CONTRACTION = "contraction"
    TROUGH = "trough"
    
    # Monetary Policy Regimes
    ACCOMMODATIVE = "accommodative"
    NEUTRAL = "neutral"
    RESTRICTIVE = "restrictive"


@dataclass
class EnhancedRegimeResult(RegimeDetectionResult):
    """Extended regime detection results with macro indicators."""
    macro_regime: MacroRegimeType = MacroRegimeType.UNDEFINED
    interest_rate_regime: MacroRegimeType = MacroRegimeType.UNDEFINED
    sentiment_regime: MacroRegimeType = MacroRegimeType.UNDEFINED
    regime_turning_point: bool = False
    turning_point_confidence: float = 0.0
    transition_signals: Dict[str, float] = field(default_factory=dict)
    timeframe_regimes: Dict[str, RegimeType] = field(default_factory=dict)


class EnhancedRegimeDetector(RegimeDetector):
    """
    Enhanced regime detector that incorporates macro market indicators and multi-timeframe analysis.
    
    This class extends the base RegimeDetector to provide:
    1. Integration with VIX and other macro indicators
    2. Timeframe-specific regime detection
    3. Market turning point identification (Jones' macro lens)
    4. Enhanced transition probability analysis
    """
    
    def __init__(self, 
                 lookback_window: int = 60,
                 volatility_threshold: float = 1.5,
                 correlation_threshold: float = 0.6,
                 stability_window: int = 30,
                 transition_window: int = 10,
                 macro_indicators: Optional[List[str]] = None,
                 vix_threshold: float = 20.0,
                 timeframes: Optional[List[str]] = None):
        """
        Initialize the enhanced regime detector.
        
        Args:
            lookback_window: Number of periods to analyze for regime detection
            volatility_threshold: Threshold for high/low volatility classification
            correlation_threshold: Threshold for high/low correlation classification
            stability_window: Window size for regime stability calculation
            transition_window: Window size for regime transition analysis
            macro_indicators: List of macro indicator column names to use
            vix_threshold: Threshold for high/low VIX classification
            timeframes: List of timeframes to analyze (e.g., ["intraday", "daily", "weekly"])
        """
        super().__init__(
            lookback_window=lookback_window,
            volatility_threshold=volatility_threshold,
            correlation_threshold=correlation_threshold,
            stability_window=stability_window,
            transition_window=transition_window
        )
        
        # Initialize macro indicator settings
        self.macro_indicators = macro_indicators or ["VIX", "USD_index", "yield_curve", "SPX"]
        self.vix_threshold = vix_threshold
        
        # Initialize timeframe settings
        self.timeframes = timeframes or ["intraday", "daily", "weekly"]
        
        # Track historical macro data
        self.macro_history = pd.DataFrame()
        
        # Initialize turning point detection
        self.turning_point_window = 20
        self.turning_point_threshold = 0.7
    
    def detect_regime(self, 
                      market_data: pd.DataFrame,
                      macro_data: Optional[pd.DataFrame] = None,
                      reference_data: Optional[pd.DataFrame] = None) -> EnhancedRegimeResult:
        """
        Detect the current market regime with enhanced macro analysis.
        
        Args:
            market_data: DataFrame with market prices, volumes, and other metrics
            macro_data: DataFrame with macro indicators (VIX, yield curve, etc.)
            reference_data: Optional historical data for comparative analysis
            
        Returns:
            EnhancedRegimeResult containing detailed regime classification
        """
        # First, get base regime detection result
        base_result = super().detect_regime(market_data, reference_data)
        
        # Process macro data if provided
        macro_regime = MacroRegimeType.UNDEFINED
        interest_rate_regime = MacroRegimeType.UNDEFINED
        sentiment_regime = MacroRegimeType.UNDEFINED
        timeframe_regimes = {}
        transition_signals = {}
        
        if macro_data is not None:
            # Update macro data history
            self._update_macro_history(macro_data)
            
            # Analyze macro conditions
            macro_regime = self._analyze_macro_conditions(macro_data)
            interest_rate_regime = self._analyze_interest_rates(macro_data)
            sentiment_regime = self._analyze_market_sentiment(macro_data)
            
            # Calculate transition signals
            transition_signals = self._calculate_transition_signals(macro_data)
        
        # Analyze different timeframes
        for timeframe in self.timeframes:
            timeframe_data = self._get_timeframe_data(market_data, timeframe)
            if timeframe_data is not None and len(timeframe_data) > self.lookback_window:
                # Detect regime for this timeframe
                timeframe_result = super().detect_regime(timeframe_data)
                timeframe_regimes[timeframe] = timeframe_result.primary_regime
        
        # Detect potential turning points
        regime_turning_point, turning_point_confidence = self._detect_turning_point(
            market_data, macro_data
        )
        
        # Create enhanced result
        enhanced_result = EnhancedRegimeResult(
            primary_regime=base_result.primary_regime,
            secondary_regime=base_result.secondary_regime,
            confidence=base_result.confidence,
            volatility_regime=base_result.volatility_regime,
            correlation_regime=base_result.correlation_regime,
            liquidity_regime=base_result.liquidity_regime,
            trend_regime=base_result.trend_regime,
            regime_start_date=base_result.regime_start_date,
            stability_score=base_result.stability_score,
            transition_probability=base_result.transition_probability,
            features_contribution=base_result.features_contribution,
            macro_regime=macro_regime,
            interest_rate_regime=interest_rate_regime,
            sentiment_regime=sentiment_regime,
            regime_turning_point=regime_turning_point,
            turning_point_confidence=turning_point_confidence,
            transition_signals=transition_signals,
            timeframe_regimes=timeframe_regimes
        )
        
        return enhanced_result
    
    def _update_macro_history(self, macro_data: pd.DataFrame) -> None:
        """
        Update the history of macro indicators.
        
        Args:
            macro_data: DataFrame with macro indicators
        """
        # If we have existing history, append the new data
        if not self.macro_history.empty:
            # Avoid duplicate indices
            new_data = macro_data[~macro_data.index.isin(self.macro_history.index)]
            if not new_data.empty:
                self.macro_history = pd.concat([self.macro_history, new_data])
        else:
            # Initialize history
            self.macro_history = macro_data.copy()
    
    def _analyze_macro_conditions(self, macro_data: pd.DataFrame) -> MacroRegimeType:
        """
        Analyze macro economic conditions to determine the macro regime.
        
        Args:
            macro_data: DataFrame with macro indicators
            
        Returns:
            MacroRegimeType indicating the current macro regime
        """
        # Check if required indicators are available
        if not all(indicator in macro_data.columns for indicator in ["SPX", "yield_curve"]):
            return MacroRegimeType.UNDEFINED
        
        # Analyze economic cycle using SPX and yield curve
        spx_trend = self._calculate_trend(macro_data["SPX"])
        yield_curve = macro_data["yield_curve"].iloc[-1]
        
        # Simplified economic cycle classification
        if spx_trend > 0.5 and yield_curve > 0:
            return MacroRegimeType.EXPANSION
        elif spx_trend > 0 and yield_curve < 0:
            return MacroRegimeType.PEAK
        elif spx_trend < -0.5 and yield_curve < 0:
            return MacroRegimeType.CONTRACTION
        elif spx_trend < 0 and yield_curve > 0:
            return MacroRegimeType.TROUGH
        else:
            return MacroRegimeType.UNDEFINED
    
    def _analyze_interest_rates(self, macro_data: pd.DataFrame) -> MacroRegimeType:
        """
        Analyze interest rate environment.
        
        Args:
            macro_data: DataFrame with macro indicators
            
        Returns:
            MacroRegimeType indicating the interest rate regime
        """
        if "interest_rate" not in macro_data.columns:
            return MacroRegimeType.UNDEFINED
        
        # Get current interest rate and trend
        current_rate = macro_data["interest_rate"].iloc[-1]
        rate_trend = self._calculate_trend(macro_data["interest_rate"])
        
        # Classify based on level and direction
        if current_rate < 2.0:
            return MacroRegimeType.ACCOMMODATIVE
        elif current_rate > 4.0:
            return MacroRegimeType.RESTRICTIVE
        else:
            if rate_trend > 0.3:
                return MacroRegimeType.RESTRICTIVE
            elif rate_trend < -0.3:
                return MacroRegimeType.ACCOMMODATIVE
            else:
                return MacroRegimeType.NEUTRAL
    
    def _analyze_market_sentiment(self, macro_data: pd.DataFrame) -> MacroRegimeType:
        """
        Analyze market sentiment using VIX and other indicators.
        
        Args:
            macro_data: DataFrame with macro indicators
            
        Returns:
            MacroRegimeType indicating the sentiment regime
        """
        if "VIX" not in macro_data.columns:
            return MacroRegimeType.UNDEFINED
        
        # Get VIX value and its recent trend
        vix = macro_data["VIX"].iloc[-1]
        vix_ma = macro_data["VIX"].rolling(10).mean().iloc[-1]
        
        # Check if VIX is significantly higher than threshold
        if vix > self.vix_threshold * 1.5:
            return MacroRegimeType.RISK_AVERSE
        elif vix > self.vix_threshold:
            if vix > vix_ma * 1.1:  # Rising VIX
                return MacroRegimeType.RISK_AVERSE
            else:
                return MacroRegimeType.RISK_NEUTRAL
        else:
            return MacroRegimeType.RISK_SEEKING
    
    def _calculate_transition_signals(self, macro_data: pd.DataFrame) -> Dict[str, float]:
        """
        Calculate signals that may indicate imminent regime transitions.
        
        Args:
            macro_data: DataFrame with macro indicators
            
        Returns:
            Dictionary of signal names to signal strengths
        """
        signals = {}
        
        # Calculate VIX transition signal if available
        if "VIX" in macro_data.columns and len(macro_data) > 20:
            vix = macro_data["VIX"]
            vix_percentile = (vix.iloc[-1] - vix.min()) / (vix.max() - vix.min())
            vix_zscore = (vix.iloc[-1] - vix.mean()) / vix.std() if vix.std() > 0 else 0
            vix_rate_of_change = vix.pct_change(3).iloc[-1] * 100 if len(vix) > 3 else 0
            
            signals["vix_percentile"] = vix_percentile
            signals["vix_zscore"] = vix_zscore
            signals["vix_rate_of_change"] = vix_rate_of_change
        
        # Calculate yield curve transition signal if available
        if "yield_curve" in macro_data.columns and len(macro_data) > 20:
            yield_curve = macro_data["yield_curve"]
            yield_curve_trend = self._calculate_trend(yield_curve)
            recent_sign_change = (yield_curve.iloc[-1] * yield_curve.iloc[-10]) < 0 if len(yield_curve) > 10 else False
            
            signals["yield_curve_trend"] = yield_curve_trend
            signals["yield_curve_sign_change"] = float(recent_sign_change)
        
        # Calculate SPX trend signals if available
        if "SPX" in macro_data.columns and len(macro_data) > 50:
            spx = macro_data["SPX"]
            spx_ma_20 = spx.rolling(20).mean()
            spx_ma_50 = spx.rolling(50).mean()
            
            if len(spx_ma_20) > 20 and len(spx_ma_50) > 50:
                # Golden cross / death cross detection
                current_relation = spx_ma_20.iloc[-1] > spx_ma_50.iloc[-1]
                previous_relation = spx_ma_20.iloc[-2] > spx_ma_50.iloc[-2]
                
                if current_relation and not previous_relation:
                    signals["golden_cross"] = 1.0
                elif not current_relation and previous_relation:
                    signals["death_cross"] = 1.0
                    
                # Calculate trend strength
                spx_trend = self._calculate_trend(spx)
                signals["spx_trend_strength"] = spx_trend
        
        return signals
    
    def _get_timeframe_data(self, 
                           market_data: pd.DataFrame, 
                           timeframe: str) -> Optional[pd.DataFrame]:
        """
        Resample data to different timeframes.
        
        Args:
            market_data: DataFrame with market data
            timeframe: Target timeframe
            
        Returns:
            Resampled DataFrame or None if not possible
        """
        if timeframe == "intraday":
            # Assuming market_data is already intraday
            return market_data
        
        # Ensure we have a datetime index
        if not isinstance(market_data.index, pd.DatetimeIndex):
            return None
        
        # Resample based on timeframe
        try:
            if timeframe == "daily":
                # Resample to daily
                resampled = market_data.resample('D').agg({
                    'open': 'first',
                    'high': 'max',
                    'low': 'min',
                    'close': 'last',
                    'volume': 'sum'
                }).dropna()
                return resampled
            elif timeframe == "weekly":
                # Resample to weekly
                resampled = market_data.resample('W').agg({
                    'open': 'first',
                    'high': 'max',
                    'low': 'min',
                    'close': 'last',
                    'volume': 'sum'
                }).dropna()
                return resampled
            else:
                return None
        except Exception as e:
            return None
    
    def _detect_turning_point(self, 
                            market_data: pd.DataFrame,
                            macro_data: Optional[pd.DataFrame] = None) -> Tuple[bool, float]:
        """
        Detect potential market regime turning points using Paul Tudor Jones' methodology.
        
        Args:
            market_data: DataFrame with market data
            macro_data: Optional DataFrame with macro indicators
            
        Returns:
            Tuple of (is_turning_point, confidence)
        """
        # Jones often looks for divergences between price and indicators
        turning_point_signals = []
        confidence_scores = []
        
        # Check for price trend exhaustion
        if 'close' in market_data.columns and len(market_data) > self.turning_point_window:
            close_prices = market_data['close']
            
            # Calculate momentum indicators
            rsi = self._calculate_rsi(close_prices, 14)
            
            if len(rsi) > 30:
                # Look for divergence between price and RSI
                price_higher_high = close_prices[-1] > close_prices[-15] and close_prices[-15] > close_prices[-30]
                rsi_lower_high = rsi[-1] < rsi[-15] and close_prices[-1] > close_prices[-15]
                
                if price_higher_high and rsi_lower_high:
                    # Bearish divergence - possible turning point from uptrend to downtrend
                    turning_point_signals.append(True)
                    confidence_scores.append(0.7)
                else:
                    turning_point_signals.append(False)
                    confidence_scores.append(0.3)
        
        # Add macro indicator checks if available
        if macro_data is not None:
            # Check VIX for sentiment shifts
            if 'VIX' in macro_data.columns and len(macro_data) > 10:
                vix = macro_data['VIX']
                vix_spike = vix.iloc[-1] > 1.5 * vix.iloc[-10:-1].mean()
                vix_collapse = vix.iloc[-1] < 0.7 * vix.iloc[-10:-1].mean()
                
                if vix_spike or vix_collapse:
                    turning_point_signals.append(True)
                    confidence_scores.append(0.8 if vix_spike else 0.7)
                else:
                    turning_point_signals.append(False)
                    confidence_scores.append(0.4)
            
            # Check yield curve for economic shifts
            if 'yield_curve' in macro_data.columns and len(macro_data) > 20:
                yield_curve = macro_data['yield_curve']
                # Check for recent inversion or steepening
                yield_crossing_zero = (yield_curve.iloc[-1] * yield_curve.iloc[-10]) < 0
                
                if yield_crossing_zero:
                    turning_point_signals.append(True)
                    confidence_scores.append(0.9)  # Strong signal
                else:
                    turning_point_signals.append(False)
                    confidence_scores.append(0.3)
        
        # Determine overall turning point assessment
        is_turning_point = any(turning_point_signals)
        
        # Calculate confidence based on signals
        if len(confidence_scores) > 0:
            # Weight the positive signals higher
            positive_scores = [score for signal, score in zip(turning_point_signals, confidence_scores) if signal]
            if positive_scores:
                confidence = sum(positive_scores) / len(positive_scores)
            else:
                confidence = 0.0
        else:
            confidence = 0.0
        
        return is_turning_point, confidence
    
    def _calculate_trend(self, series: pd.Series) -> float:
        """
        Calculate the trend strength and direction of a time series.
        
        Args:
            series: Time series data
            
        Returns:
            Float between -1.0 and 1.0 indicating trend strength and direction
        """
        if len(series) < 20:
            return 0.0
        
        # Calculate short and long-term moving averages
        short_ma = series.rolling(5).mean().iloc[-1]
        medium_ma = series.rolling(10).mean().iloc[-1]
        long_ma = series.rolling(20).mean().iloc[-1]
        
        # Calculate direction and strength based on MA relationships
        if short_ma > medium_ma > long_ma:
            # Strong uptrend
            strength = min(1.0, (short_ma / long_ma - 1) * 5)
            return strength
        elif short_ma < medium_ma < long_ma:
            # Strong downtrend
            strength = min(1.0, (1 - short_ma / long_ma) * 5)
            return -strength
        elif short_ma > medium_ma and medium_ma < long_ma:
            # Potential bottoming
            return 0.2
        elif short_ma < medium_ma and medium_ma > long_ma:
            # Potential topping
            return -0.2
        else:
            # No clear trend
            return 0.0
    
    def _calculate_rsi(self, prices: pd.Series, window: int = 14) -> pd.Series:
        """
        Calculate the Relative Strength Index (RSI).
        
        Args:
            prices: Price series data
            window: RSI calculation window
            
        Returns:
            Series containing RSI values
        """
        delta = prices.diff()
        gain = delta.where(delta > 0, 0)
        loss = -delta.where(delta < 0, 0)
        
        avg_gain = gain.rolling(window=window).mean()
        avg_loss = loss.rolling(window=window).mean()
        
        rs = avg_gain / avg_loss
        rsi = 100 - (100 / (1 + rs))
        
        return rsi
    
    def analyze_multiple_timeframes(self, 
                                   market_data: Dict[str, pd.DataFrame],
                                   macro_data: Optional[pd.DataFrame] = None) -> Dict[str, EnhancedRegimeResult]:
        """
        Analyze regimes across multiple timeframes.
        
        Args:
            market_data: Dictionary mapping timeframes to DataFrames
            macro_data: Optional DataFrame with macro indicators
            
        Returns:
            Dictionary mapping timeframes to regime detection results
        """
        results = {}
        
        for timeframe, data in market_data.items():
            results[timeframe] = self.detect_regime(data, macro_data)
            
        return results
    
    def get_regime_alignment(self, timeframe_results: Dict[str, EnhancedRegimeResult]) -> float:
        """
        Calculate how aligned regimes are across timeframes.
        
        Args:
            timeframe_results: Dictionary of regime results by timeframe
            
        Returns:
            Alignment score between 0.0 and 1.0
        """
        if not timeframe_results:
            return 0.0
            
        # Extract primary regimes
        regimes = [result.primary_regime for result in timeframe_results.values()]
        
        # Count occurrences of each regime
        regime_counts = {}
        for regime in regimes:
            if regime in regime_counts:
                regime_counts[regime] += 1
            else:
                regime_counts[regime] = 1
                
        # Calculate alignment score
        if not regime_counts:
            return 0.0
            
        max_count = max(regime_counts.values())
        alignment = max_count / len(regimes)
        
        return alignment
