"""
Market Regime Detection Module

This module provides functionality to detect market regimes based on various market indicators.
It analyzes volatility, correlation, liquidity, and trend patterns to classify current market conditions
and predict regime transitions.
"""

import enum
import numpy as np
import pandas as pd
from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional, Union
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from statsmodels.tsa.stattools import adfuller

class RegimeType(enum.Enum):
    """Enumeration of different market regime types."""
    HIGH_VOLATILITY = "high_volatility"
    LOW_VOLATILITY = "low_volatility"
    MEAN_REVERTING = "mean_reverting"
    TRENDING = "trending"
    RISK_ON = "risk_on"
    RISK_OFF = "risk_off"
    HIGH_LIQUIDITY = "high_liquidity"
    LOW_LIQUIDITY = "low_liquidity"
    UNDEFINED = "undefined"


@dataclass
class RegimeDetectionResult:
    """Container for regime detection results."""
    primary_regime: RegimeType
    secondary_regime: Optional[RegimeType]
    confidence: float
    volatility_regime: RegimeType
    correlation_regime: RegimeType
    liquidity_regime: RegimeType
    trend_regime: RegimeType
    regime_start_date: pd.Timestamp
    stability_score: float
    transition_probability: Dict[RegimeType, float]
    features_contribution: Dict[str, float]


class RegimeDetector:
    """
    Detects market regimes based on statistical analysis of market data.
    
    This class analyzes multiple dimensions of market behavior including
    volatility, correlation structure, liquidity conditions, and trend
    characteristics to classify the current market regime.
    """
    
    def __init__(self, 
                 lookback_window: int = 60, 
                 volatility_threshold: float = 1.5,
                 correlation_threshold: float = 0.6,
                 stability_window: int = 30,
                 transition_window: int = 10):
        """
        Initialize the regime detector with configuration parameters.
        
        Args:
            lookback_window: Number of periods to analyze for regime detection
            volatility_threshold: Threshold for high/low volatility classification
            correlation_threshold: Threshold for high/low correlation classification
            stability_window: Window size for regime stability calculation
            transition_window: Window size for regime transition analysis
        """
        self.lookback_window = lookback_window
        self.volatility_threshold = volatility_threshold
        self.correlation_threshold = correlation_threshold
        self.stability_window = stability_window
        self.transition_window = transition_window
        self.regime_history = []
        self.scaler = StandardScaler()
        self.classifier = RandomForestClassifier(n_estimators=100, random_state=42)
        self.fitted = False
        
    def detect_regime(self, 
                      market_data: pd.DataFrame, 
                      reference_data: Optional[pd.DataFrame] = None) -> RegimeDetectionResult:
        """
        Detect the current market regime based on provided market data.
        
        Args:
            market_data: DataFrame with market prices, volumes, and other metrics
            reference_data: Optional historical data for comparative analysis
            
        Returns:
            RegimeDetectionResult containing the detected regime and associated metrics
        """
        # Extract features from market data
        features = self._extract_features(market_data)
        
        # Analyze different regime dimensions
        volatility_regime = self._analyze_volatility(market_data)
        correlation_regime = self._analyze_correlation(market_data)
        liquidity_regime = self._analyze_liquidity(market_data)
        trend_regime = self._analyze_trend(market_data)
        
        # Combine dimensional analyses into overall regime classification
        primary_regime, confidence = self._classify_regime(features, volatility_regime, 
                                                          correlation_regime, liquidity_regime, 
                                                          trend_regime)
        
        # Identify secondary regime if present
        secondary_regime = self._identify_secondary_regime(features, primary_regime)
        
        # Calculate regime stability
        stability_score = self._calculate_stability(market_data)
        
        # Calculate transition probabilities
        transition_probs = self._calculate_transition_probabilities(market_data)
        
        # Determine regime start date
        regime_start = self._determine_regime_start_date(market_data, primary_regime)
        
        # Calculate feature contributions to regime classification
        feature_contributions = self._calculate_feature_contributions(features)
        
        # Create and return result
        result = RegimeDetectionResult(
            primary_regime=primary_regime,
            secondary_regime=secondary_regime,
            confidence=confidence,
            volatility_regime=volatility_regime,
            correlation_regime=correlation_regime,
            liquidity_regime=liquidity_regime,
            trend_regime=trend_regime,
            regime_start_date=regime_start,
            stability_score=stability_score,
            transition_probability=transition_probs,
            features_contribution=feature_contributions
        )
        
        # Update regime history
        self.regime_history.append(result)
        
        return result
    
    def _extract_features(self, market_data: pd.DataFrame) -> pd.DataFrame:
        """
        Extract relevant features from market data for regime detection.
        
        Args:
            market_data: DataFrame with market data
            
        Returns:
            DataFrame with extracted features
        """
        features = pd.DataFrame()
        
        # Ensure we have enough data
        if len(market_data) < self.lookback_window:
            raise ValueError(f"Market data must contain at least {self.lookback_window} periods")
        
        # Use the most recent data based on lookback window
        recent_data = market_data.iloc[-self.lookback_window:]
        
        # Calculate volatility features
        features['realized_volatility'] = recent_data['close'].pct_change().rolling(20).std() * np.sqrt(252)
        features['high_low_range'] = (recent_data['high'] - recent_data['low']) / recent_data['close']
        features['volatility_of_volatility'] = features['realized_volatility'].rolling(10).std()
        
        # Calculate correlation features
        if 'volume' in recent_data.columns:
            features['price_volume_corr'] = recent_data['close'].rolling(20).corr(recent_data['volume'])
        
        # Calculate liquidity features
        if 'volume' in recent_data.columns:
            features['volume_ma_ratio'] = recent_data['volume'] / recent_data['volume'].rolling(20).mean()
            features['volume_trend'] = recent_data['volume'].rolling(5).mean() / recent_data['volume'].rolling(20).mean()
        
        # Calculate trend features
        features['ma_5_20_ratio'] = recent_data['close'].rolling(5).mean() / recent_data['close'].rolling(20).mean()
        features['ma_20_50_ratio'] = recent_data['close'].rolling(20).mean() / recent_data['close'].rolling(50).mean()
        
        # Calculate mean reversion indicators
        recent_close = recent_data['close'].values
        features['hurst_exponent'] = self._calculate_hurst_exponent(recent_close)
        
        # Drop NaN values resulting from rolling windows
        features = features.dropna()
        
        return features
    
    def _analyze_volatility(self, market_data: pd.DataFrame) -> RegimeType:
        """
        Analyze the volatility regime of the market.
        
        Args:
            market_data: DataFrame with market data
            
        Returns:
            RegimeType indicating the volatility regime
        """
        # Calculate realized volatility
        recent_data = market_data.iloc[-self.lookback_window:]
        volatility = recent_data['close'].pct_change().rolling(20).std() * np.sqrt(252)
        
        # Compare current volatility with historical levels
        current_vol = volatility.iloc[-1]
        historical_vol = volatility.mean()
        
        if current_vol > historical_vol * self.volatility_threshold:
            return RegimeType.HIGH_VOLATILITY
        else:
            return RegimeType.LOW_VOLATILITY
    
    def _analyze_correlation(self, market_data: pd.DataFrame) -> RegimeType:
        """
        Analyze the correlation regime of the market.
        
        Args:
            market_data: DataFrame with market data
            
        Returns:
            RegimeType indicating the correlation regime
        """
        # For simplicity, we'll use price-volume correlation as a proxy
        # In a real implementation, this would include cross-asset correlations
        if 'volume' not in market_data.columns:
            return RegimeType.UNDEFINED
        
        recent_data = market_data.iloc[-self.lookback_window:]
        correlation = recent_data['close'].rolling(20).corr(recent_data['volume']).iloc[-1]
        
        if correlation > self.correlation_threshold:
            return RegimeType.RISK_ON
        else:
            return RegimeType.RISK_OFF
    
    def _analyze_liquidity(self, market_data: pd.DataFrame) -> RegimeType:
        """
        Analyze the liquidity regime of the market.
        
        Args:
            market_data: DataFrame with market data
            
        Returns:
            RegimeType indicating the liquidity regime
        """
        if 'volume' not in market_data.columns:
            return RegimeType.UNDEFINED
        
        recent_data = market_data.iloc[-self.lookback_window:]
        volume_ma = recent_data['volume'].rolling(20).mean()
        current_volume_ratio = recent_data['volume'].iloc[-1] / volume_ma.iloc[-1]
        
        if current_volume_ratio > 1.2:  # 20% above average
            return RegimeType.HIGH_LIQUIDITY
        elif current_volume_ratio < 0.8:  # 20% below average
            return RegimeType.LOW_LIQUIDITY
        else:
            return RegimeType.UNDEFINED
    
    def _analyze_trend(self, market_data: pd.DataFrame) -> RegimeType:
        """
        Analyze the trend regime of the market.
        
        Args:
            market_data: DataFrame with market data
            
        Returns:
            RegimeType indicating the trend regime
        """
        recent_data = market_data.iloc[-self.lookback_window:]
        
        # Calculate moving averages
        ma_5 = recent_data['close'].rolling(5).mean()
        ma_20 = recent_data['close'].rolling(20).mean()
        
        # Calculate Hurst exponent to distinguish between trending and mean-reverting
        price_series = recent_data['close'].values
        hurst = self._calculate_hurst_exponent(price_series)
        
        if hurst > 0.6:  # Trending
            # Check direction
            if ma_5.iloc[-1] > ma_20.iloc[-1]:
                return RegimeType.TRENDING
            else:
                return RegimeType.TRENDING
        elif hurst < 0.4:  # Mean-reverting
            return RegimeType.MEAN_REVERTING
        else:
            # Random walk, indeterminate
            return RegimeType.UNDEFINED
    
    def _classify_regime(self, 
                        features: pd.DataFrame, 
                        volatility_regime: RegimeType,
                        correlation_regime: RegimeType,
                        liquidity_regime: RegimeType,
                        trend_regime: RegimeType) -> Tuple[RegimeType, float]:
        """
        Combine individual regime analyses into an overall classification.
        
        Args:
            features: DataFrame with extracted features
            volatility_regime: Volatility regime classification
            correlation_regime: Correlation regime classification
            liquidity_regime: Liquidity regime classification
            trend_regime: Trend regime classification
            
        Returns:
            Tuple of (primary regime, confidence score)
        """
        # For initial implementation, use a rule-based approach
        # Later, this could be replaced with a machine learning model
        
        # Start with the trend regime as the base
        if trend_regime != RegimeType.UNDEFINED:
            primary_regime = trend_regime
            confidence = 0.7
        elif volatility_regime == RegimeType.HIGH_VOLATILITY:
            primary_regime = RegimeType.HIGH_VOLATILITY
            confidence = 0.8
        elif correlation_regime != RegimeType.UNDEFINED:
            primary_regime = correlation_regime
            confidence = 0.6
        elif liquidity_regime != RegimeType.UNDEFINED:
            primary_regime = liquidity_regime
            confidence = 0.5
        else:
            primary_regime = RegimeType.UNDEFINED
            confidence = 0.3
            
        return primary_regime, confidence
    
    def _identify_secondary_regime(self, 
                                 features: pd.DataFrame, 
                                 primary_regime: RegimeType) -> Optional[RegimeType]:
        """
        Identify a secondary market regime if applicable.
        
        Args:
            features: DataFrame with extracted features
            primary_regime: The primary regime classification
            
        Returns:
            Optional secondary regime
        """
        # Simple heuristic - if volatility is high but trend is the primary, 
        # report volatility as secondary
        if primary_regime == RegimeType.TRENDING and features['realized_volatility'].iloc[-1] > 0.2:
            return RegimeType.HIGH_VOLATILITY
            
        # If mean-reverting but liquidity is low, report liquidity as secondary
        if primary_regime == RegimeType.MEAN_REVERTING and 'volume_ma_ratio' in features.columns:
            if features['volume_ma_ratio'].iloc[-1] < 0.8:
                return RegimeType.LOW_LIQUIDITY
                
        return None
    
    def _calculate_stability(self, market_data: pd.DataFrame) -> float:
        """
        Calculate the stability of the current regime.
        
        Args:
            market_data: DataFrame with market data
            
        Returns:
            Stability score between 0 and 1
        """
        # If we don't have enough history, return moderate stability
        if len(self.regime_history) < 2:
            return 0.5
            
        # Check consistency of recent regime classifications
        recent_regimes = [result.primary_regime for result in self.regime_history[-self.stability_window:]]
        if len(recent_regimes) < self.stability_window:
            # Not enough history, use what we have
            regime_counts = pd.Series(recent_regimes).value_counts()
            most_common = regime_counts.idxmax()
            stability = regime_counts[most_common] / len(recent_regimes)
        else:
            regime_counts = pd.Series(recent_regimes).value_counts()
            most_common = regime_counts.idxmax()
            stability = regime_counts[most_common] / self.stability_window
            
        return stability
    
    def _calculate_transition_probabilities(self, market_data: pd.DataFrame) -> Dict[RegimeType, float]:
        """
        Calculate the probability of transitioning to different regimes.
        
        Args:
            market_data: DataFrame with market data
            
        Returns:
            Dictionary mapping regime types to transition probabilities
        """
        # Initialize default probabilities
        transition_probs = {regime: 0.0 for regime in RegimeType}
        
        # If we don't have enough history, return uniform distribution
        if len(self.regime_history) < self.transition_window:
            for regime in RegimeType:
                transition_probs[regime] = 1.0 / len(RegimeType)
            return transition_probs
            
        # Analyze recent feature trends to predict transitions
        recent_features = self._extract_features(market_data)
        current_regime = self.regime_history[-1].primary_regime
        
        # Volatility trend
        vol_trend = recent_features['realized_volatility'].diff().mean()
        if vol_trend > 0:
            # Increasing volatility
            transition_probs[RegimeType.HIGH_VOLATILITY] = 0.3
        else:
            # Decreasing volatility
            transition_probs[RegimeType.LOW_VOLATILITY] = 0.3
            
        # Trend-following vs mean-reversion
        hurst = recent_features['hurst_exponent'].iloc[-1]
        if hurst > 0.6:
            transition_probs[RegimeType.TRENDING] = 0.3
        elif hurst < 0.4:
            transition_probs[RegimeType.MEAN_REVERTING] = 0.3
            
        # Ensure current regime has some probability
        transition_probs[current_regime] = max(0.4, transition_probs[current_regime])
        
        # Normalize probabilities
        total_prob = sum(transition_probs.values())
        if total_prob > 0:
            for regime in transition_probs:
                transition_probs[regime] /= total_prob
                
        return transition_probs
    
    def _determine_regime_start_date(self, market_data: pd.DataFrame, current_regime: RegimeType) -> pd.Timestamp:
        """
        Determine when the current regime started.
        
        Args:
            market_data: DataFrame with market data
            current_regime: The current regime classification
            
        Returns:
            Timestamp of regime start
        """
        # If no history, return most recent date
        if len(self.regime_history) < 2:
            return market_data.index[-1]
            
        # Look back through history to find regime change
        for i in range(len(self.regime_history) - 2, -1, -1):
            if self.regime_history[i].primary_regime != current_regime:
                # Regime changed after this point
                return market_data.index[-(len(self.regime_history) - i)]
                
        # No change found, return earliest date in window
        return market_data.index[-min(len(self.regime_history), len(market_data.index))]
    
    def _calculate_feature_contributions(self, features: pd.DataFrame) -> Dict[str, float]:
        """
        Calculate the contribution of each feature to the regime classification.
        
        Args:
            features: DataFrame with extracted features
            
        Returns:
            Dictionary mapping feature names to contribution scores
        """
        # Simple implementation - normalize feature values
        feature_values = features.iloc[-1].to_dict()
        feature_contributions = {}
        
        # Normalize values between 0 and 1 for each feature
        for feature, value in feature_values.items():
            if feature in features.columns:
                feature_min = features[feature].min()
                feature_max = features[feature].max()
                if feature_max > feature_min:
                    normalized = (value - feature_min) / (feature_max - feature_min)
                    feature_contributions[feature] = normalized
                else:
                    feature_contributions[feature] = 0.5
                    
        return feature_contributions
    
    def _calculate_hurst_exponent(self, price_series: np.ndarray, lags: Optional[List[int]] = None) -> float:
        """
        Calculate the Hurst exponent to determine if a time series is:
        - Trending (H > 0.5)
        - Random walk (H = 0.5)
        - Mean-reverting (H < 0.5)
        
        Args:
            price_series: NumPy array of prices
            lags: Optional list of lag values to use
            
        Returns:
            Hurst exponent value
        """
        if lags is None:
            lags = [2, 4, 8, 16, 32]
            
        # Convert to returns
        returns = np.log(price_series[1:] / price_series[:-1])
        
        # Calculate statistics for each lag
        tau = []
        lagvec = []
        
        # Step through different lags
        for lag in lags:
            # Skip if lag is too large for data
            if lag >= len(returns):
                continue
                
            # Calculate standard deviation of differences
            pp = np.subtract(returns[lag:], returns[:-lag])
            lagvec.append(lag)
            tau.append(np.sqrt(np.std(pp)))
            
        # Linear fit to estimate Hurst
        if len(tau) > 1 and len(lagvec) > 1:
            m = np.polyfit(np.log(lagvec), np.log(tau), 1)
            hurst = m[0] / 2.0
        else:
            hurst = 0.5  # Default to random walk
            
        return hurst
    
    def train_classifier(self, historical_data: pd.DataFrame, labels: List[RegimeType]):
        """
        Train the regime classifier on historical data with known labels.
        
        Args:
            historical_data: DataFrame with historical market data
            labels: List of regime labels corresponding to historical data periods
        """
        features_list = []
        
        # Extract features from each period
        for i in range(len(historical_data) - self.lookback_window + 1):
            window_data = historical_data.iloc[i:i+self.lookback_window]
            try:
                features = self._extract_features(window_data)
                if not features.empty:
                    features_list.append(features.iloc[-1].values)
            except Exception as e:
                print(f"Error extracting features: {e}")
                continue
                
        if len(features_list) != len(labels):
            raise ValueError("Number of feature sets must match number of labels")
            
        # Convert features and labels to numpy arrays
        X = np.array(features_list)
        y = np.array([label.value for label in labels])
        
        # Scale features
        X_scaled = self.scaler.fit_transform(X)
        
        # Train classifier
        self.classifier.fit(X_scaled, y)
        self.fitted = True
        
    def detect_regime_shifts(self, market_data: pd.DataFrame) -> List[Tuple[pd.Timestamp, RegimeType]]:
        """
        Detect points where the market regime shifted.
        
        Args:
            market_data: DataFrame with market data
            
        Returns:
            List of (timestamp, new_regime) tuples indicating regime shifts
        """
        shifts = []
        
        # We need a minimum amount of data
        if len(market_data) < self.lookback_window * 2:
            return shifts
            
        # Detect regimes for each window
        regimes = []
        timestamps = []
        
        for i in range(len(market_data) - self.lookback_window + 1):
            window_data = market_data.iloc[i:i+self.lookback_window]
            try:
                result = self.detect_regime(window_data)
                regimes.append(result.primary_regime)
                timestamps.append(window_data.index[-1])
            except Exception as e:
                print(f"Error detecting regime: {e}")
                continue
                
        # Find regime shifts
        current_regime = regimes[0]
        for i in range(1, len(regimes)):
            if regimes[i] != current_regime:
                shifts.append((timestamps[i], regimes[i]))
                current_regime = regimes[i]
                
        return shifts
    
    def analyze_regime_persistence(self, market_data: pd.DataFrame) -> Dict[RegimeType, float]:
        """
        Analyze how long different regimes tend to persist.
        
        Args:
            market_data: DataFrame with market data
            
        Returns:
            Dictionary mapping regime types to average duration in periods
        """
        # Detect regimes
        shifts = self.detect_regime_shifts(market_data)
        
        if not shifts:
            return {regime: 0.0 for regime in RegimeType}
            
        # Calculate durations
        durations = {regime: [] for regime in RegimeType}
        
        # Add first regime
        current_regime = shifts[0][1]
        start_idx = 0
        
        # Process regime shifts
        for i, (timestamp, regime) in enumerate(shifts[1:], 1):
            # Find index of timestamp
            try:
                idx = market_data.index.get_loc(timestamp)
                duration = idx - start_idx
                durations[current_regime].append(duration)
                start_idx = idx
                current_regime = regime
            except KeyError:
                # Timestamp not found in index
                continue
                
        # Add last regime
        durations[current_regime].append(len(market_data) - start_idx)
        
        # Calculate average durations
        avg_durations = {}
        for regime, regime_durations in durations.items():
            if regime_durations:
                avg_durations[regime] = sum(regime_durations) / len(regime_durations)
            else:
                avg_durations[regime] = 0.0
                
        return avg_durations
