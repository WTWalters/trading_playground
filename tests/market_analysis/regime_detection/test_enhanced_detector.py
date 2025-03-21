"""
Tests for the Enhanced Regime Detector module.

This test suite validates the functionality of the EnhancedRegimeDetector class
with a focus on:
1. Multi-timeframe regime detection
2. Macro indicator integration
3. Market turning point identification
4. Transition probability modeling

It includes test scenarios for various market conditions and regime transitions.
"""

import unittest
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import pytest

from src.market_analysis.regime_detection.detector import RegimeType
from src.market_analysis.regime_detection.enhanced_detector import (
    EnhancedRegimeDetector, MacroRegimeType, EnhancedRegimeResult
)


class TestEnhancedRegimeDetector(unittest.TestCase):
    """Test cases for the EnhancedRegimeDetector class."""
    
    def setUp(self):
        """Set up test fixtures."""
        # Initialize detector with default settings
        self.detector = EnhancedRegimeDetector(
            lookback_window=20,
            volatility_threshold=1.5,
            vix_threshold=20.0
        )
        
        # Generate sample market data
        self.market_data = self._generate_sample_market_data()
        
        # Generate sample macro data
        self.macro_data = self._generate_sample_macro_data()
        
    def _generate_sample_market_data(self):
        """Generate sample market data for testing."""
        # Create date range
        end_date = datetime.now()
        start_date = end_date - timedelta(days=100)
        dates = pd.date_range(start=start_date, end=end_date, freq='D')
        
        # Generate price series with trend
        prices = [100.0]
        for i in range(1, len(dates)):
            # Add trend and noise
            trend = 0.001 if i < len(dates) // 2 else -0.001  # Up then down
            noise = np.random.normal(0, 0.015)
            prices.append(prices[-1] * (1 + trend + noise))
        
        # Create DataFrame
        df = pd.DataFrame({
            'close': prices,
            'open': [p * (1 + np.random.normal(0, 0.005)) for p in prices],
            'high': [p * (1 + abs(np.random.normal(0, 0.01))) for p in prices],
            'low': [p * (1 - abs(np.random.normal(0, 0.01))) for p in prices],
            'volume': np.random.lognormal(15, 0.5, len(dates))
        }, index=dates)
        
        # Add some volatility changes
        vol_factor = [1.0] * len(dates)
        vol_factor[len(dates)//3:2*len(dates)//3] = [2.0] * (len(dates)//3)
        
        df['volatility'] = np.array(vol_factor) * np.random.normal(0.01, 0.005, len(dates))
        
        return df
    
    def _generate_sample_macro_data(self):
        """Generate sample macro data for testing."""
        # Use the same dates as market data
        dates = self.market_data.index
        
        # Generate VIX data (inverse correlation with market)
        close_returns = self.market_data['close'].pct_change().fillna(0)
        vix_base = 15.0
        vix = [vix_base]
        
        for i in range(1, len(dates)):
            # VIX tends to spike when market drops
            vix_change = -5.0 * close_returns.iloc[i] + np.random.normal(0, 0.03)
            # Add a spike in the middle
            if i == len(dates) // 2:
                vix_change += 0.5
            vix.append(max(10, vix[-1] * (1 + vix_change)))
        
        # Generate other macro indicators
        yield_curve = [0.5] * len(dates)
        # Make yield curve negative in the last third to simulate inversion
        yield_curve[-len(dates)//3:] = [-0.2] * (len(dates)//3)
        
        # Create interest rates (rising then falling)
        interest_rate = np.concatenate([
            np.linspace(2.0, 4.0, len(dates)//2),
            np.linspace(4.0, 3.0, len(dates) - len(dates)//2)
        ])
        
        # Create USD index with some trend
        usd_index = 100 + np.cumsum(np.random.normal(0, 0.002, len(dates)))
        
        # Add SPX index (correlated with our market data but with different scale)
        spx = self.market_data['close'] * 30 + 3000
        
        # Create DataFrame
        df = pd.DataFrame({
            'VIX': vix,
            'yield_curve': yield_curve,
            'interest_rate': interest_rate,
            'USD_index': usd_index,
            'SPX': spx
        }, index=dates)
        
        return df
        
    def test_basic_regime_detection(self):
        """Test basic regime detection functionality."""
        # Run regime detection on our sample data
        result = self.detector.detect_regime(self.market_data)
        
        # Basic sanity checks
        self.assertIsNotNone(result)
        self.assertIsInstance(result, EnhancedRegimeResult)
        self.assertIsNotNone(result.primary_regime)
        self.assertIsNotNone(result.stability_score)
        self.assertIsNotNone(result.transition_probability)
        
    def test_macro_regime_detection(self):
        """Test integration with macro indicators."""
        # Run regime detection with macro data
        result = self.detector.detect_regime(self.market_data, self.macro_data)
        
        # Check macro regime fields
        self.assertIsNotNone(result.macro_regime)
        self.assertIsNotNone(result.sentiment_regime)
        self.assertIsNotNone(result.interest_rate_regime)
        
        # Check transition signals
        self.assertGreater(len(result.transition_signals), 0)
        
    def test_turning_point_detection(self):
        """Test market turning point detection."""
        # Create a specific dataset with a turning point
        dates = pd.date_range(start=datetime.now() - timedelta(days=60), periods=60, freq='D')
        
        # Create prices with a clear trend reversal
        prices = []
        for i in range(30):
            prices.append(100 + i)  # Rising trend
        for i in range(30):
            prices.append(130 - i)  # Falling trend
            
        # Create the market data
        market_data = pd.DataFrame({
            'close': prices,
            'open': [p - 0.5 for p in prices],
            'high': [p + 1 for p in prices],
            'low': [p - 1 for p in prices],
            'volume': [1000000] * 60
        }, index=dates)
        
        # Create macro data with a VIX spike at the turning point
        vix = [15] * 29 + [35] + [25] * 30
        macro_data = pd.DataFrame({
            'VIX': vix,
            'yield_curve': [0.5] * 25 + [0.0] + [-0.1] * 34,
            'SPX': [p * 30 + 3000 for p in prices]
        }, index=dates)
        
        # Run detection
        result = self.detector.detect_regime(market_data, macro_data)
        
        # Check turning point detection
        self.assertTrue(result.regime_turning_point)
        self.assertGreater(result.turning_point_confidence, 0.5)
        
    def test_multi_timeframe_detection(self):
        """Test regime detection across multiple timeframes."""
        # Create timeframe data
        # Daily data is our original market data
        daily_data = self.market_data.copy()
        
        # Weekly data by resampling
        weekly_data = daily_data.resample('W').agg({
            'open': 'first',
            'high': 'max', 
            'low': 'min',
            'close': 'last',
            'volume': 'sum'
        }).dropna()
        
        # Create a dictionary of timeframe data
        timeframe_data = {
            'daily': daily_data,
            'weekly': weekly_data
        }
        
        # Run multi-timeframe analysis
        results = self.detector.analyze_multiple_timeframes(timeframe_data, self.macro_data)
        
        # Validate results
        self.assertEqual(len(results), 2)
        self.assertIn('daily', results)
        self.assertIn('weekly', results)
        
        # Check alignment score
        alignment = self.detector.get_regime_alignment(results)
        self.assertGreaterEqual(alignment, 0)
        self.assertLessEqual(alignment, 1)

    def test_different_market_regimes(self):
        """Test detection across different simulated market regimes."""
        # Create test scenarios for different regimes
        regime_scenarios = {
            "trending": self._create_trending_market(),
            "mean_reverting": self._create_mean_reverting_market(),
            "high_volatility": self._create_high_volatility_market(),
            "low_liquidity": self._create_low_liquidity_market()
        }
        
        # Test each scenario
        for regime_name, data in regime_scenarios.items():
            market_data = data['market']
            macro_data = data['macro']
            
            # Run regime detection
            result = self.detector.detect_regime(market_data, macro_data)
            
            # Log the result for debugging
            print(f"Scenario: {regime_name}")
            print(f"Detected primary regime: {result.primary_regime}")
            print(f"Detected secondary regime: {result.secondary_regime}")
            print(f"Detected macro regime: {result.macro_regime}")
            
            # Perform appropriate assertions based on scenario
            if regime_name == "trending":
                # In trending market, we expect trending or momentum regime
                self.assertIn(result.primary_regime, 
                             [RegimeType.TRENDING, RegimeType.MOMENTUM])
            elif regime_name == "mean_reverting":
                # In mean-reverting market, we expect mean-reverting regime
                self.assertEqual(result.primary_regime, RegimeType.MEAN_REVERTING)
            elif regime_name == "high_volatility":
                # In high volatility market, we expect high volatility regime
                self.assertEqual(result.primary_regime, RegimeType.HIGH_VOLATILITY)
                
    def _create_trending_market(self):
        """Create a simulated trending market."""
        dates = pd.date_range(start=datetime.now() - timedelta(days=60), periods=60, freq='D')
        
        # Create strong uptrend
        prices = [100.0]
        for i in range(1, 60):
            prices.append(prices[-1] * (1 + 0.005 + np.random.normal(0, 0.003)))
            
        market_data = pd.DataFrame({
            'close': prices,
            'open': [p * (1 - 0.005) for p in prices],
            'high': [p * (1 + 0.01) for p in prices],
            'low': [p * (1 - 0.01) for p in prices],
            'volume': [1000000 + np.random.normal(0, 50000)] * 60
        }, index=dates)
        
        # Low VIX due to trending market
        vix = [12 + np.random.normal(0, 1) for _ in range(60)]
        
        macro_data = pd.DataFrame({
            'VIX': vix,
            'yield_curve': [0.5 + np.random.normal(0, 0.05) for _ in range(60)],
            'SPX': [p * 30 + 3000 for p in prices]
        }, index=dates)
        
        return {'market': market_data, 'macro': macro_data}
        
    def _create_mean_reverting_market(self):
        """Create a simulated mean-reverting market."""
        dates = pd.date_range(start=datetime.now() - timedelta(days=60), periods=60, freq='D')
        
        # Create oscillating prices
        center = 100.0
        amplitude = 10.0
        period = 20
        
        prices = [center + amplitude * np.sin(2 * np.pi * i / period) + np.random.normal(0, 1) 
                 for i in range(60)]
            
        market_data = pd.DataFrame({
            'close': prices,
            'open': [p + np.random.normal(0, 0.5) for p in prices],
            'high': [p + abs(np.random.normal(0, 1)) for p in prices],
            'low': [p - abs(np.random.normal(0, 1)) for p in prices],
            'volume': [800000 + np.random.normal(0, 50000)] * 60
        }, index=dates)
        
        # Moderate VIX
        vix = [15 + np.random.normal(0, 2) for _ in range(60)]
        
        macro_data = pd.DataFrame({
            'VIX': vix,
            'yield_curve': [0.2 + np.random.normal(0, 0.05) for _ in range(60)],
            'SPX': [3000 + 300 * np.sin(2 * np.pi * i / period) + np.random.normal(0, 30) 
                   for i in range(60)]
        }, index=dates)
        
        return {'market': market_data, 'macro': macro_data}
        
    def _create_high_volatility_market(self):
        """Create a simulated high volatility market."""
        dates = pd.date_range(start=datetime.now() - timedelta(days=60), periods=60, freq='D')
        
        # Create volatile prices
        prices = [100.0]
        for i in range(1, 60):
            prices.append(prices[-1] * (1 + np.random.normal(0, 0.03)))
            
        market_data = pd.DataFrame({
            'close': prices,
            'open': [p * (1 + np.random.normal(0, 0.02)) for p in prices],
            'high': [p * (1 + abs(np.random.normal(0, 0.04))) for p in prices],
            'low': [p * (1 - abs(np.random.normal(0, 0.04))) for p in prices],
            'volume': [1500000 + np.random.normal(0, 200000)] * 60
        }, index=dates)
        
        # High VIX
        vix = [30 + np.random.normal(0, 5) for _ in range(60)]
        
        macro_data = pd.DataFrame({
            'VIX': vix,
            'yield_curve': [-0.1 + np.random.normal(0, 0.05) for _ in range(60)],
            'SPX': [3000 + np.random.normal(0, 100) for _ in range(60)]
        }, index=dates)
        
        return {'market': market_data, 'macro': macro_data}
        
    def _create_low_liquidity_market(self):
        """Create a simulated low liquidity market."""
        dates = pd.date_range(start=datetime.now() - timedelta(days=60), periods=60, freq='D')
        
        # Create choppy prices with low volume
        prices = [100.0]
        for i in range(1, 60):
            # More noise, less trend
            prices.append(prices[-1] * (1 + np.random.normal(0, 0.01)))
            
        market_data = pd.DataFrame({
            'close': prices,
            'open': [p * (1 + np.random.normal(0, 0.01)) for p in prices],
            'high': [p * (1 + abs(np.random.normal(0, 0.015))) for p in prices],
            'low': [p * (1 - abs(np.random.normal(0, 0.015))) for p in prices],
            'volume': [300000 + np.random.normal(0, 30000)] * 60  # Low volume
        }, index=dates)
        
        # Moderate VIX
        vix = [18 + np.random.normal(0, 2) for _ in range(60)]
        
        macro_data = pd.DataFrame({
            'VIX': vix,
            'yield_curve': [0.3 + np.random.normal(0, 0.05) for _ in range(60)],
            'SPX': [3000 + np.random.normal(0, 20) for _ in range(60)]
        }, index=dates)
        
        return {'market': market_data, 'macro': macro_data}


if __name__ == '__main__':
    unittest.main()
