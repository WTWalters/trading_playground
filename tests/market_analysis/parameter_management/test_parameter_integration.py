"""
Tests for the Parameter Integration module.

This test suite validates the functionality of the ParameterIntegration class,
which integrates regime detection, position sizing, and risk management
to provide a unified parameter adaptation system.
"""

import unittest
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import tempfile
import os
from pathlib import Path

from src.market_analysis.regime_detection.detector import RegimeType
from src.market_analysis.regime_detection.enhanced_detector import EnhancedRegimeDetector
from src.market_analysis.parameter_management.position_sizing import KellyPositionSizer
from src.market_analysis.parameter_management.risk_controls import RiskControls, AdaptiveRiskControls
from src.market_analysis.parameter_management.risk_manager import RiskManager
from src.market_analysis.parameter_management.integration import ParameterIntegration, AdaptiveParameterManager


class TestParameterIntegration(unittest.TestCase):
    """Test cases for the ParameterIntegration class."""
    
    def setUp(self):
        """Set up test fixtures."""
        # Initialize core components
        self.regime_detector = EnhancedRegimeDetector(
            lookback_window=30,
            volatility_threshold=1.5,
            vix_threshold=20.0
        )
        
        self.risk_controls = AdaptiveRiskControls(
            default_max_position_size=0.1,
            default_max_pair_exposure=0.2,
            default_stop_loss=0.05
        )
        
        self.position_sizer = KellyPositionSizer(
            default_fraction=0.5,
            max_kelly_fraction=0.8,
            min_kelly_fraction=0.1
        )
        
        self.risk_manager = RiskManager(
            regime_detector=self.regime_detector,
            risk_controls=self.risk_controls,
            position_sizer=self.position_sizer
        )
        
        # Initialize integration module
        self.param_integration = ParameterIntegration(
            regime_detector=self.regime_detector,
            risk_manager=self.risk_manager
        )
        
        # Generate sample market data
        self.market_data = self._generate_sample_market_data()
        
        # Create temporary directory for parameter storage
        self.temp_dir = tempfile.TemporaryDirectory()
        self.storage_path = Path(self.temp_dir.name)
        
    def tearDown(self):
        """Clean up after tests."""
        self.temp_dir.cleanup()
    
    def _generate_sample_market_data(self):
        """Generate sample market data for testing."""
        # Create date range
        end_date = datetime.now()
        start_date = end_date - timedelta(days=100)
        dates = pd.date_range(start=start_date, end=end_date, freq='D')
        
        # Generate price series with mixed regimes
        prices = [100.0]
        
        # First third: low volatility
        for i in range(1, len(dates)//3):
            prices.append(prices[-1] * (1 + np.random.normal(0.0005, 0.005)))
        
        # Middle third: high volatility
        for i in range(len(dates)//3, 2*len(dates)//3):
            prices.append(prices[-1] * (1 + np.random.normal(0, 0.02)))
        
        # Last third: trending
        for i in range(2*len(dates)//3, len(dates)):
            prices.append(prices[-1] * (1 + 0.002 + np.random.normal(0, 0.008)))
        
        # Create DataFrame with returns
        df = pd.DataFrame({
            'close': prices,
            'open': [p * (1 + np.random.normal(0, 0.005)) for p in prices],
            'high': [p * (1 + abs(np.random.normal(0, 0.01))) for p in prices],
            'low': [p * (1 - abs(np.random.normal(0, 0.01))) for p in prices],
            'volume': np.random.lognormal(15, 0.5, len(dates))
        }, index=dates)
        
        # Calculate returns
        df['returns'] = df['close'].pct_change().fillna(0)
        
        return df
    
    def test_basic_integration(self):
        """Test basic parameter integration functionality."""
        # Use a window of the market data
        window = self.market_data.iloc[-30:]
        
        # Set up initial parameters
        initial_params = {
            'max_position_size': 0.1,
            'stop_loss': 0.05,
            'kelly_fraction': 0.5,
            'z_score_entry': 2.0,
            'z_score_exit': 0.0,
            'window_size': 20
        }
        
        # Run parameter adaptation
        adapted_params = self.param_integration.adapt_parameters(
            returns=window['returns'],
            volatility=window['returns'].std() * np.sqrt(252),  # Annualized
            sharpe_ratio=0.8,
            win_rate=0.6,
            current_parameters=initial_params
        )
        
        # Basic checks
        self.assertIsNotNone(adapted_params)
        self.assertIsInstance(adapted_params, dict)
        
        # Check that parameters have been modified
        self.assertNotEqual(adapted_params['max_position_size'], initial_params['max_position_size'])
        self.assertNotEqual(adapted_params['stop_loss'], initial_params['stop_loss'])
        
        # Check that z-score parameters have been adapted
        self.assertIn('z_score_entry', adapted_params)
        self.assertIn('z_score_exit', adapted_params)
        
    def test_volatile_market_adaptation(self):
        """Test parameter adaptation in volatile markets."""
        # Generate a volatile market window
        volatile_window = self.market_data.iloc[len(self.market_data)//3:2*len(self.market_data)//3]
        
        # Initial parameters
        initial_params = {
            'max_position_size': 0.1,
            'stop_loss': 0.05,
            'kelly_fraction': 0.5,
            'z_score_entry': 2.0,
            'z_score_exit': 0.0
        }
        
        # Run parameter adaptation
        volatile_params = self.param_integration.adapt_parameters(
            returns=volatile_window['returns'],
            volatility=volatile_window['returns'].std() * np.sqrt(252),
            sharpe_ratio=0.6,  # Lower in volatile markets
            win_rate=0.5,
            current_parameters=initial_params
        )
        
        # In volatile markets:
        # - Position sizes should be reduced
        # - Stop losses should be wider
        # - Z-score entries might be higher (more conservative)
        self.assertLess(volatile_params['max_position_size'], initial_params['max_position_size'])
        self.assertGreater(volatile_params['z_score_entry'], initial_params['z_score_entry'])
        
    def test_trending_market_adaptation(self):
        """Test parameter adaptation in trending markets."""
        # Generate a trending market window
        trending_window = self.market_data.iloc[-len(self.market_data)//3:]
        
        # Initial parameters
        initial_params = {
            'max_position_size': 0.05,  # Start small
            'stop_loss': 0.08,  # Wide stop
            'kelly_fraction': 0.3,
            'trend_following': False,
            'mean_reversion': True
        }
        
        # Run parameter adaptation
        trending_params = self.param_integration.adapt_parameters(
            returns=trending_window['returns'],
            volatility=trending_window['returns'].std() * np.sqrt(252),
            sharpe_ratio=1.2,  # Better in a trend
            win_rate=0.65,
            current_parameters=initial_params
        )
        
        # Just check that we get valid parameters back
        self.assertIsInstance(trending_params, dict)
        self.assertIn('max_position_size', trending_params)
        self.assertIn('stop_loss', trending_params)
        
    def test_different_regime_adaptations(self):
        """Test adaptation across different market regimes."""
        # Define regimes to test
        regimes = [
            (RegimeType.LOW_VOLATILITY, 0.01, 1.5, 0.6),  # Low vol, good Sharpe, decent win rate
            (RegimeType.HIGH_VOLATILITY, 0.03, 0.6, 0.5),  # High vol, poor Sharpe, average win rate
            (RegimeType.TRENDING, 0.015, 1.2, 0.65),      # Trending, good Sharpe, good win rate
            (RegimeType.MEAN_REVERTING, 0.012, 1.0, 0.55)  # Mean reverting, decent Sharpe, average win rate
        ]
        
        # Use the same initial parameters for comparison
        initial_params = {
            'max_position_size': 0.1,
            'stop_loss': 0.05,
            'kelly_fraction': 0.5,
            'z_score_entry': 2.0,
            'z_score_exit': 0.0,
            'trend_following_weight': 0.5,
            'mean_reversion_weight': 0.5
        }
        
        adaptations = {}
        
        # Test adaptation for each regime
        for regime, volatility, sharpe, win_rate in regimes:
            # Force regime classification
            self.param_integration.regime_detector.detect_regime = lambda x, y=None: MockRegimeResult(regime)
            
            # Generate returns based on volatility
            returns = pd.Series(np.random.normal(0, volatility / np.sqrt(252), 30))  # Daily volatility
            
            # Run adaptation
            adapted = self.param_integration.adapt_parameters(
                returns=returns,
                volatility=volatility,
                sharpe_ratio=sharpe,
                win_rate=win_rate,
                current_parameters=initial_params.copy()  # Copy to avoid modifying original
            )
            
            adaptations[regime] = adapted
        
        # Verify regime-specific adaptations
        # Low volatility should allow larger positions than high volatility
        self.assertGreater(
            adaptations[RegimeType.LOW_VOLATILITY]['max_position_size'],
            adaptations[RegimeType.HIGH_VOLATILITY]['max_position_size']
        )
        
        # Trending should favor trend following over mean reversion
        if 'trend_following_weight' in adaptations[RegimeType.TRENDING]:
            self.assertGreater(
                adaptations[RegimeType.TRENDING]['trend_following_weight'],
                adaptations[RegimeType.TRENDING]['mean_reversion_weight']
            )
        
        # Mean reverting should favor mean reversion over trend following
        if 'mean_reversion_weight' in adaptations[RegimeType.MEAN_REVERTING]:
            self.assertGreater(
                adaptations[RegimeType.MEAN_REVERTING]['mean_reversion_weight'],
                adaptations[RegimeType.MEAN_REVERTING]['trend_following_weight']
            )
        
        # High volatility should have tighter stop losses (relative to volatility)
        vol_ratio_high = adaptations[RegimeType.HIGH_VOLATILITY]['stop_loss'] / 0.03
        vol_ratio_low = adaptations[RegimeType.LOW_VOLATILITY]['stop_loss'] / 0.01
        self.assertLess(vol_ratio_high, vol_ratio_low)
        
    def test_smooth_parameter_transitions(self):
        """Test that parameters transition smoothly between regimes."""
        # Initial parameters
        initial_params = {
            'max_position_size': 0.1,
            'stop_loss': 0.05,
            'kelly_fraction': 0.5,
            'z_score_entry': 2.0,
            'z_score_exit': 0.0
        }
        
        # Create series of adaptations with increasing volatility
        volatilities = np.linspace(0.01, 0.04, 10)  # From low to high volatility
        adapted_params = []
        
        for vol in volatilities:
            # Generate returns based on volatility
            returns = pd.Series(np.random.normal(0, vol / np.sqrt(252), 30))  # Daily volatility
            
            # Use most recent adaptation as starting point
            current = adapted_params[-1] if adapted_params else initial_params
            
            # Run adaptation
            adapted = self.param_integration.adapt_parameters(
                returns=returns,
                volatility=vol,
                sharpe_ratio=1.0,
                win_rate=0.55,
                current_parameters=current.copy()  # Copy to avoid modifying original
            )
            
            adapted_params.append(adapted)
        
        # Check for smooth transitions in key parameters
        position_sizes = [p['max_position_size'] for p in adapted_params]
        stop_losses = [p['stop_loss'] for p in adapted_params]
        
        # Calculate maximum step size
        max_position_step = max([abs(position_sizes[i] - position_sizes[i-1]) 
                                for i in range(1, len(position_sizes))])
        max_stop_loss_step = max([abs(stop_losses[i] - stop_losses[i-1]) 
                                for i in range(1, len(stop_losses))])
        
        # Smooth transitions should have reasonably small step sizes
        self.assertLess(max_position_step, 0.05, "Position size changes are too abrupt")
        self.assertLess(max_stop_loss_step, 0.05, "Stop loss changes are too abrupt")
        
        # Check trend in position sizes (should decrease with increasing volatility)
        self.assertLess(position_sizes[-1], position_sizes[0], 
                       "Position sizes should decrease with increasing volatility")
        
    def test_adaptive_parameter_manager_integration(self):
        """Test integration with the AdaptiveParameterManager."""
        # Create AdaptiveParameterManager
        manager = AdaptiveParameterManager(
            risk_manager=self.risk_manager,
            config={
                'max_portfolio_risk': 0.12,
                'max_strategy_risk': 0.04,
                'max_concentration': 0.25,
                'min_strategies': 3,
                'backtesting_mode': True
            }
        )
        
        # Register a strategy
        strategy_id = "test_strategy"
        base_parameters = {
            'max_position_size': 0.1,
            'stop_loss': 0.05,
            'kelly_fraction': 0.5,
            'z_score_entry': 2.0,
            'z_score_exit': 0.0,
            'window_size': 20
        }
        
        manager.register_strategy(
            strategy_id=strategy_id,
            base_parameters=base_parameters,
            strategy_metrics={
                'win_rate': 0.6,
                'win_loss_ratio': 1.5,
                'volatility': 0.2,
                'correlation': 0.0,
                'sharpe': 1.2
            }
        )
        
        # Update market state
        manager.update_market_state(
            market_data=self.market_data,
            portfolio_value=100000.0
        )
        
        # Get optimized parameters
        optimized_params = manager.get_optimized_parameters(
            strategy_id=strategy_id,
            signal_strength=0.8  # Strong signal
        )
        
        # Basic checks
        self.assertIsNotNone(optimized_params)
        self.assertIsInstance(optimized_params, dict)
        
        # Verify adaptive parameters
        self.assertIn('position_size_pct', optimized_params)
        self.assertIn('position_size', optimized_params)
        self.assertIn('stop_loss_pct', optimized_params)
        self.assertIn('take_profit_pct', optimized_params)
        
        # Test performance update
        manager.update_performance(
            strategy_id=strategy_id,
            trade_result=True,  # Winning trade
            trade_metrics={
                'pnl': 1500.0,
                'pnl_pct': 0.015,
                'duration': 3,
                'max_drawdown': 0.005
            }
        )
        
        # Get risk profile
        risk_profile = manager.get_strategy_risk_profile(strategy_id)
        
        # Check risk profile
        self.assertIsNotNone(risk_profile)
        self.assertIn('metrics', risk_profile)
        self.assertIn('regime_distribution', risk_profile)
        self.assertIn('risk_level', risk_profile)
        
        # Run stress test
        stress_results = manager.run_stress_test(
            strategy_id=strategy_id,
            scenario_names=["Market Crash"]
        )
        
        # Check stress test results
        self.assertIsNotNone(stress_results)
        self.assertIn("Market Crash", stress_results)
        self.assertIn('position_size', stress_results["Market Crash"])
        self.assertIn('parameters', stress_results["Market Crash"])
        
        # Position size should be reduced in market crash scenario
        self.assertLess(
            stress_results["Market Crash"]["position_size"],
            optimized_params['position_size_pct']
        )


class MockRegimeResult:
    """Mock regime detection result for testing."""
    
    def __init__(self, primary_regime):
        self.primary_regime = primary_regime
        self.secondary_regime = None
        self.stability_score = 0.8
        self.volatility_regime = RegimeType.HIGH_VOLATILITY if primary_regime == RegimeType.HIGH_VOLATILITY else RegimeType.LOW_VOLATILITY
        self.correlation_regime = RegimeType.UNDEFINED
        self.liquidity_regime = RegimeType.UNDEFINED
        self.trend_regime = primary_regime if primary_regime in [RegimeType.TRENDING, RegimeType.MEAN_REVERTING] else RegimeType.UNDEFINED
        self.regime_turning_point = False
        self.turning_point_confidence = 0.0
        self.transition_signals = {}
        self.timeframe_regimes = {}


if __name__ == '__main__':
    unittest.main()
