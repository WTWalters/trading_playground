"""
Tests for the Risk Controls module.

This test suite validates the functionality of the RiskControls class
with a focus on:
1. Dynamic stop-loss and take-profit adjustment
2. Risk level assessment
3. Parameter adjustment based on risk levels
4. Psychological feedback mechanisms

It includes test scenarios for various market conditions and risk levels.
"""

import unittest
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import pytest

from src.market_analysis.regime_detection.detector import RegimeType
from src.market_analysis.parameter_management.risk_controls import (
    RiskControls, RiskAdjustment, RiskLevel
)


class TestRiskControls(unittest.TestCase):
    """Test cases for the RiskControls class."""
    
    def setUp(self):
        """Set up test fixtures."""
        # Initialize risk controls with default settings
        self.risk_controls = RiskControls(
            default_stop_loss_pct=0.20,
            default_take_profit_pct=0.40,
            max_drawdown_limit=0.25
        )
        
        # Define sample strategy parameters
        self.strategy_params = {
            'entry_threshold': 0.7,
            'exit_threshold': 0.3,
            'lookback_period': 20,
            'volatility_lookback': 10,
            'z_entry': 2.0,
            'z_exit': 0.5,
            'stop_loss_pct': 0.15,
            'take_profit_pct': 0.30,
            'max_holding_period': 10,
            'min_holding_period': 1,
        }
        
        # Create sample market data
        self.market_data = self._generate_sample_market_data()
        
    def _generate_sample_market_data(self):
        """Generate sample market data for testing."""
        # Create date range
        end_date = datetime.now()
        start_date = end_date - timedelta(days=30)
        dates = pd.date_range(start=start_date, end=end_date, freq='D')
        
        # Generate price series
        prices = [100.0]
        for i in range(1, len(dates)):
            prices.append(prices[-1] * (1 + np.random.normal(0, 0.015)))
        
        # Create DataFrame
        df = pd.DataFrame({
            'close': prices,
            'open': [p * (1 + np.random.normal(0, 0.005)) for p in prices],
            'high': [p * (1 + abs(np.random.normal(0, 0.01))) for p in prices],
            'low': [p * (1 - abs(np.random.normal(0, 0.01))) for p in prices],
            'volume': np.random.lognormal(15, 0.5, len(dates)),
            'volatility': [0.015 + np.random.normal(0, 0.003) for _ in range(len(dates))]
        }, index=dates)
        
        return df
        
    def test_basic_risk_adjustment(self):
        """Test basic risk adjustment functionality."""
        # Run risk adjustment
        adjustment = self.risk_controls.adjust_risk(
            strategy_params=self.strategy_params,
            market_data=self.market_data,
            current_regime=RegimeType.NEUTRAL
        )
        
        # Basic validations
        self.assertIsNotNone(adjustment)
        self.assertIsInstance(adjustment, RiskAdjustment)
        self.assertIsNotNone(adjustment.risk_level)
        self.assertIsNotNone(adjustment.adjusted_params)
        self.assertIsNotNone(adjustment.stop_loss_pct)
        self.assertIsNotNone(adjustment.take_profit_pct)
        
        # Parameter validations
        self.assertIn('stop_loss_pct', adjustment.adjusted_params)
        self.assertIn('take_profit_pct', adjustment.adjusted_params)
        
    def test_regime_specific_adjustments(self):
        """Test risk adjustments for specific regimes."""
        # Define regimes to test
        regimes = [
            RegimeType.HIGH_VOLATILITY,
            RegimeType.MEAN_REVERTING,
            RegimeType.TRENDING,
            RegimeType.LOW_LIQUIDITY
        ]
        
        for regime in regimes:
            # Run risk adjustment
            adjustment = self.risk_controls.adjust_risk(
                strategy_params=self.strategy_params,
                market_data=self.market_data,
                current_regime=regime
            )
            
            # Validate regime-specific adjustments
            if regime == RegimeType.HIGH_VOLATILITY:
                # High volatility should have tighter stops
                self.assertLess(adjustment.stop_loss_pct, self.strategy_params['stop_loss_pct'])
                self.assertGreater(adjustment.take_profit_pct, self.strategy_params['take_profit_pct'])
            elif regime == RegimeType.MEAN_REVERTING:
                # Mean reverting should adjust z-scores
                self.assertNotEqual(adjustment.adjusted_params['z_entry'], 
                                   self.strategy_params['z_entry'])
            elif regime == RegimeType.TRENDING:
                # Trending should adjust holding periods
                self.assertNotEqual(adjustment.adjusted_params['max_holding_period'], 
                                   self.strategy_params['max_holding_period'])
                
    def test_risk_level_classification(self):
        """Test risk level classification."""
        # Create scenarios with different volatility
        low_vol_data = self.market_data.copy()
        low_vol_data['volatility'] = low_vol_data['volatility'] * 0.5
        
        high_vol_data = self.market_data.copy()
        high_vol_data['volatility'] = high_vol_data['volatility'] * 2.0
        
        extreme_vol_data = self.market_data.copy()
        extreme_vol_data['volatility'] = extreme_vol_data['volatility'] * 4.0
        
        # Test all scenarios
        low_risk = self.risk_controls.assess_risk_level(low_vol_data)
        medium_risk = self.risk_controls.assess_risk_level(self.market_data)
        high_risk = self.risk_controls.assess_risk_level(high_vol_data)
        extreme_risk = self.risk_controls.assess_risk_level(extreme_vol_data)
        
        # Validate risk level ordering
        self.assertLess(low_risk.value, medium_risk.value)
        self.assertLess(medium_risk.value, high_risk.value)
        self.assertLess(high_risk.value, extreme_risk.value)
        
    def test_drawdown_impact(self):
        """Test impact of current drawdown on risk adjustments."""
        # Test scenarios with different drawdowns
        drawdowns = [0.05, 0.15, 0.25]
        
        for drawdown in drawdowns:
            # Run risk adjustment
            adjustment = self.risk_controls.adjust_risk(
                strategy_params=self.strategy_params,
                market_data=self.market_data,
                current_regime=RegimeType.NEUTRAL,
                current_drawdown=drawdown
            )
            
            # Validate drawdown impact
            if drawdown > 0.20:
                # High drawdown should reduce position size
                self.assertIn('position_size_factor', adjustment.adjusted_params)
                self.assertLess(adjustment.adjusted_params['position_size_factor'], 1.0)
                
                # And tighten stops
                self.assertLess(adjustment.stop_loss_pct, self.strategy_params['stop_loss_pct'])
                
    def test_stress_test_adjustments(self):
        """Test risk adjustments during stress tests."""
        # Run risk adjustment with stress test flag
        adjustment = self.risk_controls.adjust_risk(
            strategy_params=self.strategy_params,
            market_data=self.market_data,
            current_regime=RegimeType.HIGH_VOLATILITY,
            stress_test=True
        )
        
        # Validate conservative adjustments during stress tests
        self.assertLess(adjustment.stop_loss_pct, self.strategy_params['stop_loss_pct'])
        self.assertLess(adjustment.adjusted_params.get('position_size_factor', 1.0), 0.5)
        
    def test_consistency_checking(self):
        """Test consistency checking of risk parameters."""
        # Create invalid parameters (take-profit less than stop-loss)
        invalid_params = self.strategy_params.copy()
        invalid_params['take_profit_pct'] = 0.10  # Less than stop_loss_pct
        
        # Run adjustment - should fix the inconsistency
        adjustment = self.risk_controls.adjust_risk(
            strategy_params=invalid_params,
            market_data=self.market_data,
            current_regime=RegimeType.NEUTRAL
        )
        
        # Validate consistency has been enforced
        self.assertGreater(adjustment.take_profit_pct, adjustment.stop_loss_pct)
        
    def test_psychological_feedback(self):
        """Test psychological feedback mechanisms."""
        # Create scenarios with different performance history
        winning_streak = [True, True, True, True]
        losing_streak = [False, False, False, False]
        mixed_results = [True, False, True, False]
        
        # Test each scenario
        for streak in [winning_streak, losing_streak, mixed_results]:
            # Run adjustment with performance history
            adjustment = self.risk_controls.adjust_risk(
                strategy_params=self.strategy_params,
                market_data=self.market_data,
                current_regime=RegimeType.NEUTRAL,
                recent_trade_results=streak
            )
            
            # Validate psychological adjustments
            if streak == winning_streak:
                # Should warn about overconfidence with warnings
                self.assertIn('overconfidence', adjustment.warnings)
            elif streak == losing_streak:
                # Should adjust for risk aversion
                self.assertIn('risk_aversion', adjustment.warnings)
                
    def test_correlation_based_adjustments(self):
        """Test correlation-based adjustments."""
        # Create scenarios with different correlation structures
        correlations = {
            'low': 0.2,
            'medium': 0.5,
            'high': 0.8
        }
        
        for corr_name, corr_value in correlations.items():
            # Run adjustment with correlations
            adjustment = self.risk_controls.adjust_risk(
                strategy_params=self.strategy_params,
                market_data=self.market_data,
                current_regime=RegimeType.NEUTRAL,
                correlation_with_portfolio=corr_value
            )
            
            # Validate correlation-based adjustments
            if corr_value > 0.7:
                # High correlation should warn about diversification
                self.assertIn('high_correlation', adjustment.warnings)
                # And potentially reduce position size
                self.assertIn('position_size_factor', adjustment.adjusted_params)
                self.assertLess(adjustment.adjusted_params['position_size_factor'], 1.0)
                
    def test_adaptive_risk_params(self):
        """Test adaptation of risk parameters over time."""
        # Create sequence of market conditions
        market_conditions = [
            {'regime': RegimeType.NEUTRAL, 'drawdown': 0.05},
            {'regime': RegimeType.HIGH_VOLATILITY, 'drawdown': 0.15},
            {'regime': RegimeType.MEAN_REVERTING, 'drawdown': 0.10},
            {'regime': RegimeType.TRENDING, 'drawdown': 0.05},
        ]
        
        # Track adjustments
        stop_loss_history = []
        take_profit_history = []
        
        for condition in market_conditions:
            # Run adjustment for this condition
            adjustment = self.risk_controls.adjust_risk(
                strategy_params=self.strategy_params,
                market_data=self.market_data,
                current_regime=condition['regime'],
                current_drawdown=condition['drawdown']
            )
            
            # Track adjustments
            stop_loss_history.append(adjustment.stop_loss_pct)
            take_profit_history.append(adjustment.take_profit_pct)
            
        # Validate adaptive behavior
        self.assertNotEqual(len(set(stop_loss_history)), 1)
        self.assertNotEqual(len(set(take_profit_history)), 1)
        
        # High volatility should have smallest stop loss
        high_vol_index = [i for i, c in enumerate(market_conditions) 
                         if c['regime'] == RegimeType.HIGH_VOLATILITY][0]
        self.assertEqual(min(stop_loss_history), stop_loss_history[high_vol_index])


if __name__ == '__main__':
    unittest.main()
