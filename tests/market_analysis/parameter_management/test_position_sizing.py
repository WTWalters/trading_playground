"""
Tests for the Kelly-based Position Sizing module.

This test suite validates the functionality of the KellyPositionSizer class
with a focus on:
1. Kelly criterion implementation
2. Regime-specific Kelly fraction adjustments
3. Portfolio allocation functionality
4. Expected value and edge calculations

It includes test scenarios for various market conditions and signal strengths.
"""

import unittest
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import pytest

from src.market_analysis.regime_detection.detector import RegimeType
from src.market_analysis.parameter_management.position_sizing import (
    KellyPositionSizer, PositionSizeResult
)


class TestKellyPositionSizer(unittest.TestCase):
    """Test cases for the KellyPositionSizer class."""
    
    def setUp(self):
        """Set up test fixtures."""
        # Initialize position sizer with default settings
        self.position_sizer = KellyPositionSizer(
            default_kelly_fraction=0.5,
            max_position_pct=0.05  # 5% max position
        )
        
        # Define sample market data
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
            'volume': np.random.lognormal(15, 0.5, len(dates))
        }, index=dates)
        
        return df
        
    def test_basic_position_sizing(self):
        """Test basic position sizing functionality."""
        # Define a simple signal
        signal = {
            'win_probability': 0.6,
            'win_loss_ratio': 2.0,
            'strength': 0.7
        }
        
        # Calculate position size
        result = self.position_sizer.calculate_position_size(
            signal=signal,
            portfolio_value=100000,
            current_regime=RegimeType.NEUTRAL
        )
        
        # Basic validations
        self.assertIsNotNone(result)
        self.assertIsInstance(result, PositionSizeResult)
        self.assertGreater(result.position_pct, 0)
        self.assertLess(result.position_pct, self.position_sizer.max_position_pct)
        self.assertGreater(result.position_size, 0)
        
    def test_kelly_calculation(self):
        """Test Kelly criterion calculation."""
        # Test a range of win probabilities and win/loss ratios
        test_cases = [
            {'win_prob': 0.55, 'win_loss_ratio': 1.2},  # Slight edge
            {'win_prob': 0.60, 'win_loss_ratio': 2.0},  # Good edge
            {'win_prob': 0.70, 'win_loss_ratio': 3.0},  # Strong edge
            {'win_prob': 0.40, 'win_loss_ratio': 4.0},  # Low prob, high payoff
            {'win_prob': 0.80, 'win_loss_ratio': 1.1}   # High prob, low payoff
        ]
        
        for case in test_cases:
            # Calculate raw Kelly
            kelly = self.position_sizer._calculate_kelly(
                win_probability=case['win_prob'],
                win_loss_ratio=case['win_loss_ratio']
            )
            
            # Calculate expected value
            expected_value = case['win_prob'] * case['win_loss_ratio'] - (1 - case['win_prob'])
            
            # Validations
            if expected_value > 0:
                # Should recommend a positive position
                self.assertGreater(kelly, 0)
            else:
                # Should recommend no position
                self.assertLessEqual(kelly, 0)
                
            # Kelly should never exceed 1.0
            self.assertLessEqual(kelly, 1.0)
            
    def test_regime_specific_adjustments(self):
        """Test position sizing adjustments for specific regimes."""
        # Define a signal
        signal = {
            'win_probability': 0.60,
            'win_loss_ratio': 2.0,
            'strength': 0.7
        }
        
        # Test all relevant regimes
        regimes = [
            RegimeType.NEUTRAL,
            RegimeType.HIGH_VOLATILITY,
            RegimeType.MEAN_REVERTING,
            RegimeType.TRENDING,
            RegimeType.LOW_LIQUIDITY
        ]
        
        position_sizes = {}
        
        for regime in regimes:
            # Calculate position size for this regime
            result = self.position_sizer.calculate_position_size(
                signal=signal,
                portfolio_value=100000,
                current_regime=regime
            )
            
            # Store position size
            position_sizes[regime] = result.position_pct
            
        # Validate regime-specific adjustments
        # High volatility should be more conservative
        self.assertLess(position_sizes[RegimeType.HIGH_VOLATILITY], 
                       position_sizes[RegimeType.NEUTRAL])
        
        # Low liquidity should also be more conservative
        self.assertLess(position_sizes[RegimeType.LOW_LIQUIDITY], 
                       position_sizes[RegimeType.NEUTRAL])
        
    def test_signal_strength_impact(self):
        """Test impact of signal strength on position sizing."""
        # Test a range of signal strengths
        strengths = [0.3, 0.5, 0.7, 0.9]
        
        base_signal = {
            'win_probability': 0.60,
            'win_loss_ratio': 2.0
        }
        
        position_sizes = {}
        
        for strength in strengths:
            # Create signal with this strength
            signal = base_signal.copy()
            signal['strength'] = strength
            
            # Calculate position size
            result = self.position_sizer.calculate_position_size(
                signal=signal,
                portfolio_value=100000,
                current_regime=RegimeType.NEUTRAL
            )
            
            # Store position size
            position_sizes[strength] = result.position_pct
            
        # Validate strength impact - should be monotonically increasing
        for i in range(1, len(strengths)):
            self.assertGreaterEqual(
                position_sizes[strengths[i]],
                position_sizes[strengths[i-1]]
            )
            
    def test_maximum_position_constraint(self):
        """Test enforcement of maximum position size constraint."""
        # Create a very strong signal that would exceed max position
        signal = {
            'win_probability': 0.95,
            'win_loss_ratio': 10.0,
            'strength': 1.0
        }
        
        # Calculate position size
        result = self.position_sizer.calculate_position_size(
            signal=signal,
            portfolio_value=100000,
            current_regime=RegimeType.NEUTRAL
        )
        
        # Validate max position constraint
        self.assertLessEqual(result.position_pct, self.position_sizer.max_position_pct)
        self.assertTrue(result.max_size_reached)
        
    def test_portfolio_allocation(self):
        """Test portfolio allocation functionality."""
        # Define multiple signals
        signals = {
            'strategy_1': {
                'win_probability': 0.60,
                'win_loss_ratio': 2.0,
                'strength': 0.7,
                'expected_value': 0.2  # Pre-calculated for convenience
            },
            'strategy_2': {
                'win_probability': 0.55,
                'win_loss_ratio': 3.0,
                'strength': 0.6,
                'expected_value': 0.1
            },
            'strategy_3': {
                'win_probability': 0.70,
                'win_loss_ratio': 1.5,
                'strength': 0.8,
                'expected_value': 0.05
            }
        }
        
        # Calculate portfolio allocation
        allocations = self.position_sizer.get_portfolio_allocations(
            signals=signals,
            portfolio_value=100000,
            current_regime=RegimeType.NEUTRAL,
            max_allocation=0.8  # 80% max total allocation
        )
        
        # Validate allocations
        self.assertEqual(len(allocations), len(signals))
        
        # Calculate total allocation
        total_allocation = sum(alloc['allocation'] for alloc in allocations.values())
        
        # Validate constraints
        self.assertLessEqual(total_allocation, 0.8)  # Should respect max_allocation
        
        # Strategy with highest expected value should have largest allocation
        self.assertGreater(
            allocations['strategy_1']['allocation'],
            allocations['strategy_2']['allocation']
        )
        
    def test_expected_value_calculation(self):
        """Test expected value and edge calculations."""
        # Test various combinations
        test_cases = [
            {'win_prob': 0.60, 'win_loss_ratio': 2.0, 'expected': 0.60 * 2.0 - (1 - 0.60)},
            {'win_prob': 0.40, 'win_loss_ratio': 1.0, 'expected': 0.40 - 0.60},  # Negative EV
            {'win_prob': 0.50, 'win_loss_ratio': 2.0, 'expected': 0.50 * 2.0 - 0.50}
        ]
        
        for case in test_cases:
            # Calculate expected value
            ev = self.position_sizer._calculate_expected_value(
                win_probability=case['win_prob'],
                win_loss_ratio=case['win_loss_ratio']
            )
            
            # Validate calculation
            self.assertAlmostEqual(ev, case['expected'], places=6)
            
    def test_risk_of_ruin_consideration(self):
        """Test consideration of risk of ruin in position sizing."""
        # Create a signal with moderate edge
        signal = {
            'win_probability': 0.55,
            'win_loss_ratio': 2.0,
            'strength': 0.7
        }
        
        # Calculate position size with different risk of ruin settings
        result_standard = self.position_sizer.calculate_position_size(
            signal=signal,
            portfolio_value=100000,
            current_regime=RegimeType.NEUTRAL
        )
        
        # Create conservative position sizer
        conservative_sizer = KellyPositionSizer(
            default_kelly_fraction=0.25,  # Quarter-Kelly
            max_position_pct=0.05
        )
        
        result_conservative = conservative_sizer.calculate_position_size(
            signal=signal,
            portfolio_value=100000,
            current_regime=RegimeType.NEUTRAL
        )
        
        # Validate more conservative sizing
        self.assertLess(result_conservative.position_pct, result_standard.position_pct)
        
    def test_strategy_correlation_impact(self):
        """Test impact of strategy correlations on portfolio allocation."""
        # Define signals
        signals = {
            'strategy_1': {
                'win_probability': 0.60,
                'win_loss_ratio': 2.0,
                'strength': 0.7,
                'expected_value': 0.2
            },
            'strategy_2': {
                'win_probability': 0.55,
                'win_loss_ratio': 3.0,
                'strength': 0.6,
                'expected_value': 0.1
            }
        }
        
        # Define correlation matrix
        correlations = {
            ('strategy_1', 'strategy_2'): 0.8  # High correlation
        }
        
        # Calculate portfolio allocation with correlations
        allocations_with_corr = self.position_sizer.get_portfolio_allocations(
            signals=signals,
            portfolio_value=100000,
            current_regime=RegimeType.NEUTRAL,
            correlations=correlations
        )
        
        # Calculate without correlations
        allocations_no_corr = self.position_sizer.get_portfolio_allocations(
            signals=signals,
            portfolio_value=100000,
            current_regime=RegimeType.NEUTRAL
        )
        
        # Validate that correlations reduce total allocation
        total_with_corr = sum(alloc['allocation'] for alloc in allocations_with_corr.values())
        total_no_corr = sum(alloc['allocation'] for alloc in allocations_no_corr.values())
        
        self.assertLess(total_with_corr, total_no_corr)


if __name__ == '__main__':
    unittest.main()
