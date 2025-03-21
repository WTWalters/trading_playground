"""
Tests for the Risk Manager module.

This test suite validates the functionality of the RiskManager class
with a focus on:
1. Integration of regime detection, position sizing, and risk controls
2. Comprehensive risk management plan generation
3. Stress testing functionality
4. Portfolio-level monitoring and management

It includes test scenarios for various market conditions and strategy configurations.
"""

import unittest
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import pytest
from unittest.mock import Mock, patch

from src.market_analysis.regime_detection.detector import RegimeType
from src.market_analysis.regime_detection.enhanced_detector import EnhancedRegimeDetector
from src.market_analysis.parameter_management.position_sizing import KellyPositionSizer
from src.market_analysis.parameter_management.risk_controls import RiskControls
from src.market_analysis.parameter_management.risk_manager import (
    RiskManager, RiskManagementPlan
)


class TestRiskManager(unittest.TestCase):
    """Test cases for the RiskManager class."""
    
    def setUp(self):
        """Set up test fixtures."""
        # Create component instances
        self.regime_detector = EnhancedRegimeDetector(
            lookback_window=20,
            volatility_threshold=1.5,
            vix_threshold=20.0
        )
        
        self.position_sizer = KellyPositionSizer(
            default_kelly_fraction=0.5,
            max_position_pct=0.05
        )
        
        self.risk_controls = RiskControls(
            default_stop_loss_pct=0.20,
            default_take_profit_pct=0.40,
            max_drawdown_limit=0.25
        )
        
        # Initialize risk manager
        self.risk_manager = RiskManager(
            regime_detector=self.regime_detector,
            position_sizer=self.position_sizer,
            risk_controls=self.risk_controls,
            max_portfolio_risk=0.10,
            max_strategy_risk=0.04
        )
        
        # Generate sample data
        self.market_data = self._generate_sample_market_data()
        self.macro_data = self._generate_sample_macro_data()
        
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
        
        # Define sample strategy metrics
        self.strategy_metrics = {
            'win_rate': 0.58,
            'win_loss_ratio': 1.7,
            'volatility': 0.18,
            'correlation': 0.0,
            'sharpe': 1.3
        }
        
    def _generate_sample_market_data(self):
        """Generate sample market data for testing."""
        # Create date range
        end_date = datetime.now()
        start_date = end_date - timedelta(days=60)
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
            vix.append(max(10, vix[-1] * (1 + vix_change)))
        
        # Generate other macro indicators
        yield_curve = [0.5] * len(dates)
        # Make yield curve negative in the last third to simulate inversion
        yield_curve[-len(dates)//3:] = [-0.2] * (len(dates)//3)
        
        # Create interest rates
        interest_rate = np.linspace(2.0, 3.5, len(dates))
        
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
        
    def test_basic_risk_management_plan(self):
        """Test basic risk management plan functionality."""
        # Create a risk management plan
        plan = self.risk_manager.create_risk_management_plan(
            strategy_id="test_strategy",
            market_data=self.market_data,
            strategy_params=self.strategy_params,
            signal_strength=0.6,
            strategy_metrics=self.strategy_metrics,
            macro_data=self.macro_data,
            portfolio_value=100000
        )
        
        # Basic validations
        self.assertIsNotNone(plan)
        self.assertIsInstance(plan, RiskManagementPlan)
        self.assertEqual(plan.strategy_id, "test_strategy")
        self.assertIsNotNone(plan.regime_assessment)
        self.assertIsNotNone(plan.position_sizing)
        self.assertIsNotNone(plan.risk_adjustments)
        
        # Check plan consistency
        self.assertEqual(plan.regime_assessment.primary_regime, 
                        plan.position_sizing.regime)
        self.assertEqual(plan.regime_assessment.primary_regime, 
                        plan.risk_adjustments.regime)
        
    def test_regime_assessment(self):
        """Test regime assessment functionality."""
        # Assess market regime
        regime_result = self.risk_manager.assess_market_regime(
            market_data=self.market_data,
            macro_data=self.macro_data
        )
        
        # Validations
        self.assertIsNotNone(regime_result)
        self.assertIsNotNone(regime_result.primary_regime)
        self.assertIsNotNone(regime_result.secondary_regime)
        self.assertIsNotNone(regime_result.stability_score)
        self.assertIsNotNone(regime_result.transition_probability)
        
        # If macro data is provided, macro regimes should be detected
        self.assertIsNotNone(regime_result.macro_regime)
        self.assertIsNotNone(regime_result.sentiment_regime)
        
    def test_position_sizing_integration(self):
        """Test position sizing integration."""
        # Create a risk management plan
        plan = self.risk_manager.create_risk_management_plan(
            strategy_id="test_strategy",
            market_data=self.market_data,
            strategy_params=self.strategy_params,
            signal_strength=0.6,
            strategy_metrics=self.strategy_metrics,
            portfolio_value=100000
        )
        
        # Validation
        self.assertGreater(plan.position_sizing.position_pct, 0)
        self.assertLessEqual(plan.position_sizing.position_pct, 
                            self.position_sizer.max_position_pct)
        self.assertEqual(plan.position_sizing.position_size, 
                        plan.position_sizing.position_pct * 100000)
        
    def test_risk_control_integration(self):
        """Test risk control integration."""
        # Create a risk management plan
        plan = self.risk_manager.create_risk_management_plan(
            strategy_id="test_strategy",
            market_data=self.market_data,
            strategy_params=self.strategy_params,
            signal_strength=0.6,
            strategy_metrics=self.strategy_metrics,
            portfolio_value=100000
        )
        
        # Validation
        self.assertGreater(plan.risk_adjustments.stop_loss_pct, 0)
        self.assertGreater(plan.risk_adjustments.take_profit_pct, 
                          plan.risk_adjustments.stop_loss_pct)
        self.assertEqual(plan.risk_adjustments.risk_level, 
                        self.risk_controls.assess_risk_level(self.market_data))
                        
        # Check parameter adjustments
        self.assertNotEqual(plan.risk_adjustments.adjusted_params, 
                           self.strategy_params)
        
    def test_portfolio_risk_management(self):
        """Test portfolio-level risk management."""
        # Initialize portfolio state
        self.risk_manager.update_portfolio_state(
            portfolio_value=100000,
            current_drawdown=0.05,
            strategies={
                "strategy_1": {
                    "allocation": 0.2,
                    "drawdown": 0.1,
                    "volatility": 0.2
                },
                "strategy_2": {
                    "allocation": 0.3,
                    "drawdown": 0.15,
                    "volatility": 0.25
                }
            }
        )
        
        # Create risk management plan
        plan = self.risk_manager.create_risk_management_plan(
            strategy_id="strategy_3",
            market_data=self.market_data,
            strategy_params=self.strategy_params,
            signal_strength=0.6,
            strategy_metrics=self.strategy_metrics,
            portfolio_value=100000
        )
        
        # Validate portfolio risk considerations
        self.assertLessEqual(
            plan.position_sizing.position_pct,
            self.risk_manager.max_strategy_risk
        )
        
        # Total portfolio risk should be less than max
        total_allocation = 0.2 + 0.3 + plan.position_sizing.position_pct
        self.assertLessEqual(total_allocation, 
                            self.risk_manager.max_portfolio_risk * 1.1)  # Allow small buffer
        
    def test_stress_testing(self):
        """Test stress testing functionality."""
        # Run stress tests
        stress_results = self.risk_manager.run_stress_test(
            strategy_id="test_strategy",
            original_params=self.strategy_params,
            market_data=self.market_data,
            macro_data=self.macro_data
        )
        
        # Validation
        self.assertIsNotNone(stress_results)
        self.assertGreaterEqual(len(stress_results), 3)  # Should have multiple scenarios
        
        # Check common scenarios
        scenario_names = stress_results.keys()
        expected_scenarios = ["market_crash", "volatility_spike", "liquidity_crisis"]
        
        for scenario in expected_scenarios:
            self.assertIn(scenario, scenario_names)
            
        # Validate scenario specifics
        crash_result = stress_results.get("market_crash")
        self.assertIsNotNone(crash_result)
        self.assertLess(crash_result.position_sizing.position_pct, 
                       plan.position_sizing.position_pct)
        self.assertLess(crash_result.risk_adjustments.stop_loss_pct, 
                       plan.risk_adjustments.stop_loss_pct)
        
    def test_drawdown_constraints(self):
        """Test drawdown constraint enforcement."""
        # Update portfolio with high drawdown
        self.risk_manager.update_portfolio_state(
            portfolio_value=100000,
            current_drawdown=0.20,  # High drawdown
            strategies={
                "strategy_1": {
                    "allocation": 0.2,
                    "drawdown": 0.25,  # Extreme strategy drawdown
                    "volatility": 0.2
                }
            }
        )
        
        # Create risk management plan
        plan = self.risk_manager.create_risk_management_plan(
            strategy_id="strategy_1",
            market_data=self.market_data,
            strategy_params=self.strategy_params,
            signal_strength=0.6,
            strategy_metrics=self.strategy_metrics,
            portfolio_value=100000
        )
        
        # Validate drawdown constraints
        self.assertLess(plan.position_sizing.position_pct, 
                       self.position_sizer.max_position_pct / 2)
        self.assertGreater(len(plan.warning_messages), 0)
        self.assertTrue(any("drawdown" in msg.lower() for msg in plan.warning_messages))
        
    def test_trade_results_feedback(self):
        """Test incorporation of recent trade results."""
        # Update with recent losing trades
        self.risk_manager.update_portfolio_state(
            portfolio_value=100000,
            trade_results={"strategy_1": False, "strategy_1": False}
        )
        
        # Update again with one more loss
        self.risk_manager.update_portfolio_state(
            portfolio_value=95000,  # Portfolio value dropped
            trade_results={"strategy_1": False}
        )
        
        # Create risk management plan
        plan = self.risk_manager.create_risk_management_plan(
            strategy_id="strategy_1",
            market_data=self.market_data,
            strategy_params=self.strategy_params,
            signal_strength=0.6,
            strategy_metrics=self.strategy_metrics,
            portfolio_value=95000
        )
        
        # Validate psychological adjustments
        self.assertLess(plan.position_sizing.position_pct, 
                       self.position_sizer.max_position_pct / 2)
        self.assertTrue(any("losing streak" in msg.lower() for msg in plan.warning_messages))
        
    def test_correlation_constraints(self):
        """Test correlation constraint enforcement."""
        # Update portfolio with highly correlated strategies
        self.risk_manager.update_portfolio_state(
            portfolio_value=100000,
            strategies={
                "strategy_1": {
                    "allocation": 0.2,
                    "correlation": 0.8,  # High correlation with existing strategies
                    "volatility": 0.2
                },
                "strategy_2": {
                    "allocation": 0.3,
                    "correlation": 0.9,  # High correlation with existing strategies
                    "volatility": 0.25
                }
            }
        )
        
        # Create risk management plan with high correlation
        high_correlation_metrics = self.strategy_metrics.copy()
        high_correlation_metrics['correlation'] = 0.85  # High correlation
        
        plan = self.risk_manager.create_risk_management_plan(
            strategy_id="strategy_3",
            market_data=self.market_data,
            strategy_params=self.strategy_params,
            signal_strength=0.6,
            strategy_metrics=high_correlation_metrics,
            portfolio_value=100000
        )
        
        # Validate correlation constraints
        self.assertLess(plan.position_sizing.position_pct, self.position_sizer.max_position_pct / 2)
        self.assertTrue(any("correlation" in msg.lower() for msg in plan.warning_messages))
        
    def test_consistency_across_strategies(self):
        """Test consistency of regime assessment across strategies."""
        # Create plans for multiple strategies
        plan1 = self.risk_manager.create_risk_management_plan(
            strategy_id="strategy_1",
            market_data=self.market_data,
            strategy_params=self.strategy_params,
            signal_strength=0.6,
            strategy_metrics=self.strategy_metrics,
            portfolio_value=100000,
            macro_data=self.macro_data
        )
        
        plan2 = self.risk_manager.create_risk_management_plan(
            strategy_id="strategy_2",
            market_data=self.market_data,
            strategy_params=self.strategy_params,
            signal_strength=0.7,  # Different signal strength
            strategy_metrics={**self.strategy_metrics, 'win_rate': 0.62},  # Different metrics
            portfolio_value=100000,
            macro_data=self.macro_data
        )
        
        # Regime assessment should be the same for both
        self.assertEqual(plan1.regime_assessment.primary_regime, 
                        plan2.regime_assessment.primary_regime)
        self.assertEqual(plan1.regime_assessment.macro_regime, 
                        plan2.regime_assessment.macro_regime)
        
        # But position sizing and risk adjustments should differ
        self.assertNotEqual(plan1.position_sizing.position_pct, 
                           plan2.position_sizing.position_pct)
        
    def test_parameter_sensitivity(self):
        """Test sensitivity to input parameters."""
        base_signal = 0.6
        sensitivity_tests = [
            {'signal_strength': base_signal - 0.2},
            {'signal_strength': base_signal + 0.2},
            {'portfolio_value': 50000},
            {'portfolio_value': 200000},
            {'win_rate': 0.48},
            {'win_rate': 0.68},
        ]
        
        for test in sensitivity_tests:
            # Create modified inputs
            signal_strength = test.get('signal_strength', base_signal)
            portfolio_value = test.get('portfolio_value', 100000)
            strategy_metrics = self.strategy_metrics.copy()
            
            if 'win_rate' in test:
                strategy_metrics['win_rate'] = test['win_rate']
                
            # Create risk management plan
            plan = self.risk_manager.create_risk_management_plan(
                strategy_id="test_strategy",
                market_data=self.market_data,
                strategy_params=self.strategy_params,
                signal_strength=signal_strength,
                strategy_metrics=strategy_metrics,
                portfolio_value=portfolio_value
            )
            
            # Just validating that the plan is created without errors
            self.assertIsNotNone(plan)
            self.assertIsNotNone(plan.position_sizing)
            
    def test_risk_level_integration(self):
        """Test integration of risk level assessment."""
        # Create high volatility market data
        high_vol_data = self.market_data.copy()
        high_vol_data['volatility'] = high_vol_data['volatility'] * 3.0
        
        # Assess risk level directly
        risk_level = self.risk_controls.assess_risk_level(high_vol_data)
        
        # Create risk management plan
        plan = self.risk_manager.create_risk_management_plan(
            strategy_id="test_strategy",
            market_data=high_vol_data,
            strategy_params=self.strategy_params,
            signal_strength=0.6,
            strategy_metrics=self.strategy_metrics,
            portfolio_value=100000
        )
        
        # Validate risk level integration
        self.assertEqual(plan.risk_adjustments.risk_level, risk_level)
        
        # High risk should lead to smaller position size
        self.assertLessEqual(plan.position_sizing.position_pct, 
                            self.position_sizer.max_position_pct / 2)


if __name__ == '__main__':
    unittest.main()
