"""
Tests for the Integration module.

This test suite validates the functionality of the AdaptiveParameterManager class
with a focus on:
1. Central control of adaptive parameters
2. Strategy registration and parameter optimization
3. Performance tracking and feedback system
4. Portfolio allocation optimization
5. Integration of all components (regime detection, position sizing, risk controls)

It includes comprehensive test scenarios for the entire adaptive parameter management system.
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
from src.market_analysis.parameter_management.risk_manager import RiskManager
from src.market_analysis.parameter_management.integration import AdaptiveParameterManager


class TestAdaptiveParameterManager(unittest.TestCase):
    """Test cases for the AdaptiveParameterManager class."""
    
    def setUp(self):
        """Set up test fixtures."""
        # Create component instances
        regime_detector = EnhancedRegimeDetector(
            lookback_window=20,
            volatility_threshold=1.5,
            vix_threshold=20.0
        )
        
        position_sizer = KellyPositionSizer(
            default_kelly_fraction=0.5,
            max_position_pct=0.05
        )
        
        risk_controls = RiskControls(
            default_stop_loss_pct=0.20,
            default_take_profit_pct=0.40,
            max_drawdown_limit=0.25
        )
        
        risk_manager = RiskManager(
            regime_detector=regime_detector,
            position_sizer=position_sizer,
            risk_controls=risk_controls,
            max_portfolio_risk=0.10,
            max_strategy_risk=0.04
        )
        
        # Initialize adaptive parameter manager
        self.param_manager = AdaptiveParameterManager(
            risk_manager=risk_manager,
            config={
                'max_portfolio_risk': 0.12,
                'max_strategy_risk': 0.04, 
                'max_concentration': 0.25,
                'min_strategies': 3,
                'max_drawdown': 0.25,
                'default_stop_loss_pct': 0.20,
                'default_take_profit_pct': 0.40,
                'backtesting_mode': True  # Enable backtesting mode for testing
            }
        )
        
        # Generate sample data
        self.market_data = self._generate_sample_market_data()
        self.macro_data = self._generate_sample_macro_data()
        
        # Define sample strategy parameters
        self.strategy1_params = {
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
        
        # Create a second set of strategy parameters
        self.strategy2_params = dict(self.strategy1_params)
        self.strategy2_params.update({
            'entry_threshold': 0.8,
            'z_entry': 2.5,
            'stop_loss_pct': 0.10,
        })
        
        # Define strategy metrics
        self.strategy1_metrics = {
            'win_rate': 0.58,
            'win_loss_ratio': 1.7,
            'volatility': 0.18,
            'correlation': 0.0,
            'sharpe': 1.3
        }
        
        self.strategy2_metrics = {
            'win_rate': 0.52,
            'win_loss_ratio': 2.1,
            'volatility': 0.15,
            'correlation': 0.3,
            'sharpe': 1.6
        }
        
        # Register strategies
        self.param_manager.register_strategy(
            strategy_id="mean_reversion_etf",
            base_parameters=self.strategy1_params,
            strategy_metrics=self.strategy1_metrics
        )
        
        self.param_manager.register_strategy(
            strategy_id="momentum_sectors",
            base_parameters=self.strategy2_params,
            strategy_metrics=self.strategy2_metrics
        )
        
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
        
    def test_strategy_registration(self):
        """Test strategy registration functionality."""
        # Test if strategies were registered correctly
        self.assertIn("mean_reversion_etf", self.param_manager.strategy_parameters)
        self.assertIn("momentum_sectors", self.param_manager.strategy_parameters)
        
        # Validate parameter storage
        stored_params = self.param_manager.strategy_parameters["mean_reversion_etf"]["base_parameters"]
        self.assertEqual(stored_params, self.strategy1_params)
        
        # Validate metrics storage
        stored_metrics = self.param_manager.strategy_parameters["mean_reversion_etf"]["metrics"]
        self.assertEqual(stored_metrics, self.strategy1_metrics)
        
    def test_market_state_update(self):
        """Test market state update functionality."""
        # Update market state
        result = self.param_manager.update_market_state(
            market_data=self.market_data,
            portfolio_value=1000000.0,
            macro_data=self.macro_data
        )
        
        # Basic validations
        self.assertIsNotNone(result)
        self.assertIsNotNone(result.primary_regime)
        self.assertIsNotNone(self.param_manager.current_market_state)
        self.assertEqual(self.param_manager.current_market_state['portfolio_value'], 1000000.0)
        
        # Validate that market data was stored
        self.assertIn('market_data', self.param_manager.current_market_state)
        self.assertIn('macro_data', self.param_manager.current_market_state)
        
    def test_parameter_optimization(self):
        """Test parameter optimization functionality."""
        # Update market state first
        self.param_manager.update_market_state(
            market_data=self.market_data,
            portfolio_value=1000000.0,
            macro_data=self.macro_data
        )
        
        # Get optimized parameters
        optimized_params = self.param_manager.get_optimized_parameters(
            strategy_id="mean_reversion_etf",
            signal_strength=0.65
        )
        
        # Basic validations
        self.assertIsNotNone(optimized_params)
        self.assertIn('position_size_pct', optimized_params)
        self.assertIn('position_size', optimized_params)
        self.assertIn('stop_loss_pct', optimized_params)
        self.assertIn('take_profit_pct', optimized_params)
        
        # Validate that parameters were adapted based on market conditions
        self.assertNotEqual(optimized_params['z_entry'], self.strategy1_params['z_entry'])
        
        # Test different signal strength
        weak_signal_params = self.param_manager.get_optimized_parameters(
            strategy_id="mean_reversion_etf",
            signal_strength=0.3  # Weak signal
        )
        
        # Weaker signal should lead to smaller position size
        self.assertLess(weak_signal_params['position_size_pct'], 
                       optimized_params['position_size_pct'])
        
    def test_parameter_overrides(self):
        """Test parameter override functionality."""
        # Update market state
        self.param_manager.update_market_state(
            market_data=self.market_data,
            portfolio_value=1000000.0,
            macro_data=self.macro_data
        )
        
        # Define overrides
        overrides = {
            'z_entry': 3.0,
            'max_holding_period': 5
        }
        
        # Get optimized parameters with overrides
        optimized_params = self.param_manager.get_optimized_parameters(
            strategy_id="mean_reversion_etf",
            signal_strength=0.65,
            override_parameters=overrides
        )
        
        # Validate that overrides were applied
        self.assertEqual(optimized_params['z_entry'], 3.0)
        self.assertEqual(optimized_params['max_holding_period'], 5)
        
    def test_performance_update(self):
        """Test performance update functionality."""
        # Update market state
        self.param_manager.update_market_state(
            market_data=self.market_data,
            portfolio_value=1000000.0,
            macro_data=self.macro_data
        )
        
        # Add some trade results
        trade_metrics = {
            'pnl': 0.05,
            'pnl_pct': 0.05,
            'duration': 3,
            'regime': 'neutral',
            'timestamp': datetime.now()
        }
        
        # Update with a winning trade
        self.param_manager.update_performance(
            strategy_id="mean_reversion_etf",
            trade_result=True,
            trade_metrics=trade_metrics
        )
        
        # Update with a losing trade
        losing_metrics = trade_metrics.copy()
        losing_metrics['pnl'] = -0.03
        losing_metrics['pnl_pct'] = -0.03
        
        self.param_manager.update_performance(
            strategy_id="mean_reversion_etf",
            trade_result=False,
            trade_metrics=losing_metrics
        )
        
        # Validate performance history
        strategy_info = self.param_manager.strategy_parameters["mean_reversion_etf"]
        self.assertEqual(len(strategy_info['performance_history']), 2)
        
        # First trade should be a win
        self.assertTrue(strategy_info['performance_history'][0]['result'])
        
        # Second trade should be a loss
        self.assertFalse(strategy_info['performance_history'][1]['result'])
        
        # Check metrics update (we need at least 5 trades to update metrics, so this should not change much)
        # Ensuring the method doesn't crash is sufficient for this test
        self.assertIsNotNone(strategy_info['metrics'])
        
    def test_stress_testing(self):
        """Test stress testing functionality."""
        # Update market state
        self.param_manager.update_market_state(
            market_data=self.market_data,
            portfolio_value=1000000.0,
            macro_data=self.macro_data
        )
        
        # Run stress tests
        stress_results = self.param_manager.run_stress_test(
            strategy_id="mean_reversion_etf"
        )
        
        # Basic validations
        self.assertIsNotNone(stress_results)
        self.assertGreaterEqual(len(stress_results), 3)  # Should have multiple scenarios
        
        # Test specific scenario
        self.assertIn('market_crash', stress_results)
        crash_scenario = stress_results['market_crash']
        
        # Validate scenario format
        self.assertIn('regime', crash_scenario)
        self.assertIn('position_size', crash_scenario)
        self.assertIn('stop_loss', crash_scenario)
        self.assertIn('take_profit', crash_scenario)
        
        # Crash scenario should be more conservative
        normal_params = self.param_manager.get_optimized_parameters(
            strategy_id="mean_reversion_etf",
            signal_strength=0.65
        )
        
        self.assertLess(crash_scenario['position_size'], normal_params['position_size_pct'])
        self.assertLess(crash_scenario['stop_loss'], normal_params['stop_loss_pct'])
        
    def test_portfolio_allocation(self):
        """Test portfolio allocation functionality."""
        # Update market state
        self.param_manager.update_market_state(
            market_data=self.market_data,
            portfolio_value=1000000.0,
            macro_data=self.macro_data
        )
        
        # Get portfolio allocation
        allocations = self.param_manager.get_portfolio_allocation(
            strategy_ids=["mean_reversion_etf", "momentum_sectors"],
            portfolio_value=1000000.0,
            max_allocation=0.9
        )
        
        # Basic validations
        self.assertIsNotNone(allocations)
        self.assertEqual(len(allocations), 2)
        self.assertIn("mean_reversion_etf", allocations)
        self.assertIn("momentum_sectors", allocations)
        
        # Validate allocation format
        self.assertIn('allocation', allocations["mean_reversion_etf"])
        self.assertIn('position_size', allocations["mean_reversion_etf"])
        
        # Test allocation constraints
        total_allocation = sum(alloc['allocation'] for alloc in allocations.values())
        self.assertLessEqual(total_allocation, 0.9)  # Max allocation constraint
        
    def test_risk_profile(self):
        """Test risk profile functionality."""
        # Update market state
        self.param_manager.update_market_state(
            market_data=self.market_data,
            portfolio_value=1000000.0,
            macro_data=self.macro_data
        )
        
        # Add some performance history
        for i in range(5):
            # Winning trade
            trade_metrics = {
                'pnl': 0.05,
                'pnl_pct': 0.05,
                'duration': 3,
                'regime': 'neutral',
                'timestamp': datetime.now() - timedelta(days=i)
            }
            
            self.param_manager.update_performance(
                strategy_id="mean_reversion_etf",
                trade_result=True,
                trade_metrics=trade_metrics
            )
            
            # Losing trade for momentum strategy
            losing_metrics = trade_metrics.copy()
            losing_metrics['pnl'] = -0.03
            losing_metrics['pnl_pct'] = -0.03
            
            self.param_manager.update_performance(
                strategy_id="momentum_sectors",
                trade_result=False,
                trade_metrics=losing_metrics
            )
        
        # Get risk profiles
        risk_profile1 = self.param_manager.get_strategy_risk_profile("mean_reversion_etf")
        risk_profile2 = self.param_manager.get_strategy_risk_profile("momentum_sectors")
        
        # Basic validations
        self.assertIsNotNone(risk_profile1)
        self.assertIsNotNone(risk_profile2)
        
        # Validate risk profile format
        self.assertIn('metrics', risk_profile1)
        self.assertIn('current_drawdown', risk_profile1)
        self.assertIn('risk_level', risk_profile1)
        
        # Winning strategy should have lower risk than losing strategy
        self.assertNotEqual(risk_profile1['risk_level'], risk_profile2['risk_level'])
        
    def test_end_to_end_workflow(self):
        """Test the complete adaptive parameter management workflow."""
        # Define a complete workflow:
        # 1. Update market state
        # 2. Get optimized parameters
        # 3. Simulate some trades
        # 4. Update performance
        # 5. Get updated parameters
        
        # Step 1: Update market state
        self.param_manager.update_market_state(
            market_data=self.market_data,
            portfolio_value=1000000.0,
            macro_data=self.macro_data
        )
        
        # Step 2: Get initial optimized parameters
        initial_params = self.param_manager.get_optimized_parameters(
            strategy_id="mean_reversion_etf",
            signal_strength=0.65
        )
        
        # Step 3 & 4: Simulate trades and update performance
        # Simulate a winning streak
        for i in range(3):
            trade_metrics = {
                'pnl': 0.05 + 0.01 * i,  # Increasing profits
                'pnl_pct': 0.05 + 0.01 * i,
                'duration': 3,
                'regime': str(self.param_manager.current_market_state['regime'].primary_regime),
                'timestamp': datetime.now() - timedelta(days=i)
            }
            
            self.param_manager.update_performance(
                strategy_id="mean_reversion_etf",
                trade_result=True,
                trade_metrics=trade_metrics
            )
            
        # Step 5: Get updated parameters
        updated_params = self.param_manager.get_optimized_parameters(
            strategy_id="mean_reversion_etf",
            signal_strength=0.65
        )
        
        # Validate workflow results
        # After a winning streak, we might expect larger position sizes but more conservative stops
        # due to psychological factors (overconfidence)
        
        # Check for warnings related to psychological factors
        self.assertTrue('warnings' in updated_params)
        warnings = updated_params.get('warnings', [])
        self.assertTrue(any('confidence' in warning.lower() for warning in warnings))
        
    def test_error_handling(self):
        """Test error handling in the adaptive parameter manager."""
        # Test non-existent strategy
        with self.assertRaises(ValueError):
            self.param_manager.get_optimized_parameters(
                strategy_id="non_existent_strategy",
                signal_strength=0.5
            )
            
        # Test missing market state
        # Reset market state
        self.param_manager.current_market_state = {}
        
        # Should return base parameters without error
        params = self.param_manager.get_optimized_parameters(
            strategy_id="mean_reversion_etf",
            signal_strength=0.5
        )
        
        # Validate that it returned base parameters
        self.assertEqual(params['z_entry'], self.strategy1_params['z_entry'])
        
    def test_parameter_adaptation_across_regimes(self):
        """Test parameter adaptation across different market regimes."""
        # Create data for different regimes
        regimes_data = {
            "trending": self._create_trending_market(),
            "mean_reverting": self._create_mean_reverting_market(),
            "high_volatility": self._create_high_volatility_market()
        }
        
        params_by_regime = {}
        
        # Test each regime
        for regime_name, data in regimes_data.items():
            # Update market state with this regime's data
            self.param_manager.update_market_state(
                market_data=data['market'],
                portfolio_value=1000000.0,
                macro_data=data['macro']
            )
            
            # Get optimized parameters for this regime
            params = self.param_manager.get_optimized_parameters(
                strategy_id="mean_reversion_etf",
                signal_strength=0.65
            )
            
            # Store parameters
            params_by_regime[regime_name] = params
            
        # Validate regime-specific adaptations
        # High volatility should have more conservative position sizing
        self.assertLess(
            params_by_regime["high_volatility"]["position_size_pct"],
            params_by_regime["trending"]["position_size_pct"]
        )
        
        # Mean reverting should have optimized z-scores
        self.assertNotEqual(
            params_by_regime["mean_reverting"]["z_entry"],
            params_by_regime["trending"]["z_entry"]
        )
        
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
            'volume': [1000000 + np.random.normal(0, 50000)] * 60,
            'volatility': [0.01 + np.random.normal(0, 0.002) for _ in range(60)]
        }, index=dates)
        
        # Low VIX due to trending market
        vix = [12 + np.random.normal(0, 1) for _ in range(60)]
        
        macro_data = pd.DataFrame({
            'VIX': vix,
            'yield_curve': [0.5 + np.random.normal(0, 0.05) for _ in range(60)],
            'SPX': [p * 30 + 3000 for p in prices],
            'interest_rate': np.linspace(2.0, 2.1, 60),
            'USD_index': 100 + np.cumsum(np.random.normal(0, 0.001, 60))
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
            'volume': [800000 + np.random.normal(0, 50000)] * 60,
            'volatility': [0.015 + np.random.normal(0, 0.002) for _ in range(60)]
        }, index=dates)
        
        # Moderate VIX
        vix = [15 + np.random.normal(0, 2) for _ in range(60)]
        
        macro_data = pd.DataFrame({
            'VIX': vix,
            'yield_curve': [0.2 + np.random.normal(0, 0.05) for _ in range(60)],
            'SPX': [3000 + 300 * np.sin(2 * np.pi * i / period) + np.random.normal(0, 30) 
                   for i in range(60)],
            'interest_rate': np.linspace(2.5, 2.6, 60),
            'USD_index': 100 + np.cumsum(np.random.normal(0, 0.001, 60))
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
            'volume': [1500000 + np.random.normal(0, 200000)] * 60,
            'volatility': [0.03 + np.random.normal(0, 0.005) for _ in range(60)]
        }, index=dates)
        
        # High VIX
        vix = [30 + np.random.normal(0, 5) for _ in range(60)]
        
        macro_data = pd.DataFrame({
            'VIX': vix,
            'yield_curve': [-0.1 + np.random.normal(0, 0.05) for _ in range(60)],
            'SPX': [3000 + np.random.normal(0, 100) for _ in range(60)],
            'interest_rate': np.linspace(3.0, 3.5, 60),
            'USD_index': 100 + np.cumsum(np.random.normal(0, 0.002, 60))
        }, index=dates)
        
        return {'market': market_data, 'macro': macro_data}


if __name__ == '__main__':
    unittest.main()
