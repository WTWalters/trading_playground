"""
Tests for the Walk-Forward Testing Framework.

This test suite validates the functionality of walk-forward testing and optimization,
a key component for evaluating the adaptative parameter management system across
different market regimes.
"""

import unittest
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import tempfile
import os
from pathlib import Path

from src.market_analysis.regime_detection.detector import RegimeType
from src.market_analysis.parameter_management.parameters import AdaptiveParameterManager, ParameterSet


class TestWalkForwardFramework(unittest.TestCase):
    """Test cases for the walk-forward testing framework."""
    
    def setUp(self):
        """Set up test fixtures."""
        # Create temporary directory for parameter storage
        self.temp_dir = tempfile.TemporaryDirectory()
        self.storage_path = Path(self.temp_dir.name)
        
        # Define base parameters for testing
        self.base_parameters = {
            'window_size': 20,
            'z_score_threshold': 2.0,
            'max_position_size': 100,
            'stop_loss_pct': 0.05,
            'take_profit_pct': 0.10
        }
        
        # Define parameter ranges for optimization
        self.parameter_ranges = {
            'window_size': (10, 50, 5),
            'z_score_threshold': (1.0, 3.0, 0.1),
            'max_position_size': (50, 200, 10),
            'stop_loss_pct': (0.02, 0.10, 0.01),
            'take_profit_pct': (0.05, 0.20, 0.01)
        }
        
        # Create parameter manager
        self.manager = AdaptiveParameterManager(
            strategy_name='test_walk_forward',
            base_parameters=self.base_parameters,
            parameter_ranges=self.parameter_ranges,
            storage_path=self.storage_path
        )
        
        # Generate synthetic market data with regime shifts
        self.market_data = self._generate_market_data_with_regimes()
        
    def tearDown(self):
        """Clean up after tests."""
        self.temp_dir.cleanup()
        
    def _generate_market_data_with_regimes(self):
        """Generate synthetic market data with different regimes."""
        # Create date range
        end_date = datetime.now()
        start_date = end_date - timedelta(days=365)  # One year of data
        dates = pd.date_range(start=start_date, end=end_date, freq='D')
        
        # Generate price series with different regimes
        prices = [100.0]
        
        # Define regime segments (each ~3 months)
        segments = len(dates) // 4
        
        # Low volatility regime (first quarter)
        for i in range(1, segments):
            prices.append(prices[-1] * (1 + np.random.normal(0.0005, 0.005)))
        
        # Trending regime (second quarter)
        for i in range(segments, 2*segments):
            prices.append(prices[-1] * (1 + 0.001 + np.random.normal(0, 0.007)))
        
        # High volatility regime (third quarter)
        for i in range(2*segments, 3*segments):
            prices.append(prices[-1] * (1 + np.random.normal(0, 0.02)))
            
        # Mean-reverting regime (fourth quarter)
        center = prices[-1]
        amplitude = center * 0.1  # 10% oscillation
        period = 20
        for i in range(3*segments, len(dates)):
            oscillation = amplitude * np.sin(2 * np.pi * (i - 3*segments) / period)
            noise = np.random.normal(0, center * 0.005)
            prices.append(center + oscillation + noise)
        
        # Ensure we have the right length
        prices = prices[:len(dates)]
        
        # Create DataFrame
        df = pd.DataFrame({
            'close': prices,
            'open': [p * (1 + np.random.normal(0, 0.005)) for p in prices],
            'high': [p * (1 + abs(np.random.normal(0, 0.01))) for p in prices],
            'low': [p * (1 - abs(np.random.normal(0, 0.01))) for p in prices],
            'volume': np.random.lognormal(15, 0.5, len(dates))
        }, index=dates)
        
        # Add regime labels for validation
        regimes = []
        for i in range(len(dates)):
            if i < segments:
                regimes.append('low_volatility')
            elif i < 2*segments:
                regimes.append('trending')
            elif i < 3*segments:
                regimes.append('high_volatility')
            else:
                regimes.append('mean_reverting')
        
        df['regime'] = regimes
        
        # Calculate returns
        df['returns'] = df['close'].pct_change().fillna(0)
        
        return df
        
    def test_walk_forward_validation(self):
        """Test basic walk-forward validation functionality."""
        # Create a simple evaluation function
        def evaluate_parameters(params, data_window):
            """Simple evaluation function that prefers smaller window_size in high vol and larger in low vol."""
            window_size = params['window_size']
            z_score = params['z_score_threshold']
            
            # Calculate volatility
            volatility = data_window['returns'].std() * np.sqrt(252)
            
            # Calculate a score based on parameters and regime
            if 'regime' in data_window.columns:
                regime = data_window['regime'].iloc[-1]
                
                if regime == 'high_volatility':
                    # In high volatility, prefer lower window sizes and higher z-scores
                    window_score = 1.0 - (window_size - 10) / 40  # Higher score for smaller windows
                    z_score_score = (z_score - 1.0) / 2.0  # Higher score for higher z-scores
                elif regime == 'trending':
                    # In trending, prefer medium window sizes
                    window_score = 1.0 - abs(window_size - 30) / 20
                    z_score_score = 1.0 - abs(z_score - 1.5) / 1.5
                else:
                    # In low vol or mean-reverting, prefer larger window sizes and lower z-scores
                    window_score = (window_size - 10) / 40  # Higher score for larger windows
                    z_score_score = 1.0 - (z_score - 1.0) / 2.0  # Higher score for lower z-scores
            else:
                # Default scoring if no regime label
                window_score = 0.5
                z_score_score = 0.5
                
            # Calculate overall score
            score = 0.6 * window_score + 0.4 * z_score_score
            
            # Mock performance metrics
            return {
                'sharpe_ratio': 1.0 + score,
                'max_drawdown': 0.1 - 0.05 * score,
                'win_rate': 0.5 + 0.1 * score
            }
            
        # Create parameter sets optimized for different regimes
        regime_parameters = {
            RegimeType.HIGH_VOLATILITY: {
                'window_size': 15,
                'z_score_threshold': 2.5
            },
            RegimeType.TRENDING: {
                'window_size': 30,
                'z_score_threshold': 1.5
            },
            RegimeType.LOW_VOLATILITY: {
                'window_size': 40,
                'z_score_threshold': 1.2
            },
            RegimeType.MEAN_REVERTING: {
                'window_size': 35,
                'z_score_threshold': 1.0
            }
        }
        
        # Add parameter sets to the manager
        for regime, params in regime_parameters.items():
            # Create full parameter set with default values for missing parameters
            full_params = self.base_parameters.copy()
            full_params.update(params)
            
            parameter_set = ParameterSet(
                regime_type=regime,
                parameters=full_params,
                performance_metrics={'sharpe_ratio': 1.0, 'max_drawdown': 0.1}
            )
            
            self.manager.add_parameter_set(parameter_set)
            
        # Run walk-forward validation
        window_size = 60  # 60-day windows
        step_size = 30    # 30-day steps
        
        metrics, confidence = self.manager.walk_forward_validation(
            parameter_set=self.manager.parameter_sets[RegimeType.HIGH_VOLATILITY],
            evaluation_func=evaluate_parameters,
            data=self.market_data,
            window_size=window_size,
            step_size=step_size
        )
        
        # Check that metrics were returned
        self.assertIsNotNone(metrics)
        self.assertIn('sharpe_ratio', metrics)
        self.assertIn('max_drawdown', metrics)
        self.assertIn('win_rate', metrics)
        
        # Check confidence score is reasonable
        self.assertGreaterEqual(confidence, 0.0)
        self.assertLessEqual(confidence, 1.0)
        
    def test_regime_specific_optimization(self):
        """Test that parameters are optimized differently for different regimes."""
        # Create regime-specific datasets
        low_vol_data = self.market_data[self.market_data['regime'] == 'low_volatility']
        high_vol_data = self.market_data[self.market_data['regime'] == 'high_volatility']
        trending_data = self.market_data[self.market_data['regime'] == 'trending']
        mean_rev_data = self.market_data[self.market_data['regime'] == 'mean_reverting']
        
        # Store current regime for the evaluation function
        global current_regime
        current_regime = RegimeType.LOW_VOLATILITY
        
        # Evaluation function that has different optimal parameters for each regime
        def evaluate_parameters(params):
            window_size = params['window_size']
            z_score = params['z_score_threshold']
            
            # Different optimal parameters for each regime
            if current_regime == RegimeType.HIGH_VOLATILITY:
                # High volatility prefers small windows and high z-scores
                window_score = 1.0 - (window_size - 10) / 40
                z_score_score = (z_score - 1.0) / 2.0
            elif current_regime == RegimeType.TRENDING:
                # Trending prefers medium windows and medium z-scores
                window_score = 1.0 - abs(window_size - 30) / 20
                z_score_score = 1.0 - abs(z_score - 1.5) / 1.5
            elif current_regime == RegimeType.MEAN_REVERTING:
                # Mean-reverting prefers larger windows and low z-scores
                window_score = (window_size - 10) / 40
                z_score_score = 1.0 - (z_score - 1.0) / 2.0
            else:  # Low volatility
                # Low volatility prefers larger windows and medium z-scores
                window_score = (window_size - 10) / 40
                z_score_score = 1.0 - abs(z_score - 1.8) / 1.8
                
            # Calculate score
            score = 0.6 * window_score + 0.4 * z_score_score
            return 1.0 + score  # Higher is better
        
        # Optimize for each regime
        regime_data_map = {
            RegimeType.LOW_VOLATILITY: low_vol_data,
            RegimeType.HIGH_VOLATILITY: high_vol_data,
            RegimeType.TRENDING: trending_data,
            RegimeType.MEAN_REVERTING: mean_rev_data
        }
        
        optimized_params = {}
        
        for regime, data in regime_data_map.items():
            # Skip if not enough data
            if len(data) < 30:
                continue
                
            # Set current regime for the evaluation function
            current_regime = regime
            
            # Optimize parameters for this regime
            param_set = self.manager.optimize_parameters(
                regime=regime,
                evaluation_func=evaluate_parameters,
                method="grid_search",
                n_iterations=20
            )
            
            optimized_params[regime] = param_set.parameters
        
        # Check that parameters were optimized differently for each regime
        if RegimeType.HIGH_VOLATILITY in optimized_params and RegimeType.LOW_VOLATILITY in optimized_params:
            # High volatility should have smaller window size than low volatility
            self.assertLess(
                optimized_params[RegimeType.HIGH_VOLATILITY]['window_size'],
                optimized_params[RegimeType.LOW_VOLATILITY]['window_size']
            )
            
            # High volatility should have larger z-score threshold than low volatility
            self.assertGreater(
                optimized_params[RegimeType.HIGH_VOLATILITY]['z_score_threshold'],
                optimized_params[RegimeType.LOW_VOLATILITY]['z_score_threshold']
            )
        
        if RegimeType.TRENDING in optimized_params and RegimeType.MEAN_REVERTING in optimized_params:
            # Trending should have different z-score threshold than mean-reverting
            self.assertNotEqual(
                optimized_params[RegimeType.TRENDING]['z_score_threshold'],
                optimized_params[RegimeType.MEAN_REVERTING]['z_score_threshold']
            )
            
    def test_walk_forward_performance(self):
        """Test walk-forward performance with static vs. adaptive parameters."""
        # Define synthetic trading strategy for testing
        def run_strategy(params, data_window):
            """Run a simple mean-reversion trading strategy and return metrics."""
            window_size = params['window_size']
            z_score_threshold = params['z_score_threshold']
            max_position_size = params['max_position_size']
            stop_loss_pct = params['stop_loss_pct']
            take_profit_pct = params['take_profit_pct']
            
            # Calculate moving average and standard deviation
            if len(data_window) <= window_size:
                return {'sharpe_ratio': 0.0, 'max_drawdown': 1.0, 'win_rate': 0.0, 'total_return': 0.0}
                
            rolling_mean = data_window['close'].rolling(window=window_size).mean()
            rolling_std = data_window['close'].rolling(window=window_size).std()
            
            # Calculate z-scores
            z_scores = (data_window['close'] - rolling_mean) / rolling_std
            
            # Generate signals
            # Long when price is below mean by z_score_threshold
            long_signals = z_scores < -z_score_threshold
            # Short when price is above mean by z_score_threshold
            short_signals = z_scores > z_score_threshold
            # Exit when price crosses back to mean
            exit_signals = abs(z_scores) < 0.5
            
            # Initialize portfolio
            position = 0
            entry_price = 0
            equity = [1000.0]  # Start with $1000
            trades = []
            
            # Simulate trading
            for i in range(window_size + 1, len(data_window)):
                current_price = data_window['close'].iloc[i]
                prev_price = data_window['close'].iloc[i-1]
                
                # Check for stop loss or take profit if in position
                if position != 0:
                    price_change = (current_price / entry_price - 1) * position  # Positive if profitable
                    
                    # Check stop loss
                    if price_change < -stop_loss_pct:
                        # Exit position due to stop loss
                        pnl = -stop_loss_pct * position  # Negative for long, positive for short
                        position = 0
                        trades.append((pnl, "stop_loss"))
                    
                    # Check take profit
                    elif price_change > take_profit_pct:
                        # Exit position due to take profit
                        pnl = take_profit_pct * position  # Positive for long, negative for short
                        position = 0
                        trades.append((pnl, "take_profit"))
                
                # Check for exit signal
                if position != 0 and exit_signals.iloc[i]:
                    # Calculate P&L
                    pnl = (current_price / entry_price - 1) * position
                    position = 0
                    trades.append((pnl, "signal"))
                
                # Check for entry signals if not in position
                elif position == 0:
                    if long_signals.iloc[i]:
                        position = 1
                        entry_price = current_price
                    elif short_signals.iloc[i]:
                        position = -1
                        entry_price = current_price
                
                # Update equity
                if position != 0:
                    # Mark-to-market P&L
                    daily_return = (current_price / prev_price - 1) * position
                    equity.append(equity[-1] * (1 + daily_return * max_position_size/100))
                else:
                    equity.append(equity[-1])
            
            # Calculate performance metrics
            returns = np.diff(equity) / equity[:-1]
            sharpe = np.mean(returns) / np.std(returns) * np.sqrt(252) if np.std(returns) > 0 else 0
            max_drawdown = 0
            peak = equity[0]
            
            for value in equity:
                if value > peak:
                    peak = value
                drawdown = (peak - value) / peak
                max_drawdown = max(max_drawdown, drawdown)
            
            win_rate = sum(1 for pnl, reason in trades if pnl > 0) / len(trades) if trades else 0
            total_return = (equity[-1] / equity[0]) - 1
            
            return {
                'sharpe_ratio': sharpe,
                'max_drawdown': max_drawdown,
                'win_rate': win_rate,
                'total_return': total_return
            }
        
        # Create parameter sets for each regime
        regime_parameters = {
            'low_volatility': {
                'window_size': 40,
                'z_score_threshold': 1.2,
                'max_position_size': 100,
                'stop_loss_pct': 0.05,
                'take_profit_pct': 0.10
            },
            'high_volatility': {
                'window_size': 15,
                'z_score_threshold': 2.5,
                'max_position_size': 50,
                'stop_loss_pct': 0.08,
                'take_profit_pct': 0.12
            },
            'trending': {
                'window_size': 30,
                'z_score_threshold': 1.8,
                'max_position_size': 80,
                'stop_loss_pct': 0.07,
                'take_profit_pct': 0.15
            },
            'mean_reverting': {
                'window_size': 25,
                'z_score_threshold': 1.0,
                'max_position_size': 120,
                'stop_loss_pct': 0.04,
                'take_profit_pct': 0.08
            }
        }
        
        # Add parameter sets to the manager
        for regime_name, params in regime_parameters.items():
            # Map string regime name to RegimeType
            if regime_name == 'low_volatility':
                regime_type = RegimeType.LOW_VOLATILITY
            elif regime_name == 'high_volatility':
                regime_type = RegimeType.HIGH_VOLATILITY
            elif regime_name == 'trending':
                regime_type = RegimeType.TRENDING
            elif regime_name == 'mean_reverting':
                regime_type = RegimeType.MEAN_REVERTING
            else:
                continue
                
            parameter_set = ParameterSet(
                regime_type=regime_type,
                parameters=params,
                performance_metrics={}
            )
            
            self.manager.add_parameter_set(parameter_set)
        
        # Set up walk-forward test
        window_size = 60  # 60-day windows
        step_size = 30    # 30-day steps
        
        # Run with static parameters (use low_volatility as default)
        static_results = []
        
        for i in range(0, len(self.market_data) - window_size, step_size):
            test_window = self.market_data.iloc[i:i+window_size]
            static_metrics = run_strategy(regime_parameters['low_volatility'], test_window)
            static_results.append(static_metrics)
        
        # Run with adaptive parameters
        adaptive_results = []
        
        for i in range(0, len(self.market_data) - window_size, step_size):
            test_window = self.market_data.iloc[i:i+window_size]
            
            # Determine the current regime
            regime_name = test_window['regime'].iloc[-1]
            
            # Map to RegimeType
            if regime_name == 'low_volatility':
                regime_type = RegimeType.LOW_VOLATILITY
            elif regime_name == 'high_volatility':
                regime_type = RegimeType.HIGH_VOLATILITY
            elif regime_name == 'trending':
                regime_type = RegimeType.TRENDING
            elif regime_name == 'mean_reverting':
                regime_type = RegimeType.MEAN_REVERTING
            else:
                regime_type = RegimeType.LOW_VOLATILITY  # Default
                
            # Get parameters for this regime
            if regime_type in self.manager.parameter_sets:
                params = self.manager.parameter_sets[regime_type].parameters
            else:
                params = regime_parameters['low_volatility']
                
            # Run strategy with adaptive parameters
            adaptive_metrics = run_strategy(params, test_window)
            adaptive_results.append(adaptive_metrics)
        
        # Calculate average performance metrics
        static_avg = {
            'sharpe_ratio': np.mean([r['sharpe_ratio'] for r in static_results]),
            'max_drawdown': np.mean([r['max_drawdown'] for r in static_results]),
            'win_rate': np.mean([r['win_rate'] for r in static_results]),
            'total_return': np.mean([r['total_return'] for r in static_results])
        }
        
        adaptive_avg = {
            'sharpe_ratio': np.mean([r['sharpe_ratio'] for r in adaptive_results]),
            'max_drawdown': np.mean([r['max_drawdown'] for r in adaptive_results]),
            'win_rate': np.mean([r['win_rate'] for r in adaptive_results]),
            'total_return': np.mean([r['total_return'] for r in adaptive_results])
        }
        
        # Just verify we got valid metrics back
        self.assertIsInstance(adaptive_avg['sharpe_ratio'], float)
        self.assertIsInstance(static_avg['sharpe_ratio'], float)
        self.assertIsInstance(adaptive_avg['max_drawdown'], float)
        self.assertIsInstance(static_avg['max_drawdown'], float)
        
    def test_fine_tuning_in_walk_forward(self):
        """Test fine-tuning parameters within walk-forward framework."""
        # Initial parameter set
        initial_params = {
            'window_size': 20,
            'z_score_threshold': 2.0,
            'max_position_size': 100,
            'stop_loss_pct': 0.05,
            'take_profit_pct': 0.10
        }
        
        # Add initial parameter set
        initial_set = ParameterSet(
            regime_type=RegimeType.LOW_VOLATILITY,
            parameters=initial_params,
            performance_metrics={}
        )
        
        self.manager.add_parameter_set(initial_set)
        
        # Define evaluation function
        def evaluate_parameters(params, data_window):
            """Evaluation function that prefers specific parameters for each volatility level."""
            window_size = params['window_size']
            z_score = params['z_score_threshold']
            
            # Calculate volatility
            volatility = data_window['returns'].std() * np.sqrt(252)
            
            # Define optimal parameters based on volatility
            if volatility > 0.3:  # High volatility
                optimal_window = 15
                optimal_z_score = 2.5
            elif volatility > 0.15:  # Medium volatility
                optimal_window = 25
                optimal_z_score = 2.0
            else:  # Low volatility
                optimal_window = 40
                optimal_z_score = 1.5
                
            # Calculate score based on distance from optimal
            window_score = 1.0 - abs(window_size - optimal_window) / 30
            z_score_score = 1.0 - abs(z_score - optimal_z_score) / 1.5
            
            # Overall score
            score = 0.6 * window_score + 0.4 * z_score_score
            
            return {
                'sharpe_ratio': 1.0 + score,
                'max_drawdown': 0.1 - 0.05 * score,
                'win_rate': 0.5 + 0.1 * score
            }
        
        # Set up walk-forward framework
        window_size = 60
        step_size = 30
        
        # First, validate current parameters
        metrics, confidence = self.manager.walk_forward_validation(
            parameter_set=self.manager.parameter_sets[RegimeType.LOW_VOLATILITY],
            evaluation_func=evaluate_parameters,
            data=self.market_data,
            window_size=window_size,
            step_size=step_size
        )
        
        initial_sharpe = metrics['sharpe_ratio']
        
        # Fine-tune parameters
        fine_tuned_set = self.manager.fine_tune_parameters(
            regime=RegimeType.LOW_VOLATILITY,
            evaluation_func=lambda params: evaluate_parameters(params, self.market_data.iloc[-60:])
        )
        
        # Run validation again with fine-tuned parameters
        metrics, confidence = self.manager.walk_forward_validation(
            parameter_set=fine_tuned_set,
            evaluation_func=evaluate_parameters,
            data=self.market_data,
            window_size=window_size,
            step_size=step_size
        )
        
        tuned_sharpe = metrics['sharpe_ratio']
        
        # Just verify we got valid metrics back
        self.assertIsInstance(tuned_sharpe, float)
        self.assertIsInstance(initial_sharpe, float)
        
        # Make sure fine-tuned parameters exist with expected attributes
        self.assertIsNotNone(fine_tuned_set)
        self.assertIn('window_size', fine_tuned_set.parameters)
        self.assertIn('z_score_threshold', fine_tuned_set.parameters)


if __name__ == '__main__':
    unittest.main()
