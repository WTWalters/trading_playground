"""
Tests for the Adaptive Parameter Management module.
"""

import os
import tempfile
import unittest
import pandas as pd
import numpy as np
from pathlib import Path

from src.market_analysis.regime_detection.detector import RegimeType
from src.market_analysis.parameter_management.parameters import AdaptiveParameterManager, ParameterSet

class TestAdaptiveParameterManager(unittest.TestCase):
    """Test cases for AdaptiveParameterManager class."""
    
    def setUp(self):
        """Set up test fixtures."""
        # Create a temporary directory for parameter storage
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
            strategy_name='test_strategy',
            base_parameters=self.base_parameters,
            parameter_ranges=self.parameter_ranges,
            storage_path=self.storage_path
        )
        
        # Create sample parameter sets for different regimes
        self.high_vol_params = ParameterSet(
            regime_type=RegimeType.HIGH_VOLATILITY,
            parameters={
                'window_size': 15,
                'z_score_threshold': 2.5,
                'max_position_size': 50,
                'stop_loss_pct': 0.07,
                'take_profit_pct': 0.15
            },
            performance_metrics={'sharpe_ratio': 1.2, 'max_drawdown': 0.12}
        )
        
        self.mean_rev_params = ParameterSet(
            regime_type=RegimeType.MEAN_REVERTING,
            parameters={
                'window_size': 25,
                'z_score_threshold': 1.8,
                'max_position_size': 120,
                'stop_loss_pct': 0.04,
                'take_profit_pct': 0.08
            },
            performance_metrics={'sharpe_ratio': 1.5, 'max_drawdown': 0.08}
        )
        
    def tearDown(self):
        """Clean up after tests."""
        self.temp_dir.cleanup()
        
    def test_add_parameter_set(self):
        """Test adding parameter set to manager."""
        self.manager.add_parameter_set(self.high_vol_params)
        self.assertIn(RegimeType.HIGH_VOLATILITY, self.manager.parameter_sets)
        self.assertEqual(
            self.high_vol_params.parameters, 
            self.manager.parameter_sets[RegimeType.HIGH_VOLATILITY].parameters
        )
        
    def test_get_parameters_no_blend(self):
        """Test getting parameters without blending."""
        # Add parameter sets
        self.manager.add_parameter_set(self.high_vol_params)
        self.manager.add_parameter_set(self.mean_rev_params)
        
        # Get parameters without blending
        params = self.manager.get_parameters(RegimeType.HIGH_VOLATILITY, blend=False)
        self.assertEqual(params, self.high_vol_params.parameters)
        
    def test_get_parameters_with_blend(self):
        """Test getting parameters with blending."""
        # Add parameter sets
        self.manager.add_parameter_set(self.high_vol_params)
        self.manager.add_parameter_set(self.mean_rev_params)
        
        # Set current regime and parameters
        self.manager.current_regime = RegimeType.HIGH_VOLATILITY
        self.manager.current_parameters = self.high_vol_params.parameters.copy()
        
        # Get parameters with blending - first call should return target directly
        params1 = self.manager.get_parameters(RegimeType.MEAN_REVERTING, blend=True)
        self.assertEqual(params1, self.mean_rev_params.parameters)
        
        # Second call to same regime should return current parameters
        params2 = self.manager.get_parameters(RegimeType.MEAN_REVERTING, blend=True)
        self.assertEqual(params2, self.mean_rev_params.parameters)
        
    def test_reset_parameters(self):
        """Test resetting parameters."""
        # Add parameter sets
        self.manager.add_parameter_set(self.high_vol_params)
        self.manager.add_parameter_set(self.mean_rev_params)
        
        # Reset specific parameter set
        self.manager.reset_parameters(RegimeType.HIGH_VOLATILITY)
        self.assertNotIn(RegimeType.HIGH_VOLATILITY, self.manager.parameter_sets)
        self.assertIn(RegimeType.MEAN_REVERTING, self.manager.parameter_sets)
        
        # Reset all parameter sets
        self.manager.reset_parameters()
        self.assertEqual(len(self.manager.parameter_sets), 0)
        
    def test_performance_comparison(self):
        """Test getting performance comparison."""
        # Add parameter sets
        self.manager.add_parameter_set(self.high_vol_params)
        self.manager.add_parameter_set(self.mean_rev_params)
        
        # Get performance comparison
        comparison = self.manager.get_performance_comparison()
        self.assertEqual(len(comparison), 2)
        self.assertIn('sharpe_ratio', comparison.columns)
        self.assertIn('max_drawdown', comparison.columns)

    def test_save_and_load_parameter_set(self):
        """Test saving and loading parameter set."""
        # Add parameter set
        self.manager.add_parameter_set(self.high_vol_params)
        
        # Create new manager with same storage path
        new_manager = AdaptiveParameterManager(
            strategy_name='test_strategy',
            base_parameters=self.base_parameters,
            parameter_ranges=self.parameter_ranges,
            storage_path=self.storage_path
        )
        
        # Check loaded parameter set
        self.assertIn(RegimeType.HIGH_VOLATILITY, new_manager.parameter_sets)
        self.assertEqual(
            self.high_vol_params.parameters,
            new_manager.parameter_sets[RegimeType.HIGH_VOLATILITY].parameters
        )
        
if __name__ == '__main__':
    unittest.main()
