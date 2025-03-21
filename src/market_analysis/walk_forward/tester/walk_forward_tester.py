"""
Walk-Forward Testing Module - Main Class Implementation

This module provides a framework for walk-forward testing of trading strategies,
implementing proper training/validation separation to avoid lookahead bias
and provide realistic strategy performance assessment.
"""

import logging
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Any, Union, Callable
from datetime import datetime
import json
from pathlib import Path
from enum import Enum

from src.market_analysis.walk_forward.parameter_optimizer import ParameterOptimizer
from src.market_analysis.regime_detection.detector import RegimeType
from src.market_analysis.regime_detection.enhanced_detector import EnhancedRegimeDetector

# Import components
from src.market_analysis.walk_forward.tester.walk_forward_methods import (
    standard_walk_forward, anchored_walk_forward
)
from src.market_analysis.walk_forward.tester.regime_based_walkforward import (
    regime_based_walk_forward, detect_regimes_for_dataset, process_regime_window
)
from src.market_analysis.walk_forward.tester.processor import (
    process_window, optimize_parameters, calculate_metrics, aggregate_results,
    calculate_parameter_stability, calculate_performance_consistency
)

# Configure logging
logger = logging.getLogger(__name__)


class WalkForwardMethod(Enum):
    """Enumeration of different walk-forward testing methods."""
    STANDARD = "standard"
    ANCHORED = "anchored"
    REGIME_BASED = "regime_based"


class WalkForwardTester:
    """
    Implements walk-forward testing methodology for trading strategies.
    
    Walk-forward testing provides a robust framework for strategy assessment by:
    1. Properly separating training and validation data
    2. Optimizing parameters on in-sample data
    3. Testing strategy performance on out-of-sample data
    4. Moving through the dataset in a sequential manner
    """
    
    def __init__(self,
                 strategy_class: Any,
                 parameter_ranges: Dict[str, Tuple],
                 optimization_metric: str = "sharpe_ratio",
                 method: Union[str, WalkForwardMethod] = WalkForwardMethod.STANDARD,
                 is_window_size: int = 252,  # Default to 1 year
                 oos_window_size: int = 63,  # Default to 3 months
                 step_size: int = 21,  # Default to 1 month
                 optimization_method: str = "bayesian",
                 optimization_iterations: int = 50,
                 regime_detector: Optional[EnhancedRegimeDetector] = None,
                 min_regime_data: int = 30,
                 random_seed: int = 42):
        """
        Initialize the walk-forward tester.
        
        Args:
            strategy_class: Class of the trading strategy to test
            parameter_ranges: Dictionary mapping parameter names to ranges for optimization
            optimization_metric: Metric to maximize during optimization
            method: Walk-forward testing method (standard, anchored, or regime-based)
            is_window_size: Size of in-sample window for optimization (periods)
            oos_window_size: Size of out-of-sample window for validation (periods)
            step_size: Size of steps for standard walk-forward (periods)
            optimization_method: Method to use for optimization
            optimization_iterations: Number of iterations for optimization
            regime_detector: Optional regime detector for regime-based walk-forward
            min_regime_data: Minimum data points required for regime-specific optimization
            random_seed: Random seed for reproducibility
        """
        self.strategy_class = strategy_class
        self.parameter_ranges = parameter_ranges
        self.optimization_metric = optimization_metric
        
        # Convert string method to enum if needed
        if isinstance(method, str):
            try:
                self.method = WalkForwardMethod(method.lower())
            except ValueError:
                logger.warning(f"Unknown method: {method}, falling back to standard walk-forward")
                self.method = WalkForwardMethod.STANDARD
        else:
            self.method = method
            
        # Set window sizes and step size
        self.is_window_size = is_window_size
        self.oos_window_size = oos_window_size
        self.step_size = step_size
        
        # Set optimization parameters
        self.optimization_method = optimization_method
        self.optimization_iterations = optimization_iterations
        self.random_seed = random_seed
        
        # Set regime detection parameters
        self.regime_detector = regime_detector
        self.min_regime_data = min_regime_data
        
        # Store test results
        self.window_results = []
        self.regime_parameters = {}
        self.windows = []
    
    def run_test(self, 
                data: pd.DataFrame, 
                macro_data: Optional[pd.DataFrame] = None) -> Dict[str, Any]:
        """
        Run walk-forward test on the provided market data.
        
        Args:
            data: DataFrame of market data (must have 'close' column at minimum)
            macro_data: Optional DataFrame of macro indicator data for regime detection
            
        Returns:
            Dict containing test results
        """
        # Choose the appropriate method based on configuration
        if self.method == WalkForwardMethod.STANDARD:
            results = standard_walk_forward(self, data)
        elif self.method == WalkForwardMethod.ANCHORED:
            results = anchored_walk_forward(self, data)
        elif self.method == WalkForwardMethod.REGIME_BASED:
            if self.regime_detector is None:
                logger.warning("Regime detector not provided for regime-based testing, falling back to standard")
                results = standard_walk_forward(self, data)
            else:
                results = regime_based_walk_forward(self, data, macro_data)
        else:
            logger.error(f"Unknown walk-forward method: {self.method}")
            raise ValueError(f"Unknown walk-forward method: {self.method}")
            
        return results
    
    # Link to processor functions
    def _process_window(self, window_id, is_data, oos_data):
        return process_window(self, window_id, is_data, oos_data)
        
    def _optimize_parameters(self, data):
        return optimize_parameters(self, data)
        
    def _calculate_metrics(self, returns, trades):
        return calculate_metrics(returns, trades)
        
    def _aggregate_results(self, window_results):
        return aggregate_results(window_results)
        
    def _detect_regimes_for_dataset(self, data, macro_data=None):
        return detect_regimes_for_dataset(self, data, macro_data)
        
    def _process_regime_window(self, window_id, is_data, oos_data, regime_params, oos_regimes):
        return process_regime_window(self, window_id, is_data, oos_data, regime_params, oos_regimes)
    
    def save_results(self, output_path: str) -> None:
        """
        Save the walk-forward test results to a file.
        
        Args:
            output_path: Path to save the results
        """
        if not self.window_results:
            logger.warning("No results to save")
            return
            
        # Create directory if it doesn't exist
        output_dir = Path(output_path).parent
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Prepare results for serialization
        serializable_results = {
            "method": self.method.value,
            "parameters": {
                "is_window_size": self.is_window_size,
                "oos_window_size": self.oos_window_size,
                "step_size": self.step_size,
                "optimization_metric": self.optimization_metric,
                "optimization_method": self.optimization_method,
                "optimization_iterations": self.optimization_iterations
            },
            "window_results": [
                {
                    k: (v.isoformat() if isinstance(v, pd.Timestamp) else v)
                    for k, v in window.items()
                    if k != "trades"  # Exclude trades for simplicity
                }
                for window in self.window_results
            ],
            "aggregated_results": self._aggregate_results(self.window_results),
            "timestamp": datetime.now().isoformat()
        }
        
        # Save to file
        with open(output_path, 'w') as f:
            json.dump(serializable_results, f, indent=2)
            
        logger.info(f"Saved walk-forward test results to {output_path}")
    
    def load_results(self, input_path: str) -> Dict[str, Any]:
        """
        Load walk-forward test results from a file.
        
        Args:
            input_path: Path to load the results from
            
        Returns:
            Dict containing the loaded results
        """
        try:
            with open(input_path, 'r') as f:
                results = json.load(f)
                
            # Update instance variables from loaded results
            self.method = WalkForwardMethod(results.get("method", "standard"))
            self.window_results = results.get("window_results", [])
            
            # Convert ISO-format timestamps back to pandas Timestamps
            for window in self.window_results:
                for key in ["is_start", "is_end", "oos_start", "oos_end"]:
                    if key in window and isinstance(window[key], str):
                        window[key] = pd.Timestamp(window[key])
            
            logger.info(f"Loaded walk-forward test results from {input_path}")
            return results
        except Exception as e:
            logger.error(f"Error loading walk-forward test results: {e}")
            return {}
