"""
Integrated Adaptive Parameter Management System

This module provides a unified interface for regime detection, parameter optimization,
and adaptive parameter management, combining all components into a cohesive system.
"""

import logging
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Any, Union
from datetime import datetime, timedelta
from pathlib import Path
import json

from src.market_analysis.regime_detection.detector import RegimeType
from src.market_analysis.regime_detection.enhanced_detector import EnhancedRegimeDetector
from src.market_analysis.walk_forward.tester.walk_forward_tester import WalkForwardTester, WalkForwardMethod
from src.market_analysis.parameter_management.adaptive_manager import AdaptiveParameterManager, RegimeParameters

# Configure logging
logger = logging.getLogger(__name__)


class AdaptiveSystemConfig:
    """Configuration settings for the integrated adaptive system."""
    
    def __init__(self, 
                strategy_class: Any,
                base_parameters: Dict[str, Any],
                parameter_ranges: Dict[str, Tuple],
                optimization_metric: str = "sharpe_ratio",
                regime_detection_lookback: int = 60,
                transition_window: int = 5,
                is_window_size: int = 252,
                oos_window_size: int = 63,
                optimization_iterations: int = 50,
                storage_path: Optional[str] = None,
                seed: int = 42):
        """
        Initialize configuration.
        
        Args:
            strategy_class: Trading strategy class
            base_parameters: Default parameters
            parameter_ranges: Parameter ranges for optimization
            optimization_metric: Metric to optimize
            regime_detection_lookback: Lookback window for regime detection
            transition_window: Window for parameter transitions
            is_window_size: In-sample window size for walk-forward testing
            oos_window_size: Out-of-sample window size for walk-forward testing
            optimization_iterations: Number of optimization iterations
            storage_path: Path for storing data
            seed: Random seed
        """
        self.strategy_class = strategy_class
        self.base_parameters = base_parameters
        self.parameter_ranges = parameter_ranges
        self.optimization_metric = optimization_metric
        self.regime_detection_lookback = regime_detection_lookback
        self.transition_window = transition_window
        self.is_window_size = is_window_size
        self.oos_window_size = oos_window_size
        self.optimization_iterations = optimization_iterations
        self.storage_path = storage_path
        self.seed = seed


class IntegratedAdaptiveSystem:
    """
    Integrated system for adaptive parameter management.
    
    This class combines regime detection, walk-forward testing, and adaptive parameter
    management into a cohesive system for optimizing and adapting trading strategy
    parameters based on changing market conditions.
    """
    
    def __init__(self, config: AdaptiveSystemConfig):
        """
        Initialize the integrated system.
        
        Args:
            config: System configuration
        """
        self.config = config
        
        # Set storage path
        if config.storage_path is None:
            self.storage_path = Path("./config/adaptive_system")
        else:
            self.storage_path = Path(config.storage_path)
        
        # Create storage directory if it doesn't exist
        self.storage_path.mkdir(parents=True, exist_ok=True)
        
        # Initialize regime detector
        self.regime_detector = EnhancedRegimeDetector(
            lookback_window=config.regime_detection_lookback
        )
        
        # Initialize walk-forward tester
        self.walk_forward_tester = WalkForwardTester(
            strategy_class=config.strategy_class,
            parameter_ranges=config.parameter_ranges,
            optimization_metric=config.optimization_metric,
            method=WalkForwardMethod.REGIME_BASED,
            is_window_size=config.is_window_size,
            oos_window_size=config.oos_window_size,
            optimization_iterations=config.optimization_iterations,
            regime_detector=self.regime_detector,
            random_seed=config.seed
        )
        
        # Initialize adaptive parameter manager
        self.parameter_manager = AdaptiveParameterManager(
            strategy_class=config.strategy_class,
            base_parameters=config.base_parameters,
            parameter_ranges=config.parameter_ranges,
            regime_detector=self.regime_detector,
            optimization_metric=config.optimization_metric,
            transition_window=config.transition_window,
            storage_path=self.storage_path / "parameters",
            seed=config.seed
        )
        
        # Internal state
        self.current_market_data = None
        self.current_macro_data = None
        self.current_regime = None
        self.last_update_time = None
        self.last_optimization_time = None
        self.continuous_optimization = False
        
        # Performance tracking
        self.performance_history = []
    
    def initialize(self, 
                  historical_data: pd.DataFrame,
                  macro_data: Optional[pd.DataFrame] = None,
                  optimize_regimes: bool = True) -> None:
        """
        Initialize the system with historical data.
        
        Args:
            historical_data: Historical market data
            macro_data: Optional macro economic data
            optimize_regimes: Whether to optimize parameters for detected regimes
        """
        # Store data
        self.current_market_data = historical_data
        self.current_macro_data = macro_data
        
        # Detect current regime
        regime_result = self.regime_detector.detect_regime(
            historical_data.iloc[-self.config.regime_detection_lookback:], 
            macro_data
        )
        self.current_regime = regime_result.primary_regime
        
        # Optimize parameters for different regimes if requested
        if optimize_regimes:
            # First, detect all regimes in the historical data
            all_regimes = self._detect_all_regimes(historical_data, macro_data)
            unique_regimes = set(all_regimes)
            
            # Optimize for each regime
            for regime in unique_regimes:
                if regime != RegimeType.UNDEFINED:  # Skip undefined regime
                    logger.info(f"Optimizing parameters for regime: {regime.value}")
                    self.parameter_manager.optimize_for_regime(
                        regime, historical_data, macro_data,
                        is_window_size=self.config.is_window_size,
                        oos_window_size=self.config.oos_window_size,
                        optimization_iterations=self.config.optimization_iterations
                    )
        
        # Set initialization timestamps
        now = datetime.now()
        self.last_update_time = now
        self.last_optimization_time = now
        
        logger.info(f"Initialized adaptive system, current regime: {self.current_regime.value}")
    
    def process_data_update(self,
                           new_data: pd.DataFrame,
                           macro_data: Optional[pd.DataFrame] = None,
                           detect_regime_change: bool = True,
                           smooth_transition: bool = True) -> Dict[str, Any]:
        """
        Process new market data and update parameters if needed.
        
        Args:
            new_data: New market data
            macro_data: Optional new macro data
            detect_regime_change: Whether to detect regime changes
            smooth_transition: Whether to smooth parameter transitions
            
        Returns:
            Dictionary with updated parameters and regime information
        """
        # Update internal data
        if self.current_market_data is not None:
            # Append new data to existing data
            self.current_market_data = pd.concat([self.current_market_data, new_data])
            
            # Keep only recent data (last 2 years) to limit memory usage
            cutoff_date = new_data.index[-1] - pd.Timedelta(days=365*2)
            self.current_market_data = self.current_market_data[self.current_market_data.index >= cutoff_date]
        else:
            self.current_market_data = new_data
            
        # Update macro data if provided
        if macro_data is not None:
            if self.current_macro_data is not None:
                self.current_macro_data = pd.concat([self.current_macro_data, macro_data])
                
                # Keep only recent macro data
                cutoff_date = macro_data.index[-1] - pd.Timedelta(days=365*2)
                self.current_macro_data = self.current_macro_data[self.current_macro_data.index >= cutoff_date]
            else:
                self.current_macro_data = macro_data
                
        # Detect current regime if requested
        if detect_regime_change:
            lookback_window = min(self.config.regime_detection_lookback, len(self.current_market_data))
            recent_data = self.current_market_data.iloc[-lookback_window:]
            
            # Get recent macro data if available
            recent_macro = None
            if self.current_macro_data is not None:
                macro_indices = self.current_macro_data.index.intersection(recent_data.index)
                if len(macro_indices) > 0:
                    recent_macro = self.current_macro_data.loc[macro_indices]
            
            # Detect regime
            regime_result = self.regime_detector.detect_regime(recent_data, recent_macro)
            previous_regime = self.current_regime
            self.current_regime = regime_result.primary_regime
            
            # Log regime change
            if previous_regime != self.current_regime:
                logger.info(f"Regime change detected: {previous_regime.value} -> {self.current_regime.value}")
        
        # Get optimal parameters for current conditions
        optimal_params = self.parameter_manager.get_parameters(
            self.current_market_data.iloc[-self.config.regime_detection_lookback:],
            self.current_macro_data,
            smooth_transition=smooth_transition
        )
        
        # Update timestamps
        self.last_update_time = datetime.now()
        
        # Check if continuous optimization is enabled and it's time for optimization
        if (self.continuous_optimization and self.last_optimization_time is not None and
            datetime.now() - self.last_optimization_time > timedelta(days=7)):
            
            # Run optimization in background (simplified here)
            logger.info("Running scheduled parameter optimization")
            self._optimize_current_regime()
            self.last_optimization_time = datetime.now()
        
        # Return results
        return {
            "parameters": optimal_params,
            "regime": self.current_regime.value,
            "regime_confidence": regime_result.confidence if detect_regime_change else None,
            "transition_progress": self.parameter_manager.transition_progress,
            "timestamp": datetime.now().isoformat()
        }
    
    def update_performance_metrics(self,
                                  parameters: Dict[str, Any],
                                  performance_metrics: Dict[str, float],
                                  regime: Optional[RegimeType] = None) -> None:
        """
        Update performance metrics for a parameter set.
        
        Args:
            parameters: Parameter set that was used
            performance_metrics: Performance metrics achieved
            regime: Optional specific regime to update (defaults to current)
        """
        if regime is None:
            regime = self.current_regime
            
        # Update parameter manager
        self.parameter_manager.update_with_performance(
            regime, parameters, performance_metrics
        )
        
        # Store in performance history
        self.performance_history.append({
            "timestamp": datetime.now().isoformat(),
            "regime": regime.value,
            "metrics": performance_metrics
        })
        
        logger.info(f"Updated performance metrics for regime {regime.value}: "
                  f"{self.config.optimization_metric}={performance_metrics.get(self.config.optimization_metric, 0):.4f}")
    
    def optimize_for_regime(self, 
                           regime: RegimeType,
                           force: bool = False) -> RegimeParameters:
        """
        Optimize parameters for a specific regime.
        
        Args:
            regime: Regime to optimize for
            force: Whether to force optimization even if recently optimized
            
        Returns:
            Updated RegimeParameters
        """
        # Check if we have recent optimization and not forced
        if not force:
            if regime in self.parameter_manager.regime_parameters:
                param_set = self.parameter_manager.regime_parameters[regime]
                days_since_update = (datetime.now() - param_set.last_updated).days
                
                if days_since_update < 7 and param_set.confidence > 0.7:
                    logger.info(f"Skipping optimization for regime {regime.value} - "
                              f"recently updated ({days_since_update} days ago)")
                    return param_set
        
        # Run optimization
        return self.parameter_manager.optimize_for_regime(
            regime, 
            self.current_market_data,
            self.current_macro_data,
            is_window_size=self.config.is_window_size,
            oos_window_size=self.config.oos_window_size,
            optimization_iterations=self.config.optimization_iterations
        )
    
    def _optimize_current_regime(self) -> RegimeParameters:
        """Optimize parameters for the current regime."""
        return self.optimize_for_regime(self.current_regime)
    
    def _detect_all_regimes(self, 
                           data: pd.DataFrame,
                           macro_data: Optional[pd.DataFrame] = None) -> List[RegimeType]:
        """
        Detect regimes for the entire dataset.
        
        Args:
            data: Market data
            macro_data: Optional macro data
            
        Returns:
            List of detected regimes
        """
        return self.parameter_manager._detect_regimes_for_dataset(data, macro_data)
    
    def enable_continuous_optimization(self, enabled: bool = True) -> None:
        """Enable or disable continuous parameter optimization."""
        self.continuous_optimization = enabled
        logger.info(f"Continuous optimization {'enabled' if enabled else 'disabled'}")
    
    def get_regime_analysis(self) -> Dict[str, Any]:
        """Get comprehensive regime analysis."""
        return self.parameter_manager.get_regime_analysis()
    
    def save_state(self, file_path: Optional[str] = None) -> str:
        """
        Save the current state of the integrated system.
        
        Args:
            file_path: Optional path to save to
            
        Returns:
            Path where state was saved
        """
        if file_path is None:
            file_path = self.storage_path / "integrated_system_state.json"
        else:
            file_path = Path(file_path)
            
        # Save parameter manager state
        param_manager_path = self.storage_path / "parameter_manager_state.json"
        self.parameter_manager.save_state(param_manager_path)
        
        # Prepare system state
        state = {
            "current_regime": self.current_regime.value if self.current_regime else None,
            "last_update_time": self.last_update_time.isoformat() if self.last_update_time else None,
            "last_optimization_time": self.last_optimization_time.isoformat() if self.last_optimization_time else None,
            "continuous_optimization": self.continuous_optimization,
            "performance_history": self.performance_history[-100:],  # Last 100 entries
            "timestamp": datetime.now().isoformat()
        }
        
        # Save to file
        try:
            with open(file_path, 'w') as f:
                json.dump(state, f, indent=2)
            logger.info(f"Saved integrated system state to {file_path}")
            return str(file_path)
        except Exception as e:
            logger.error(f"Error saving state: {e}")
            return ""
    
    def load_state(self, file_path: Optional[str] = None) -> bool:
        """
        Load the state of the integrated system.
        
        Args:
            file_path: Optional path to load from
            
        Returns:
            True if successful, False otherwise
        """
        if file_path is None:
            file_path = self.storage_path / "integrated_system_state.json"
        else:
            file_path = Path(file_path)
            
        # Load parameter manager state
        param_manager_path = self.storage_path / "parameter_manager_state.json"
        self.parameter_manager.load_state(param_manager_path)
        
        if not file_path.exists():
            logger.warning(f"State file {file_path} not found")
            return False
            
        try:
            with open(file_path, 'r') as f:
                state = json.load(f)
                
            # Load current regime
            curr_regime_str = state.get("current_regime")
            if curr_regime_str:
                self.current_regime = RegimeType(curr_regime_str)
                
            # Load timestamps
            last_update = state.get("last_update_time")
            last_optimization = state.get("last_optimization_time")
            
            if last_update:
                self.last_update_time = datetime.fromisoformat(last_update)
            if last_optimization:
                self.last_optimization_time = datetime.fromisoformat(last_optimization)
                
            # Load other state variables
            self.continuous_optimization = state.get("continuous_optimization", False)
            self.performance_history = state.get("performance_history", [])
            
            logger.info(f"Loaded integrated system state from {file_path}")
            return True
        except Exception as e:
            logger.error(f"Error loading state: {e}")
            return False
