"""
Adaptive Parameter Management System

This module provides a comprehensive solution for dynamically adjusting trading
strategy parameters based on changing market conditions. It integrates regime detection,
parameter optimization, and walk-forward testing to create a robust adaptive system.
"""

import logging
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Any, Union, Callable
from enum import Enum
from dataclasses import dataclass, field
import json
from pathlib import Path
from datetime import datetime

from src.market_analysis.regime_detection.detector import RegimeType
from src.market_analysis.regime_detection.enhanced_detector import EnhancedRegimeDetector, MacroRegimeType
from src.market_analysis.walk_forward.tester.walk_forward_tester import WalkForwardTester, WalkForwardMethod

# Configure logging
logger = logging.getLogger(__name__)


@dataclass
class RegimeParameters:
    """Container for regime-specific parameters and performance metrics."""
    regime: RegimeType
    parameters: Dict[str, Any]
    performance_metrics: Dict[str, float] = field(default_factory=dict)
    creation_date: datetime = field(default_factory=datetime.now)
    last_updated: datetime = field(default_factory=datetime.now)
    confidence: float = 0.0
    sample_size: int = 0
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "regime": self.regime.value,
            "parameters": self.parameters,
            "performance_metrics": self.performance_metrics,
            "creation_date": self.creation_date.isoformat(),
            "last_updated": self.last_updated.isoformat(),
            "confidence": self.confidence,
            "sample_size": self.sample_size
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'RegimeParameters':
        """Create from dictionary."""
        return cls(
            regime=RegimeType(data["regime"]),
            parameters=data["parameters"],
            performance_metrics=data["performance_metrics"],
            creation_date=datetime.fromisoformat(data["creation_date"]),
            last_updated=datetime.fromisoformat(data["last_updated"]),
            confidence=data["confidence"],
            sample_size=data["sample_size"]
        )


class AdaptiveParameterManager:
    """
    Adaptive parameter management system that automatically adjusts strategy parameters
    based on detected market regimes.
    
    Key features:
    1. Integration with regime detection for market condition analysis
    2. Parameter optimization for each regime using walk-forward testing
    3. Smooth parameter transitions between regimes
    4. Performance tracking and continuous adaptation
    5. Parameter validation and confidence scoring
    """
    
    def __init__(self,
                 strategy_class: Any,
                 base_parameters: Dict[str, Any],
                 parameter_ranges: Dict[str, Tuple],
                 regime_detector: Optional[EnhancedRegimeDetector] = None,
                 optimization_metric: str = "sharpe_ratio",
                 transition_window: int = 5,
                 storage_path: Optional[str] = None,
                 seed: int = 42):
        """
        Initialize the adaptive parameter manager.
        
        Args:
            strategy_class: Trading strategy class to manage parameters for
            base_parameters: Default parameters to use when no regime-specific params available
            parameter_ranges: Dictionary mapping parameter names to optimization ranges
            regime_detector: Regime detector for market condition analysis
            optimization_metric: Metric to optimize for (e.g., "sharpe_ratio")
            transition_window: Number of periods to blend parameters during transitions
            storage_path: Path to store parameter sets
            seed: Random seed for reproducibility
        """
        self.strategy_class = strategy_class
        self.base_parameters = base_parameters
        self.parameter_ranges = parameter_ranges
        self.regime_detector = regime_detector or EnhancedRegimeDetector()
        self.optimization_metric = optimization_metric
        self.transition_window = transition_window
        self.seed = seed
        
        # Set storage path
        if storage_path is None:
            self.storage_path = Path("./config/adaptive_parameters")
        else:
            self.storage_path = Path(storage_path)
        
        # Create storage directory if it doesn't exist
        self.storage_path.mkdir(parents=True, exist_ok=True)
        
        # Initialize parameter sets for different regimes
        self.regime_parameters: Dict[RegimeType, RegimeParameters] = {}
        
        # Track current state
        self.current_regime: Optional[RegimeType] = None
        self.current_parameters: Dict[str, Any] = base_parameters.copy()
        self.previous_regime: Optional[RegimeType] = None
        self.transition_progress: float = 0.0  # 0.0 = start of transition, 1.0 = completed
        
        # Track regime history
        self.regime_history: List[Tuple[datetime, RegimeType]] = []
        
        # Initialize walk-forward tester
        self.walk_forward_tester = WalkForwardTester(
            strategy_class=strategy_class,
            parameter_ranges=parameter_ranges,
            optimization_metric=optimization_metric,
            method=WalkForwardMethod.REGIME_BASED,
            regime_detector=regime_detector,
            random_seed=seed
        )
        
        # Load existing parameter sets
        self._load_all_parameters()
    
    def get_parameters(self, 
                      market_data: pd.DataFrame,
                      macro_data: Optional[pd.DataFrame] = None,
                      smooth_transition: bool = True) -> Dict[str, Any]:
        """
        Get the optimal parameters for the current market conditions.
        
        Args:
            market_data: Recent market data for regime detection
            macro_data: Optional macro economic data for enhanced detection
            smooth_transition: Whether to smooth parameter transitions between regimes
            
        Returns:
            Dictionary of optimal parameters for current conditions
        """
        # Detect current regime
        regime_result = self.regime_detector.detect_regime(market_data, macro_data)
        detected_regime = regime_result.primary_regime
        
        # Record in history
        self.regime_history.append((datetime.now(), detected_regime))
        
        # Check if regime has changed
        regime_changed = (self.current_regime != detected_regime)
        if regime_changed:
            logger.info(f"Regime change detected: {self.current_regime} -> {detected_regime}")
            self.previous_regime = self.current_regime
            self.current_regime = detected_regime
            self.transition_progress = 0.0
        elif smooth_transition and self.transition_progress < 1.0:
            # Continue ongoing transition
            self.transition_progress = min(1.0, self.transition_progress + 1.0 / self.transition_window)
        
        # Get regime-specific parameters if available
        if detected_regime in self.regime_parameters:
            target_params = self.regime_parameters[detected_regime].parameters
            logger.debug(f"Using optimized parameters for regime {detected_regime.value}")
        else:
            # Fall back to base parameters if no regime-specific params available
            target_params = self.base_parameters
            logger.debug(f"No optimized parameters for regime {detected_regime.value}, using base")
        
        # Handle parameter transition
        if smooth_transition and regime_changed and self.previous_regime is not None:
            # Get previous parameters
            if self.previous_regime in self.regime_parameters:
                previous_params = self.regime_parameters[self.previous_regime].parameters
            else:
                previous_params = self.base_parameters
                
            # Blend parameters for smooth transition
            self.current_parameters = self._blend_parameters(
                previous_params, target_params, self.transition_progress)
            logger.debug(f"Blending parameters: {self.transition_progress:.2f} progress")
        else:
            # Use target parameters directly
            self.current_parameters = target_params.copy()
        
        return self.current_parameters
    
    def update_with_performance(self, 
                               regime: RegimeType,
                               parameters: Dict[str, Any],
                               performance_metrics: Dict[str, float],
                               sample_size: int = 1) -> None:
        """
        Update parameter set with new performance metrics.
        
        Args:
            regime: Market regime these parameters were used in
            parameters: Parameter set that was used
            performance_metrics: Performance metrics achieved with these parameters
            sample_size: Number of samples these metrics are based on
        """
        if regime in self.regime_parameters:
            # Update existing parameter set
            param_set = self.regime_parameters[regime]
            
            # Calculate weighted average for metrics
            total_samples = param_set.sample_size + sample_size
            weight_existing = param_set.sample_size / total_samples
            weight_new = sample_size / total_samples
            
            updated_metrics = {}
            for metric, value in performance_metrics.items():
                if metric in param_set.performance_metrics:
                    # Weighted average
                    updated_metrics[metric] = (
                        param_set.performance_metrics[metric] * weight_existing +
                        value * weight_new
                    )
                else:
                    updated_metrics[metric] = value
            
            # Update parameter set
            param_set.performance_metrics.update(updated_metrics)
            param_set.sample_size = total_samples
            param_set.last_updated = datetime.now()
            
            # Recalculate confidence based on sample size
            param_set.confidence = min(0.95, 0.5 + 0.05 * np.log10(total_samples + 1))
            
            logger.info(f"Updated parameter set for regime {regime.value}, "
                      f"sample size {total_samples}, confidence {param_set.confidence:.2f}")
        else:
            # Create new parameter set
            param_set = RegimeParameters(
                regime=regime,
                parameters=parameters,
                performance_metrics=performance_metrics,
                creation_date=datetime.now(),
                last_updated=datetime.now(),
                confidence=0.5,  # Initial confidence
                sample_size=sample_size
            )
            self.regime_parameters[regime] = param_set
            logger.info(f"Created new parameter set for regime {regime.value}")
        
        # Save updated parameter set
        self._save_parameter_set(param_set)
    
    def optimize_for_regime(self,
                           regime: RegimeType,
                           historical_data: pd.DataFrame,
                           macro_data: Optional[pd.DataFrame] = None,
                           is_window_size: int = 252,
                           oos_window_size: int = 63,
                           optimization_iterations: int = 50) -> RegimeParameters:
        """
        Optimize parameters for a specific market regime using walk-forward testing.
        
        Args:
            regime: Market regime to optimize for
            historical_data: Historical market data for optimization
            macro_data: Optional macro economic data
            is_window_size: In-sample window size for walk-forward testing
            oos_window_size: Out-of-sample window size for walk-forward testing
            optimization_iterations: Number of optimization iterations
            
        Returns:
            RegimeParameters object with optimized parameters
        """
        # Filter data to get regime-specific periods if possible
        if self.regime_detector is not None and len(historical_data) > is_window_size:
            # Detect regimes throughout historical data
            all_regimes = self._detect_regimes_for_dataset(historical_data, macro_data)
            
            # Filter for periods matching the target regime
            regime_indices = [i for i, detected in enumerate(all_regimes) 
                             if detected == regime]
            
            if len(regime_indices) >= is_window_size // 2:  # Enough data for regime-specific optimization
                regime_data = historical_data.iloc[regime_indices]
                logger.info(f"Optimizing for regime {regime.value} with "
                          f"{len(regime_data)} specific data points")
            else:
                regime_data = historical_data
                logger.warning(f"Insufficient regime-specific data for {regime.value}, "
                             f"using all historical data")
        else:
            regime_data = historical_data
            
        # Configure walk-forward tester for this optimization
        self.walk_forward_tester.is_window_size = is_window_size
        self.walk_forward_tester.oos_window_size = oos_window_size
        self.walk_forward_tester.optimization_iterations = optimization_iterations
        
        # Run walk-forward test
        results = self.walk_forward_tester.run_test(regime_data, macro_data)
        
        # Extract optimized parameters for this regime
        if self.walk_forward_tester.method == WalkForwardMethod.REGIME_BASED:
            # Get regime-specific parameters if available
            if regime in results.get("regime_parameters", {}):
                optimized_params = results["regime_parameters"][regime]
            else:
                # Fall back to base parameters
                optimized_params = self.base_parameters
        else:
            # For non-regime-based methods, use the best parameters from aggregated results
            if results.get("aggregated_results", {}).get("best_parameters"):
                optimized_params = results["aggregated_results"]["best_parameters"]
            else:
                # Fall back to base parameters
                optimized_params = self.base_parameters
        
        # Calculate aggregate performance metrics
        performance_metrics = {}
        for key, value in results.get("aggregated_results", {}).get("metrics", {}).items():
            if isinstance(value, dict) and "mean" in value:
                performance_metrics[key] = value["mean"]
        
        # Create or update parameter set
        if regime in self.regime_parameters:
            # Update existing set
            param_set = self.regime_parameters[regime]
            param_set.parameters = optimized_params
            param_set.performance_metrics.update(performance_metrics)
            param_set.last_updated = datetime.now()
            # Increase confidence and sample size
            param_set.sample_size += len(results.get("individual_windows", []))
            param_set.confidence = min(0.95, 0.5 + 0.05 * np.log10(param_set.sample_size + 1))
        else:
            # Create new set
            param_set = RegimeParameters(
                regime=regime,
                parameters=optimized_params,
                performance_metrics=performance_metrics,
                creation_date=datetime.now(),
                last_updated=datetime.now(),
                confidence=0.7,  # Higher initial confidence due to systematic optimization
                sample_size=len(results.get("individual_windows", []))
            )
            self.regime_parameters[regime] = param_set
        
        # Save parameter set
        self._save_parameter_set(param_set)
        
        logger.info(f"Optimized parameters for regime {regime.value} with "
                  f"confidence {param_set.confidence:.2f}")
        
        return param_set
    
    def optimize_all_regimes(self,
                            historical_data: pd.DataFrame,
                            macro_data: Optional[pd.DataFrame] = None,
                            regimes_to_optimize: Optional[List[RegimeType]] = None) -> Dict[RegimeType, RegimeParameters]:
        """
        Optimize parameters for all detected regimes in historical data.
        
        Args:
            historical_data: Historical market data for optimization
            macro_data: Optional macro economic data
            regimes_to_optimize: Optional list of specific regimes to optimize
            
        Returns:
            Dictionary mapping regimes to their optimized parameters
        """
        # Detect all regimes in the dataset
        all_regimes = self._detect_regimes_for_dataset(historical_data, macro_data)
        
        # Get unique regimes
        unique_regimes = set(all_regimes)
        
        # Filter regimes if specified
        if regimes_to_optimize is not None:
            unique_regimes = unique_regimes.intersection(set(regimes_to_optimize))
        
        # Optimize for each regime
        results = {}
        for regime in unique_regimes:
            try:
                param_set = self.optimize_for_regime(regime, historical_data, macro_data)
                results[regime] = param_set
            except Exception as e:
                logger.error(f"Error optimizing for regime {regime.value}: {e}")
        
        return results
    
    def _blend_parameters(self,
                         source_params: Dict[str, Any],
                         target_params: Dict[str, Any],
                         blend_ratio: float) -> Dict[str, Any]:
        """
        Blend parameters for smooth transition between regimes.
        
        Args:
            source_params: Source parameter values
            target_params: Target parameter values
            blend_ratio: Blending ratio (0.0 = source, 1.0 = target)
            
        Returns:
            Dictionary of blended parameters
        """
        blended_params = {}
        
        # Ensure blend ratio is between 0 and 1
        blend_ratio = max(0.0, min(1.0, blend_ratio))
        
        # Apply easing function for smoother transitions
        # Using cubic easing: y = x^3
        eased_ratio = blend_ratio ** 3
        
        # Blend parameter values
        for key in set(source_params.keys()).union(target_params.keys()):
            if key in source_params and key in target_params:
                source_value = source_params[key]
                target_value = target_params[key]
                
                # For numeric parameters, interpolate
                if isinstance(source_value, (int, float)) and isinstance(target_value, (int, float)):
                    blended_value = source_value + eased_ratio * (target_value - source_value)
                    
                    # Maintain integer type if both are integers
                    if isinstance(source_value, int) and isinstance(target_value, int):
                        blended_value = int(round(blended_value))
                        
                    blended_params[key] = blended_value
                else:
                    # For non-numeric parameters, switch at midpoint
                    blended_params[key] = target_value if eased_ratio > 0.5 else source_value
            elif key in source_params:
                blended_params[key] = source_params[key]
            else:
                blended_params[key] = target_params[key]
        
        return blended_params
    
    def _detect_regimes_for_dataset(self,
                                   data: pd.DataFrame,
                                   macro_data: Optional[pd.DataFrame] = None) -> List[RegimeType]:
        """
        Detect regimes for an entire dataset.
        
        Args:
            data: Market data
            macro_data: Optional macro data
            
        Returns:
            List of detected regimes for each data point
        """
        # If detector not available, return default
        if self.regime_detector is None:
            return [RegimeType.UNDEFINED] * len(data)
        
        regimes = []
        window_size = min(60, len(data) // 10)  # 60 days or 10% of data
        
        # Detect regime for each window
        for i in range(0, len(data) - window_size + 1, window_size // 2):
            window_data = data.iloc[i:i+window_size]
            
            # Get corresponding macro data if available
            window_macro = None
            if macro_data is not None:
                macro_indices = macro_data.index.intersection(window_data.index)
                if len(macro_indices) > 0:
                    window_macro = macro_data.loc[macro_indices]
            
            try:
                result = self.regime_detector.detect_regime(window_data, window_macro)
                regime = result.primary_regime
            except Exception as e:
                logger.warning(f"Error detecting regime: {e}")
                regime = RegimeType.UNDEFINED
            
            # Assign this regime to all days in the window
            regimes.extend([regime] * window_size)
            
            # Adjust if we went beyond data length
            if len(regimes) > len(data):
                regimes = regimes[:len(data)]
        
        # Fill any missing days at the end
        if len(regimes) < len(data):
            regimes.extend([regimes[-1]] * (len(data) - len(regimes)))
        
        return regimes
    
    def _save_parameter_set(self, param_set: RegimeParameters) -> None:
        """Save a parameter set to disk."""
        file_path = self.storage_path / f"regime_{param_set.regime.value}.json"
        
        try:
            with open(file_path, 'w') as f:
                json.dump(param_set.to_dict(), f, indent=2)
            logger.debug(f"Saved parameter set for regime {param_set.regime.value}")
        except Exception as e:
            logger.error(f"Error saving parameter set: {e}")
    
    def _load_parameter_set(self, regime: RegimeType) -> Optional[RegimeParameters]:
        """Load a parameter set from disk."""
        file_path = self.storage_path / f"regime_{regime.value}.json"
        
        if not file_path.exists():
            return None
        
        try:
            with open(file_path, 'r') as f:
                data = json.load(f)
            param_set = RegimeParameters.from_dict(data)
            self.regime_parameters[regime] = param_set
            logger.debug(f"Loaded parameter set for regime {regime.value}")
            return param_set
        except Exception as e:
            logger.error(f"Error loading parameter set: {e}")
            return None
    
    def _load_all_parameters(self) -> None:
        """Load all parameter sets from disk."""
        # Find all parameter files
        pattern = "regime_*.json"
        json_files = list(self.storage_path.glob(pattern))
        
        # Load each file
        for file_path in json_files:
            try:
                regime_str = file_path.stem.split('_', 1)[1]
                regime = RegimeType(regime_str)
                self._load_parameter_set(regime)
            except Exception as e:
                logger.error(f"Error loading parameter set from {file_path}: {e}")
    
    def get_regime_transition_probability(self, 
                                         from_regime: RegimeType, 
                                         to_regime: RegimeType,
                                         window_size: int = 30) -> float:
        """
        Calculate the probability of transitioning from one regime to another.
        
        Args:
            from_regime: Source regime
            to_regime: Target regime
            window_size: Number of recent regime transitions to consider
            
        Returns:
            Transition probability (0.0-1.0)
        """
        if len(self.regime_history) < 2:
            return 0.5  # Default when history is insufficient
        
        # Look at recent transitions in the history
        history_window = self.regime_history[-window_size:]
        transition_count = 0
        from_count = 0
        
        # Count transitions
        for i in range(1, len(history_window)):
            prev_regime = history_window[i-1][1]
            curr_regime = history_window[i][1]
            
            if prev_regime == from_regime:
                from_count += 1
                if curr_regime == to_regime:
                    transition_count += 1
        
        # Calculate probability
        if from_count > 0:
            return transition_count / from_count
        else:
            return 0.0
    
    def get_regime_analysis(self) -> Dict[str, Any]:
        """
        Get comprehensive analysis of regime history and parameters.
        
        Returns:
            Dictionary with regime analysis
        """
        # Ensure we have regime history
        if not self.regime_history:
            return {"error": "No regime history available"}
        
        # Calculate regime statistics
        regime_counts = {}
        for _, regime in self.regime_history:
            if regime not in regime_counts:
                regime_counts[regime] = 0
            regime_counts[regime] += 1
        
        total_counts = sum(regime_counts.values())
        regime_frequencies = {r.value: count / total_counts 
                             for r, count in regime_counts.items()}
        
        # Calculate regime persistence (average duration)
        regime_durations = {}
        current_regime = None
        current_duration = 0
        
        for _, regime in self.regime_history:
            if regime == current_regime:
                current_duration += 1
            else:
                if current_regime is not None:
                    if current_regime not in regime_durations:
                        regime_durations[current_regime] = []
                    regime_durations[current_regime].append(current_duration)
                current_regime = regime
                current_duration = 1
        
        # Add final regime duration
        if current_regime is not None:
            if current_regime not in regime_durations:
                regime_durations[current_regime] = []
            regime_durations[current_regime].append(current_duration)
        
        avg_durations = {r.value: np.mean(durations) for r, durations in regime_durations.items()}
        
        # Gather parameter information
        parameter_info = {}
        for regime, param_set in self.regime_parameters.items():
            parameter_info[regime.value] = {
                "confidence": param_set.confidence,
                "sample_size": param_set.sample_size,
                "last_updated": param_set.last_updated.isoformat(),
                "key_metrics": {k: v for k, v in param_set.performance_metrics.items()
                              if k in [self.optimization_metric, "total_return", "max_drawdown"]}
            }
        
        # Calculate transition matrix
        transition_matrix = {}
        for from_regime in RegimeType:
            transition_matrix[from_regime.value] = {}
            for to_regime in RegimeType:
                probability = self.get_regime_transition_probability(from_regime, to_regime)
                transition_matrix[from_regime.value][to_regime.value] = probability
        
        return {
            "regime_frequencies": regime_frequencies,
            "average_durations": avg_durations,
            "parameter_info": parameter_info,
            "transition_matrix": transition_matrix,
            "current_regime": self.current_regime.value if self.current_regime else None,
            "previous_regime": self.previous_regime.value if self.previous_regime else None,
            "transition_progress": self.transition_progress
        }

    def validate_parameters(self, 
                           parameters: Dict[str, Any],
                           validation_data: pd.DataFrame) -> Dict[str, float]:
        """
        Validate a parameter set on unseen data.
        
        Args:
            parameters: Parameters to validate
            validation_data: Data to validate on
            
        Returns:
            Dictionary of performance metrics
        """
        try:
            # Instantiate strategy with parameters
            strategy = self.strategy_class(**parameters)
            
            # Run backtest on validation data
            strategy.backtest(validation_data)
            
            # Get performance metrics
            return strategy.performance_metrics()
        except Exception as e:
            logger.error(f"Error validating parameters: {e}")
            return {}
        
    def save_state(self, file_path: Optional[str] = None) -> str:
        """
        Save the entire state of the adaptive parameter manager.
        
        Args:
            file_path: Optional path to save to
            
        Returns:
            Path where state was saved
        """
        if file_path is None:
            file_path = self.storage_path / "adaptive_manager_state.json"
        else:
            file_path = Path(file_path)
            
        # Prepare state dictionary
        state = {
            "base_parameters": self.base_parameters,
            "current_regime": self.current_regime.value if self.current_regime else None,
            "previous_regime": self.previous_regime.value if self.previous_regime else None,
            "transition_progress": self.transition_progress,
            "regime_parameters": {
                regime.value: param_set.to_dict()
                for regime, param_set in self.regime_parameters.items()
            },
            "regime_history": [
                [dt.isoformat(), r.value] for dt, r in self.regime_history[-100:]  # Last 100 entries
            ],
            "timestamp": datetime.now().isoformat()
        }
        
        # Save to file
        try:
            with open(file_path, 'w') as f:
                json.dump(state, f, indent=2)
            logger.info(f"Saved adaptive parameter manager state to {file_path}")
            return str(file_path)
        except Exception as e:
            logger.error(f"Error saving state: {e}")
            return ""
    
    def load_state(self, file_path: Optional[str] = None) -> bool:
        """
        Load the state of the adaptive parameter manager.
        
        Args:
            file_path: Optional path to load from
            
        Returns:
            True if successful, False otherwise
        """
        if file_path is None:
            file_path = self.storage_path / "adaptive_manager_state.json"
        else:
            file_path = Path(file_path)
            
        if not file_path.exists():
            logger.warning(f"State file {file_path} not found")
            return False
            
        try:
            with open(file_path, 'r') as f:
                state = json.load(f)
                
            # Load base parameters
            self.base_parameters = state.get("base_parameters", self.base_parameters)
            
            # Load current state
            curr_regime_str = state.get("current_regime")
            prev_regime_str = state.get("previous_regime")
            
            if curr_regime_str:
                self.current_regime = RegimeType(curr_regime_str)
            if prev_regime_str:
                self.previous_regime = RegimeType(prev_regime_str)
                
            self.transition_progress = state.get("transition_progress", 0.0)
            
            # Load regime parameters
            regime_params = state.get("regime_parameters", {})
            for regime_str, param_dict in regime_params.items():
                try:
                    regime = RegimeType(regime_str)
                    param_set = RegimeParameters.from_dict(param_dict)
                    self.regime_parameters[regime] = param_set
                except Exception as e:
                    logger.error(f"Error loading parameter set for {regime_str}: {e}")
            
            # Load regime history
            regime_history = state.get("regime_history", [])
            self.regime_history = []
            for dt_str, regime_str in regime_history:
                try:
                    dt = datetime.fromisoformat(dt_str)
                    regime = RegimeType(regime_str)
                    self.regime_history.append((dt, regime))
                except Exception as e:
                    logger.error(f"Error loading regime history entry: {e}")
            
            logger.info(f"Loaded adaptive parameter manager state from {file_path}")
            return True
        except Exception as e:
            logger.error(f"Error loading state: {e}")
            return False
