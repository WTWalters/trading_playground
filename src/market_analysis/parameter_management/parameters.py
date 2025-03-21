"""
Adaptive Parameter Management Module

This module provides functionality for managing strategy parameters across different
market regimes. It handles parameter optimization, storage, and smooth transitions between
parameter sets as market conditions change.
"""

import json
import pickle
import os
import logging
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Any, Union, Callable
from dataclasses import dataclass, field
from skopt import gp_minimize
from skopt.space import Real, Integer, Categorical
import itertools
from pathlib import Path

from src.market_analysis.regime_detection.detector import RegimeType

# Configure logging
logger = logging.getLogger(__name__)

@dataclass
class ParameterSet:
    """
    Container for a set of strategy parameters optimized for a specific market regime.
    """
    regime_type: RegimeType
    parameters: Dict[str, Any]
    performance_metrics: Dict[str, float] = field(default_factory=dict)
    creation_date: str = field(default_factory=lambda: pd.Timestamp.now().strftime("%Y-%m-%d"))
    last_updated: str = field(default_factory=lambda: pd.Timestamp.now().strftime("%Y-%m-%d"))
    confidence_score: float = 0.0
    
    def update_metrics(self, metrics: Dict[str, float]):
        """Update performance metrics for this parameter set."""
        self.performance_metrics.update(metrics)
        self.last_updated = pd.Timestamp.now().strftime("%Y-%m-%d")
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "regime_type": self.regime_type.value,
            "parameters": self.parameters,
            "performance_metrics": self.performance_metrics,
            "creation_date": self.creation_date,
            "last_updated": self.last_updated,
            "confidence_score": self.confidence_score
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'ParameterSet':
        """Create a ParameterSet from a dictionary."""
        return cls(
            regime_type=RegimeType(data["regime_type"]),
            parameters=data["parameters"],
            performance_metrics=data["performance_metrics"],
            creation_date=data["creation_date"],
            last_updated=data["last_updated"],
            confidence_score=data["confidence_score"]
        )


class AdaptiveParameterManager:
    """
    Manages strategy parameters that adapt to different market regimes.
    
    This class provides functionality for:
    1. Storing optimized parameter sets for different regimes
    2. Selecting appropriate parameters based on current market conditions
    3. Smoothly transitioning between parameter sets during regime changes
    4. Optimizing parameters using grid search or Bayesian optimization
    5. Validating parameters using walk-forward testing
    """
    
    def __init__(self, 
                 strategy_name: str,
                 base_parameters: Dict[str, Any],
                 parameter_ranges: Dict[str, Tuple],
                 optimization_metric: str = "sharpe_ratio",
                 storage_path: Optional[str] = None,
                 blend_window: int = 5):
        """
        Initialize the parameter manager.
        
        Args:
            strategy_name: Name of the strategy for file storage
            base_parameters: Default parameters to use when no specific regime is detected
            parameter_ranges: Dictionary mapping parameter names to (min, max, step) for optimization
            optimization_metric: Metric to maximize during optimization
            storage_path: Path to store parameter sets (default to ./config/parameters)
            blend_window: Number of periods to blend parameters during regime transitions
        """
        self.strategy_name = strategy_name
        self.base_parameters = base_parameters
        self.parameter_ranges = parameter_ranges
        self.optimization_metric = optimization_metric
        self.blend_window = blend_window
        
        # Set default storage path if not provided
        if storage_path is None:
            self.storage_path = Path("./config/parameters")
        else:
            self.storage_path = Path(storage_path)
            
        # Ensure storage directory exists
        os.makedirs(self.storage_path, exist_ok=True)
        
        # Initialize parameter sets dictionary
        self.parameter_sets: Dict[RegimeType, ParameterSet] = {}
        
        # Track current regime and parameters
        self.current_regime: Optional[RegimeType] = None
        self.current_parameters: Dict[str, Any] = base_parameters.copy()
        
        # Track regime history for blending
        self.regime_history: List[Tuple[pd.Timestamp, RegimeType]] = []
        
        # Load any existing parameter sets
        self._load_parameter_sets()
        
    def get_parameters(self, 
                      regime: RegimeType, 
                      blend: bool = True,
                      regime_transition_info: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Get the parameters appropriate for the specified market regime.
        
        Args:
            regime: The detected market regime
            blend: Whether to blend parameters during regime transitions
            regime_transition_info: Additional info about regime transition
            
        Returns:
            Dictionary of parameters optimized for the current conditions
        """
        # Update regime history
        timestamp = pd.Timestamp.now()
        self.regime_history.append((timestamp, regime))
        
        # Check if we have parameters for this regime
        if regime not in self.parameter_sets:
            logger.info(f"No optimized parameters for {regime.value}, using base parameters")
            # Use base parameters if no specific ones exist
            self.current_regime = regime
            self.current_parameters = self.base_parameters.copy()
            return self.current_parameters
        
        # Get optimized parameters for this regime
        target_parameters = self.parameter_sets[regime].parameters
        
        # If blending is disabled or this is the first regime, use target directly
        if not blend or len(self.regime_history) <= 1:
            self.current_regime = regime
            self.current_parameters = target_parameters.copy()
            return self.current_parameters
            
        # If regime hasn't changed, use current parameters
        if self.current_regime == regime:
            return self.current_parameters
            
        # Handle transition through parameter blending
        blended_parameters = self._blend_parameters(
            self.current_parameters,
            target_parameters,
            regime_transition_info
        )
        
        # Update current state
        self.current_regime = regime
        self.current_parameters = blended_parameters
        
        return blended_parameters
    
    def _grid_search(self, 
                   evaluation_func: Callable[[Dict[str, Any]], float],
                   n_iterations: int = 50) -> Tuple[Dict[str, Any], float]:
        """
        Perform grid search optimization.
        
        Args:
            evaluation_func: Function that takes parameters and returns performance metric
            n_iterations: Maximum number of parameter combinations to evaluate
            
        Returns:
            Tuple of (best parameters, best score)
        """
        param_space = {}
        for param_name, (min_val, max_val, step) in self.parameter_ranges.items():
            if isinstance(min_val, int) and isinstance(max_val, int):
                # Integer parameter
                values = list(range(min_val, max_val + 1, max(1, step)))
            elif isinstance(min_val, float) or isinstance(max_val, float):
                # Float parameter
                num_values = min(10, int((max_val - min_val) / step) + 1)
                values = list(np.linspace(min_val, max_val, num_values))
            else:
                # Assume categorical
                values = min_val  # min_val is actually the list of choices
            param_space[param_name] = values
            
        # Generate parameter combinations
        param_names = list(param_space.keys())
        param_values = list(param_space.values())
        
        # Limit the number of combinations to evaluate
        all_combinations = list(itertools.product(*param_values))
        if len(all_combinations) > n_iterations:
            # Sample randomly if too many combinations
            np.random.shuffle(all_combinations)
            combinations = all_combinations[:n_iterations]
        else:
            combinations = all_combinations
            
        best_score = float('-inf')
        best_params = None
        
        # Evaluate each combination
        for i, values in enumerate(combinations):
            params = {param_names[j]: values[j] for j in range(len(param_names))}
            
            try:
                score = evaluation_func(params)
                
                if score > best_score:
                    best_score = score
                    best_params = params
                    
                logger.debug(f"Grid search iteration {i+1}/{len(combinations)}: score={score:.4f}")
            except Exception as e:
                logger.warning(f"Error evaluating parameters {params}: {e}")
                
        if best_params is None:
            logger.warning("Grid search failed to find valid parameters")
            return self.base_parameters, 0.0
            
        return best_params, best_score
    
    def _save_parameter_sets(self) -> None:
        """Save all parameter sets to storage."""
        for regime, param_set in self.parameter_sets.items():
            self.save_parameter_set(param_set)
    
    def _load_parameter_sets(self) -> None:
        """Load all parameter sets from storage."""
        if not self.storage_path.exists():
            return
            
        # Find all parameter files
        pattern = f"{self.strategy_name}_*.json"
        json_files = list(self.storage_path.glob(pattern))
        
        for file_path in json_files:
            try:
                # Extract regime type from filename
                regime_str = file_path.stem.split('_')[-1]
                try:
                    regime = RegimeType(regime_str)
                    self.load_parameter_set(regime)
                except ValueError:
                    logger.warning(f"Unknown regime type in filename: {regime_str}")
            except Exception as e:
                logger.error(f"Error loading parameter set from {file_path}: {e}")
                
    def reset_parameters(self, regime: Optional[RegimeType] = None) -> None:
        """
        Reset parameters to base values for a specific regime or all regimes.
        
        Args:
            regime: Optional specific regime to reset, or None for all
        """
        if regime is None:
            # Reset all parameter sets
            self.parameter_sets = {}
            self.current_parameters = self.base_parameters.copy()
            self.current_regime = None
            logger.info("Reset all parameter sets to base values")
        elif regime in self.parameter_sets:
            # Reset specific regime
            del self.parameter_sets[regime]
            # If current regime was reset, reset current parameters too
            if self.current_regime == regime:
                self.current_parameters = self.base_parameters.copy()
                self.current_regime = None
            logger.info(f"Reset parameters for {regime.value}")
        else:
            logger.warning(f"Cannot reset: no parameters for {regime.value}")
    
    def _bayesian_optimization(self, 
                             evaluation_func: Callable[[Dict[str, Any]], float],
                             n_iterations: int = 50) -> Tuple[Dict[str, Any], float]:
        """
        Perform Bayesian optimization for parameter tuning.
        
        Args:
            evaluation_func: Function that takes parameters and returns performance metric
            n_iterations: Number of iterations for optimization
            
        Returns:
            Tuple of (best parameters, best score)
        """
        # Define the search space for Bayesian optimization
        space = []
        param_names = []
        
        for param_name, (min_val, max_val, step) in self.parameter_ranges.items():
            param_names.append(param_name)
            
            if isinstance(min_val, int) and isinstance(max_val, int):
                # Integer parameter
                space.append(Integer(min_val, max_val, name=param_name))
            elif isinstance(min_val, float) or isinstance(max_val, float):
                # Float parameter
                space.append(Real(min_val, max_val, name=param_name))
            else:
                # Categorical parameter
                space.append(Categorical(min_val, name=param_name))
        
        # Define objective function wrapper
        def objective(param_values):
            # Convert parameter values to dictionary
            params = {param_names[i]: param_values[i] for i in range(len(param_names))}
            
            try:
                # We want to maximize, but gp_minimize minimizes
                score = -evaluation_func(params)
                return score
            except Exception as e:
                logger.warning(f"Error evaluating parameters {params}: {e}")
                return float('inf')  # Penalize errors heavily
        
        # Run Bayesian optimization
        try:
            result = gp_minimize(
                objective,
                space,
                n_calls=n_iterations,
                random_state=42,
                verbose=False
            )
            
            # Convert result to parameters dictionary
            best_params = {param_names[i]: result.x[i] for i in range(len(param_names))}
            
            # Calculate true score (not negated)
            best_score = -result.fun
            
            return best_params, best_score
        except Exception as e:
            logger.error(f"Bayesian optimization failed: {e}")
            return self.base_parameters, 0.0
    
    def add_parameter_set(self, parameter_set: ParameterSet) -> None:
        """
        Add a new parameter set for a specific regime.
        
        Args:
            parameter_set: ParameterSet to add
        """
        self.parameter_sets[parameter_set.regime_type] = parameter_set
        self._save_parameter_sets()
        logger.info(f"Added parameter set for {parameter_set.regime_type.value}")
    
    def optimize_parameters(self, 
                           regime: RegimeType,
                           evaluation_func: Callable[[Dict[str, Any]], float],
                           method: str = "grid_search",
                           n_iterations: int = 50) -> ParameterSet:
        """
        Optimize parameters for a specific regime.
        
        Args:
            regime: The market regime to optimize for
            evaluation_func: Function that takes parameters and returns performance metric
            method: Optimization method ('grid_search' or 'bayesian')
            n_iterations: Number of iterations for optimization
            
        Returns:
            ParameterSet with optimized parameters
        """
        if method == "grid_search":
            best_params, best_score = self._grid_search(evaluation_func, n_iterations)
        elif method == "bayesian":
            best_params, best_score = self._bayesian_optimization(evaluation_func, n_iterations)
        else:
            raise ValueError(f"Unsupported optimization method: {method}")
            
        # Create new parameter set
        parameter_set = ParameterSet(
            regime_type=regime,
            parameters=best_params,
            performance_metrics={self.optimization_metric: best_score},
            confidence_score=0.8  # Default confidence
        )
        
        # Add to our collection
        self.add_parameter_set(parameter_set)
        
        return parameter_set
    
    def walk_forward_validation(self,
                              parameter_set: ParameterSet,
                              evaluation_func: Callable[[Dict[str, Any], pd.DataFrame], Dict[str, float]],
                              data: pd.DataFrame,
                              window_size: int = 60,
                              step_size: int = 20) -> Tuple[Dict[str, float], float]:
        """
        Perform walk-forward validation on a parameter set.
        
        Args:
            parameter_set: ParameterSet to validate
            evaluation_func: Function that takes parameters and data window and returns metrics
            data: Historical market data for validation
            window_size: Size of each testing window
            step_size: Steps between windows
            
        Returns:
            Tuple of (average metrics, confidence score)
        """
        if len(data) < window_size:
            raise ValueError(f"Data must contain at least {window_size} periods")
            
        # Track metrics across windows
        all_metrics = []
        
        # Walk forward through the data
        for start_idx in range(0, len(data) - window_size, step_size):
            window_data = data.iloc[start_idx:start_idx + window_size]
            window_metrics = evaluation_func(parameter_set.parameters, window_data)
            all_metrics.append(window_metrics)
            
        # Calculate average metrics
        avg_metrics = {}
        for metric in all_metrics[0].keys():
            values = [m[metric] for m in all_metrics if metric in m]
            avg_metrics[metric] = sum(values) / len(values)
            
        # Calculate confidence score based on consistency
        consistency = {}
        for metric in avg_metrics.keys():
            values = [m[metric] for m in all_metrics if metric in m]
            if len(values) > 1:
                # Coefficient of variation (lower is more consistent)
                mean_val = sum(values) / len(values)
                std_val = np.std(values)
                if mean_val != 0:
                    consistency[metric] = 1.0 - min(1.0, std_val / abs(mean_val))
                else:
                    consistency[metric] = 0.0
            else:
                consistency[metric] = 0.5  # Default with insufficient data
                
        # Overall confidence score - weighted average of key metrics
        if self.optimization_metric in consistency:
            confidence_score = consistency[self.optimization_metric]
        else:
            # Take average of available metrics
            confidence_score = sum(consistency.values()) / len(consistency)
            
        # Update parameter set with new metrics and confidence
        parameter_set.update_metrics(avg_metrics)
        parameter_set.confidence_score = confidence_score
        
        # Save updated parameter set
        self._save_parameter_sets()
        
        return avg_metrics, confidence_score
    
    def fine_tune_parameters(self,
                           regime: RegimeType,
                           evaluation_func: Callable[[Dict[str, Any]], float],
                           parameter_space: Optional[Dict[str, Tuple]] = None,
                           n_iterations: int = 30) -> ParameterSet:
        """
        Fine-tune an existing parameter set with a narrow search.
        
        Args:
            regime: The market regime to optimize for
            evaluation_func: Function that evaluates parameter performance
            parameter_space: Optional custom parameter ranges (narrower than default)
            n_iterations: Number of iterations for optimization
            
        Returns:
            Updated ParameterSet with fine-tuned parameters
        """
        # Check if we have existing parameters to start from
        if regime not in self.parameter_sets:
            logger.warning(f"No existing parameters for {regime.value}, performing regular optimization")
            return self.optimize_parameters(regime, evaluation_func)
            
        # Get existing parameters as starting point
        base_params = self.parameter_sets[regime].parameters
        
        # Create narrow parameter space around current values if not provided
        if parameter_space is None:
            parameter_space = {}
            for param_name, current_value in base_params.items():
                if param_name in self.parameter_ranges:
                    min_val, max_val, step = self.parameter_ranges[param_name]
                    
                    # Determine type of parameter
                    if isinstance(current_value, int):
                        # For integers, create a narrow range
                        margin = max(1, int((max_val - min_val) * 0.1))  # 10% of range
                        new_min = max(min_val, current_value - margin)
                        new_max = min(max_val, current_value + margin)
                        parameter_space[param_name] = (new_min, new_max, step)
                    elif isinstance(current_value, float):
                        # For floats, create a narrow range
                        margin = (max_val - min_val) * 0.1  # 10% of range
                        new_min = max(min_val, current_value - margin)
                        new_max = min(max_val, current_value + margin)
                        parameter_space[param_name] = (new_min, new_max, step)
                    else:
                        # For categorical, keep the original range
                        parameter_space[param_name] = self.parameter_ranges[param_name]
        
        # Store original ranges
        original_ranges = self.parameter_ranges
        
        # Replace with narrow ranges for fine-tuning
        self.parameter_ranges = parameter_space
        
        # Run optimization with narrow ranges
        try:
            best_params, best_score = self._bayesian_optimization(evaluation_func, n_iterations)
            
            # Create new parameter set
            updated_set = ParameterSet(
                regime_type=regime,
                parameters=best_params,
                performance_metrics={self.optimization_metric: best_score},
                confidence_score=self.parameter_sets[regime].confidence_score,
                creation_date=self.parameter_sets[regime].creation_date
            )
            
            # Add to our collection
            self.add_parameter_set(updated_set)
            
            return updated_set
        finally:
            # Restore original parameter ranges
            self.parameter_ranges = original_ranges
    
    def save_parameter_set(self, parameter_set: ParameterSet, custom_path: Optional[str] = None) -> str:
        """
        Save a parameter set to a specific location.
        
        Args:
            parameter_set: ParameterSet to save
            custom_path: Optional custom path for saving
            
        Returns:
            Path where the parameter set was saved
        """
        if custom_path is None:
            # Use default path based on regime
            file_path = self.storage_path / f"{self.strategy_name}_{parameter_set.regime_type.value}.json"
        else:
            file_path = Path(custom_path)
            
        # Create directory if it doesn't exist
        os.makedirs(file_path.parent, exist_ok=True)
        
        # Convert to dict and save as JSON
        with open(file_path, 'w') as f:
            json.dump(parameter_set.to_dict(), f, indent=2)
            
        logger.info(f"Saved parameter set to {file_path}")
        return str(file_path)
    
    def load_parameter_set(self, regime: RegimeType, custom_path: Optional[str] = None) -> Optional[ParameterSet]:
        """
        Load a parameter set for a specific regime.
        
        Args:
            regime: RegimeType to load parameters for
            custom_path: Optional custom path to load from
            
        Returns:
            Loaded ParameterSet or None if not found
        """
        if custom_path is None:
            # Use default path based on regime
            file_path = self.storage_path / f"{self.strategy_name}_{regime.value}.json"
        else:
            file_path = Path(custom_path)
            
        if not file_path.exists():
            logger.warning(f"Parameter set file not found: {file_path}")
            return None
            
        try:
            with open(file_path, 'r') as f:
                data = json.load(f)
                
            parameter_set = ParameterSet.from_dict(data)
            self.parameter_sets[regime] = parameter_set
            return parameter_set
        except Exception as e:
            logger.error(f"Error loading parameter set: {e}")
            return None
    
    def get_performance_comparison(self) -> pd.DataFrame:
        """
        Get a comparison of performance metrics across different regime parameter sets.
        
        Returns:
            DataFrame with performance metrics for each regime
        """
        data = []
        
        for regime, param_set in self.parameter_sets.items():
            row = {
                "regime": regime.value,
                "confidence": param_set.confidence_score,
                **param_set.performance_metrics
            }
            data.append(row)
            
        if not data:
            return pd.DataFrame()
            
        return pd.DataFrame(data)
    
    def export_all_parameters(self, export_path: Optional[str] = None) -> Dict[str, str]:
        """
        Export all parameter sets to files.
        
        Args:
            export_path: Optional directory to export to (default: storage_path)
            
        Returns:
            Dictionary mapping regime names to export file paths
        """
        if export_path is None:
            export_dir = self.storage_path
        else:
            export_dir = Path(export_path)
            os.makedirs(export_dir, exist_ok=True)
            
        export_paths = {}
        
        for regime, param_set in self.parameter_sets.items():
            file_path = export_dir / f"{self.strategy_name}_{regime.value}.json"
            with open(file_path, 'w') as f:
                json.dump(param_set.to_dict(), f, indent=2)
            export_paths[regime.value] = str(file_path)
            
        return export_paths
    
    def import_parameters(self, 
                        import_path: str,
                        regime_mapping: Optional[Dict[str, RegimeType]] = None) -> List[RegimeType]:
        """
        Import parameter sets from files.
        
        Args:
            import_path: Directory containing parameter set files
            regime_mapping: Optional mapping from file names to regime types
            
        Returns:
            List of RegimeType for successfully imported parameter sets
        """
        import_dir = Path(import_path)
        if not import_dir.exists() or not import_dir.is_dir():
            raise ValueError(f"Import path is not a valid directory: {import_path}")
            
        imported_regimes = []
        
        # Find JSON files in the directory
        json_files = list(import_dir.glob(f"{self.strategy_name}_*.json"))
        
        for file_path in json_files:
            try:
                with open(file_path, 'r') as f:
                    data = json.load(f)
                    
                # Extract regime from filename if not in the data
                if "regime_type" not in data:
                    file_regime = file_path.stem.split('_')[-1]
                    try:
                        regime = RegimeType(file_regime)
                    except ValueError:
                        if regime_mapping and file_regime in regime_mapping:
                            regime = regime_mapping[file_regime]
                        else:
                            logger.warning(f"Could not determine regime type for {file_path}")
                            continue
                    data["regime_type"] = regime.value
                    
                parameter_set = ParameterSet.from_dict(data)
                self.parameter_sets[parameter_set.regime_type] = parameter_set
                imported_regimes.append(parameter_set.regime_type)
                
            except Exception as e:
                logger.error(f"Error importing parameter set from {file_path}: {e}")
                
        # Save the updated parameter sets
        self._save_parameter_sets()
        
        return imported_regimes
    
    def _blend_parameters(self, 
                        current_params: Dict[str, Any],
                        target_params: Dict[str, Any],
                        transition_info: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Blend parameters for smooth transition between regimes.
        
        Args:
            current_params: Current parameter values
            target_params: Target parameter values
            transition_info: Optional information about the transition
            
        Returns:
            Dictionary of blended parameters
        """
        # Default to linear blending
        blend_ratio = 0.2  # Move 20% toward target by default
        
        # Use transition info if provided
        if transition_info:
            if "blend_ratio" in transition_info:
                blend_ratio = transition_info["blend_ratio"]
            elif "confidence" in transition_info:
                # Higher confidence means faster transition
                blend_ratio = min(0.8, transition_info["confidence"])
                
        # Create blended parameters
        blended = {}
        for param_name, current_value in current_params.items():
            if param_name in target_params:
                target_value = target_params[param_name]
                
                # Blend numerical parameters
                if isinstance(current_value, (int, float)) and isinstance(target_value, (int, float)):
                    blended_value = current_value + blend_ratio * (target_value - current_value)
                    # Keep integer parameters as integers
                    if isinstance(current_value, int):
                        blended_value = int(round(blended_value))
                    blended[param_name] = blended_value
                else:
                    # For non-numeric, use target with probability proportional to blend_ratio
                    use_target = np.random.random() < blend_ratio
                    blended[param_name] = target_value if use_target else current_value
            else:
                # Parameter exists in current but not target, keep it
                blended[param_name] = current_value
                
        # Add parameters that exist in target but not current
        for param_name, target_value in target_params.items():
            if param_name not in current_params:
                blended[param_name] = target_value
                
        return blended
    
    def quick_optimize_parameters(self,
                                regime: RegimeType,
                                evaluation_func: Callable[[Dict[str, Any]], float],
                                n_iterations: int = 10) -> ParameterSet:
        """
        Quickly optimize a subset of parameters for rapid adaptation.
        
        Args:
            regime: The market regime to optimize for
            evaluation_func: Function that evaluates parameter performance
            n_iterations: Number of iterations for quick optimization
            
        Returns:
            Updated ParameterSet with optimized parameters
        """
        # Start with base parameters if we don't have any for this regime
        if regime not in self.parameter_sets:
            start_params = self.base_parameters
        else:
            start_params = self.parameter_sets[regime].parameters
            
        # Identify most sensitive parameters (for simplicity, choose a subset)
        # In a real implementation, this would be based on sensitivity analysis
        sensitive_params = {}
        for i, (param_name, range_tuple) in enumerate(self.parameter_ranges.items()):
            if i < len(self.parameter_ranges) // 3:  # Take ~1/3 of parameters
                sensitive_params[param_name] = range_tuple
                
        # Store original ranges
        original_ranges = self.parameter_ranges
        
        # Replace with sensitive parameter subset
        self.parameter_ranges = sensitive_params
        
        # Run optimization with subset of parameters
        try:
            # Fix non-sensitive parameters
            fixed_params = {k: v for k, v in start_params.items() if k not in sensitive_params}
            
            # Create a wrapper for the evaluation function that includes fixed params
            def wrapped_eval_func(params_dict):
                full_params = {**fixed_params, **params_dict}
                return evaluation_func(full_params)
                
            best_params, best_score = self._grid_search(wrapped_eval_func, n_iterations)
            
            # Combine with fixed parameters
            full_params = {**fixed_params, **best_params}
            
            # Create new parameter set
            parameter_set = ParameterSet(
                regime_type=regime,
                parameters=full_params,
                performance_metrics={self.optimization_metric: best_score},
                confidence_score=0.6  # Lower confidence for quick optimization
            )
            
            # Add to our collection
            self.add_parameter_set(parameter_set)
            
            return parameter_set
        finally:
            # Restore original parameter ranges
            self.parameter_ranges = original_ranges
