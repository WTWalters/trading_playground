"""
Parameter Optimizer Module

This module provides optimization functionality for strategy parameters within 
the walk-forward testing framework.
"""

import logging
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Any, Union, Callable
import itertools
from datetime import datetime
import json
from pathlib import Path

# Try to import optional optimization libraries
try:
    from skopt import gp_minimize
    from skopt.space import Real, Integer, Categorical
    SKOPT_AVAILABLE = True
except ImportError:
    SKOPT_AVAILABLE = False

logger = logging.getLogger(__name__)

class ParameterOptimizer:
    """
    Optimizes strategy parameters for a training window.
    
    This class provides multiple optimization methods including:
    1. Grid search
    2. Random search 
    3. Bayesian optimization (if scikit-optimize is available)
    
    It evaluates parameter combinations by running backtests and selecting
    the parameters that maximize a specified performance metric.
    """
    
    def __init__(self, 
                 parameter_ranges: Dict[str, Tuple],
                 optimization_metric: str = "sharpe_ratio",
                 method: str = "grid_search",
                 max_evaluations: int = 50,
                 random_seed: int = 42):
        """
        Initialize the parameter optimizer.
        
        Args:
            parameter_ranges: Dictionary mapping parameter names to (min, max, step) for numerical
                             parameters or list of values for categorical parameters
            optimization_metric: Metric to maximize during optimization
            method: Optimization method ('grid_search', 'random_search', or 'bayesian')
            max_evaluations: Maximum number of parameter combinations to evaluate
            random_seed: Random seed for reproducibility
        """
        self.parameter_ranges = parameter_ranges
        self.optimization_metric = optimization_metric
        self.method = method
        self.max_evaluations = max_evaluations
        self.random_seed = random_seed
        
        # Set random seed
        np.random.seed(random_seed)
        
        # Validate optimization method
        if method == "bayesian" and not SKOPT_AVAILABLE:
            logger.warning("scikit-optimize not available, falling back to grid search")
            self.method = "grid_search"
        elif method not in ["grid_search", "random_search", "bayesian"]:
            logger.warning(f"Unknown optimization method: {method}, falling back to grid search")
            self.method = "grid_search"
            
    def optimize(self, 
                evaluation_func: Callable[[Dict[str, Any]], Dict[str, float]],
                base_parameters: Optional[Dict[str, Any]] = None) -> Tuple[Dict[str, Any], Dict[str, float]]:
        """
        Optimize strategy parameters using the specified method.
        
        Args:
            evaluation_func: Function that takes parameters dict and returns metrics dict
            base_parameters: Optional base parameters to use for parameters not being optimized
            
        Returns:
            Tuple of (optimal parameters dict, performance metrics dict)
        """
        logger.info(f"Starting parameter optimization using {self.method}")
        
        # Select optimization method
        if self.method == "grid_search":
            return self._grid_search(evaluation_func, base_parameters)
        elif self.method == "random_search":
            return self._random_search(evaluation_func, base_parameters)
        elif self.method == "bayesian":
            return self._bayesian_optimization(evaluation_func, base_parameters)
        else:
            logger.error(f"Unknown optimization method: {self.method}")
            raise ValueError(f"Unknown optimization method: {self.method}")
            
    def _grid_search(self, 
                    evaluation_func: Callable[[Dict[str, Any]], Dict[str, float]],
                    base_parameters: Optional[Dict[str, Any]] = None) -> Tuple[Dict[str, Any], Dict[str, float]]:
        """
        Perform grid search optimization.
        
        Args:
            evaluation_func: Function that takes parameters dict and returns metrics dict
            base_parameters: Optional base parameters to use for parameters not being optimized
            
        Returns:
            Tuple of (optimal parameters dict, performance metrics dict)
        """
        # Generate parameter space
        param_space = {}
        for param_name, range_tuple in self.parameter_ranges.items():
            if isinstance(range_tuple, (list, tuple)):
                if len(range_tuple) == 3:
                    # Assume (min, max, step) format
                    min_val, max_val, step = range_tuple
                    
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
                else:
                    # Assume it's a list of discrete values
                    values = range_tuple
            else:
                # Single value, no optimization needed
                values = [range_tuple]
                
            param_space[param_name] = values
            
        # Generate parameter combinations
        param_names = list(param_space.keys())
        param_values = list(param_space.values())
        
        # Calculate total combinations
        total_combinations = 1
        for values in param_values:
            total_combinations *= len(values)
            
        logger.info(f"Grid search: {total_combinations} potential combinations")
        
        # Limit the number of combinations to evaluate
        all_combinations = list(itertools.product(*param_values))
        if len(all_combinations) > self.max_evaluations:
            # Sample randomly if too many combinations
            logger.info(f"Limiting to {self.max_evaluations} random combinations")
            np.random.shuffle(all_combinations)
            combinations = all_combinations[:self.max_evaluations]
        else:
            combinations = all_combinations
            
        return self._evaluate_combinations(combinations, param_names, evaluation_func, base_parameters)
    
    def _random_search(self, 
                      evaluation_func: Callable[[Dict[str, Any]], Dict[str, float]],
                      base_parameters: Optional[Dict[str, Any]] = None) -> Tuple[Dict[str, Any], Dict[str, float]]:
        """
        Perform random search optimization.
        
        Args:
            evaluation_func: Function that takes parameters dict and returns metrics dict
            base_parameters: Optional base parameters to use for parameters not being optimized
            
        Returns:
            Tuple of (optimal parameters dict, performance metrics dict)
        """
        # Generate random parameter combinations
        combinations = []
        param_names = list(self.parameter_ranges.keys())
        
        for _ in range(self.max_evaluations):
            combination = []
            for param_name in param_names:
                range_tuple = self.parameter_ranges[param_name]
                
                if isinstance(range_tuple, (list, tuple)):
                    if len(range_tuple) == 3:
                        # Assume (min, max, step) format
                        min_val, max_val, step = range_tuple
                        
                        if isinstance(min_val, int) and isinstance(max_val, int):
                            # Integer parameter
                            value = np.random.randint(min_val, max_val + 1)
                        elif isinstance(min_val, float) or isinstance(max_val, float):
                            # Float parameter
                            value = np.random.uniform(min_val, max_val)
                        else:
                            # Assume categorical
                            value = np.random.choice(min_val)  # min_val is the list of choices
                    else:
                        # Assume it's a list of discrete values
                        value = np.random.choice(range_tuple)
                else:
                    # Single value, no optimization needed
                    value = range_tuple
                    
                combination.append(value)
                
            combinations.append(tuple(combination))
            
        logger.info(f"Random search: {len(combinations)} random combinations")
        
        return self._evaluate_combinations(combinations, param_names, evaluation_func, base_parameters)
    
    def _bayesian_optimization(self, 
                             evaluation_func: Callable[[Dict[str, Any]], Dict[str, float]],
                             base_parameters: Optional[Dict[str, Any]] = None) -> Tuple[Dict[str, Any], Dict[str, float]]:
        """
        Perform Bayesian optimization using scikit-optimize.
        
        Args:
            evaluation_func: Function that takes parameters dict and returns metrics dict
            base_parameters: Optional base parameters to use for parameters not being optimized
            
        Returns:
            Tuple of (optimal parameters dict, performance metrics dict)
        """
        if not SKOPT_AVAILABLE:
            raise ImportError("scikit-optimize is required for Bayesian optimization")
            
        # Define the search space for Bayesian optimization
        space = []
        param_names = []
        
        for param_name, range_tuple in self.parameter_ranges.items():
            param_names.append(param_name)
            
            if isinstance(range_tuple, (list, tuple)):
                if len(range_tuple) == 3:
                    # Assume (min, max, step) format
                    min_val, max_val, step = range_tuple
                    
                    if isinstance(min_val, int) and isinstance(max_val, int):
                        # Integer parameter
                        space.append(Integer(min_val, max_val, name=param_name))
                    elif isinstance(min_val, float) or isinstance(max_val, float):
                        # Float parameter
                        space.append(Real(min_val, max_val, name=param_name))
                    else:
                        # Categorical parameter (assuming min_val is the list of choices)
                        space.append(Categorical(min_val, name=param_name))
                else:
                    # Assume it's a list of discrete values
                    space.append(Categorical(range_tuple, name=param_name))
            else:
                # Single value, no optimization needed
                space.append(Categorical([range_tuple], name=param_name))
                
        # Initialize tracking for best result
        best_parameters = None
        best_metrics = None
        best_score = float('-inf')
        
        # Prepare base parameters if provided
        final_base_params = {} if base_parameters is None else base_parameters.copy()
        
        # Define the objective function
        def objective(param_values):
            # Convert to dictionary
            params = {param_names[i]: param_values[i] for i in range(len(param_names))}
            
            # Combine with base parameters
            full_params = {**final_base_params, **params}
            
            try:
                # Evaluate the parameters
                metrics = evaluation_func(full_params)
                score = metrics.get(self.optimization_metric, float('-inf'))
                
                # Update tracking for best result
                nonlocal best_parameters, best_metrics, best_score
                if score > best_score:
                    best_score = score
                    best_parameters = full_params.copy()
                    best_metrics = metrics.copy()
                    
                # Log progress
                logger.debug(f"Parameters: {params}, Score: {score:.6f}")
                
                # Return negative score for minimization
                return -score
            except Exception as e:
                logger.warning(f"Error evaluating parameters {params}: {e}")
                return float('inf')  # Penalize errors
                
        # Run Bayesian optimization
        logger.info(f"Bayesian optimization: max {self.max_evaluations} evaluations")
        
        try:
            result = gp_minimize(
                objective,
                space,
                n_calls=self.max_evaluations,
                random_state=self.random_seed,
                verbose=False
            )
            
            # Return best result found
            if best_parameters is None:
                raise ValueError("Optimization failed to find valid parameters")
                
            logger.info(f"Bayesian optimization complete. Best score: {best_score:.6f}")
            return best_parameters, best_metrics
        except Exception as e:
            logger.error(f"Bayesian optimization failed: {e}")
            
            # Fall back to grid search
            logger.info("Falling back to grid search")
            return self._grid_search(evaluation_func, base_parameters)
            
    def _evaluate_combinations(self,
                              combinations: List[Tuple],
                              param_names: List[str],
                              evaluation_func: Callable[[Dict[str, Any]], Dict[str, float]],
                              base_parameters: Optional[Dict[str, Any]] = None) -> Tuple[Dict[str, Any], Dict[str, float]]:
        """
        Evaluate a list of parameter combinations.
        
        Args:
            combinations: List of parameter value tuples
            param_names: List of parameter names corresponding to combination values
            evaluation_func: Function that takes parameters dict and returns metrics dict
            base_parameters: Optional base parameters to use for parameters not being optimized
            
        Returns:
            Tuple of (optimal parameters dict, performance metrics dict)
        """
        best_score = float('-inf')
        best_params = None
        best_metrics = None
        
        # Prepare base parameters if provided
        final_base_params = {} if base_parameters is None else base_parameters.copy()
        
        # Evaluate each combination
        for i, values in enumerate(combinations):
            # Convert to dictionary
            params = {param_names[j]: values[j] for j in range(len(param_names))}
            
            # Combine with base parameters
            full_params = {**final_base_params, **params}
            
            try:
                # Evaluate the parameters
                metrics = evaluation_func(full_params)
                score = metrics.get(self.optimization_metric, float('-inf'))
                
                if score > best_score:
                    best_score = score
                    best_params = full_params.copy()
                    best_metrics = metrics.copy()
                    
                # Log progress
                logger.debug(f"Combination {i+1}/{len(combinations)}: {params}, Score: {score:.6f}")
                
                if (i + 1) % 10 == 0:
                    logger.info(f"Evaluated {i+1}/{len(combinations)} combinations. Best score: {best_score:.6f}")
            except Exception as e:
                logger.warning(f"Error evaluating parameters {params}: {e}")
                
        if best_params is None:
            logger.warning("Optimization failed to find valid parameters")
            
            # Return base parameters or empty dict
            if base_parameters:
                return base_parameters, {self.optimization_metric: 0.0}
            else:
                # Create default parameters from the middle of each range
                default_params = {}
                for param_name, range_tuple in self.parameter_ranges.items():
                    if isinstance(range_tuple, (list, tuple)):
                        if len(range_tuple) == 3:
                            min_val, max_val, _ = range_tuple
                            default_params[param_name] = (min_val + max_val) / 2
                        else:
                            default_params[param_name] = range_tuple[0]
                    else:
                        default_params[param_name] = range_tuple
                
                return default_params, {self.optimization_metric: 0.0}
                
        logger.info(f"Optimization complete. Best score: {best_score:.6f}")
        return best_params, best_metrics
    
    def save_results(self, 
                    parameters: Dict[str, Any], 
                    metrics: Dict[str, float],
                    output_path: str) -> None:
        """
        Save optimization results to a file.
        
        Args:
            parameters: Optimized parameters dictionary
            metrics: Performance metrics dictionary
            output_path: Path to save results
        """
        # Create result dictionary
        result = {
            "parameters": parameters,
            "metrics": metrics,
            "optimization_info": {
                "method": self.method,
                "metric": self.optimization_metric,
                "evaluations": self.max_evaluations,
                "timestamp": datetime.now().isoformat()
            }
        }
        
        # Create directory if needed
        output_dir = Path(output_path).parent
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Save to file
        with open(output_path, 'w') as f:
            json.dump(result, f, indent=2)
            
        logger.info(f"Saved optimization results to {output_path}")
        
    @classmethod
    def load_parameters(cls, input_path: str) -> Dict[str, Any]:
        """
        Load optimized parameters from a file.
        
        Args:
            input_path: Path to load parameters from
            
        Returns:
            Dictionary of optimized parameters
        """
        try:
            with open(input_path, 'r') as f:
                result = json.load(f)
                
            return result.get("parameters", {})
        except Exception as e:
            logger.error(f"Error loading parameters from {input_path}: {e}")
            return {}
