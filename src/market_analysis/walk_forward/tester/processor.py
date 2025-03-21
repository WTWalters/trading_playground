"""
Walk-Forward Processing Module

This module provides processing functionality for walk-forward testing,
including window processing, parameter optimization, and metrics calculation.
"""

import logging
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Any

# Configure logging
logger = logging.getLogger(__name__)


def process_window(tester, 
                  window_id: int,
                  is_data: pd.DataFrame,
                  oos_data: pd.DataFrame) -> Dict[str, Any]:
    """
    Process a single window in standard or anchored walk-forward testing.
    
    Args:
        tester: WalkForwardTester instance
        window_id: Identifier for the window
        is_data: In-sample data for parameter optimization
        oos_data: Out-of-sample data for validation
        
    Returns:
        Dict containing window results
    """
    # Optimize parameters on in-sample data
    optimal_params, optimization_metrics = optimize_parameters(tester, is_data)
    
    # Evaluate on out-of-sample data
    strategy = tester.strategy_class(**optimal_params)
    strategy.backtest(oos_data)
    
    # Get performance metrics
    performance_metrics = strategy.performance_metrics()
    
    # Store window result
    window_result = {
        "window_id": window_id,
        "is_start": is_data.index[0],
        "is_end": is_data.index[-1],
        "oos_start": oos_data.index[0],
        "oos_end": oos_data.index[-1],
        "parameters": optimal_params,
        "optimization_metrics": optimization_metrics,
        "metrics": performance_metrics,
        "trades": getattr(strategy, "trades", [])
    }
    
    return window_result


def optimize_parameters(tester, data: pd.DataFrame) -> Tuple[Dict[str, Any], Dict[str, float]]:
    """
    Optimize strategy parameters on the given data.
    
    Args:
        tester: WalkForwardTester instance
        data: Market data for optimization
        
    Returns:
        Tuple of (optimal parameters, optimization metrics)
    """
    from src.market_analysis.walk_forward.parameter_optimizer import ParameterOptimizer
    
    # Create parameter optimizer
    optimizer = ParameterOptimizer(
        parameter_ranges=tester.parameter_ranges,
        optimization_metric=tester.optimization_metric,
        method=tester.optimization_method,
        max_evaluations=tester.optimization_iterations,
        random_seed=tester.random_seed
    )
    
    # Define evaluation function
    def evaluation_func(params):
        strategy = tester.strategy_class(**params)
        strategy.backtest(data)
        return strategy.performance_metrics()
        
    # Run optimization
    optimal_params, metrics = optimizer.optimize(evaluation_func)
    
    return optimal_params, metrics


def calculate_metrics(returns: List[float], trades: List[Dict]) -> Dict[str, float]:
    """
    Calculate performance metrics from returns and trades.
    
    Args:
        returns: List of daily returns
        trades: List of trade dictionaries
        
    Returns:
        Dictionary of performance metrics
    """
    if not returns:
        return {"total_return": 0.0, "sharpe_ratio": 0.0, "max_drawdown": 0.0}
        
    # Convert returns to numpy array
    returns_arr = np.array(returns)
    
    # Calculate key metrics
    total_return = np.prod(1 + returns_arr) - 1
    daily_volatility = np.std(returns_arr)
    annual_volatility = daily_volatility * np.sqrt(252)
    
    # Calculate Sharpe ratio
    if annual_volatility > 0:
        sharpe_ratio = (total_return / len(returns_arr) * 252) / annual_volatility
    else:
        sharpe_ratio = 0.0
        
    # Calculate maximum drawdown
    cumulative_returns = np.cumprod(1 + returns_arr) - 1
    running_max = np.maximum.accumulate(cumulative_returns)
    drawdowns = running_max - cumulative_returns
    max_drawdown = np.max(drawdowns) if len(drawdowns) > 0 else 0.0
    
    # Calculate trade-based metrics
    if trades:
        winning_trades = [t for t in trades if t.get("pnl", 0) > 0]
        win_rate = len(winning_trades) / len(trades) if trades else 0.0
        
        # Calculate profit factor
        gross_profit = sum(t.get("pnl", 0) for t in winning_trades)
        losing_trades = [t for t in trades if t.get("pnl", 0) <= 0]
        gross_loss = abs(sum(t.get("pnl", 0) for t in losing_trades))
        profit_factor = gross_profit / gross_loss if gross_loss > 0 else float('inf')
    else:
        win_rate = 0.0
        profit_factor = 0.0
        
    return {
        "total_return": total_return,
        "sharpe_ratio": sharpe_ratio,
        "max_drawdown": max_drawdown,
        "win_rate": win_rate,
        "profit_factor": profit_factor,
        "annualized_return": total_return / len(returns_arr) * 252,
        "daily_volatility": daily_volatility,
        "annual_volatility": annual_volatility
    }


def aggregate_results(window_results: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Aggregate results from multiple walk-forward windows.
    
    Args:
        window_results: List of results from individual windows
        
    Returns:
        Dictionary of aggregated results
    """
    if not window_results:
        return {"metrics": {}, "parameter_stability": {}}
        
    # Extract all trades
    all_trades = []
    for window in window_results:
        if "trades" in window:
            all_trades.extend(window["trades"])
            
    # Sort trades by date if possible
    try:
        all_trades.sort(key=lambda x: x.get("entry_time", 0))
    except Exception:
        pass
        
    # Aggregate metrics
    metrics = {}
    for key in window_results[0].get("metrics", {}).keys():
        values = [w["metrics"].get(key, 0) for w in window_results]
        metrics[key] = {
            "mean": np.mean(values),
            "median": np.median(values),
            "std": np.std(values),
            "min": np.min(values),
            "max": np.max(values)
        }
        
    # Calculate parameter stability
    parameter_stability = calculate_parameter_stability(window_results)
    
    # Calculate performance consistency
    consistency = calculate_performance_consistency(window_results)
    
    return {
        "metrics": metrics,
        "trades": all_trades,
        "parameter_stability": parameter_stability,
        "performance_consistency": consistency
    }


def calculate_parameter_stability(window_results: List[Dict[str, Any]]) -> Dict[str, float]:
    """
    Calculate parameter stability across windows.
    
    Args:
        window_results: List of results from individual windows
        
    Returns:
        Dictionary mapping parameter names to stability scores
    """
    # Only relevant for standard and anchored methods, not regime-based
    if not window_results or "parameters" not in window_results[0]:
        return {}
        
    # Extract parameters from each window
    all_params = {}
    for window in window_results:
        for param, value in window.get("parameters", {}).items():
            if param not in all_params:
                all_params[param] = []
            all_params[param].append(value)
            
    # Calculate stability metrics (lower is more stable)
    stability = {}
    for param, values in all_params.items():
        if not values:
            continue
            
        mean = np.mean(values)
        std = np.std(values)
        
        # Coefficient of variation as stability measure
        if mean != 0:
            stability[param] = std / abs(mean)
        else:
            stability[param] = float('inf')
            
    return stability


def calculate_performance_consistency(window_results: List[Dict[str, Any]], 
                                     metric: str = "sharpe_ratio") -> float:
    """
    Calculate performance consistency across windows.
    
    Args:
        window_results: List of results from individual windows
        metric: Performance metric to evaluate
        
    Returns:
        Consistency score (0-1)
    """
    if not window_results:
        return 0.0
        
    # Extract metric values
    values = [w.get("metrics", {}).get(metric, 0) for w in window_results]
    
    # Calculate consistency (percentage of windows with positive metric)
    consistency = sum(1 for v in values if v > 0) / len(values) if values else 0.0
    
    return consistency
