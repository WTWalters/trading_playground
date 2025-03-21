"""
Regime-Based Walk-Forward Testing

This module provides regime-based walk-forward testing implementation,
which adjusts strategy parameters based on detected market regimes.
"""

import logging
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Any
from src.market_analysis.regime_detection.detector import RegimeType

# Configure logging
logger = logging.getLogger(__name__)


def regime_based_walk_forward(tester, 
                             data: pd.DataFrame,
                             macro_data: Optional[pd.DataFrame] = None) -> Dict[str, Any]:
    """
    Perform regime-based walk-forward testing.
    
    Regime-based walk-forward adjusts the optimization and validation based on 
    detected market regimes.
    
    Args:
        tester: WalkForwardTester instance
        data: DataFrame of market data
        macro_data: Optional DataFrame of macro indicator data
        
    Returns:
        Dict containing test results
    """
    if tester.regime_detector is None:
        raise ValueError("Regime detector must be provided for regime-based walk-forward testing")
        
    # Initialize results
    oos_results = []
    windows = []
    regime_params = {}
    
    # Calculate number of windows
    total_data_size = len(data)
    window_size = tester.is_window_size + tester.oos_window_size
    n_windows = (total_data_size - window_size) // tester.step_size + 1
    
    logger.info(f"Running regime-based walk-forward test with {n_windows} windows")
    
    # First, detect regimes across entire dataset
    all_regimes = detect_regimes_for_dataset(tester, data, macro_data)
    
    # Iterate through windows
    for i in range(n_windows):
        # Calculate window indices
        start_idx = i * tester.step_size
        is_end_idx = start_idx + tester.is_window_size
        oos_end_idx = is_end_idx + tester.oos_window_size
        
        # Skip if this would exceed the data
        if oos_end_idx > total_data_size:
            logger.info(f"Skipping window {i+1} as it exceeds available data")
            continue
            
        # Extract window data
        is_data = data.iloc[start_idx:is_end_idx]
        oos_data = data.iloc[is_end_idx:oos_end_idx]
        
        # Skip if insufficient data
        if len(is_data) < tester.is_window_size or len(oos_data) < tester.oos_window_size:
            logger.warning(f"Insufficient data for window {i+1}, skipping")
            continue
            
        # Get regimes for this window
        is_dates = is_data.index
        oos_dates = oos_data.index
        
        # Filter regimes for these windows
        is_regimes = {date: regime for date, regime in all_regimes.items() if date in is_dates}
        oos_regimes = {date: regime for date, regime in all_regimes.items() if date in oos_dates}
        
        # Get unique regimes in the out-of-sample window
        unique_oos_regimes = set(oos_regimes.values())
        
        # For each unique regime in OOS window, ensure we have optimized parameters
        for regime in unique_oos_regimes:
            if regime not in regime_params:
                # Get all in-sample data for this regime
                regime_is_dates = [date for date, r in is_regimes.items() if r == regime]
                
                if len(regime_is_dates) < tester.min_regime_data:
                    logger.warning(f"Insufficient data for regime {regime.value} optimization, using default")
                    continue
                    
                # Get in-sample data for this regime
                mask = is_data.index.isin(regime_is_dates)
                regime_is_data = is_data[mask]
                
                # Skip if insufficient data
                if len(regime_is_data) < tester.min_regime_data:
                    logger.warning(f"Insufficient data for regime {regime.value}, skipping optimization")
                    continue
                    
                # Optimize parameters for this regime
                optimal_params, _ = tester._optimize_parameters(regime_is_data)
                regime_params[regime] = optimal_params
                logger.info(f"Optimized parameters for regime {regime.value}")
        
        # Process this window with regime-specific parameters
        window_result = process_regime_window(tester, i, is_data, oos_data, regime_params, oos_regimes)
        
        # Store results
        oos_results.append(window_result)
        windows.append((is_data.index[0], oos_data.index[-1]))
        
        logger.info(f"Completed window {i+1}/{n_windows}")
        
    # Store results for later reference
    tester.window_results = oos_results
    tester.windows = windows
    tester.regime_parameters = regime_params
    
    # Aggregate results across windows
    aggregated_results = tester._aggregate_results(oos_results)
    
    return {
        "individual_windows": oos_results,
        "aggregated_results": aggregated_results,
        "windows": windows,
        "regime_parameters": regime_params,
        "method": "regime_based"
    }


def detect_regimes_for_dataset(tester, 
                              data: pd.DataFrame, 
                              macro_data: Optional[pd.DataFrame] = None) -> Dict[pd.Timestamp, RegimeType]:
    """
    Detect regimes for the entire dataset.
    
    Args:
        tester: WalkForwardTester instance
        data: Market data
        macro_data: Optional macro indicator data
        
    Returns:
        Dictionary mapping dates to regime types
    """
    if tester.regime_detector is None:
        raise ValueError("Regime detector not provided for regime detection")
        
    regimes = {}
    
    # Use a sliding window to detect regimes through the dataset
    window_size = min(60, len(data) // 10)  # Use 10% of data or 60 days, whichever is smaller
    
    for i in range(0, len(data) - window_size, window_size // 2):
        window_data = data.iloc[i:i+window_size]
        
        # Get macro data for this window if available
        window_macro = None
        if macro_data is not None:
            macro_indices = macro_data.index.intersection(window_data.index)
            if len(macro_indices) > 0:
                window_macro = macro_data.loc[macro_indices]
                
        # Detect regime for this window
        try:
            result = tester.regime_detector.detect_regime(window_data, window_macro)
            
            # Map the last day in the window to the detected regime
            midpoint = i + window_size // 2
            if midpoint < len(data):
                regimes[data.index[midpoint]] = result.primary_regime
        except Exception as e:
            logger.warning(f"Error detecting regime: {e}")
            continue
            
    # Interpolate regimes for missing days
    all_dates = data.index
    regime_dates = list(regimes.keys())
    
    if len(regime_dates) < 2:
        # Not enough regimes detected, use a default
        default_regime = RegimeType.UNDEFINED
        return {date: default_regime for date in all_dates}
        
    # Fill in missing dates with nearest regime
    filled_regimes = {}
    
    for date in all_dates:
        if date in regimes:
            filled_regimes[date] = regimes[date]
        else:
            # Find nearest detected regime date
            nearest_idx = np.argmin([abs((date - regime_date).total_seconds()) 
                                    for regime_date in regime_dates])
            filled_regimes[date] = regimes[regime_dates[nearest_idx]]
            
    return filled_regimes


def process_regime_window(tester,
                         window_id: int,
                         is_data: pd.DataFrame,
                         oos_data: pd.DataFrame,
                         regime_params: Dict[RegimeType, Dict[str, Any]],
                         oos_regimes: Dict[pd.Timestamp, RegimeType]) -> Dict[str, Any]:
    """
    Process a single window in regime-based walk-forward testing.
    
    Args:
        tester: WalkForwardTester instance
        window_id: Identifier for the window
        is_data: In-sample data for parameter optimization
        oos_data: Out-of-sample data for validation
        regime_params: Dictionary mapping regimes to optimal parameters
        oos_regimes: Dictionary mapping dates to regimes in the out-of-sample window
        
    Returns:
        Dict containing window results
    """
    # Evaluate strategy performance on out-of-sample data with regime-specific parameters
    trades = []
    daily_returns = []
    regime_transitions = []
    
    # Track current regime and parameters
    current_regime = None
    current_params = None
    
    # Process each day in the out-of-sample window
    for i in range(len(oos_data)):
        # Get current date and data
        current_date = oos_data.index[i]
        current_data = oos_data.iloc[i:i+1]
        
        # Get current regime
        if current_date in oos_regimes:
            this_regime = oos_regimes[current_date]
            
            # Check if regime changed
            if this_regime != current_regime:
                regime_transitions.append((current_date, current_regime, this_regime))
                current_regime = this_regime
                
                # Log regime transition
                logger.debug(f"Regime transition at {current_date}: {this_regime.value}")
                
            # Use regime-specific parameters if available
            if this_regime in regime_params:
                current_params = regime_params[this_regime]
            else:
                # If no parameters for this regime, skip or use defaults
                logger.debug(f"No parameters for regime {this_regime.value}, skipping day")
                continue
        elif current_params is None:
            # If we don't have a regime or parameters, skip this day
            logger.debug(f"No regime detected for {current_date}, skipping day")
            continue
            
        # Instantiate strategy with current parameters
        strategy = tester.strategy_class(**current_params)
        
        # Process this day
        result = strategy.process_day(current_data)
        
        # Collect results
        if result.get("trade"):
            trades.append(result["trade"])
        if result.get("return"):
            daily_returns.append(result["return"])
            
    # Calculate overall performance metrics
    performance_metrics = tester._calculate_metrics(daily_returns, trades)
    
    # Store window result
    window_result = {
        "window_id": window_id,
        "is_start": is_data.index[0],
        "is_end": is_data.index[-1],
        "oos_start": oos_data.index[0],
        "oos_end": oos_data.index[-1],
        "regime_parameters": regime_params.copy(),
        "metrics": performance_metrics,
        "trades": trades,
        "regime_transitions": regime_transitions
    }
    
    return window_result
