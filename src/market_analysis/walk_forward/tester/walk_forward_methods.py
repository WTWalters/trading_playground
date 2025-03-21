"""
Walk-Forward Testing Methods

This module provides the core walk-forward testing method implementations
for the WalkForwardTester class.
"""

import logging
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Any

# Configure logging
logger = logging.getLogger(__name__)


def standard_walk_forward(tester, data: pd.DataFrame) -> Dict[str, Any]:
    """
    Perform standard walk-forward testing.
    
    Standard walk-forward uses a sliding window of in-sample and out-of-sample periods.
    
    Args:
        tester: WalkForwardTester instance
        data: DataFrame of market data
        
    Returns:
        Dict containing test results
    """
    # Initialize results
    oos_results = []
    windows = []
    
    # Calculate number of windows
    total_data_size = len(data)
    window_size = tester.is_window_size + tester.oos_window_size
    n_windows = (total_data_size - window_size) // tester.step_size + 1
    
    logger.info(f"Running standard walk-forward test with {n_windows} windows")
    
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
            
        # Process this window
        window_result = tester._process_window(i, is_data, oos_data)
        
        # Store results
        oos_results.append(window_result)
        windows.append((is_data.index[0], oos_data.index[-1]))
        
        logger.info(f"Completed window {i+1}/{n_windows}")
        
    # Store results for later reference
    tester.window_results = oos_results
    tester.windows = windows
    
    # Aggregate results across windows
    aggregated_results = tester._aggregate_results(oos_results)
    
    return {
        "individual_windows": oos_results,
        "aggregated_results": aggregated_results,
        "windows": windows,
        "method": "standard"
    }


def anchored_walk_forward(tester, data: pd.DataFrame) -> Dict[str, Any]:
    """
    Perform anchored walk-forward testing.
    
    Anchored walk-forward keeps the beginning of the in-sample period fixed
    while extending the end date.
    
    Args:
        tester: WalkForwardTester instance
        data: DataFrame of market data
        
    Returns:
        Dict containing test results
    """
    # Initialize results
    oos_results = []
    windows = []
    
    # Calculate number of windows
    total_data_size = len(data)
    n_windows = (total_data_size - tester.is_window_size) // tester.oos_window_size
    
    logger.info(f"Running anchored walk-forward test with {n_windows} windows")
    
    # Iterate through windows
    for i in range(n_windows):
        # Calculate window indices
        is_start_idx = 0  # Anchored at the beginning
        is_end_idx = tester.is_window_size + i * tester.oos_window_size
        oos_start_idx = is_end_idx
        oos_end_idx = oos_start_idx + tester.oos_window_size
        
        # Skip if this would exceed the data
        if oos_end_idx > total_data_size:
            logger.info(f"Skipping window {i+1} as it exceeds available data")
            break
            
        # Extract window data
        is_data = data.iloc[is_start_idx:is_end_idx]
        oos_data = data.iloc[oos_start_idx:oos_end_idx]
        
        # Skip if insufficient data
        if len(oos_data) < tester.oos_window_size:
            logger.warning(f"Insufficient data for window {i+1}, skipping")
            continue
            
        # Process this window
        window_result = tester._process_window(i, is_data, oos_data)
        
        # Store results
        oos_results.append(window_result)
        windows.append((is_data.index[0], oos_data.index[-1]))
        
        logger.info(f"Completed window {i+1}/{n_windows}")
        
    # Store results for later reference
    tester.window_results = oos_results
    tester.windows = windows
    
    # Aggregate results across windows
    aggregated_results = tester._aggregate_results(oos_results)
    
    return {
        "individual_windows": oos_results,
        "aggregated_results": aggregated_results,
        "windows": windows,
        "method": "anchored"
    }
