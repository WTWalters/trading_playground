"""
Parallelized cointegration testing module for statistical arbitrage strategies.

This module provides optimized tools for testing cointegration between time series,
leveraging parallel processing for improved performance on multi-core systems.
"""

import logging
import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Union
from statsmodels.tsa.stattools import adfuller, coint
import statsmodels.api as sm
import asyncio
from datetime import datetime, timedelta

from ..database.manager import DatabaseManager

class ParallelCointegrationTester:
    """Optimized version of CointegrationTester that leverages parallel processing.
    
    This class implements various statistical tests to identify cointegrated
    relationships between financial time series, using asyncio to parallelize
    the most computationally intensive operations.
    """
    
    def __init__(self, db_manager: DatabaseManager):
        """Initialize the ParallelCointegrationTester.
        
        Args:
            db_manager: DatabaseManager instance for data access
        """
        self.db_manager = db_manager
        self.logger = logging.getLogger(__name__)
        
    async def adf_test(self, residuals: np.ndarray, significance_level: float = 0.05) -> Dict:
        """Perform Augmented Dickey-Fuller test for stationarity.
        
        The ADF test checks if a time series is stationary (mean-reverting).
        For cointegrated pairs, the residuals should be stationary.
        
        Args:
            residuals: Residuals from the cointegration regression
            significance_level: Threshold for statistical significance (default: 0.05)
            
        Returns:
            Dictionary with test results including:
                - is_stationary: Boolean indicating stationarity
                - p_value: P-value of the test
                - test_statistic: ADF test statistic
                - critical_values: Critical values at different significance levels
        """
        try:
            # Perform ADF test
            result = adfuller(residuals, autolag='AIC')
            
            # Extract results
            test_statistic = result[0]
            p_value = result[1]
            critical_values = result[4]
            
            return {
                'is_stationary': p_value < significance_level,
                'p_value': p_value,
                'test_statistic': test_statistic,
                'critical_values': critical_values
            }
        except Exception as e:
            self.logger.error(f"Error in ADF test: {e}")
            raise
            
    async def engle_granger_test(
        self, 
        series1: np.ndarray, 
        series2: np.ndarray, 
        significance_level: float = 0.05
    ) -> Dict:
        """Perform Engle-Granger two-step method for cointegration testing.
        
        This is a simpler approach to test for cointegration:
        1. Run OLS regression between the two series
        2. Test residuals for stationarity using ADF test
        
        Args:
            series1: First time series
            series2: Second time series
            significance_level: Threshold for statistical significance
            
        Returns:
            Dictionary with test results including:
                - is_cointegrated: Boolean indicating cointegration
                - hedge_ratio: Beta coefficient from regression
                - residuals: Residuals from regression
                - adf_results: Results from ADF test on residuals
        """
        try:
            # Step 1: Run OLS regression to find hedge ratio
            X = sm.add_constant(series1)
            model = sm.OLS(series2, X)
            results = model.fit()
            
            # Extract hedge ratio (beta coefficient)
            hedge_ratio = results.params[1]
            
            # Calculate spread/residuals
            residuals = series2 - (hedge_ratio * series1)
            
            # Step 2: Test residuals for stationarity
            adf_results = await self.adf_test(residuals, significance_level)
            
            return {
                'is_cointegrated': adf_results['is_stationary'],
                'hedge_ratio': hedge_ratio,
                'residuals': residuals,
                'adf_results': adf_results,
                'regression_results': {
                    'params': results.params.tolist(),
                    'r_squared': results.rsquared,
                    'p_value': results.f_pvalue
                }
            }
        except Exception as e:
            self.logger.error(f"Error in Engle-Granger test: {e}")
            raise
            
    async def johansen_test(
        self, 
        series1: np.ndarray, 
        series2: np.ndarray, 
        significance_level: float = 0.05
    ) -> Dict:
        """Perform Johansen test for cointegration.
        
        The Johansen test is more robust for identifying cointegration
        relationships, especially for multiple time series.
        
        Args:
            series1: First time series
            series2: Second time series
            significance_level: Threshold for statistical significance
            
        Returns:
            Dictionary with test results including cointegration status and parameters
        """
        try:
            # Prepare data for Johansen test
            data = np.column_stack([series1, series2])
            
            # Use simplified statsmodels coint function
            t_stat, p_value, crit_values = coint(series1, series2)
            
            return {
                'is_cointegrated': p_value < significance_level,
                't_statistic': t_stat,
                'p_value': p_value,
                'critical_values': crit_values,
                'eigenvalues': None,
                'eigenvectors': None
            }
        except Exception as e:
            self.logger.error(f"Error in Johansen test: {e}")
            raise

    async def test_pair(
        self,
        symbol1: str,
        symbol2: str,
        price_df: pd.DataFrame,
        significance_level: float = 0.05
    ) -> Optional[Dict]:
        """Test a single pair for cointegration.
        
        Args:
            symbol1: First symbol
            symbol2: Second symbol
            price_df: DataFrame with price data
            significance_level: Significance level for tests
            
        Returns:
            Dictionary with cointegration results or None if not cointegrated
        """
        try:
            # Get price series
            series1 = price_df[symbol1].values
            series2 = price_df[symbol2].values
            
            # Calculate correlation
            correlation = np.corrcoef(series1, series2)[0, 1]
            
            # Run tests in parallel
            eg_future = self.engle_granger_test(series1, series2, significance_level)
            johansen_future = self.johansen_test(series1, series2, significance_level)
            
            eg_result, johansen_result = await asyncio.gather(eg_future, johansen_future)
            
            # If either test confirms cointegration, add to results
            if eg_result['is_cointegrated'] or johansen_result['is_cointegrated']:
                return {
                    'symbol1': symbol1,
                    'symbol2': symbol2,
                    'correlation': correlation,
                    'hedge_ratio': eg_result['hedge_ratio'],
                    'engle_granger_result': eg_result,
                    'johansen_result': johansen_result,
                    'timeframe': '1d',  # Using daily by default
                    'period': {
                        'start': None,  # Will be filled later
                        'end': None     # Will be filled later
                    }
                }
            return None
        except Exception as e:
            self.logger.error(f"Error testing pair {symbol1}/{symbol2}: {e}")
            return None
            
    async def select_pairs_parallel(
        self, 
        symbols: List[str], 
        start_date, 
        end_date, 
        timeframe: str = '1d',
        min_correlation: float = 0.6,
        significance_level: float = 0.05,
        source: Optional[str] = None
    ) -> List[Dict]:
        """Select cointegrated pairs using parallel processing.
        
        This function:
        1. Fetches price data for all symbols concurrently
        2. Calculates correlation between all pairs
        3. Tests for cointegration on highly correlated pairs in parallel
        4. Returns pairs that are cointegrated
        
        Args:
            symbols: List of symbols to analyze
            start_date: Start date for analysis period
            end_date: End date for analysis period
            timeframe: Data timeframe (e.g., '1d' for daily)
            min_correlation: Minimum correlation threshold
            significance_level: Significance level for cointegration test
            source: Data source (optional)
            
        Returns:
            List of dictionaries with cointegrated pair information
        """
        try:
            # Step 1: Fetch data for all symbols in parallel
            self.logger.info(f"Fetching data for {len(symbols)} symbols from {start_date} to {end_date}")
            
            fetch_tasks = []
            for symbol in symbols:
                self.logger.info(f"Fetching data for {symbol}...")
                fetch_tasks.append(
                    self.db_manager.get_market_data(symbol, start_date, end_date, timeframe, source)
                )
            
            # Wait for all fetch tasks to complete
            results = await asyncio.gather(*fetch_tasks)
            
            # Process results
            symbol_data = {}
            for i, df in enumerate(results):
                if not df.empty:
                    symbol = symbols[i]
                    self.logger.info(f"Found {len(df)} records for {symbol}")
                    symbol_data[symbol] = df['close']
                else:
                    self.logger.warning(f"No data found for {symbols[i]}")
            
            self.logger.info(f"Successfully fetched data for {len(symbol_data)} symbols")
            
            if len(symbol_data) < 2:
                self.logger.warning("Insufficient data for pair selection: need at least 2 symbols with data")
                return []
            
            # Create a DataFrame with all close prices
            price_df = pd.DataFrame(symbol_data)
            
            # Step 2: Calculate correlation matrix
            correlation_matrix = price_df.corr()
            
            # Step 3: Find pairs with high correlation
            potential_pairs = []
            for i, symbol1 in enumerate(correlation_matrix.columns):
                for j, symbol2 in enumerate(correlation_matrix.columns):
                    if i < j:  # Avoid duplicate pairs
                        correlation = correlation_matrix.loc[symbol1, symbol2]
                        if abs(correlation) >= min_correlation:
                            potential_pairs.append((symbol1, symbol2, correlation))
            
            # Step 4: Test for cointegration in parallel
            self.logger.info(f"Testing {len(potential_pairs)} potential pairs for cointegration...")
            
            test_tasks = []
            for symbol1, symbol2, _ in potential_pairs:
                test_tasks.append(
                    self.test_pair(symbol1, symbol2, price_df, significance_level)
                )
            
            # Wait for all test tasks to complete
            results = await asyncio.gather(*test_tasks)
            
            # Filter out None results
            cointegrated_pairs = [r for r in results if r is not None]
            
            # Update period information
            for pair in cointegrated_pairs:
                pair['period']['start'] = start_date
                pair['period']['end'] = end_date
            
            self.logger.info(
                f"Found {len(cointegrated_pairs)} cointegrated pairs "
                f"out of {len(potential_pairs)} potential pairs"
            )
            return cointegrated_pairs
            
        except Exception as e:
            self.logger.error(f"Error in parallel pair selection: {e}")
            raise
            
    async def calculate_window_cointegration(
        self,
        symbol1: str,
        symbol2: str, 
        window_data: pd.DataFrame,
        window_start,
        window_end
    ) -> Dict:
        """Calculate cointegration for a specific time window.
        
        Args:
            symbol1: First symbol
            symbol2: Second symbol
            window_data: DataFrame with price data for the window
            window_start: Start of window
            window_end: End of window
            
        Returns:
            Dictionary with window cointegration results
        """
        try:
            # Test cointegration in this window
            eg_result = await self.engle_granger_test(
                window_data[symbol1].values,
                window_data[symbol2].values
            )
            
            return {
                'start_date': window_start,
                'end_date': window_end,
                'is_cointegrated': eg_result['is_cointegrated'],
                'hedge_ratio': eg_result['hedge_ratio'],
                'p_value': eg_result['adf_results']['p_value']
            }
        except Exception as e:
            self.logger.error(f"Error in window cointegration calculation: {e}")
            # Return a placeholder with error information
            return {
                'start_date': window_start,
                'end_date': window_end,
                'is_cointegrated': False,
                'hedge_ratio': 0,
                'p_value': 1.0,
                'error': str(e)
            }
            
    async def calculate_cointegration_stability_parallel(
        self,
        symbol1: str,
        symbol2: str,
        start_date,
        end_date,
        window_size: int = 60,
        step_size: int = 20,
        timeframe: str = '1d',
        source: Optional[str] = None
    ) -> Dict:
        """Calculate stability of cointegration relationship using parallel processing.
        
        Tests cointegration over multiple rolling windows to assess stability.
        
        Args:
            symbol1: First symbol
            symbol2: Second symbol
            start_date: Start date for analysis
            end_date: End date for analysis
            window_size: Size of rolling window in days
            step_size: Step size for rolling window
            timeframe: Data timeframe
            source: Data source (optional)
            
        Returns:
            Dictionary with cointegration stability metrics
        """
        try:
            # Fetch data for both symbols
            fetch_tasks = [
                self.db_manager.get_market_data(symbol1, start_date, end_date, timeframe, source),
                self.db_manager.get_market_data(symbol2, start_date, end_date, timeframe, source)
            ]
            
            df1, df2 = await asyncio.gather(*fetch_tasks)
            
            if df1.empty or df2.empty:
                self.logger.warning("Insufficient data for cointegration stability analysis")
                return {'is_stable': False, 'windows': []}
            
            # Align both series on the same dates
            joined = pd.concat([df1['close'], df2['close']], axis=1, join='inner')
            joined.columns = [symbol1, symbol2]
            
            # Set up rolling windows
            dates = joined.index
            window_tasks = []
            
            # Process each window
            for i in range(0, len(joined) - window_size, step_size):
                window_data = joined.iloc[i:i+window_size]
                window_start = dates[i]
                window_end = dates[min(i+window_size-1, len(dates)-1)]
                
                window_tasks.append(
                    self.calculate_window_cointegration(
                        symbol1, symbol2, window_data, window_start, window_end
                    )
                )
            
            # Execute all window tasks in parallel
            windows = await asyncio.gather(*window_tasks)
            
            # Calculate stability metrics
            hedge_ratios = [w['hedge_ratio'] for w in windows]
            is_cointegrated = [w['is_cointegrated'] for w in windows]
            p_values = [w['p_value'] for w in windows]
            
            stability_ratio = sum(is_cointegrated) / len(windows) if windows else 0
            hedge_ratio_std = np.std(hedge_ratios) if hedge_ratios else 0
            hedge_ratio_mean = np.mean(hedge_ratios) if hedge_ratios else 0
            
            return {
                'is_stable': stability_ratio >= 0.7,
                'stability_ratio': stability_ratio,
                'hedge_ratio': {
                    'mean': float(hedge_ratio_mean),
                    'std': float(hedge_ratio_std),
                    'values': hedge_ratios
                },
                'p_values': p_values,
                'windows': windows
            }
            
        except Exception as e:
            self.logger.error(f"Error in parallel cointegration stability analysis: {e}")
            raise
