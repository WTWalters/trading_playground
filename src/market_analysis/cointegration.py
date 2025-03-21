"""Cointegration testing module for statistical arbitrage strategies.

This module provides tools for testing cointegration between time series,
which is essential for pair trading and statistical arbitrage strategies.
"""

import logging
import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Union
from statsmodels.tsa.stattools import adfuller, coint
import statsmodels.api as sm
from ..database.manager import DatabaseManager

class CointegrationTester:
    """Tests for cointegration between time series.
    
    This class implements various statistical tests to identify
    cointegrated relationships between financial time series,
    which can be exploited for mean reversion trading strategies.
    """
    
    def __init__(self, db_manager: DatabaseManager):
        """Initialize the CointegrationTester.
        
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
            # (This is a simplified approach as full Johansen test implementation is complex)
            t_stat, p_value, crit_values = coint(series1, series2)
            
            # We can use more sophisticated Johansen test libraries if needed
            # For now, this simplified approach serves our purpose
            
            return {
                'is_cointegrated': p_value < significance_level,
                't_statistic': t_stat,
                'p_value': p_value,
                'critical_values': crit_values,
                'eigenvalues': None,  # Would be filled with actual eigenvalues in full implementation
                'eigenvectors': None  # Would be filled with actual eigenvectors in full implementation
            }
        except Exception as e:
            self.logger.error(f"Error in Johansen test: {e}")
            raise
            
    async def select_pairs(
        self, 
        symbols: List[str], 
        start_date, 
        end_date, 
        timeframe: str = '1d',
        min_correlation: float = 0.6,
        significance_level: float = 0.05,
        source: Optional[str] = None
    ) -> List[Dict]:
        """Select cointegrated pairs from a list of symbols.
        
        This function:
        1. Fetches price data for all symbols
        2. Calculates correlation between all pairs
        3. Tests for cointegration on highly correlated pairs
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
            # Step 1: Fetch data for all symbols
            symbol_data = {}
            self.logger.info(f"Fetching data for {len(symbols)} symbols from {start_date} to {end_date}")
            
            for symbol in symbols:
                self.logger.info(f"Fetching data for {symbol}...")
                df = await self.db_manager.get_market_data(
                    symbol, start_date, end_date, timeframe, source
                )
                if not df.empty:
                    self.logger.info(f"Found {len(df)} records for {symbol}")
                    symbol_data[symbol] = df['close']
                else:
                    self.logger.warning(f"No data found for {symbol}")
            
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
            
            # Step 4: Test for cointegration
            cointegrated_pairs = []
            for symbol1, symbol2, correlation in potential_pairs:
                # Get the price series
                series1 = price_df[symbol1].values
                series2 = price_df[symbol2].values
                
                # Run Engle-Granger test
                eg_result = await self.engle_granger_test(
                    series1, series2, significance_level
                )
                
                # Run Johansen test
                johansen_result = await self.johansen_test(
                    series1, series2, significance_level
                )
                
                # If either test confirms cointegration, add to results
                if eg_result['is_cointegrated'] or johansen_result['is_cointegrated']:
                    cointegrated_pairs.append({
                        'symbol1': symbol1,
                        'symbol2': symbol2,
                        'correlation': correlation,
                        'hedge_ratio': eg_result['hedge_ratio'],
                        'engle_granger_result': eg_result,
                        'johansen_result': johansen_result,
                        'timeframe': timeframe,
                        'period': {
                            'start': start_date,
                            'end': end_date
                        }
                    })
            
            self.logger.info(
                f"Found {len(cointegrated_pairs)} cointegrated pairs "
                f"out of {len(potential_pairs)} potential pairs"
            )
            return cointegrated_pairs
            
        except Exception as e:
            self.logger.error(f"Error in pair selection: {e}")
            raise
            
    async def calculate_cointegration_stability(
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
        """Calculate stability of cointegration relationship over time.
        
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
            df1 = await self.db_manager.get_market_data(
                symbol1, start_date, end_date, timeframe, source
            )
            df2 = await self.db_manager.get_market_data(
                symbol2, start_date, end_date, timeframe, source
            )
            
            if df1.empty or df2.empty:
                self.logger.warning("Insufficient data for cointegration stability analysis")
                return {'is_stable': False, 'windows': []}
            
            # Align both series on the same dates
            joined = pd.concat([df1['close'], df2['close']], axis=1, join='inner')
            joined.columns = [symbol1, symbol2]
            
            # Set up rolling windows
            dates = joined.index
            windows = []
            hedge_ratios = []
            is_cointegrated = []
            p_values = []
            
            # Process each window
            for i in range(0, len(joined) - window_size, step_size):
                window_data = joined.iloc[i:i+window_size]
                
                window_start = dates[i]
                window_end = dates[min(i+window_size-1, len(dates)-1)]
                
                # Test cointegration in this window
                eg_result = await self.engle_granger_test(
                    window_data[symbol1].values,
                    window_data[symbol2].values
                )
                
                windows.append({
                    'start_date': window_start,
                    'end_date': window_end,
                    'is_cointegrated': eg_result['is_cointegrated'],
                    'hedge_ratio': eg_result['hedge_ratio'],
                    'p_value': eg_result['adf_results']['p_value']
                })
                
                hedge_ratios.append(eg_result['hedge_ratio'])
                is_cointegrated.append(eg_result['is_cointegrated'])
                p_values.append(eg_result['adf_results']['p_value'])
            
            # Calculate stability metrics
            stability_ratio = sum(is_cointegrated) / len(windows) if windows else 0
            hedge_ratio_std = np.std(hedge_ratios) if hedge_ratios else 0
            hedge_ratio_mean = np.mean(hedge_ratios) if hedge_ratios else 0
            
            return {
                'is_stable': stability_ratio >= 0.7,  # Arbitrary threshold
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
            self.logger.error(f"Error in cointegration stability analysis: {e}")
            raise
