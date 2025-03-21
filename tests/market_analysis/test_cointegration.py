"""
Unit tests for the cointegration framework.

These tests validate the functionality of the CointegrationTester and
MeanReversionAnalyzer classes.
"""

import pytest
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from unittest.mock import AsyncMock, MagicMock, patch

from src.database.manager import DatabaseManager
from src.market_analysis.cointegration import CointegrationTester
from src.market_analysis.mean_reversion import MeanReversionAnalyzer

# Sample data for testing
@pytest.fixture
def sample_price_data():
    # Create sample price data with known cointegration properties
    np.random.seed(42)
    dates = pd.date_range(start='2023-01-01', periods=100)
    
    # Create prices for symbol1 with random walk
    symbol1_prices = np.cumprod(1 + np.random.normal(0.0005, 0.01, 100))
    
    # Create prices for symbol2 based on symbol1 with a fixed relationship
    # plus some noise - this guarantees cointegration
    symbol2_prices = 1.5 * symbol1_prices + np.random.normal(0, 5, 100)
    
    # Create a non-cointegrated random series for symbol3
    symbol3_prices = np.cumprod(1 + np.random.normal(0.0007, 0.015, 100))
    
    # Create DataFrames
    df1 = pd.DataFrame({
        'open': symbol1_prices,
        'high': symbol1_prices * 1.01,
        'low': symbol1_prices * 0.99,
        'close': symbol1_prices,
        'volume': np.random.randint(1000, 10000, 100)
    }, index=dates)
    
    df2 = pd.DataFrame({
        'open': symbol2_prices,
        'high': symbol2_prices * 1.01,
        'low': symbol2_prices * 0.99,
        'close': symbol2_prices,
        'volume': np.random.randint(1000, 10000, 100)
    }, index=dates)
    
    df3 = pd.DataFrame({
        'open': symbol3_prices,
        'high': symbol3_prices * 1.01,
        'low': symbol3_prices * 0.99,
        'close': symbol3_prices,
        'volume': np.random.randint(1000, 10000, 100)
    }, index=dates)
    
    return {
        'symbol1': df1,
        'symbol2': df2,
        'symbol3': df3,
        'dates': dates
    }

@pytest.fixture
def mock_db_manager(sample_price_data):
    """Create a mock DatabaseManager with predefined data."""
    mock_manager = AsyncMock(spec=DatabaseManager)
    
    # Set up the mock get_market_data method
    async def mock_get_market_data(symbol, start_date, end_date, timeframe, source=None):
        if symbol in sample_price_data:
            df = sample_price_data[symbol]
            # Filter based on date range
            mask = (df.index >= start_date) & (df.index <= end_date)
            return df[mask]
        return pd.DataFrame()
    
    mock_manager.get_market_data.side_effect = mock_get_market_data
    return mock_manager

@pytest.mark.asyncio
async def test_adf_test(mock_db_manager, sample_price_data):
    """Test the ADF test for stationarity."""
    # Create a CointegrationTester with mock DB manager
    tester = CointegrationTester(mock_db_manager)
    
    # Create stationary residuals (random noise around zero)
    np.random.seed(42)
    stationary_residuals = np.random.normal(0, 1, 100)
    
    # Create non-stationary residuals (random walk)
    non_stationary_residuals = np.cumsum(np.random.normal(0, 1, 100))
    
    # Test stationary series
    result = await tester.adf_test(stationary_residuals)
    assert 'is_stationary' in result
    assert 'p_value' in result
    assert 'test_statistic' in result
    assert 'critical_values' in result
    
    # Test non-stationary series
    result = await tester.adf_test(non_stationary_residuals)
    assert 'is_stationary' in result
    assert 'p_value' in result
    assert 'test_statistic' in result
    assert 'critical_values' in result

@pytest.mark.asyncio
async def test_engle_granger_test(mock_db_manager, sample_price_data):
    """Test the Engle-Granger cointegration test."""
    # Create a CointegrationTester with mock DB manager
    tester = CointegrationTester(mock_db_manager)
    
    # Get price data
    symbol1_prices = sample_price_data['symbol1']['close'].values
    symbol2_prices = sample_price_data['symbol2']['close'].values
    symbol3_prices = sample_price_data['symbol3']['close'].values
    
    # Test cointegrated pair (symbol1 and symbol2)
    result = await tester.engle_granger_test(symbol1_prices, symbol2_prices)
    assert 'is_cointegrated' in result
    assert 'hedge_ratio' in result
    assert 'residuals' in result
    assert 'adf_results' in result
    assert 'regression_results' in result
    
    # Test non-cointegrated pair (symbol1 and symbol3)
    result = await tester.engle_granger_test(symbol1_prices, symbol3_prices)
    assert 'is_cointegrated' in result
    assert 'hedge_ratio' in result
    assert 'residuals' in result
    assert 'adf_results' in result
    assert 'regression_results' in result

@pytest.mark.asyncio
async def test_johansen_test(mock_db_manager, sample_price_data):
    """Test the Johansen cointegration test."""
    # Create a CointegrationTester with mock DB manager
    tester = CointegrationTester(mock_db_manager)
    
    # Get price data
    symbol1_prices = sample_price_data['symbol1']['close'].values
    symbol2_prices = sample_price_data['symbol2']['close'].values
    symbol3_prices = sample_price_data['symbol3']['close'].values
    
    # Test cointegrated pair (symbol1 and symbol2)
    result = await tester.johansen_test(symbol1_prices, symbol2_prices)
    assert 'is_cointegrated' in result
    assert 'p_value' in result
    assert 't_statistic' in result
    assert 'critical_values' in result
    
    # Test non-cointegrated pair (symbol1 and symbol3)
    result = await tester.johansen_test(symbol1_prices, symbol3_prices)
    assert 'is_cointegrated' in result
    assert 'p_value' in result
    assert 't_statistic' in result
    assert 'critical_values' in result

@pytest.mark.asyncio
async def test_select_pairs(mock_db_manager, sample_price_data):
    """Test the pair selection functionality."""
    # Create a CointegrationTester with mock DB manager
    tester = CointegrationTester(mock_db_manager)
    
    # Define test parameters
    symbols = ['symbol1', 'symbol2', 'symbol3']
    start_date = datetime.now() - timedelta(days=100)
    end_date = datetime.now()
    
    # Mock the component methods to isolate the test
    tester.engle_granger_test = AsyncMock(return_value={
        'is_cointegrated': True,
        'hedge_ratio': 1.5,
        'adf_results': {'p_value': 0.01}
    })
    
    tester.johansen_test = AsyncMock(return_value={
        'is_cointegrated': True,
        'p_value': 0.01
    })
    
    # Test pair selection
    result = await tester.select_pairs(symbols, start_date, end_date)
    
    # Check that the result is a list
    assert isinstance(result, list)
    assert tester.engle_granger_test.called
    assert tester.johansen_test.called

@pytest.mark.asyncio
async def test_calculate_half_life(mock_db_manager, sample_price_data):
    """Test the half-life calculation."""
    # Create a MeanReversionAnalyzer with mock DB manager
    analyzer = MeanReversionAnalyzer(mock_db_manager)
    
    # Create a stationary, mean-reverting series
    np.random.seed(42)
    mean = 100
    n = 100
    halflife = 20  # Known half-life value
    
    # Generate an AR(1) process with known half-life
    lambda_coefficient = -np.log(2) / halflife
    
    prices = np.zeros(n)
    prices[0] = mean
    for t in range(1, n):
        # Mean-reverting process: X_t = X_t-1 + lambda*(mean - X_t-1) + random_noise
        prices[t] = prices[t-1] + lambda_coefficient * (mean - prices[t-1]) + np.random.normal(0, 1)
    
    # Calculate half-life
    result = await analyzer.calculate_half_life(prices)
    
    # Check the result
    assert 'half_life' in result
    assert 'lambda' in result
    assert 'is_mean_reverting' in result
    assert 'regression_results' in result
    
    # The calculated half-life should be close to the true value
    assert 10 <= result['half_life'] <= 30  # Allow some margin due to random noise

@pytest.mark.asyncio
async def test_calculate_zscore(mock_db_manager, sample_price_data):
    """Test the Z-score calculation."""
    # Create a MeanReversionAnalyzer with mock DB manager
    analyzer = MeanReversionAnalyzer(mock_db_manager)
    
    # Create a sample spread
    np.random.seed(42)
    spread = np.random.normal(100, 10, 100)
    
    # Test full-series Z-score
    zscore = await analyzer.calculate_zscore(spread)
    assert len(zscore) == len(spread)
    assert abs(np.mean(zscore)) < 0.01  # Mean should be close to 0
    assert abs(np.std(zscore) - 1.0) < 0.01  # Std should be close to 1
    
    # Test rolling window Z-score
    window = 20
    rolling_zscore = await analyzer.calculate_zscore(spread, window)
    assert len(rolling_zscore) == len(spread)
    
    # First window-1 values should be 0 or NaN
    for i in range(window):
        assert np.isnan(rolling_zscore[i]) or rolling_zscore[i] == 0

@pytest.mark.asyncio
async def test_generate_mean_reversion_signals(mock_db_manager, sample_price_data):
    """Test the signal generation functionality."""
    # Create a MeanReversionAnalyzer with mock DB manager
    analyzer = MeanReversionAnalyzer(mock_db_manager)
    
    # Mock the calculate_zscore and calculate_half_life methods
    analyzer.calculate_zscore = AsyncMock(return_value=np.random.normal(0, 1, 100))
    analyzer.calculate_half_life = AsyncMock(return_value={
        'half_life': 15.0,
        'lambda': -0.05,
        'is_mean_reverting': True,
        'regression_results': {}
    })
    
    # Define test parameters
    symbol1 = 'symbol1'
    symbol2 = 'symbol2'
    hedge_ratio = 1.5
    start_date = datetime.now() - timedelta(days=100)
    end_date = datetime.now()
    
    # Test signal generation
    result = await analyzer.generate_mean_reversion_signals(
        symbol1, symbol2, hedge_ratio, start_date, end_date
    )
    
    # Check the result
    assert 'signals' in result
    assert 'metadata' in result
    assert analyzer.calculate_zscore.called
    assert analyzer.calculate_half_life.called
    
    # Check metadata
    assert result['metadata']['symbol1'] == symbol1
    assert result['metadata']['symbol2'] == symbol2
    assert result['metadata']['hedge_ratio'] == hedge_ratio
    assert 'half_life' in result['metadata']
    assert 'entry_threshold' in result['metadata']
    assert 'exit_threshold' in result['metadata']

@pytest.mark.asyncio
async def test_analyze_pair_statistics(mock_db_manager, sample_price_data):
    """Test the pair statistics analysis."""
    # Create a MeanReversionAnalyzer with mock DB manager
    analyzer = MeanReversionAnalyzer(mock_db_manager)
    
    # Mock the calculate_zscore and calculate_half_life methods
    analyzer.calculate_zscore = AsyncMock(return_value=np.random.normal(0, 1, 100))
    analyzer.calculate_half_life = AsyncMock(return_value={
        'half_life': 15.0,
        'lambda': -0.05,
        'is_mean_reverting': True,
        'regression_results': {}
    })
    
    # Define test parameters
    symbol1 = 'symbol1'
    symbol2 = 'symbol2'
    hedge_ratio = 1.5
    start_date = datetime.now() - timedelta(days=100)
    end_date = datetime.now()
    
    # Test pair statistics analysis
    result = await analyzer.analyze_pair_statistics(
        symbol1, symbol2, hedge_ratio, start_date, end_date
    )
    
    # Check the result
    assert 'correlation' in result
    assert 'half_life' in result
    assert 'is_mean_reverting' in result
    assert 'volatility' in result
    assert 'zscore_thresholds' in result
    assert 'return_potential' in result
    assert 'hedge_ratio' in result
    assert analyzer.calculate_zscore.called
    assert analyzer.calculate_half_life.called
