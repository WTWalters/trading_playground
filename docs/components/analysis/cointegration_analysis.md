# Cointegration Analysis

**Type**: Component Documentation  
**Last Updated**: 2025-03-16  
**Status**: Completed

## Related Documents

- [Parameter Management](./parameter_management.md)
- [Regime Detection](./regime_detection.md)
- [Backtesting Framework](../backtesting/backtesting_framework.md)
- [Data Ingestion Pipeline](../data/data_ingestion.md)

## Overview

The Cointegration Analysis component is a core element of the TITAN Trading System's statistical arbitrage strategy. It identifies pairs of securities that exhibit a statistically significant long-term equilibrium relationship, making them suitable candidates for mean-reversion trading.

## Theoretical Background

Cointegration is a statistical property of time series where two or more non-stationary series can be combined to form a stationary series. In financial terms, this means that while the individual price series might wander extensively (like random walks), there exists a linear combination that behaves as a stationary, mean-reverting process.

This component implements the Engle-Granger two-step method for cointegration testing:

1. **Unit Root Testing**: Confirm that individual price series are non-stationary
2. **Cointegration Testing**: Test if a linear combination of the series is stationary
3. **Parameter Estimation**: Estimate the hedge ratio and other parameters

## Implementation

### Main Components

The cointegration analysis module consists of several key subcomponents:

#### Pair Screener

Identifies potential pairs for cointegration testing based on:
- Sector/industry relationship
- Historical correlation
- Liquidity and trading volume
- Market capitalization similarity

#### Cointegration Tester

Tests pairs for cointegration using:
- Augmented Dickey-Fuller (ADF) test for stationarity
- Johansen test for multivariate cointegration
- Phillips-Ouliaris test as a robustness check

#### Parameter Estimator

Estimates key parameters for pairs trading:
- Hedge ratio (coefficient in cointegrating relationship)
- Half-life of mean reversion
- Historical spread volatility
- Z-score thresholds for entry/exit

#### Pair Monitor

Monitors the stability of cointegrated pairs over time:
- Rolling-window cointegration tests
- Structural break detection
- Regime-specific parameter adjustments

### Implementation Details

#### Data Preparation

```python
def prepare_data(symbol1, symbol2, start_date, end_date, source='synthetic'):
    """
    Retrieve and prepare price data for cointegration analysis.
    
    Args:
        symbol1 (str): First symbol in the pair
        symbol2 (str): Second symbol in the pair
        start_date (datetime): Start date for analysis
        end_date (datetime): End date for analysis
        source (str): Data source identifier
        
    Returns:
        tuple: (prices1, prices2) - prepared price series
    """
    # Retrieve prices from database
    prices1 = database.get_prices(symbol1, start_date, end_date, source)
    prices2 = database.get_prices(symbol2, start_date, end_date, source)
    
    # Align dates and handle missing values
    combined = pd.merge(prices1, prices2, on='date', how='inner')
    
    # Log transform prices
    log_prices1 = np.log(combined[f'{symbol1}_close'])
    log_prices2 = np.log(combined[f'{symbol2}_close'])
    
    return log_prices1, log_prices2
```

#### Cointegration Testing

```python
def test_cointegration(prices1, prices2, significance_level=0.05):
    """
    Test for cointegration between two price series.
    
    Args:
        prices1 (array-like): First price series
        prices2 (array-like): Second price series
        significance_level (float): P-value threshold for significance
        
    Returns:
        dict: Cointegration test results
    """
    # Step 1: Test for unit roots in individual series
    adf_result1 = adfuller(prices1)
    adf_result2 = adfuller(prices2)
    
    # Only proceed if both series are non-stationary
    if adf_result1[1] < significance_level or adf_result2[1] < significance_level:
        return {"is_cointegrated": False, "reason": "At least one series is stationary"}
    
    # Step 2: Estimate cointegrating relationship
    X = sm.add_constant(prices1)
    model = sm.OLS(prices2, X).fit()
    hedge_ratio = model.params[1]
    
    # Step 3: Test if residuals are stationary
    spread = prices2 - hedge_ratio * prices1
    coint_result = adfuller(spread)
    
    is_cointegrated = coint_result[1] < significance_level
    
    return {
        "is_cointegrated": is_cointegrated,
        "hedge_ratio": hedge_ratio,
        "p_value": coint_result[1],
        "adf_statistic": coint_result[0],
        "spread": spread
    }
```

#### Half-life Estimation

```python
def estimate_half_life(spread):
    """
    Estimate the half-life of mean reversion for a spread series.
    
    Args:
        spread (array-like): The spread series
        
    Returns:
        float: Estimated half-life in days
    """
    # Calculate lag-1 spread
    lag_spread = spread.shift(1)
    lag_spread = lag_spread.dropna()
    spread = spread.iloc[1:]
    
    # Estimate AR(1) coefficient
    X = sm.add_constant(lag_spread)
    model = sm.OLS(spread, X).fit()
    ar_coef = model.params[1]
    
    # Calculate half-life
    half_life = -np.log(2) / np.log(ar_coef)
    
    return half_life
```

## Performance Considerations

The cointegration analysis component is designed for both batch processing and incremental updates:

- **Batch Mode**: Analyzes the entire universe of pairs at scheduled intervals
- **Incremental Mode**: Updates existing pairs with new data as it becomes available

Computational optimizations include:

- Parallel processing for testing multiple pairs
- Caching of intermediate results
- Prioritization of pairs based on pre-screening metrics
- Early termination for pairs that clearly fail tests

## Configuration Options

The component can be configured through a YAML configuration file:

```yaml
cointegration:
  # Statistical parameters
  significance_level: 0.05
  min_half_life: 1.0
  max_half_life: 30.0
  
  # Pair screening
  correlation_threshold: 0.5
  min_liquidity: 100000
  max_market_cap_ratio: 5.0
  
  # Stability requirements
  min_stability_period: 60
  rolling_window_size: 180
  
  # Computational settings
  parallel_processing: true
  max_workers: 8
  caching_enabled: true
```

## Usage Example

```python
from titan.analysis.cointegration import CointegrationAnalyzer

# Initialize the analyzer
analyzer = CointegrationAnalyzer(config_path='config/cointegration.yaml')

# Screen the universe for potential pairs
potential_pairs = analyzer.screen_universe(
    symbols=['AAPL', 'MSFT', 'GOOG', 'AMZN', 'FB'],
    start_date='2024-01-01',
    end_date='2025-01-01'
)

# Test pairs for cointegration
cointegrated_pairs = analyzer.test_pairs(potential_pairs)

# Save results to database
analyzer.save_pairs(cointegrated_pairs)

# Generate pair summary
pair_summary = analyzer.generate_summary(cointegrated_pairs)
print(pair_summary)
```

## Pair Stability Monitoring

Once cointegrated pairs are identified, they are continuously monitored for stability:

```python
def monitor_pair_stability(pair_id, lookback_days=30):
    """
    Monitor the stability of a cointegrated pair.
    
    Args:
        pair_id (int): ID of the pair to monitor
        lookback_days (int): Number of days to look back
        
    Returns:
        dict: Stability metrics
    """
    # Retrieve pair information
    pair = database.get_pair(pair_id)
    
    # Get recent spread data
    recent_spread = database.get_pair_spread(
        pair_id, 
        days=lookback_days
    )
    
    # Recalculate cointegration statistics
    coint_result = adfuller(recent_spread)
    
    # Check if still cointegrated
    is_still_cointegrated = coint_result[1] < 0.05
    
    # Calculate stability metrics
    spread_volatility = np.std(recent_spread)
    current_half_life = estimate_half_life(recent_spread)
    half_life_change = abs(current_half_life - pair['half_life']) / pair['half_life']
    
    return {
        "is_stable": is_still_cointegrated and half_life_change < 0.3,
        "current_p_value": coint_result[1],
        "current_half_life": current_half_life,
        "half_life_change": half_life_change,
        "spread_volatility": spread_volatility
    }
```

## Future Improvements

Planned enhancements to the cointegration analysis component:

1. **Adaptive Significance Levels**: Adjust significance thresholds based on market regimes
2. **Machine Learning Integration**: Use ML models to predict cointegration breakdown
3. **Multi-factor Cointegration**: Extend beyond pairs to multi-asset cointegration
4. **Kalman Filter Implementation**: Dynamic estimation of hedge ratios
5. **Advanced Structural Break Detection**: Improved methods for detecting relationship changes

## See Also

- [Backtesting Framework](../backtesting/backtesting_framework.md) - Testing pair trading strategies
- [Regime Detection](./regime_detection.md) - Market regime classification
- [Position Sizing](../trading/position_sizing.md) - Optimal position sizing for pairs
- [Risk Controls](../trading/risk_controls.md) - Risk management for pair trading
