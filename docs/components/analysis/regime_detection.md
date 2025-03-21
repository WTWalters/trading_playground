# Market Regime Detection

**Type**: Component Documentation  
**Last Updated**: 2025-03-17  
**Status**: Completed

## Related Documents

- [Parameter Management](./parameter_management.md)
- [Cointegration Analysis](./cointegration_analysis.md)
- [Backtesting Framework](../backtesting/backtesting_framework.md)
- [Performance Metrics](../../results/performance_metrics.md)

## Overview

The Market Regime Detection component of the TITAN Trading System identifies distinct market conditions and classifies them into regimes. These regime classifications drive parameter adaptation, risk management decisions, and trading strategy selection. The system employs multiple detection mechanisms and integrates various data sources to provide robust regime identification.

## Key Features

- Multi-dimensional regime classification (volatility, correlation, trend, liquidity)
- Multi-timeframe analysis for early detection of regime shifts
- Integration of technical, fundamental, and alternative data sources
- Probabilistic transition modeling for regime evolution prediction
- Adaptive threshold calibration based on historical analysis
- Visualization and explanation of regime characteristics

## Theoretical Background

### Market Regimes

Market regimes are distinct periods characterized by specific statistical properties and market behaviors. Unlike random fluctuations, regimes typically persist for meaningful periods and display recognizable patterns. Common regime types include:

1. **Volatility Regimes**: Periods of low, normal, or high volatility
2. **Correlation Regimes**: Periods of high or low correlation across assets
3. **Trend Regimes**: Periods of trending, mean-reverting, or random-walk behavior
4. **Liquidity Regimes**: Periods of high or low market liquidity
5. **Macro Economic Regimes**: Periods defined by economic conditions (growth, recession, inflation, etc.)

### Regime Detection Approaches

The system employs multiple approaches to regime detection:

1. **Statistical Approaches**: Methods based on statistical properties of market data
2. **Model-Based Approaches**: Methods using explicit models of regime transitions
3. **Machine Learning Approaches**: Data-driven methods for pattern recognition
4. **Fundamental Approaches**: Methods incorporating economic indicators
5. **Hybrid Approaches**: Combinations of different detection methods

## Implementation

### Regime Types

The TITAN system identifies the following regime types:

#### Volatility Regimes

Classified based on realized volatility metrics:

```python
def classify_volatility_regime(price_series, lookback=20):
    """
    Classify volatility regime based on realized volatility.
    
    Args:
        price_series: Series of price data
        lookback: Lookback period for volatility calculation
        
    Returns:
        str: Volatility regime classification ('LOW', 'NORMAL', 'HIGH')
    """
    # Calculate log returns
    returns = np.log(price_series / price_series.shift(1)).dropna()
    
    # Calculate realized volatility (annualized)
    realized_vol = returns.rolling(lookback).std() * np.sqrt(252)
    current_vol = realized_vol.iloc[-1]
    
    # Calculate long-term volatility statistics
    long_term_vol = returns.std() * np.sqrt(252)
    long_term_vol_std = realized_vol.std()
    
    # Classify regime
    if current_vol < long_term_vol - 0.5 * long_term_vol_std:
        return "LOW"
    elif current_vol > long_term_vol + 0.5 * long_term_vol_std:
        return "HIGH"
    else:
        return "NORMAL"
```

#### Correlation Regimes

Classified based on average pairwise correlations:

```python
def classify_correlation_regime(return_matrix, lookback=20):
    """
    Classify correlation regime based on average pairwise correlation.
    
    Args:
        return_matrix: Matrix of asset returns (rows=time, columns=assets)
        lookback: Lookback period for correlation calculation
        
    Returns:
        str: Correlation regime classification ('LOW', 'NORMAL', 'HIGH')
    """
    # Calculate rolling correlation matrix
    corr_matrix = return_matrix.rolling(lookback).corr()
    
    # Calculate average pairwise correlation (excluding self-correlations)
    n_assets = return_matrix.shape[1]
    avg_corr = (corr_matrix.sum().sum() - n_assets) / (n_assets * (n_assets - 1))
    
    # Calculate long-term correlation statistics
    long_term_corr = return_matrix.corr().values
    long_term_avg_corr = (long_term_corr.sum() - n_assets) / (n_assets * (n_assets - 1))
    
    # Estimate standard deviation of correlation
    corr_std = np.std([corr_matrix.iloc[i].values.mean() 
                      for i in range(len(corr_matrix))])
    
    # Classify regime
    if avg_corr < long_term_avg_corr - 0.5 * corr_std:
        return "LOW"
    elif avg_corr > long_term_avg_corr + 0.5 * corr_std:
        return "HIGH"
    else:
        return "NORMAL"
```

#### Trend Regimes

Classified based on trend strength indicators:

```python
def classify_trend_regime(price_series, lookback=20):
    """
    Classify trend regime based on trend strength indicators.
    
    Args:
        price_series: Series of price data
        lookback: Lookback period for trend calculation
        
    Returns:
        str: Trend regime classification ('TRENDING', 'MEAN_REVERTING', 'RANDOM')
    """
    # Calculate log returns
    returns = np.log(price_series / price_series.shift(1)).dropna()
    
    # Calculate autocorrelation
    autocorr = returns.autocorr(lag=1)
    
    # Calculate Hurst exponent (simplified calculation)
    # H > 0.5: trending, H < 0.5: mean-reverting, H ≈ 0.5: random walk
    lags = range(2, 20)
    tau = [np.sqrt(np.std(np.subtract(returns[lag:], returns[:-lag]))) 
           for lag in lags]
    poly = np.polyfit(np.log(lags), np.log(tau), 1)
    hurst = poly[0] / 2.0
    
    # Classify regime based on Hurst exponent
    if hurst > 0.55:
        return "TRENDING"
    elif hurst < 0.45:
        return "MEAN_REVERTING"
    else:
        return "RANDOM"
```

#### Liquidity Regimes

Classified based on trading volume and bid-ask spreads:

```python
def classify_liquidity_regime(volume_series, spread_series=None, lookback=20):
    """
    Classify liquidity regime based on volume and spreads.
    
    Args:
        volume_series: Series of trading volume data
        spread_series: Series of bid-ask spread data (optional)
        lookback: Lookback period for calculation
        
    Returns:
        str: Liquidity regime classification ('LOW', 'NORMAL', 'HIGH')
    """
    # Calculate normalized volume (Z-score)
    vol_mean = volume_series.rolling(252).mean()
    vol_std = volume_series.rolling(252).std()
    norm_volume = (volume_series - vol_mean) / vol_std
    
    # Use spread information if available
    if spread_series is not None:
        # Calculate normalized spread (Z-score)
        spread_mean = spread_series.rolling(252).mean()
        spread_std = spread_series.rolling(252).std()
        norm_spread = (spread_series - spread_mean) / spread_std
        
        # Combine volume and spread into liquidity score
        # High volume and low spread = high liquidity
        liquidity_score = norm_volume - norm_spread
    else:
        # Use just volume as liquidity proxy
        liquidity_score = norm_volume
    
    # Calculate recent average
    recent_liquidity = liquidity_score.rolling(lookback).mean().iloc[-1]
    
    # Classify regime
    if recent_liquidity > 0.8:
        return "HIGH"
    elif recent_liquidity < -0.8:
        return "LOW"
    else:
        return "NORMAL"
```

#### Market Sentiment Regimes

Classified based on sentiment indicators and volatility indexes:

```python
def classify_sentiment_regime(vix_series, sentiment_series=None, lookback=20):
    """
    Classify sentiment regime based on VIX and sentiment indicators.
    
    Args:
        vix_series: Series of VIX data
        sentiment_series: Series of sentiment data (optional)
        lookback: Lookback period for calculation
        
    Returns:
        str: Sentiment regime classification ('FEAR', 'NEUTRAL', 'GREED')
    """
    # Calculate normalized VIX (Z-score)
    vix_mean = vix_series.rolling(252).mean()
    vix_std = vix_series.rolling(252).std()
    norm_vix = (vix_series - vix_mean) / vix_std
    
    # Combine with sentiment data if available
    if sentiment_series is not None:
        # Normalize sentiment
        sent_mean = sentiment_series.rolling(252).mean()
        sent_std = sentiment_series.rolling(252).std()
        norm_sent = (sentiment_series - sent_mean) / sent_std
        
        # Combine VIX and sentiment (VIX is inversely related to positive sentiment)
        sentiment_score = -norm_vix + norm_sent
    else:
        # Use inverse of VIX as sentiment proxy
        sentiment_score = -norm_vix
    
    # Calculate recent average
    recent_sentiment = sentiment_score.rolling(lookback).mean().iloc[-1]
    
    # Classify regime
    if recent_sentiment > 0.8:
        return "GREED"
    elif recent_sentiment < -0.8:
        return "FEAR"
    else:
        return "NEUTRAL"
```

### Hidden Markov Model Implementation

For more sophisticated regime detection, the system uses Hidden Markov Models:

```python
from hmmlearn import hmm
import numpy as np
import pandas as pd

def detect_regimes_hmm(returns, n_regimes=3):
    """
    Detect market regimes using a Hidden Markov Model.
    
    Args:
        returns: DataFrame of asset returns
        n_regimes: Number of regimes to detect
        
    Returns:
        array: Regime classifications
    """
    # Prepare data
    X = returns.values.reshape(-1, 1)
    
    # Initialize and fit HMM
    model = hmm.GaussianHMM(
        n_components=n_regimes,
        covariance_type="full",
        n_iter=1000,
        random_state=42
    )
    model.fit(X)
    
    # Predict regimes
    hidden_states = model.predict(X)
    
    # Interpret regimes (sort by volatility)
    means = model.means_.flatten()
    variances = np.array([model.covars_[i][0][0] for i in range(n_regimes)])
    
    # Map states to regimes (0=lowest volatility, n-1=highest volatility)
    state_to_regime = np.argsort(variances)
    regime_names = ["LOW_VOL", "NORMAL_VOL", "HIGH_VOL"][:n_regimes]
    
    # Create mapping
    mapping = {state: regime_names[i] for i, state in enumerate(state_to_regime)}
    
    # Return named regimes
    return np.array([mapping[state] for state in hidden_states])
```

### Multi-Factor Regime Classification

The system combines multiple factors for a comprehensive regime assessment:

```python
def classify_combined_regime(price_data, volume_data, vix_data, sentiment_data=None):
    """
    Classify overall market regime based on multiple factors.
    
    Args:
        price_data: DataFrame of price data for multiple assets
        volume_data: DataFrame of volume data for multiple assets
        vix_data: Series of VIX data
        sentiment_data: Series of sentiment data (optional)
        
    Returns:
        dict: Combined regime classifications
    """
    # Calculate returns
    returns = price_data.pct_change().dropna()
    
    # Classify individual regimes
    vol_regime = classify_volatility_regime(price_data.iloc[:, 0])
    corr_regime = classify_correlation_regime(returns)
    trend_regime = classify_trend_regime(price_data.iloc[:, 0])
    liq_regime = classify_liquidity_regime(volume_data.iloc[:, 0])
    sent_regime = classify_sentiment_regime(vix_data, sentiment_data)
    
    # Classify overall regime
    if vol_regime == "HIGH" and sent_regime == "FEAR":
        overall = "CRISIS"
    elif vol_regime == "LOW" and sent_regime == "GREED":
        overall = "EUPHORIA"
    elif vol_regime == "NORMAL" and corr_regime == "NORMAL":
        overall = "NEUTRAL"
    elif vol_regime == "LOW" and corr_regime == "LOW":
        overall = "FAVORABLE"
    elif trend_regime == "TRENDING":
        overall = "TRENDING"
    elif trend_regime == "MEAN_REVERTING":
        overall = "MEAN_REVERTING"
    else:
        overall = "MIXED"
    
    # Return all classifications
    return {
        "overall": overall,
        "volatility": vol_regime,
        "correlation": corr_regime,
        "trend": trend_regime,
        "liquidity": liq_regime,
        "sentiment": sent_regime
    }
```

### Regime Transition Modeling

The system models regime transitions to anticipate changes:

```python
def model_regime_transitions(regime_history, lookback=252):
    """
    Model transition probabilities between regimes.
    
    Args:
        regime_history: Series of regime classifications
        lookback: Lookback period for transition modeling
        
    Returns:
        DataFrame: Transition probability matrix
    """
    # Get recent regime history
    recent_regimes = regime_history[-lookback:]
    
    # Identify unique regimes
    unique_regimes = recent_regimes.unique()
    n_regimes = len(unique_regimes)
    
    # Initialize transition matrix
    transition_matrix = pd.DataFrame(
        data=np.zeros((n_regimes, n_regimes)),
        index=unique_regimes,
        columns=unique_regimes
    )
    
    # Count transitions
    for i in range(len(recent_regimes) - 1):
        from_regime = recent_regimes.iloc[i]
        to_regime = recent_regimes.iloc[i + 1]
        transition_matrix.loc[from_regime, to_regime] += 1
    
    # Convert to probabilities
    for regime in unique_regimes:
        total = transition_matrix.loc[regime].sum()
        if total > 0:
            transition_matrix.loc[regime] = transition_matrix.loc[regime] / total
    
    return transition_matrix
```

### Regime Duration Analysis

The system analyzes the typical duration of different regimes:

```python
def analyze_regime_duration(regime_history):
    """
    Analyze the typical duration of different regimes.
    
    Args:
        regime_history: Series of regime classifications
        
    Returns:
        dict: Average duration by regime
    """
    # Initialize variables
    current_regime = None
    current_duration = 0
    regime_durations = {}
    
    # Iterate through regime history
    for regime in regime_history:
        if regime == current_regime:
            # Continue current regime
            current_duration += 1
        else:
            # Save previous regime duration
            if current_regime is not None:
                if current_regime not in regime_durations:
                    regime_durations[current_regime] = []
                regime_durations[current_regime].append(current_duration)
            
            # Start new regime
            current_regime = regime
            current_duration = 1
    
    # Save final regime duration
    if current_regime is not None:
        if current_regime not in regime_durations:
            regime_durations[current_regime] = []
        regime_durations[current_regime].append(current_duration)
    
    # Calculate average durations
    avg_durations = {
        regime: sum(durations) / len(durations)
        for regime, durations in regime_durations.items()
    }
    
    return avg_durations
```

## Usage Examples

### Basic Regime Detection

```python
# Load market data
price_data = load_price_data("SPY", "2024-01-01", "2025-01-01")
volume_data = load_volume_data("SPY", "2024-01-01", "2025-01-01")
vix_data = load_vix_data("2024-01-01", "2025-01-01")

# Detect volatility regime
vol_regime = classify_volatility_regime(price_data, lookback=20)
print(f"Current volatility regime: {vol_regime}")

# Detect overall market regime
overall_regime = classify_combined_regime(
    price_data=price_data,
    volume_data=volume_data,
    vix_data=vix_data
)
print(f"Overall market regime: {overall_regime['overall']}")
```

### Regime Transition Analysis

```python
# Load historical regime classifications
regime_history = load_regime_history("2023-01-01", "2025-01-01")

# Model regime transitions
transition_matrix = model_regime_transitions(regime_history)
print("Regime transition probabilities:")
print(transition_matrix)

# Calculate average regime durations
durations = analyze_regime_duration(regime_history)
print("Average regime durations (days):")
for regime, duration in durations.items():
    print(f"{regime}: {duration:.1f}")
```

### Visualization

```python
# Create regime visualization
plt.figure(figsize=(12, 8))

# Plot asset price
plt.subplot(3, 1, 1)
plt.plot(price_data)
plt.title("Asset Price")

# Plot regime classifications
plt.subplot(3, 1, 2)
regime_numeric = pd.Series(regime_history).map({
    "CRISIS": 0,
    "UNFAVORABLE": 1,
    "NEUTRAL": 2,
    "FAVORABLE": 3,
    "EUPHORIA": 4
})
plt.plot(regime_numeric)
plt.yticks([0, 1, 2, 3, 4], 
           ["CRISIS", "UNFAVORABLE", "NEUTRAL", "FAVORABLE", "EUPHORIA"])
plt.title("Market Regime")

# Plot regime transition heatmap
plt.subplot(3, 1, 3)
sns.heatmap(transition_matrix, annot=True, cmap="Blues")
plt.title("Regime Transition Probabilities")

plt.tight_layout()
plt.show()
```

## Integration with Parameter Management

The regime detection component integrates with parameter management:

```python
# Detect current market regime
current_regime = detect_market_regime(market_data)

# Get regime-specific parameters
if current_regime == "FAVORABLE":
    # More aggressive parameters in favorable regimes
    entry_threshold = 1.8
    position_size_factor = 1.0
elif current_regime == "CRISIS":
    # More conservative parameters in crisis regimes
    entry_threshold = 2.5
    position_size_factor = 0.4
else:
    # Default parameters for neutral regimes
    entry_threshold = 2.0
    position_size_factor = 0.7
```

## Configuration Options

The regime detection component can be configured through a YAML configuration file:

```yaml
regime_detection:
  # Classification thresholds
  volatility:
    lookback: 20
    z_score_threshold: 0.8
  
  correlation:
    lookback: 20
    z_score_threshold: 0.8
  
  trend:
    lookback: 20
    hurst_threshold_high: 0.55
    hurst_threshold_low: 0.45
  
  liquidity:
    lookback: 20
    z_score_threshold: 0.8
  
  # HMM parameters
  hmm:
    enabled: true
    n_regimes: 3
    min_history: 252
  
  # Transition modeling
  transition:
    enabled: true
    lookback: 252
    min_observations: 20
  
  # Integration settings
  integration:
    persistence: true
    database_table: "regime_history"
    notification_enabled: true
```

## Performance Considerations

The regime detection component is designed for both efficiency and accuracy:

- **Computational Efficiency**: Incremental calculations where possible
- **Scalability**: Parallel processing for multiple assets
- **Memory Usage**: Optimized data structures for large datasets
- **Storage Efficiency**: Selective persistence of regime classifications
- **Real-time Capability**: Fast updates with new market data

## Validation and Testing

The regime detection component includes comprehensive validation and testing:

### Backtest Validation

```python
def validate_regime_detection(price_data, regime_detector, known_events=None):
    """
    Validate regime detection against known market events.
    
    Args:
        price_data: Historical price data
        regime_detector: Regime detection function
        known_events: Dictionary of known market events with dates and regimes
        
    Returns:
        dict: Validation results
    """
    # Detect regimes
    detected_regimes = regime_detector(price_data)
    
    # Validate against known events
    if known_events is not None:
        correct_classifications = 0
        total_events = len(known_events)
        
        for event_date, expected_regime in known_events.items():
            if event_date in detected_regimes.index:
                detected_regime = detected_regimes.loc[event_date]
                if detected_regime == expected_regime:
                    correct_classifications += 1
        
        accuracy = correct_classifications / total_events
    else:
        accuracy = None
    
    # Calculate regime statistics
    regime_counts = detected_regimes.value_counts()
    regime_durations = analyze_regime_duration(detected_regimes)
    
    return {
        "accuracy": accuracy,
        "regime_counts": regime_counts,
        "regime_durations": regime_durations
    }
```

### Regime Stability Analysis

```python
def analyze_regime_stability(price_data, regime_detector, window_size=20):
    """
    Analyze the stability of regime classifications.
    
    Args:
        price_data: Historical price data
        regime_detector: Regime detection function
        window_size: Window size for stability analysis
        
    Returns:
        float: Stability score (0-1)
    """
    # Initialize
    n_days = len(price_data)
    stability_scores = []
    
    # Calculate regime stability for each window
    for i in range(n_days - window_size):
        window_data = price_data.iloc[i:i+window_size]
        regime = regime_detector(window_data)
        
        # Check if regime is consistent within window
        regime_changes = (regime.shift() != regime).sum()
        stability = 1 - (regime_changes / (window_size - 1))
        stability_scores.append(stability)
    
    # Calculate overall stability
    overall_stability = sum(stability_scores) / len(stability_scores)
    
    return overall_stability
```

## Future Enhancements

Planned enhancements to the regime detection component include:

1. **Machine Learning Integration**:
   - Deep learning models for pattern recognition
   - Ensemble methods for regime classification
   - Reinforcement learning for adaptive thresholds

2. **Alternative Data Sources**:
   - News sentiment analysis
   - Social media indicators
   - Economic surprise indices
   - Options market signals

3. **Predictive Regime Modeling**:
   - Early warning indicators for regime shifts
   - Regime transition prediction models
   - Probability distribution forecasts

4. **Visualization Enhancements**:
   - Interactive regime visualizations
   - Real-time regime dashboards
   - Regime transition animations

## Known Limitations

1. **Lag in Detection**: Regime shifts may be detected with a lag
2. **False Positives**: Short-term fluctuations may be misclassified as regime changes
3. **Calibration Dependency**: Performance depends on calibration parameters
4. **Data Requirements**: Requires sufficient historical data for accurate classification

## See Also

- [Parameter Management](./parameter_management.md) - Adaptive parameter system
- [Backtesting Framework](../backtesting/backtesting_framework.md) - Strategy validation
- [Performance Metrics](../../results/performance_metrics.md) - Regime-specific metrics
- [Market Analysis Tools](../../developer/api_reference.md) - Market analysis APIs
