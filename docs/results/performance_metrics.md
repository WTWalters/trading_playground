# Performance Metrics

**Type**: Results Documentation  
**Last Updated**: 2025-03-16  

## Related Documents

- [Pipeline Results](./pipeline_results.md)
- [Strategy Performance](./strategy_performance.md)
- [Backtesting Framework](../components/backtesting/backtesting_framework.md)

## Overview

This document describes the performance metrics used in the TITAN Trading System to evaluate strategies, assess risk, and measure overall system effectiveness. It provides definitions, calculation methodologies, and guidance on interpreting each metric.

## Return Metrics

### Total Return

Total return measures the overall percentage gain or loss of a strategy over a specified period.

**Formula:**
```
Total Return = (Ending Value / Beginning Value) - 1
```

**Interpretation:**
- Positive values indicate profit
- Negative values indicate loss
- Higher values suggest better performance

**Example:** A total return of 15.4% means that $10,000 invested initially would grow to $11,540.

### Annualized Return

Annualized return converts returns of different time periods to an annual basis for comparison.

**Formula:**
```
Annualized Return = (1 + Total Return)^(365/Days) - 1
```

**Interpretation:**
- Standardizes returns to an annual basis
- Allows comparison between strategies with different durations
- Higher values suggest better performance

**Example:** A 3-month return of 3.5% equates to an annualized return of approximately 14.8%.

### Compound Annual Growth Rate (CAGR)

CAGR measures the mean annual growth rate of an investment over a specified time period longer than one year.

**Formula:**
```
CAGR = (Ending Value / Beginning Value)^(1/Years) - 1
```

**Interpretation:**
- Smooths out volatility to show a steady growth rate
- More accurate for long-term performance assessment
- Higher values suggest better long-term performance

**Example:** An investment that grows from $10,000 to $16,000 over 4 years has a CAGR of approximately 12.5%.

## Risk-Adjusted Return Metrics

### Sharpe Ratio

The Sharpe ratio measures the excess return per unit of risk (volatility).

**Formula:**
```
Sharpe Ratio = (Strategy Return - Risk-Free Rate) / Strategy Standard Deviation
```

**Interpretation:**
- Higher values indicate better risk-adjusted returns
- Values > 1.0 are generally considered acceptable
- Values > 2.0 are considered excellent
- Values < 0 indicate underperformance compared to the risk-free rate

**Example:** A Sharpe ratio of 1.8 means that for each unit of risk taken, the strategy generates 1.8 units of excess return.

### Sortino Ratio

The Sortino ratio is similar to the Sharpe ratio but uses downside deviation instead of standard deviation, focusing only on harmful volatility.

**Formula:**
```
Sortino Ratio = (Strategy Return - Risk-Free Rate) / Downside Deviation
```

**Interpretation:**
- Higher values indicate better risk-adjusted returns
- More relevant than Sharpe ratio for strategies with non-normal return distributions
- Better metric for assessing strategies with positive skewness

**Example:** A Sortino ratio of 2.5 means that for each unit of downside risk, the strategy generates 2.5 units of excess return.

### Calmar Ratio

The Calmar ratio measures return relative to maximum drawdown risk.

**Formula:**
```
Calmar Ratio = Annualized Return / Maximum Drawdown
```

**Interpretation:**
- Higher values indicate better return per unit of drawdown risk
- Values > 1.0 are generally considered good
- Values > 3.0 are considered excellent

**Example:** A Calmar ratio of 2.0 means that for each 1% of maximum drawdown risk, the strategy generates 2% of annualized return.

## Drawdown Metrics

### Maximum Drawdown

Maximum drawdown measures the largest peak-to-trough decline in portfolio value.

**Formula:**
```
Maximum Drawdown = (Trough Value - Peak Value) / Peak Value
```

**Interpretation:**
- Lower values indicate less severe drawdowns
- Important for assessing worst-case risk scenarios
- Critical for psychological factors in trading

**Example:** A maximum drawdown of 15% means that at its worst point, the strategy lost 15% from its previous peak value.

### Average Drawdown

Average drawdown measures the mean of all drawdowns exceeding a specified threshold.

**Formula:**
```
Average Drawdown = Sum of All Drawdowns / Number of Drawdowns
```

**Interpretation:**
- Lower values indicate less severe typical drawdowns
- Provides a more balanced view of drawdown risk than maximum drawdown alone
- Important for assessing everyday risk rather than just extreme events

**Example:** An average drawdown of 4.2% means that, on average, the strategy experiences 4.2% declines from peak to trough.

### Drawdown Duration

Drawdown duration measures the time it takes to recover from a drawdown.

**Formula:**
```
Drawdown Duration = Recovery Date - Drawdown Start Date
```

**Interpretation:**
- Lower values indicate faster recovery
- Important for assessing opportunity cost during drawdowns
- Critical for strategies with time constraints or liquidity needs

**Example:** An average drawdown duration of 18 days means that typically, it takes 18 days to recover from a drawdown.

## Trade Metrics

### Win Rate

Win rate measures the percentage of trades that result in a profit.

**Formula:**
```
Win Rate = Number of Winning Trades / Total Number of Trades
```

**Interpretation:**
- Higher values indicate more consistent profitability
- Should be considered alongside profit/loss ratio
- Different strategies may have optimal win rates (trend following typically lower, mean reversion typically higher)

**Example:** A win rate of 62% means that 62% of trades are profitable.

### Profit Factor

Profit factor measures the ratio of gross profit to gross loss.

**Formula:**
```
Profit Factor = Gross Profit / Gross Loss
```

**Interpretation:**
- Values > 1.0 indicate overall profitability
- Values > 2.0 generally indicate strong strategies
- Values > 3.0 indicate excellent strategies

**Example:** A profit factor of 2.5 means that for every $1 of loss, the strategy generates $2.50 of profit.

### Average Trade Profit/Loss

Average trade profit/loss measures the mean return per trade.

**Formula:**
```
Average Trade P/L = Net Profit or Loss / Total Number of Trades
```

**Interpretation:**
- Higher values indicate greater profit per trade
- Should be positive for profitable strategies
- Important for assessing efficiency of capital use

**Example:** An average trade P/L of 0.8% means that, on average, each trade generates a 0.8% return on the invested capital.

### Expectancy

Expectancy measures the expected profit or loss per dollar risked on each trade.

**Formula:**
```
Expectancy = (Win Rate × Average Win) - ((1 - Win Rate) × Average Loss)
```

**Interpretation:**
- Higher values indicate greater expected return per unit of risk
- Should be positive for profitable strategies
- Critical for position sizing decisions

**Example:** An expectancy of 0.4 means that for each $1 risked, the expected return is $0.40.

## Time Metrics

### Time in Market

Time in market measures the percentage of time that capital is deployed in positions.

**Formula:**
```
Time in Market = Sum of All Trade Durations / Total Time Period
```

**Interpretation:**
- Higher values indicate greater capital utilization
- Lower values may indicate more selective entry criteria
- Neither high nor low is inherently better; optimal value depends on strategy type

**Example:** A time in market of 65% means that capital is deployed in positions 65% of the time.

### Average Holding Period

Average holding period measures the mean duration of trades.

**Formula:**
```
Average Holding Period = Sum of All Trade Durations / Number of Trades
```

**Interpretation:**
- Varies widely by strategy type
- Shorter periods typically indicate higher turnover
- Longer periods typically indicate trend-following or longer-term strategies

**Example:** An average holding period of 4.5 days means that, on average, positions are held for 4.5 days.

## Volatility Metrics

### Annualized Volatility

Annualized volatility measures the standard deviation of returns, converted to an annual basis.

**Formula:**
```
Annualized Volatility = Standard Deviation of Returns × √(Trading Days Per Year)
```

**Interpretation:**
- Lower values indicate more stable returns
- Important for risk management and position sizing
- Neither high nor low is inherently better; optimal value depends on risk tolerance

**Example:** An annualized volatility of 12% means that approximately two-thirds of annual returns fall within ±12% of the mean return.

### Beta

Beta measures the volatility of a strategy relative to a benchmark.

**Formula:**
```
Beta = Covariance(Strategy Returns, Benchmark Returns) / Variance(Benchmark Returns)
```

**Interpretation:**
- Beta = 1: Strategy moves with the benchmark
- Beta > 1: Strategy is more volatile than the benchmark
- Beta < 1: Strategy is less volatile than the benchmark
- Beta < 0: Strategy moves opposite to the benchmark

**Example:** A beta of 0.75 means that the strategy is 25% less volatile than the benchmark.

### Correlation

Correlation measures the statistical relationship between the strategy returns and a benchmark or another strategy.

**Formula:**
```
Correlation = Covariance(Strategy Returns, Benchmark Returns) / (SD(Strategy Returns) × SD(Benchmark Returns))
```

**Interpretation:**
- Values range from -1 to 1
- Values near 1 indicate strong positive correlation
- Values near -1 indicate strong negative correlation
- Values near 0 indicate no correlation

**Example:** A correlation of 0.2 with the S&P 500 indicates a weak positive relationship with the overall market.

## Regime-Specific Metrics

### Regime Performance Ratio

Regime performance ratio measures the performance in a specific market regime relative to overall performance.

**Formula:**
```
Regime Performance Ratio = Annualized Return in Regime / Overall Annualized Return
```

**Interpretation:**
- Values > 1.0 indicate outperformance in the specific regime
- Values < 1.0 indicate underperformance in the specific regime
- Important for assessing strategy robustness across different market conditions

**Example:** A regime performance ratio of 1.4 in high-volatility regimes means that the strategy performs 40% better in high-volatility environments than its overall average.

### Regime Stability

Regime stability measures the consistency of a strategy's performance metrics across different market regimes.

**Formula:**
```
Regime Stability = 1 - Coefficient of Variation of Key Metrics Across Regimes
```

**Interpretation:**
- Values closer to 1.0 indicate more stable performance across regimes
- Values closer to 0 indicate highly regime-dependent performance
- Important for assessing strategy robustness

**Example:** A regime stability of 0.85 for Sharpe ratio means that the Sharpe ratio remains relatively consistent across different market regimes.

## Custom TITAN Metrics

### Pair Stability Index

Pair stability index measures the stability of cointegrated pairs over time.

**Formula:**
```
Pair Stability Index = 1 - Average(Normalized Half-Life Changes)
```

**Interpretation:**
- Values closer to 1.0 indicate stable cointegrated relationships
- Values closer to 0 indicate unstable relationships
- Important for assessing the reliability of pair trading strategies

**Example:** A pair stability index of 0.92 indicates that the cointegrated relationship remains highly stable over time.

### Adaptation Effectiveness

Adaptation effectiveness measures the improvement in performance from parameter adaptation.

**Formula:**
```
Adaptation Effectiveness = Risk-Adjusted Return with Adaptation / Risk-Adjusted Return without Adaptation
```

**Interpretation:**
- Values > 1.0 indicate that adaptation improves performance
- Higher values indicate more effective adaptation
- Important for assessing the value of the adaptive parameter management system

**Example:** An adaptation effectiveness of 1.25 means that adaptive parameters improve risk-adjusted returns by 25% compared to fixed parameters.

## Calculation Methodology

The TITAN Trading System calculates all performance metrics using the following methodology:

1. **Data Cleaning**:
   - Remove outliers (values beyond 3 standard deviations)
   - Handle missing data through forward filling
   - Adjust for any corporate actions or splits

2. **Return Calculation**:
   - Use logarithmic returns for statistical analyses
   - Use simple returns for reporting
   - Account for transaction costs in all calculations

3. **Time Weighting**:
   - Use time-weighted return calculations for overall performance
   - Use money-weighted returns for fund management metrics
   - Apply appropriate day count conventions

4. **Benchmark Comparison**:
   - Use appropriate benchmarks for each strategy type
   - Calculate excess returns relative to both risk-free rate and benchmark
   - Apply risk adjustments based on benchmark characteristics

## Reporting Standards

The system follows these reporting standards for consistency:

1. **Time Periods**:
   - Daily, weekly, monthly, quarterly, and annual reporting options
   - Rolling window calculations for trend analysis
   - Custom date range functionality for specific analyses

2. **Confidence Intervals**:
   - Include 95% confidence intervals for key metrics
   - Report statistical significance where appropriate
   - Indicate sample size for all metric calculations

3. **Visualization**:
   - Consistent color coding (green for positive, red for negative)
   - Appropriate chart types for each metric
   - Interactive visualizations with drill-down capabilities

## See Also

- [Pipeline Results](./pipeline_results.md) - Results from running the TITAN Trading System pipeline
- [Strategy Performance](./strategy_performance.md) - Detailed performance analysis by strategy
- [Backtesting Framework](../components/backtesting/backtesting_framework.md) - Details of the backtesting methodology
