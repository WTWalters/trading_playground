# Getting Started with TITAN Trading System

**Type**: User Guide  
**Last Updated**: 2025-03-16  

## Related Documents

- [Configuration Guide](./configuration_guide.md)
- [Running the Pipeline](./running_pipeline.md)
- [Interpreting Results](./interpreting_results.md)

## Overview

This guide provides a comprehensive introduction to the TITAN Trading System for new users. It covers basic concepts, system capabilities, and steps to start using the system for statistical arbitrage trading.

## What is TITAN?

TITAN (Trading with Integrated Technical Analysis and Networks) is an advanced statistical arbitrage trading system designed to identify and exploit cointegrated pairs of securities. The system:

- Identifies pairs of securities that exhibit a stable long-term relationship
- Detects market regimes and adapts trading parameters accordingly
- Generates trading signals based on deviations from equilibrium
- Validates strategies through comprehensive backtesting
- Provides performance metrics and visualization tools

## Core Components

The TITAN system consists of several core components:

1. **Data Infrastructure**: Collects, processes, and stores market data
2. **Cointegration Analysis**: Identifies statistically significant relationships between securities
3. **Regime Detection**: Classifies market conditions into distinct regimes
4. **Parameter Management**: Adapts strategy parameters based on market conditions
5. **Backtesting Engine**: Validates strategies against historical data
6. **Signal Generation**: Creates trading signals based on statistical deviations
7. **Risk Management**: Controls risk exposure and position sizing

## System Requirements

To run the TITAN Trading System, you need:

- **Operating System**: Linux, macOS, or Windows
- **Python**: Version 3.12 or higher
- **Database**: PostgreSQL with TimescaleDB extension
- **Memory**: 8 GB RAM minimum (16 GB recommended)
- **Storage**: 50 GB free space (for market data)
- **Network**: Stable internet connection for data updates

## Installation

### Option 1: Docker (Recommended)

The easiest way to get started with TITAN is using Docker:

```bash
# Pull the TITAN Docker image
docker pull organizaton/titan-trading-system:latest

# Run the container
docker run -d \
  --name titan \
  -p 8000:8000 \
  -v /path/to/data:/app/data \
  -v /path/to/config:/app/config \
  organizaton/titan-trading-system:latest
```

### Option 2: Manual Installation

For manual installation, follow these steps:

1. **Clone the repository**:
   ```bash
   git clone https://github.com/organization/titan-trading-system.git
   cd titan-trading-system
   ```

2. **Install dependencies**:
   ```bash
   # Using Poetry (recommended)
   poetry install

   # Or using pip
   pip install -r requirements.txt
   ```

3. **Set up the database**:
   ```bash
   # Install TimescaleDB
   # Follow instructions at: https://docs.timescale.com/install/

   # Create the database
   createdb titan_trading

   # Enable TimescaleDB extension
   psql -d titan_trading -c "CREATE EXTENSION IF NOT EXISTS timescaledb CASCADE;"

   # Initialize the database
   python scripts/initialize_database.py
   ```

4. **Configure the system**:
   ```bash
   # Copy the template configuration
   cp config/template.yaml config/local.yaml

   # Edit the configuration file
   nano config/local.yaml
   ```

5. **Load test data** (optional):
   ```bash
   python scripts/load_test_data.py
   ```

## Quick Start

Once installed, follow these steps to start using the TITAN Trading System:

### Step 1: Configure Data Sources

Edit the configuration file to specify the data sources you want to use:

```yaml
data_sources:
  - type: csv
    path: /path/to/historical/data.csv
    symbol: AAPL
    source: csv
  
  - type: api
    url: https://api.example.com/data
    api_key: YOUR_API_KEY
    symbol: MSFT
    source: example_api
```

For more details, see the [Configuration Guide](./configuration_guide.md).

### Step 2: Load Market Data

Load market data into the system:

```bash
python scripts/data_ingestion.py --config config/local.yaml
```

This will:
- Retrieve data from the configured sources
- Process and clean the data
- Store it in the TimescaleDB database

### Step 3: Identify Cointegrated Pairs

Run the cointegration analysis to identify potential trading pairs:

```bash
python scripts/cointegration_analysis.py \
  --symbols "AAPL,MSFT,GOOG,AMZN,FB" \
  --start-date "2024-01-01" \
  --end-date "2025-01-01"
```

This will output a list of cointegrated pairs with their statistical properties.

### Step 4: Run a Backtest

Backtest a strategy based on the identified pairs:

```bash
python scripts/run_backtest.py \
  --pair "AAPL,MSFT" \
  --start-date "2024-01-01" \
  --end-date "2025-01-01" \
  --entry-z 2.0 \
  --exit-z 0.5
```

This will generate a performance report and visualizations of the strategy.

### Step 5: Generate Trading Signals

Generate trading signals for live trading:

```bash
python scripts/generate_signals.py \
  --pair "AAPL,MSFT" \
  --hedge-ratio 0.85 \
  --entry-z 2.0 \
  --exit-z 0.5
```

The system will continuously generate trading signals based on the specified parameters.

## Complete Workflow Example

Here's a complete example workflow for using the TITAN Trading System:

1. **Configure the system**:
   ```bash
   cp config/template.yaml config/my_strategy.yaml
   # Edit the configuration file with your parameters
   ```

2. **Load market data for a set of ETFs**:
   ```bash
   python scripts/data_ingestion.py \
     --config config/my_strategy.yaml \
     --symbols "SPY,QQQ,IWM,EEM,GLD,TLT,XLF,XLE,XLK,XLV"
   ```

3. **Run cointegration analysis**:
   ```bash
   python scripts/cointegration_analysis.py \
     --config config/my_strategy.yaml \
     --min-half-life 5 \
     --max-half-life 30 \
     --p-value-threshold 0.05
   ```

4. **Identify market regimes**:
   ```bash
   python scripts/regime_detection.py \
     --config config/my_strategy.yaml \
     --start-date "2024-01-01" \
     --end-date "2025-01-01"
   ```

5. **Optimize strategy parameters**:
   ```bash
   python scripts/parameter_optimization.py \
     --config config/my_strategy.yaml \
     --pair "SPY,QQQ" \
     --start-date "2024-01-01" \
     --end-date "2024-06-30"
   ```

6. **Run a backtest with optimized parameters**:
   ```bash
   python scripts/run_backtest.py \
     --config config/my_strategy.yaml \
     --pair "SPY,QQQ" \
     --start-date "2024-07-01" \
     --end-date "2025-01-01" \
     --adaptive-parameters
   ```

7. **Analyze backtest results**:
   ```bash
   python scripts/analyze_results.py \
     --results-file "results/backtest_SPY_QQQ_2024-07-01_2025-01-01.json"
   ```

8. **Generate live trading signals**:
   ```bash
   python scripts/generate_signals.py \
     --config config/my_strategy.yaml \
     --pair "SPY,QQQ" \
     --adaptive-parameters
   ```

## Common Use Cases

### Pairs Trading with ETFs

ETFs are excellent candidates for pairs trading due to their liquidity and sector exposure:

1. **Sector ETF Pairs**: XLE/USO (Energy), XLF/KBE (Finance), XLV/IBB (Healthcare)
2. **Index ETF Pairs**: SPY/QQQ, IWM/MDY, EEM/VWO
3. **Fixed Income Pairs**: TLT/IEF, LQD/HYG, MBB/TLT

### Cross-Asset Class Pairs

TITAN can also identify relationships between different asset classes:

1. **Equity/Commodity**: XLE/GLD, GDX/GLD
2. **Equity/Fixed Income**: SPY/TLT
3. **Currency/Commodity**: UUP/GLD, FXE/GLD

### Multi-Strategy Portfolio

Combine multiple pairs into a diversified portfolio:

1. **Configure pairs**: Set up multiple pairs in the configuration
2. **Optimize allocation**: Use the portfolio optimization tool
3. **Run a portfolio backtest**: Test the entire portfolio
4. **Generate portfolio signals**: Get signals for all pairs

## Best Practices

### Pair Selection

- Focus on related instruments with fundamental relationships
- Verify the stability of cointegration with rolling window tests
- Prefer pairs with half-life between 5 and 30 days
- Include pairs from different sectors for diversification

### Parameter Tuning

- Use walk-forward testing to validate parameters
- Adapt parameters to different market regimes
- Start conservative with position sizing
- Implement stops for risk management

### Monitoring

- Regularly check pair stability
- Watch for regime changes
- Monitor trading costs and slippage
- Track performance metrics over time

## Troubleshooting

### Common Issues

**No cointegrated pairs found:**
- Try different combinations of securities
- Adjust the p-value threshold
- Check for data quality issues
- Consider longer lookback periods

**Poor backtest performance:**
- Review parameter settings
- Check for market regime changes
- Ensure proper risk management
- Verify transaction cost assumptions

**Database connection errors:**
- Verify database is running
- Check connection parameters
- Ensure proper permissions
- Increase connection timeout

### Getting Help

If you encounter issues:

1. Check the [FAQ](../faq.md) for common questions
2. Review the error logs: `logs/titan.log`
3. Search for similar issues in the project's issue tracker
4. Post detailed questions to the user forum

## Next Steps

After you've set up and started using the TITAN Trading System, consider:

1. Reading the [Configuration Guide](./configuration_guide.md) for detailed configuration options
2. Exploring [Running the Pipeline](./running_pipeline.md) for advanced pipeline usage
3. Learning about [Interpreting Results](./interpreting_results.md) for in-depth analysis
4. Setting up automated execution using the trading API

## See Also

- [Configuration Guide](./configuration_guide.md) - Detailed configuration options
- [Running the Pipeline](./running_pipeline.md) - Advanced pipeline usage
- [Interpreting Results](./interpreting_results.md) - Analysis of system outputs
- [Backtesting Framework](../components/backtesting/backtesting_framework.md) - Details of the backtesting system
