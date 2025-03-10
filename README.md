# TITAN Trading Platform

A professional-grade, modular system for statistical arbitrage and regime-adaptive trading.

## Overview

TITAN Trading Platform is a comprehensive system for market data ingestion, analysis, and statistical arbitrage trading. It features robust data validation, pair selection, regime detection, and backtesting capabilities.

The platform is designed with several key principles:
- Event-driven, loosely coupled components
- Database-centered with comprehensive time-series capabilities
- Statistically rigorous with proper validation methodologies
- Risk-aware at every level of operation
- Adaptable to changing market regimes
- Performance-optimized for real-time decision making

## Getting Started

### Prerequisites

- Python 3.12+
- PostgreSQL 16 with TimescaleDB extension
- Poetry for dependency management

### Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/WTWalters/trading_playground.git
   cd trading_playground
   ```

2. Install dependencies using Poetry:
   ```bash
   poetry install
   ```

3. Create a `.env` file with your configuration:
   ```
   # Database settings
   DB_HOST=localhost
   DB_PORT=5432
   DB_DATABASE=trading
   DB_USER=your_username
   DB_PASSWORD=your_password

   # API Keys
   POLYGON_API_KEY=your_polygon_api_key
   ```

4. Initialize the database:
   ```bash
   psql -U your_username -d trading -f migrations/initial/001_market_data_schema.sql
   psql -U your_username -d trading -f migrations/initial/002_statistical_features_schema.sql
   ```

## Usage

### Fetching Market Data

To fetch data for a single symbol:

```bash
python scripts/fetch_and_store.py --symbol AAPL --provider yahoo --days 30 --timeframe 1d
```

To fetch data for multiple symbols:

```bash
python scripts/bulk_import.py --symbols AAPL,MSFT,GOOGL --provider yahoo --days 30 --timeframe 1d
```

### Running Statistical Analysis

*(Coming soon)*

### Pair Selection and Trading

*(Coming soon)*

## Architecture

The platform consists of several key subsystems:

### Data Subsystem
- Data ingestion from multiple providers
- Validation and cleaning
- Storage in TimescaleDB
- Data enrichment

### Alpha Subsystem
- Statistical feature generation
- Signal generation
- Alpha combination

### Portfolio Subsystem
- Position sizing
- Risk management
- Strategy allocation

### Execution Subsystem
- Order management
- Execution algorithms
- Market impact modeling

### Analytics Subsystem
- Performance tracking
- Risk analysis
- Visualization

## Development

### Project Structure

```
trading_playground/
├── config/                 # Configuration files
├── docs/                   # Documentation
├── migrations/             # Database migrations
│   └── initial/            # Initial schema setup
├── notebooks/              # Jupyter notebooks for analysis
├── scripts/                # Utility scripts
├── src/                    # Source code
│   ├── config/             # Configuration modules
│   ├── data_ingestion/     # Data ingestion components
│   │   └── providers/      # Data provider implementations
│   ├── market_analysis/    # Market analysis components
│   │   ├── microstructure/ # Market microstructure analysis
│   │   └── time_series/    # Time series analysis
│   ├── strategies/         # Trading strategy implementations
│   └── utils/              # Utility functions
└── tests/                  # Test suite
```

### Contributing

1. Create a new branch for your feature or fix
2. Write tests for your code
3. Ensure all tests pass
4. Submit a pull request

## License

This project is licensed under the MIT License - see the LICENSE file for details.