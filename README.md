# TITAN Trading Platform

## Overview

The TITAN Trading Platform is a comprehensive statistical arbitrage trading system designed for identifying and exploiting market inefficiencies through mean reversion, cointegration, and adaptive parameter management.

## Architecture

The platform consists of the following core components:

1. **Data Infrastructure**
   - TimescaleDB integration for time-series data storage
   - Market data ingestion pipeline

2. **Market Analysis**
   - Cointegration analysis framework for pair identification
   - Mean reversion signal generation
   - Regime detection and classification
   - Adaptive parameter management

3. **Strategy Execution**
   - Backtesting engine for strategy validation
   - Performance optimization with parallel processing
   - Real-time signal generation service

## Current Status

The platform has successfully implemented all core components, with recent updates including:

- Fixed data source issues in the backtesting engine
- Enhanced regime detection with macro indicator support
- Implemented adaptive parameter management
- Developed comprehensive testing framework (tests developed but pending execution)

## Technical Stack

- Python 3.12 with asyncio for asynchronous operations
- TimescaleDB for time-series data storage and retrieval
- Pandas/NumPy for data analysis and transformation
- Statsmodels for statistical testing (cointegration, stationarity)
- Matplotlib for visualization
- SQLAlchemy for database operations
- Custom modules for trading strategy and signal generation

## Development Priorities

1. **Testing and Validation**
   - Execute comprehensive test suite for all components
   - Perform historical backtesting across multiple market regimes
   - Validate the system with stress testing scenarios

2. **Performance Optimization**
   - Profile and optimize computational bottlenecks
   - Implement caching for regime detection results
   - Add parallel processing for multi-strategy optimization

3. **Documentation and Examples**
   - Expand API documentation with more examples
   - Create tutorial notebooks for different use cases
   - Add detailed explanations for key algorithms


## Documentation

For comprehensive project documentation, please refer to our reorganized documentation structure:

### Main Documentation Index

The [Documentation Index](./docs/documentation_index.md) provides a complete overview of all available documentation.

### Key Documentation Categories

- **[Architecture Documentation](./docs/architecture/)** - System design and component relationships
- **[Component Documentation](./docs/components/)** - Detailed information about system components
- **[Project Management](./docs/project/)** - Project status, roadmap, and history
- **[Developer Guides](./docs/developer/)** - Implementation details and guides
- **[Testing Documentation](./docs/testing/)** - Testing framework and procedures

### Current Project Status

For the current status of the project, see the [Project Status Dashboard](./docs/project/project_status_dashboard.md).

### Development Plans

To understand upcoming development priorities, refer to the [Development Roadmap](./docs/project/development_roadmap.md).


## Getting Started

### Prerequisites

- Python 3.12+
- TimescaleDB
- Required Python packages listed in requirements.txt

### Installation

1. Clone the repository
2. Install dependencies: `pip install -r requirements.txt`
3. Configure database connection in config/db_config.py
4. Run the data ingestion pipeline: `python -m src.data_ingestion.pipeline`

### Running Tests

To run the test suite:

```bash
python -m pytest tests/
```

To run specific test modules:

```bash
python -m pytest tests/market_analysis/regime_detection/test_detector.py
python -m pytest tests/market_analysis/parameter_management/test_parameter_integration.py
python -m pytest tests/market_analysis/parameter_management/test_walk_forward.py
python -m pytest tests/market_analysis/test_pair_stability.py
```

Note: Several test files have been recently developed but not yet executed as part of the ongoing testing and validation phase.

## License

This project is proprietary and confidential.
