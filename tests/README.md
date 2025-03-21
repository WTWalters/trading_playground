# TITAN Trading Platform - Test Suite

This directory contains the comprehensive test suite for the TITAN Trading Platform.

## Test Structure

- **tests/market_analysis/** - Tests for market analysis components
  - **regime_detection/** - Tests for regime detection system
  - **parameter_management/** - Tests for adaptive parameter management
  - **microstructure/** - Tests for market microstructure analysis
  - **time_series/** - Tests for time series analysis

- **tests/data_ingestion/** - Tests for data ingestion pipeline
- **tests/database/** - Tests for database operations

## Recently Developed Tests

The following test files have been recently developed but not yet executed:

1. **tests/market_analysis/regime_detection/test_detector.py**
   - Tests for the base RegimeDetector class
   - Validates detection of different market regimes
   - Tests transition probability modeling
   - Checks historical regime shift detection

2. **tests/market_analysis/parameter_management/test_parameter_integration.py**
   - Tests parameter adaptation across different market regimes
   - Validates integration with adaptive parameter manager
   - Verifies smooth parameter transitions between regimes

3. **tests/market_analysis/parameter_management/test_walk_forward.py**
   - Tests walk-forward validation methodology
   - Compares static vs. adaptive parameters
   - Validates regime-specific parameter optimization
   - Tests fine-tuning parameters within walk-forward framework

4. **tests/market_analysis/test_pair_stability.py**
   - Tests cointegration stability monitoring
   - Validates correlation breakdown detection
   - Tests risk adjustment for unstable pairs
   - Checks comprehensive stability reporting

## Running Tests

To run the entire test suite:

```bash
python -m pytest tests/
```

To run a specific test file:

```bash
python -m pytest tests/market_analysis/regime_detection/test_detector.py
```

To run tests with detailed output:

```bash
python -m pytest tests/market_analysis/ -v
```

To generate test coverage report:

```bash
python -m pytest --cov=src tests/
```

## Next Steps

1. Execute all test files and fix any identified issues
2. Complete missing test cases for edge scenarios
3. Add performance benchmarking tests
4. Enhance test documentation with examples
