# TITAN Trading System Testing Framework

![TITAN Testing](https://img.shields.io/badge/TITAN-Testing-blue)

A comprehensive testing framework for validating the TITAN trading system's adaptive parameter management functionality, with a focus on market regime transitions.

## 🌟 Key Features

- **Synthetic Market Data Generation**: Create realistic market scenarios for controlled testing
- **Regime Transition Testing**: Validate system behavior during market regime changes
- **Performance Validation**: Ensure trading performance is maintained during transitions
- **Correlation Breakdown Testing**: Test pair trading strategies under correlation stress
- **Comprehensive Reporting**: Generate visual reports with detailed analytics
- **End-to-End Integration Testing**: Validate all system components working together

## 📋 Requirements

- Python 3.8+
- NumPy, Pandas, Matplotlib
- TITAN trading system codebase
- Unix-based system for shell script execution (or Windows with bash capabilities)

## 🚀 Quick Start

### Installation

1. Clone this repository into your TITAN project:

   ```bash
   git clone https://github.com/your-org/titan-testing.git tests/regime_testing
   cd tests/regime_testing
   ```

2. Install required dependencies:
   ```bash
   pip install -r requirements.txt
   ```

### Running Tests

Execute all tests with the shell script:

```bash
./run_tests.sh
```

Or run specific tests:

```bash
./run_tests.sh --test test_correlation_breakdown
```

Enable debug logging:

```bash
./run_tests.sh --debug
```

## 📊 Test Structure

The framework consists of these main components:

### MarketRegimeSimulator

Generates synthetic market data for testing, including:

- Single-regime data
- Multi-regime transitions
- Correlation breakdowns for pairs
- Crisis scenarios

### TestRegimeTransition

The main test class containing multiple test methods:

- `test_regime_detection_accuracy`: Validates regime detection
- `test_parameter_adaptation`: Tests parameter adjustments
- `test_regime_transition_adaptation`: Validates smooth transitions
- `test_performance_maintenance`: Confirms performance meets thresholds
- `test_correlation_breakdown`: Tests pair correlation handling
- `test_end_to_end_integration`: Comprehensive system test

### Utilities

- Visualization generators
- HTML report creation
- Test runner and orchestration

## 📈 Test Report

After running tests, an HTML report is automatically generated with:

- Test results summary
- Detailed visualizations
- Comparative analytics

The report is saved in the `test_reports` directory and should open automatically in your browser.

## 🔍 Interpreting Results

### Pass/Fail Criteria

Tests validate that the system meets these criteria:

- **Regime detection**: ≥60% accuracy for most regimes
- **Parameter adaptation**: Appropriate adjustments for each regime
- **Transition smoothness**: No abrupt parameter changes
- **Performance maintenance**: ≥60% of steady-state performance
- **Correlation handling**: Appropriate risk reduction when correlations break down

### Visualizations

Each test generates visualizations to help identify issues:

1. **Regime Transition Charts**: Shows parameter changes during transitions
2. **Performance Comparison**: Compares performance across regimes
3. **Correlation Breakdown**: Visualizes pair correlation changes
4. **End-to-End System**: Shows all components working together

## 🧩 Integration with TITAN System

The testing framework integrates with these TITAN components:

- `EnhancedRegimeDetector`: Market regime identification
- `AdaptiveRiskControls`: Risk parameter management
- `KellyPositionSizer`: Position sizing adaptation
- `RiskManager`: Comprehensive risk management
- `ParameterIntegration`: System integration module

## 📝 Configuration

Tests can be configured by modifying the parameters in `test_regime_transition.py`:

```python
self.regime_detector = EnhancedRegimeDetector(
    lookback_period=30,  # Adjust lookback period
    volatility_threshold_high=0.015,  # Adjust volatility thresholds
    volatility_threshold_low=0.008
)
```

## 🗂️ Directory Structure

```
tests/regime_testing/
├── README.md               # This file
├── TESTING_FRAMEWORK.md    # Detailed framework documentation
├── NEXT_STEPS.md           # Future development roadmap
├── run_tests.sh            # Test execution script
├── run_regime_tests.py     # Python test runner
├── test_regime_transition.py # Main test implementation
├── test_outputs/           # Generated visualizations
└── test_reports/           # Generated HTML reports
```

## 🛠️ Extending the Framework

To add new tests:

1. Add test methods to `TestRegimeTransition` class
2. Create supporting visualization methods
3. Update documentation

See `NEXT_STEPS.md` for the planned enhancements.

## 📄 License

Copyright © 2025 TITAN Trading Systems

## 🤝 Contributing

See `CONTRIBUTING.md` for contribution guidelines.
