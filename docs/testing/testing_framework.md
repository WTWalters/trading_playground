# TITAN Trading System Testing Framework

This document provides an overview of the testing framework for the TITAN trading system's adaptive parameter management functionality, with a focus on validating the system's ability to handle market regime transitions.

## Overview

The testing framework validates the following key capabilities:

1. **Regime Detection Accuracy**: Tests if our system correctly identifies different market regimes (normal/low volatility, high volatility, trending, mean-reverting).

2. **Parameter Adaptation**: Validates that trading parameters (position sizing, stop loss levels, etc.) adjust appropriately for each regime.

3. **Transition Handling**: Confirms that parameters adapt smoothly during transitions between regimes without abrupt changes.

4. **Performance Maintenance**: Ensures the system maintains at least 60% of its performance during regime transitions compared to steady-state regimes.

5. **Correlation Breakdown Handling**: Tests if the system detects and adapts to correlation breakdowns in pairs trading.

## Test Components

### MarketRegimeSimulator

A utility class that generates synthetic market data with different regime characteristics:

- **Low Volatility**: Stable market conditions with minimal price fluctuations
- **High Volatility**: Unstable markets with large price swings
- **Trending**: Directional price movement with a consistent bias
- **Mean-Reverting**: Oscillating prices that return to a central value
- **Crisis**: Sharp downward movement followed by high volatility recovery

The simulator can create:

- Single-regime data for baseline testing
- Multi-regime data with controlled transitions
- Pair data with correlation breakdowns

### TestRegimeTransition

The main test class containing multiple test methods:

1. **test_regime_detection_accuracy**: Validates the precision of our regime detection algorithms.

2. **test_parameter_adaptation**: Ensures parameters adjust appropriately for each regime type.

3. **test_regime_transition_adaptation**: Tests if transitions between regimes are handled smoothly.

4. **test_performance_maintenance**: Confirms that performance during transitions meets our 60% threshold requirement.

5. **test_correlation_breakdown**: Validates the system's response to pair correlation breakdowns.

6. **test_end_to_end_integration**: Comprehensive test of all components working together across multiple regime transitions.

## Visualization Components

The framework automatically generates visualizations to help analyze test results:

- **Regime Transition Charts**: Shows parameter changes during transitions between regimes
- **Performance Comparison Charts**: Compares performance in stable vs. transitional regimes
- **Correlation Breakdown Charts**: Visualizes pair correlation changes and parameter adaptations
- **End-to-End Integration Charts**: Comprehensive view of all system components working together

## Running the Tests

### Prerequisites

- Python 3.8+ with required packages (numpy, pandas, matplotlib)
- TITAN trading system codebase with adaptive parameter management modules

### Basic Usage

To run all tests and generate a comprehensive report:

```bash
python run_regime_tests.py
```

To run a specific test only:

```bash
python run_regime_tests.py --test test_performance_maintenance
```

To enable debug logging:

```bash
python run_regime_tests.py --debug
```

### Test Output

The test runner generates:

1. A timestamped report directory containing:

   - Visualization PNGs for each test
   - An HTML report summarizing results
   - Detailed test logs

2. Console output showing test progress and results

## Performance Requirements

The testing framework validates that the system meets these key requirements:

1. Regime detection accuracy of at least 60% for most regimes (50% for crisis regimes)
2. Smooth parameter transitions (no more than 3 instances of parameter changes >3% in a single day)
3. Performance maintenance of at least 60% during transitions compared to stable regimes
4. Appropriate parameter adjustments during correlation breakdowns

## Extending the Framework

To add new tests:

1. Add new test methods to the `TestRegimeTransition` class
2. Create supporting visualization methods as needed
3. Update the `run_regime_tests.py` script if necessary

## Implementation Notes

- Test methodology follows standard unittest patterns
- Synthetic data generation ensures reproducible results
- Visualization utilities help identify issues visually
- HTML report generation facilitates sharing results with team members
