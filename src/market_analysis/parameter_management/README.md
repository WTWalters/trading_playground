# Adaptive Parameter Management System

This module provides a comprehensive solution for dynamically adjusting trading strategy parameters based on market regimes, implementing an antifragile approach that allows strategies to adapt to changing market conditions.

## Overview

The adaptive parameter management system consists of three main components:

1. **Regime Detection**: Identifies different market regimes using statistical analysis of market data
2. **Walk-Forward Testing**: Validates strategy parameters without lookahead bias
3. **Parameter Adaptation**: Smoothly transitions between optimal parameter sets as market regimes change

## Key Features

- **Market Regime Classification**: Multi-dimensional regime detection across volatility, trend, correlation, and liquidity
- **Regime-Specific Parameter Optimization**: Optimizes strategy parameters for each detected market regime
- **Smooth Parameter Transitions**: Gradually blends parameter values when transitioning between regimes
- **Continuous Learning**: Updates parameter sets based on observed performance
- **Walk-Forward Validation**: Implements proper in-sample/out-of-sample separation to avoid overfitting
- **Antifragile Design**: System becomes more robust as it experiences different market conditions

## Component Architecture

```
Adaptive Parameter Management System
├── RegimeDetector - Identifies market regimes
│   ├── Basic Regime Detection
│   └── Enhanced Regime Detection with macro indicators
├── WalkForwardTester - Tests strategy parameters
│   ├── Standard Walk-Forward
│   ├── Anchored Walk-Forward
│   └── Regime-Based Walk-Forward
├── AdaptiveParameterManager - Manages parameter sets
│   ├── Parameter Optimization
│   ├── Performance Tracking
│   └── Parameter Blending
└── IntegratedAdaptiveSystem - Main integration layer
    ├── System Configuration
    ├── Continuous Adaptation
    └── State Management
```

## Usage Guide

### Basic Setup

1. Define your strategy parameters and ranges:

```python
base_parameters = {
    "entry_threshold": 2.0,
    "exit_threshold": 0.5,
    "lookback_period": 20,
    "stop_loss": 0.05
}

parameter_ranges = {
    "entry_threshold": (1.0, 3.0, 0.1),  # (min, max, step)
    "exit_threshold": (0.1, 1.0, 0.1),
    "lookback_period": (10, 30, 5),
    "stop_loss": (0.02, 0.10, 0.01)
}
```

2. Create the system configuration:

```python
from src.market_analysis.parameter_management.integrated_system import AdaptiveSystemConfig

config = AdaptiveSystemConfig(
    strategy_class=YourStrategyClass,
    base_parameters=base_parameters,
    parameter_ranges=parameter_ranges,
    optimization_metric="sharpe_ratio",
    regime_detection_lookback=60,
    transition_window=5,
    is_window_size=252,
    oos_window_size=63,
    optimization_iterations=100
)
```

3. Initialize the system with historical data:

```python
from src.market_analysis.parameter_management.integrated_system import IntegratedAdaptiveSystem

system = IntegratedAdaptiveSystem(config)
system.initialize(
    historical_data=your_historical_data,
    macro_data=your_macro_data,  # Optional
    optimize_regimes=True
)
```

### Daily Usage

1. Process new market data and get updated parameters:

```python
# When new data becomes available
update_result = system.process_data_update(
    new_data=todays_data,
    macro_data=todays_macro_data,  # Optional
    detect_regime_change=True,
    smooth_transition=True
)

# Get optimized parameters for current conditions
current_parameters = update_result["parameters"]
current_regime = update_result["regime"]
```

2. Update the system with performance results:

```python
# After trading with the parameters
system.update_performance_metrics(
    parameters=current_parameters,
    performance_metrics=performance_metrics
)
```

### Advanced Features

1. Enable continuous optimization:

```python
# System will periodically re-optimize parameters
system.enable_continuous_optimization(True)
```

2. Get regime analysis:

```python
regime_analysis = system.get_regime_analysis()
print(f"Current regime: {regime_analysis['current_regime']}")
print(f"Regime frequencies: {regime_analysis['regime_frequencies']}")
```

3. Save and load system state:

```python
# Save state
state_path = system.save_state()

# Load state later
system.load_state(state_path)
```

## Example Usage

See `example_usage.py` for a complete demonstration of the adaptive parameter management system.

## Customization

### Custom Regime Detection

You can create your own custom regime detector by extending the `RegimeDetector` class:

```python
from src.market_analysis.regime_detection.detector import RegimeDetector

class CustomRegimeDetector(RegimeDetector):
    # Override methods to implement custom regime detection logic
    ...
```

### Custom Parameter Optimization

You can implement your own parameter optimization logic by extending the `ParameterOptimizer` class:

```python
from src.market_analysis.walk_forward.parameter_optimizer import ParameterOptimizer

class CustomParameterOptimizer(ParameterOptimizer):
    # Override methods to implement custom optimization logic
    ...
```

## Implementation Notes

- The system persists parameter sets and regime data between runs
- Parameter transitions are smoothed using an easing function to avoid abrupt changes
- The system includes data validation and error handling for robustness
- Logging provides visibility into system operation and parameter changes

## Requirements

- Python 3.9+
- NumPy
- Pandas
- Scikit-learn (optional, for enhanced regime detection)
- Matplotlib (optional, for visualization)

## Integration with Trading Systems

The adaptive parameter management system is designed to integrate with existing trading systems:

1. **Real-time Trading**: Get optimized parameters for current market conditions
2. **Backtesting**: Test how adaptive parameters would have performed historically
3. **Simulation**: Evaluate different regime detection and adaptation strategies

## References

1. Pardo, R. (2008). *The Evaluation and Optimization of Trading Strategies*
2. Taleb, N. N. (2012). *Antifragile: Things That Gain from Disorder*
3. Jones, P. T. (1999). *The Trading Game: Playing by the Numbers to Make Millions*
