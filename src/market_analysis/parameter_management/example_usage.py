"""
Example Usage of the Adaptive Parameter Management System

This script demonstrates how to use the integrated adaptive parameter management system
for optimizing and adapting trading strategy parameters based on market regimes.
"""

import pandas as pd
import numpy as np
import logging
from datetime import datetime, timedelta
from pathlib import Path

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Import components
from src.market_analysis.parameter_management.integrated_system import (
    IntegratedAdaptiveSystem, AdaptiveSystemConfig
)
from src.market_analysis.regime_detection.detector import RegimeType
from src.market_analysis.regime_detection.enhanced_detector import EnhancedRegimeDetector

# Import a dummy strategy for the example
# In a real implementation, replace this with your actual strategy class
class DummyMeanReversionStrategy:
    def __init__(self, entry_z_score=2.0, exit_z_score=0.5, lookback=20, 
                 stop_loss=0.05, take_profit=0.1, max_holding_days=10):
        """Initialize the strategy with parameters."""
        self.entry_z_score = entry_z_score
        self.exit_z_score = exit_z_score
        self.lookback = lookback
        self.stop_loss = stop_loss
        self.take_profit = take_profit
        self.max_holding_days = max_holding_days
        self.trades = []
        
    def backtest(self, data):
        """Run a backtest on the provided data."""
        # Simulate a backtest
        # In a real implementation, replace this with actual backtest logic
        self.trades = []
        returns = []
        
        # Simplified simulation for demonstration purposes
        for i in range(len(data) - 1):
            # Randomly generate a trade with probability related to parameters
            if np.random.random() < 0.1:  # 10% chance of trade per day
                pnl = np.random.normal(0.002 * (3.0 / self.entry_z_score), 0.01)
                trade = {
                    "entry_time": data.index[i],
                    "exit_time": data.index[min(i + int(self.max_holding_days / 2), len(data) - 1)],
                    "pnl": pnl,
                    "pnl_pct": pnl,
                    "direction": "long" if pnl > 0 else "short"
                }
                self.trades.append(trade)
                returns.append(pnl)
                
        # Always include some return data
        if not returns:
            returns = [0.0001] * (len(data) - 1)
            
        self.returns = returns
    
    def process_day(self, data):
        """Process a single day of data."""
        # In a real implementation, replace this with actual logic
        result = {}
        
        # Randomly generate a trade with low probability
        if np.random.random() < 0.05:  # 5% chance of trade
            pnl = np.random.normal(0.002 * (3.0 / self.entry_z_score), 0.01)
            trade = {
                "entry_time": data.index[0],
                "exit_time": data.index[0] + timedelta(days=int(self.max_holding_days / 2)),
                "pnl": pnl,
                "pnl_pct": pnl,
                "direction": "long" if pnl > 0 else "short"
            }
            result["trade"] = trade
            result["return"] = pnl
            
        return result
    
    def performance_metrics(self):
        """Calculate performance metrics."""
        if not self.trades:
            return {
                "total_return": 0.0,
                "sharpe_ratio": 0.0,
                "max_drawdown": 0.0,
                "win_rate": 0.0
            }
            
        # Calculate metrics
        returns = [t["pnl"] for t in self.trades]
        total_return = sum(returns)
        
        # Calculate Sharpe ratio (simplified)
        if len(returns) > 1 and np.std(returns) > 0:
            sharpe_ratio = np.mean(returns) / np.std(returns) * np.sqrt(252 / len(returns))
        else:
            sharpe_ratio = 0.0
            
        # Calculate max drawdown (simplified)
        cumulative = np.cumsum(returns)
        running_max = np.maximum.accumulate(cumulative)
        drawdown = running_max - cumulative
        max_drawdown = np.max(drawdown) if len(drawdown) > 0 else 0.0
        
        # Calculate win rate
        win_rate = len([t for t in self.trades if t["pnl"] > 0]) / len(self.trades)
        
        return {
            "total_return": total_return,
            "sharpe_ratio": sharpe_ratio,
            "max_drawdown": max_drawdown,
            "win_rate": win_rate
        }


def generate_dummy_data(days=500, start_date=None):
    """Generate dummy price data for testing."""
    if start_date is None:
        start_date = datetime(2023, 1, 1)
        
    # Create date range
    dates = [start_date + timedelta(days=i) for i in range(days)]
    
    # Generate price series
    initial_price = 100.0
    volatility = 0.015
    returns = np.random.normal(0.0002, volatility, days)
    
    # Add some regime-like behavior
    # Regime 1: Low volatility, mean-reverting (first third)
    returns[:days//3] = np.random.normal(0.0001, 0.01, days//3)
    
    # Regime 2: High volatility, trending (middle third)
    trend = np.linspace(0, 0.001, days//3)
    returns[days//3:2*days//3] = np.random.normal(trend, 0.02, days//3)
    
    # Regime 3: Medium volatility, random (last third)
    returns[2*days//3:] = np.random.normal(0.0, 0.015, days - 2*days//3)
    
    # Calculate prices
    prices = [initial_price]
    for ret in returns:
        prices.append(prices[-1] * (1 + ret))
        
    # Create DataFrame
    data = pd.DataFrame({
        'open': prices[:-1],
        'high': [p * (1 + np.random.uniform(0, 0.005)) for p in prices[:-1]],
        'low': [p * (1 - np.random.uniform(0, 0.005)) for p in prices[:-1]],
        'close': prices[1:],
        'volume': np.random.randint(100000, 1000000, len(prices) - 1)
    }, index=dates)
    
    return data


def generate_dummy_macro_data(days=500, start_date=None):
    """Generate dummy macro data for testing."""
    if start_date is None:
        start_date = datetime(2023, 1, 1)
        
    # Create date range (weekly data)
    dates = [start_date + timedelta(days=i) for i in range(0, days, 7)]
    
    # Generate VIX-like series
    vix = 20 + np.random.normal(0, 3, len(dates))
    
    # Generate yield curve-like series
    yield_curve = np.random.normal(0.5, 0.3, len(dates))
    
    # Add regime shifts
    # First third: low volatility
    vix[:len(dates)//3] = 15 + np.random.normal(0, 2, len(dates)//3)
    yield_curve[:len(dates)//3] = 1.0 + np.random.normal(0, 0.1, len(dates)//3)
    
    # Middle third: high volatility
    vix[len(dates)//3:2*len(dates)//3] = 30 + np.random.normal(0, 5, len(dates)//3)
    yield_curve[len(dates)//3:2*len(dates)//3] = -0.2 + np.random.normal(0, 0.15, len(dates)//3)
    
    # Last third: medium volatility
    vix[2*len(dates)//3:] = 22 + np.random.normal(0, 3, len(dates) - 2*len(dates)//3)
    yield_curve[2*len(dates)//3:] = 0.4 + np.random.normal(0, 0.2, len(dates) - 2*len(dates)//3)
    
    # Create DataFrame
    data = pd.DataFrame({
        'VIX': vix,
        'yield_curve': yield_curve,
        'SPX': np.cumsum(np.random.normal(0.001, 0.01, len(dates))),
        'USD_index': 90 + np.cumsum(np.random.normal(0, 0.005, len(dates)))
    }, index=dates)
    
    return data


def main():
    """Main execution function."""
    logger.info("Starting adaptive parameter management example")
    
    # Generate dummy data
    historical_data = generate_dummy_data(days=500)
    macro_data = generate_dummy_macro_data(days=500)
    
    logger.info(f"Generated {len(historical_data)} days of historical data")
    
    # Define strategy parameters
    base_parameters = {
        "entry_z_score": 2.0,
        "exit_z_score": 0.5,
        "lookback": 20,
        "stop_loss": 0.05,
        "take_profit": 0.1,
        "max_holding_days": 10
    }
    
    parameter_ranges = {
        "entry_z_score": (1.0, 3.0, 0.1),
        "exit_z_score": (0.1, 1.0, 0.1),
        "lookback": (10, 40, 5),
        "stop_loss": (0.02, 0.10, 0.01),
        "take_profit": (0.03, 0.20, 0.01),
        "max_holding_days": (5, 20, 1)
    }
    
    # Create system configuration
    config = AdaptiveSystemConfig(
        strategy_class=DummyMeanReversionStrategy,
        base_parameters=base_parameters,
        parameter_ranges=parameter_ranges,
        optimization_metric="sharpe_ratio",
        regime_detection_lookback=60,
        transition_window=5,
        is_window_size=120,  # Smaller window for faster example
        oos_window_size=30,
        optimization_iterations=20,  # Small number for demonstration
        storage_path="./config/example",
        seed=42
    )
    
    # Create integrated system
    system = IntegratedAdaptiveSystem(config)
    
    # Initialize with historical data
    logger.info("Initializing system with historical data...")
    system.initialize(
        historical_data=historical_data,
        macro_data=macro_data,
        optimize_regimes=True  # This will take some time in a real system
    )
    
    # Get initial regime and parameters
    current_regime = system.current_regime
    initial_params = system.parameter_manager.get_parameters(
        historical_data.iloc[-60:], macro_data
    )
    
    logger.info(f"Initial regime: {current_regime.value}")
    logger.info(f"Initial parameters: {initial_params}")
    
    # Simulate new data coming in
    logger.info("Simulating new data updates...")
    
    # Generate 30 days of new data
    start_date = historical_data.index[-1] + timedelta(days=1)
    new_data = generate_dummy_data(days=30, start_date=start_date)
    new_macro = generate_dummy_macro_data(days=30, start_date=start_date)
    
    # Process the new data
    update_result = system.process_data_update(
        new_data=new_data,
        macro_data=new_macro,
        detect_regime_change=True,
        smooth_transition=True
    )
    
    logger.info(f"Update result: {update_result}")
    
    # Simulate trading with the optimized parameters
    strategy = DummyMeanReversionStrategy(**update_result["parameters"])
    strategy.backtest(new_data)
    performance = strategy.performance_metrics()
    
    logger.info(f"Trading performance: {performance}")
    
    # Update performance metrics
    system.update_performance_metrics(
        parameters=update_result["parameters"],
        performance_metrics=performance,
        regime=RegimeType(update_result["regime"])
    )
    
    # Get regime analysis
    regime_analysis = system.get_regime_analysis()
    logger.info(f"Regime frequencies: {regime_analysis['regime_frequencies']}")
    logger.info(f"Average durations: {regime_analysis['average_durations']}")
    
    # Save system state
    state_path = system.save_state()
    logger.info(f"Saved system state to {state_path}")
    
    # Demonstrate continuous adaptation
    logger.info("Demonstrating continuous adaptation...")
    
    # Enable continuous optimization
    system.enable_continuous_optimization(True)
    
    # Simulate a month of daily updates
    for day in range(30):
        # Generate one day of new data
        day_start = start_date + timedelta(days=30+day)
        day_data = generate_dummy_data(days=1, start_date=day_start)
        
        # Process the new data (macro data less frequent, so only update weekly)
        if day % 7 == 0:
            day_macro = generate_dummy_macro_data(days=7, start_date=day_start)
            update_result = system.process_data_update(day_data, day_macro)
        else:
            update_result = system.process_data_update(day_data)
        
        # Every few days, simulate trading and update performance
        if day % 5 == 0:
            # Get the latest parameters
            latest_params = update_result["parameters"]
            
            # Simulate trading
            strategy = DummyMeanReversionStrategy(**latest_params)
            strategy.backtest(day_data)
            performance = strategy.performance_metrics()
            
            # Update performance metrics
            system.update_performance_metrics(
                parameters=latest_params,
                performance_metrics=performance
            )
            
            logger.info(f"Day {day}: Regime={update_result['regime']}, "
                       f"Performance: sharpe={performance['sharpe_ratio']:.4f}")
    
    # Final regime analysis
    final_analysis = system.get_regime_analysis()
    logger.info("Final regime analysis:")
    logger.info(f"Current regime: {final_analysis['current_regime']}")
    logger.info(f"Parameter info: {final_analysis['parameter_info']}")
    
    logger.info("Example completed successfully")


if __name__ == "__main__":
    main()
