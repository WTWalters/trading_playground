#!/usr/bin/env python
"""
Demo Script for Adaptive Parameter Management System

This script demonstrates how to use the integrated risk management system for:
1. Regime detection
2. Position sizing
3. Risk-adjusted parameter optimization
4. Stress testing

It provides a comprehensive example of the complete workflow.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
import time
import random

# Import TITAN system components
from src.market_analysis.parameter_management.integration import AdaptiveParameterManager
from src.market_analysis.regime_detection.enhanced_detector import EnhancedRegimeDetector
from src.market_analysis.parameter_management.position_sizing import KellyPositionSizer
from src.market_analysis.parameter_management.risk_controls import RiskControls
from src.market_analysis.parameter_management.risk_manager import RiskManager
from src.market_analysis.regime_detection.detector import RegimeType


def generate_sample_market_data(days=180, volatility=0.015, regime_changes=True):
    """Generate sample market data for demonstration."""
    # Create date range
    end_date = datetime.now()
    start_date = end_date - timedelta(days=days)
    dates = pd.date_range(start=start_date, end=end_date, freq='D')
    
    # Generate price series with regime changes
    prices = [100.0]
    regimes = []
    
    # Start with trending regime
    current_regime = "trending" if regime_changes else "neutral"
    regime_length = random.randint(20, 40)
    regime_days = 0
    
    for i in range(1, len(dates)):
        # Check if we need to change regime
        if regime_changes and regime_days >= regime_length:
            # Change regime
            if current_regime == "trending":
                current_regime = "mean_reverting"
            elif current_regime == "mean_reverting":
                current_regime = "high_volatility"
            elif current_regime == "high_volatility":
                current_regime = "trending"
            else:
                current_regime = "trending"
                
            regime_length = random.randint(20, 40)
            regime_days = 0
        
        # Generate return based on regime
        if current_regime == "trending":
            # Trending: positive drift with moderate volatility
            drift = 0.0008
            vol = volatility * 0.8
        elif current_regime == "mean_reverting":
            # Mean reverting: negative autocorrelation
            if prices[-1] > prices[0] * 1.1:
                drift = -0.0005
            elif prices[-1] < prices[0] * 0.9:
                drift = 0.0005
            else:
                drift = 0.0
            vol = volatility * 0.7
        elif current_regime == "high_volatility":
            # High volatility: higher vol, random drift
            drift = random.uniform(-0.001, 0.001)
            vol = volatility * 2.0
        else:
            # Neutral
            drift = 0.0003
            vol = volatility
            
        # Generate return
        daily_return = np.random.normal(drift, vol)
        prices.append(prices[-1] * (1 + daily_return))
        
        # Store regime
        regimes.append(current_regime)
        regime_days += 1
    
    # Create DataFrame
    df = pd.DataFrame({
        'close': prices,
        'regime': [None] + regimes  # Shift regimes to align with returns
    }, index=dates)
    
    # Add other price columns based on close
    df['open'] = df['close'].shift(1) * (1 + np.random.normal(0, 0.002, len(df)))
    df['high'] = df[['open', 'close']].max(axis=1) * (1 + abs(np.random.normal(0, 0.003, len(df))))
    df['low'] = df[['open', 'close']].min(axis=1) * (1 - abs(np.random.normal(0, 0.003, len(df))))
    df['volume'] = np.random.lognormal(15, 0.5, len(df))
    
    # Fill missing values
    df = df.fillna(method='bfill').fillna(method='ffill')
    
    return df


def generate_sample_macro_data(market_data):
    """Generate sample macro data aligned with market data."""
    # Use the same index as market data
    dates = market_data.index
    
    # Generate VIX (inversely correlated with market)
    vix_base = 20.0
    vix = []
    
    # Calculate returns from market data
    returns = market_data['close'].pct_change().fillna(0)
    
    for i in range(len(dates)):
        if i > 0:
            # VIX tends to spike when market drops
            vix_change = -5.0 * returns[i] + np.random.normal(0, 0.03)
            vix.append(max(10, vix[-1] * (1 + vix_change)))
        else:
            vix.append(vix_base)
    
    # Generate other macro indicators
    yield_curve = np.random.normal(0.5, 0.3, len(dates))
    interest_rate = np.linspace(2.0, 3.5, len(dates)) + np.random.normal(0, 0.1, len(dates))
    usd_index = 100 + np.cumsum(np.random.normal(0, 0.002, len(dates)))
    
    # Create DataFrame
    df = pd.DataFrame({
        'VIX': vix,
        'yield_curve': yield_curve,
        'interest_rate': interest_rate,
        'USD_index': usd_index
    }, index=dates)
    
    return df


def generate_sample_strategy_parameters():
    """Generate sample strategy parameters."""
    return {
        # Entry/exit thresholds
        'entry_threshold': 0.7,
        'exit_threshold': 0.3,
        
        # Lookback periods
        'lookback_period': 20,
        'volatility_lookback': 10,
        
        # Mean reversion parameters
        'z_entry': 2.0,
        'z_exit': 0.5,
        
        # Stop loss and take profit
        'stop_loss_pct': 0.15,
        'take_profit_pct': 0.30,
        
        # Other parameters
        'max_holding_period': 10,
        'min_holding_period': 1,
        'rebalance_frequency': 1
    }


def simulate_trades(strategy_id, parameters, market_data, days=10):
    """Simulate trading results using the provided parameters."""
    trade_results = []
    
    # Sample some random days from the market data
    sample_days = min(days, len(market_data) - 1)
    indices = random.sample(range(1, len(market_data)), sample_days)
    
    for idx in indices:
        # Create random trade result based on market regime
        regime = market_data['regime'].iloc[idx]
        
        # Set different win probabilities based on regime
        if regime == 'trending':
            win_prob = 0.6
        elif regime == 'mean_reverting':
            win_prob = 0.55
        elif regime == 'high_volatility':
            win_prob = 0.45
        else:
            win_prob = 0.5
            
        # Adjust based on parameters
        # If z_entry is too low or too high, decrease win probability
        z_entry = parameters.get('z_entry', 2.0)
        if z_entry < 1.0 or z_entry > 3.0:
            win_prob *= 0.8
            
        # Generate random win/loss
        won = random.random() < win_prob
        
        # Generate PnL
        if won:
            pnl = random.uniform(0.5, 2.0) * parameters.get('take_profit_pct', 0.3)
        else:
            pnl = -random.uniform(0.5, 1.0) * parameters.get('stop_loss_pct', 0.15)
            
        # Create trade metrics
        trade_metrics = {
            'pnl': pnl,
            'pnl_pct': pnl,
            'duration': random.randint(1, parameters.get('max_holding_period', 10)),
            'regime': regime,
            'timestamp': market_data.index[idx]
        }
        
        trade_results.append((won, trade_metrics))
        
    return trade_results


def main():
    """Main demonstration function."""
    print("\n" + "="*80)
    print("TITAN ADAPTIVE PARAMETER MANAGEMENT SYSTEM DEMONSTRATION")
    print("="*80 + "\n")
    
    # Step 1: Generate sample data
    print("Generating sample market data...")
    market_data = generate_sample_market_data(days=180)
    macro_data = generate_sample_macro_data(market_data)
    
    # Create more recent subset for analysis
    current_data = market_data.iloc[-30:].copy()
    current_macro = macro_data.iloc[-30:].copy()
    
    print(f"Generated {len(market_data)} days of market data with {len(current_data)} days for current analysis")
    
    # Step 2: Initialize the adaptive parameter manager
    print("\nInitializing Adaptive Parameter Management System...")
    
    # Create component instances with custom settings
    regime_detector = EnhancedRegimeDetector(
        lookback_window=20,
        volatility_threshold=1.5,
        vix_threshold=25.0
    )
    
    position_sizer = KellyPositionSizer(
        default_kelly_fraction=0.5,
        max_position_pct=0.02,  # 2% max position size
    )
    
    risk_controls = RiskControls(
        default_stop_loss_pct=0.20,
        default_take_profit_pct=0.40,
        max_drawdown_limit=0.25
    )
    
    risk_manager = RiskManager(
        regime_detector=regime_detector,
        position_sizer=position_sizer,
        risk_controls=risk_controls,
        max_portfolio_risk=0.12,
        max_strategy_risk=0.04
    )
    
    # Create adaptive parameter manager
    param_manager = AdaptiveParameterManager(
        risk_manager=risk_manager,
        config={'backtesting_mode': True}  # Enable backtesting mode
    )
    
    # Step 3: Register strategies
    print("\nRegistering sample strategies...")
    
    # Generate sample strategy parameters
    strategy1_params = generate_sample_strategy_parameters()
    strategy2_params = generate_sample_strategy_parameters()
    
    # Modify strategy2 params to be more conservative
    strategy2_params['entry_threshold'] = 0.8
    strategy2_params['z_entry'] = 2.5
    strategy2_params['stop_loss_pct'] = 0.10
    
    # Define strategy metrics
    strategy1_metrics = {
        'win_rate': 0.58,
        'win_loss_ratio': 1.7,
        'volatility': 0.18,
        'correlation': 0.0,
        'sharpe': 1.3
    }
    
    strategy2_metrics = {
        'win_rate': 0.52,
        'win_loss_ratio': 2.1,
        'volatility': 0.15,
        'correlation': 0.3,
        'sharpe': 1.6
    }
    
    # Register strategies
    param_manager.register_strategy(
        strategy_id="mean_reversion_etf",
        base_parameters=strategy1_params,
        strategy_metrics=strategy1_metrics
    )
    
    param_manager.register_strategy(
        strategy_id="momentum_sectors",
        base_parameters=strategy2_params,
        strategy_metrics=strategy2_metrics
    )
    
    print("Registered strategies: mean_reversion_etf, momentum_sectors")
    
    # Step 4: Update market state
    print("\nUpdating current market state...")
    
    # Update with current data
    portfolio_value = 1000000.0  # $1M portfolio
    regime_result = param_manager.update_market_state(
        market_data=current_data,
        portfolio_value=portfolio_value,
        macro_data=current_macro
    )
    
    print(f"Current primary regime: {regime_result.primary_regime}")
    print(f"Current secondary regime: {regime_result.secondary_regime}")
    
    if hasattr(regime_result, 'macro_regime'):
        print(f"Current macro regime: {regime_result.macro_regime}")
        print(f"Current sentiment regime: {regime_result.sentiment_regime}")
    
    print(f"Regime stability: {regime_result.stability_score:.2f}")
    print(f"Transition probability: {regime_result.transition_probability:.2f}")
    
    # Step 5: Get optimized parameters
    print("\nGenerating optimized parameters...")
    time.sleep(1)  # Small pause for demonstration effect
    
    # Get parameters for both strategies
    strategy1_opt_params = param_manager.get_optimized_parameters(
        strategy_id="mean_reversion_etf",
        signal_strength=0.65  # Strong signal
    )
    
    strategy2_opt_params = param_manager.get_optimized_parameters(
        strategy_id="momentum_sectors",
        signal_strength=0.45  # Weak signal
    )
    
    # Print parameter changes
    print("\nMean Reversion ETF Strategy - Parameter Adjustments:")
    for key in sorted(strategy1_params.keys()):
        if key in strategy1_opt_params:
            original = strategy1_params[key]
            optimized = strategy1_opt_params[key]
            if isinstance(original, (int, float)) and isinstance(optimized, (int, float)):
                change_pct = (optimized - original) / abs(original) * 100 if original != 0 else 0
                print(f"  {key:20s}: {original:.4f} → {optimized:.4f} ({change_pct:+.1f}%)")
    
    print("\nMomentum Sectors Strategy - Parameter Adjustments:")
    for key in sorted(strategy2_params.keys()):
        if key in strategy2_opt_params:
            original = strategy2_params[key]
            optimized = strategy2_opt_params[key]
            if isinstance(original, (int, float)) and isinstance(optimized, (int, float)):
                change_pct = (optimized - original) / abs(original) * 100 if original != 0 else 0
                print(f"  {key:20s}: {original:.4f} → {optimized:.4f} ({change_pct:+.1f}%)")
    
    # Print position sizing
    print(f"\nMean Reversion ETF Position Size: {strategy1_opt_params['position_size_pct']:.2%} (${strategy1_opt_params['position_size']:.2f})")
    print(f"Momentum Sectors Position Size: {strategy2_opt_params['position_size_pct']:.2%} (${strategy2_opt_params['position_size']:.2f})")
    
    # Print warnings if any
    if 'warnings' in strategy1_opt_params:
        print("\nWarnings for Mean Reversion ETF Strategy:")
        for warning in strategy1_opt_params['warnings']:
            print(f"  - {warning}")
            
    if 'warnings' in strategy2_opt_params:
        print("\nWarnings for Momentum Sectors Strategy:")
        for warning in strategy2_opt_params['warnings']:
            print(f"  - {warning}")
    
    # Step 6: Simulate trades
    print("\nSimulating trades with optimized parameters...")
    
    # Simulate trades for both strategies
    strategy1_trades = simulate_trades(
        strategy_id="mean_reversion_etf",
        parameters=strategy1_opt_params,
        market_data=current_data,
        days=5
    )
    
    strategy2_trades = simulate_trades(
        strategy_id="momentum_sectors",
        parameters=strategy2_opt_params,
        market_data=current_data,
        days=5
    )
    
    # Update performance with trade results
    print("\nUpdating performance with trade results...")
    
    for won, metrics in strategy1_trades:
        param_manager.update_performance(
            strategy_id="mean_reversion_etf",
            trade_result=won,
            trade_metrics=metrics
        )
        print(f"  Mean Reversion ETF: {'WIN' if won else 'LOSS'} with PnL: {metrics['pnl']:.2%}")
        
    for won, metrics in strategy2_trades:
        param_manager.update_performance(
            strategy_id="momentum_sectors",
            trade_result=won,
            trade_metrics=metrics
        )
        print(f"  Momentum Sectors: {'WIN' if won else 'LOSS'} with PnL: {metrics['pnl']:.2%}")
    
    # Step 7: Run stress tests
    print("\nRunning stress tests...")
    time.sleep(1)  # Small pause for demonstration effect
    
    stress_results = param_manager.run_stress_test(
        strategy_id="mean_reversion_etf"
    )
    
    print("\nStress Test Results for Mean Reversion ETF Strategy:")
    for scenario, result in stress_results.items():
        print(f"\n  Scenario: {scenario}")
        print(f"    Regime: {result['regime']}")
        print(f"    Position Size: {result['position_size']:.2%}")
        print(f"    Stop Loss: {result['stop_loss']:.2%}")
        print(f"    Take Profit: {result['take_profit']:.2%}")
        
        if result['warnings']:
            print("    Warnings:")
            for warning in result['warnings']:
                print(f"      - {warning}")
    
    # Step 8: Get portfolio allocation
    print("\nCalculating optimal portfolio allocation...")
    
    allocations = param_manager.get_portfolio_allocation(
        strategy_ids=["mean_reversion_etf", "momentum_sectors"],
        portfolio_value=portfolio_value,
        max_allocation=0.9
    )
    
    print("\nPortfolio Allocation:")
    total_allocation = 0.0
    for strategy_id, allocation in allocations.items():
        amount = allocation['position_size']
        pct = allocation['allocation']
        total_allocation += pct
        print(f"  {strategy_id}: {pct:.2%} (${amount:.2f})")
    
    print(f"\nTotal Allocation: {total_allocation:.2%}")
    
    # Step 9: Get strategy risk profiles
    print("\nGenerating strategy risk profiles...")
    
    for strategy_id in ["mean_reversion_etf", "momentum_sectors"]:
        risk_profile = param_manager.get_strategy_risk_profile(strategy_id)
        
        print(f"\n{strategy_id} Risk Profile:")
        print(f"  Overall Risk Level: {risk_profile['risk_level']}")
        print(f"  Current Drawdown: {risk_profile['current_drawdown']:.2%}")
        
        print("  Strategy Metrics:")
        for metric, value in risk_profile['metrics'].items():
            if isinstance(value, (int, float)):
                print(f"    {metric}: {value:.4f}")
    
    print("\n" + "="*80)
    print("DEMONSTRATION COMPLETE")
    print("="*80 + "\n")


if __name__ == "__main__":
    main()
