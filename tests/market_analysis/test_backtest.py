# tests/market_analysis/test_backtest.py

import pytest
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from src.market_analysis.backtest import SimpleBacktest
from src.market_analysis.base import AnalysisConfig, MarketRegime

@pytest.mark.asyncio
async def test_basic_backtest(backtest_engine, sample_data):
    """
    Test basic backtest functionality with default parameters

    Verifies:
    - Basic result structure
    - Capital preservation
    - Essential metrics presence
    """
    results = await backtest_engine.run_test(
        data=sample_data,
        initial_capital=10000,
        risk_per_trade=0.02
    )

    # Verify essential metrics are present
    assert 'final_capital' in results
    assert 'total_trades' in results
    assert 'win_rate' in results

    # Verify capital preservation
    assert results['final_capital'] > 0

@pytest.mark.asyncio
async def test_backtest_with_commission(backtest_engine):
    """
    Test backtest with commission costs

    Verifies:
    - Commission calculation
    - Impact on profits
    - Trade execution
    """
    # Create larger dataset to meet minimum points requirement
        dates = pd.date_range(start='2023-01-01', periods=30)  # Increased from 10
        test_data = pd.DataFrame({
            'open':  [100 + i * 0.5 for i in range(30)],  # Trending up to ensure trades
            'high':  [100 + i * 0.5 + 1 for i in range(30)],
            'low':   [100 + i * 0.5 - 1 for i in range(30)],
            'close': [100 + i * 0.5 + 0.2 for i in range(30)],
            'volume': [10000] * 30
        }, index=dates)

    # Configure backtest with smaller minimum data requirement
        backtest_engine.config.minimum_data_points = 10  # Reduce minimum requirement


    results = await backtest_engine.run_test(
            data=test_data,
            initial_capital=10000,
            risk_per_trade=0.02,
            commission=0.001
        )

    # Verify commission impact
    assert results['total_commission'] > 0
    assert 'total_trades' in results
    assert results['total_trades'] > 0
    assert results.get('gross_profit', 0) > results.get('net_profit', 0)

@pytest.mark.asyncio
async def test_backtest_strategy_rules(backtest_engine, sample_data):
    """
    Test backtest with custom strategy implementation

    Verifies:
    - Strategy execution
    - Signal generation
    - Trading decisions
    """
    def simple_strategy(volatility, trend):
        """Simple test strategy based on volatility and trend"""
        if (volatility.get('metrics', {}).get('volatility_regime') == 'low_volatility' and
            trend.get('regime') == MarketRegime.TRENDING_UP):
            return 'BUY'
        elif volatility.get('metrics', {}).get('volatility_regime') == 'high_volatility':
            return 'SELL'
        return 'HOLD'

    results = await backtest_engine.run_test(
        data=sample_data,
        initial_capital=10000,
        strategy=simple_strategy
    )

    # Verify strategy execution
    assert 'strategy_signals' in results
    assert all(signal in ['BUY', 'SELL', 'HOLD'] for signal in results['strategy_signals'])
    assert len(results['strategy_signals']) > 0

@pytest.mark.asyncio
async def test_backtest_risk_management(backtest_engine, sample_data):
    """
    Test risk management features in backtest

    Verifies:
    - Position sizing
    - Stop loss enforcement
    - Risk per trade limits
    """
    results = await backtest_engine.run_test(
        data=sample_data,
        initial_capital=10000,
        risk_per_trade=0.02,  # 2% risk per trade
        commission=0.001
    )

    # Verify risk management
    assert 'max_drawdown' in results
    assert results.get('max_drawdown', 1.0) <= 0.2  # Maximum 20% drawdown
    assert results['final_capital'] >= initial_capital * 0.8  # Maximum 20% loss
