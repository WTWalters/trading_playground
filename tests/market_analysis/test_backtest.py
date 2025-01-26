# tests/market_analysis/test_backtest.py

import pytest
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from src.market_analysis.backtest import SimpleBacktest
from src.market_analysis.base import AnalysisConfig, MarketRegime
from src.market_analysis.trade import TradeDirection, TradeStatus

@pytest.fixture
def sample_backtest_data():
    """Create sample data for backtest testing"""
    dates = pd.date_range(start='2023-01-01', periods=50)
    return pd.DataFrame({
        'open':  [100 + i * 0.5 for i in range(50)],
        'high':  [100 + i * 0.5 + 1 for i in range(50)],
        'low':   [100 + i * 0.5 - 1 for i in range(50)],
        'close': [100 + i * 0.5 + 0.2 for i in range(50)],
        'volume': [10000] * 50
    }, index=dates)

@pytest.fixture
def trending_data():
    """Create trending data for specific test cases"""
    dates = pd.date_range(start='2023-01-01', periods=30)
    return pd.DataFrame({
        'open':  [100 + i for i in range(30)],  # Strong uptrend
        'high':  [100 + i + 0.5 for i in range(30)],
        'low':   [100 + i - 0.5 for i in range(30)],
        'close': [100 + i + 0.3 for i in range(30)],
        'volume': [10000] * 30
    }, index=dates)

@pytest.fixture
def volatile_data():
    """Create volatile data for specific test cases"""
    dates = pd.date_range(start='2023-01-01', periods=30)
    np.random.seed(42)  # For reproducibility
    base = 100
    volatility = 2.0
    prices = base + np.random.randn(30) * volatility
    return pd.DataFrame({
        'open':  prices,
        'high':  prices + 1,
        'low':   prices - 1,
        'close': prices + 0.2,
        'volume': [10000] * 30
    }, index=dates)

@pytest.mark.asyncio
async def test_basic_backtest(backtest_engine, sample_backtest_data):
    """Test basic backtest functionality"""
    results = await backtest_engine.run_test(
        data=sample_backtest_data,
        initial_capital=10000,
        risk_per_trade=0.02
    )

    assert 'final_capital' in results
    assert 'total_trades' in results
    assert 'win_rate' in results
    assert results['final_capital'] > 0
    assert isinstance(results['equity_curve'], pd.DataFrame)

@pytest.mark.asyncio
async def test_backtest_with_commission(backtest_engine, trending_data):
    """Test backtest with commission costs"""
    results = await backtest_engine.run_test(
        data=trending_data,
        initial_capital=10000,
        risk_per_trade=0.02,
        commission=0.001  # 0.1% commission
    )

    assert results['total_commission'] > 0
    assert results['total_trades'] > 0
    assert results['final_capital'] < (10000 + results.get('trade_metrics', {}).get('total_profit', 0))

@pytest.mark.asyncio
async def test_backtest_strategy_rules(backtest_engine, sample_backtest_data):
    """Test backtest with specific strategy rules"""
    def test_strategy(volatility, trend):
        """Test strategy implementation"""
        if (volatility.get('metrics', {}).get('volatility_regime') == 'low_volatility' and
            trend.get('regime') == MarketRegime.TRENDING_UP):
            return 'BUY'
        elif volatility.get('metrics', {}).get('volatility_regime') == 'high_volatility':
            return 'SELL'
        return 'HOLD'

    results = await backtest_engine.run_test(
        data=sample_backtest_data,
        initial_capital=10000,
        strategy=test_strategy
    )

    assert 'strategy_signals' in results
    assert all(signal in ['BUY', 'SELL', 'HOLD'] for signal in results['strategy_signals'])
    assert len(results['strategy_signals']) == len(sample_backtest_data) - 1

@pytest.mark.asyncio
async def test_backtest_risk_management(backtest_engine, sample_backtest_data):
    """Test risk management features in backtest"""
    results = await backtest_engine.run_test(
        data=sample_backtest_data,
        initial_capital=10000,
        risk_per_trade=0.02,  # 2% risk per trade
        commission=0.001
    )

    assert 'max_drawdown' in results
    assert results['max_drawdown'] >= 0.0
    assert results['max_drawdown'] <= 1.0
    assert 'trade_metrics' in results
    assert isinstance(results['trade_metrics'], dict)

@pytest.mark.asyncio
async def test_backtest_position_sizing(backtest_engine, trending_data):
    """Test position sizing calculations"""
    results = await backtest_engine.run_test(
        data=trending_data,
        initial_capital=10000,
        risk_per_trade=0.01  # 1% risk per trade
    )

    # Verify position sizes are appropriate
    assert results['total_trades'] > 0
    for trade in results.get('trade_metrics', {}).get('trades', []):
        position_value = trade.entry_price * trade.position_size
        assert position_value <= 10000  # No position should be larger than initial capital

@pytest.mark.asyncio
async def test_backtest_with_volatile_market(backtest_engine, volatile_data):
    """Test backtest behavior in volatile market conditions"""
    results = await backtest_engine.run_test(
        data=volatile_data,
        initial_capital=10000,
        risk_per_trade=0.02
    )

    assert 'volatility_metrics' in results
    assert results.get('volatility_metrics', {}).get('average_volatility', 0) > 0

@pytest.mark.asyncio
async def test_backtest_edge_cases(backtest_engine):
    """Test backtest handling of edge cases"""
    # Test with minimal data
    min_data = pd.DataFrame({
        'open': [100, 101],
        'high': [102, 103],
        'low': [99, 98],
        'close': [101, 102],
        'volume': [1000, 1000]
    }, index=pd.date_range(start='2023-01-01', periods=2))

    results = await backtest_engine.run_test(
        data=min_data,
        initial_capital=10000
    )

    assert results['final_capital'] == 10000  # Should not trade with insufficient data
    assert results['total_trades'] == 0

@pytest.mark.asyncio
async def test_backtest_trade_validation(backtest_engine, sample_backtest_data):
    """Test trade validation and tracking"""
    results = await backtest_engine.run_test(
        data=sample_backtest_data,
        initial_capital=10000,
        risk_per_trade=0.02
    )

    trades = results.get('trade_metrics', {}).get('trades', [])
    for trade in trades:
        assert trade.entry_price > 0
        assert trade.position_size > 0
        assert trade.direction in [TradeDirection.LONG, TradeDirection.SHORT]
        assert trade.status == TradeStatus.CLOSED
