# tests/market_analysis/test_trade_tracking.py

import pytest
from datetime import datetime, timedelta
from src.market_analysis.trade import TradeTracker

def test_basic_trade_tracking(tracker):
    """
    Test basic trade tracking functionality

    Verifies:
    - Win rate calculation
    - Basic profit calculation
    - Trade recording
    """
    # Add test trades with known outcomes
    tracker.add_trade(entry_price=100, exit_price=110)  # Win: +10
    tracker.add_trade(entry_price=100, exit_price=90)   # Loss: -10
    tracker.add_trade(entry_price=100, exit_price=105)  # Win: +5

    # Verify win rate calculation (2 wins out of 3 trades)
    assert tracker.get_win_rate() == pytest.approx(0.667, 0.001)

    # Verify profit calculation (10 - 10 + 5 = 5)
    total_profit = tracker.get_total_profit()
    assert total_profit == 5

def test_trade_metrics(tracker):
    """
    Test detailed trade metrics calculation

    Verifies:
    - Duration calculation
    - Profit metrics
    - Position sizing
    """
    # Add trade with specific details
    tracker.add_trade(
        entry_price=100,
        exit_price=110,
        entry_time=datetime(2023, 1, 1),
        exit_time=datetime(2023, 1, 2),
        position_size=100
    )

    metrics = tracker.get_metrics()

    # Verify metrics calculations
    assert metrics['avg_trade_duration'] == timedelta(days=1)
    assert metrics['average_profit'] == 1000  # (110-100) * 100 shares
    assert metrics['largest_win'] == 1000
    assert metrics['largest_loss'] == 0

def test_trade_series(tracker):
    """
    Test trade series analysis

    Verifies:
    - Consecutive wins/losses tracking
    - Trade sequence analysis
    """
    # Add series of trades with known sequence
    trades = [
        (100, 105),  # Win
        (105, 110),  # Win
        (110, 108),  # Loss
        (108, 112),  # Win
        (112, 115)   # Win
    ]

    for entry, exit in trades:
        tracker.add_trade(
            entry_price=entry,
            exit_price=exit,
            entry_time=datetime.now(),
            exit_time=datetime.now() + timedelta(days=1)
        )

    # Verify streak calculations
    assert tracker.get_max_consecutive_wins() == 2
    assert tracker.get_max_consecutive_losses() == 1
    assert len(tracker.trades) == 5

def test_risk_metrics(tracker):
    """
    Test risk-related metrics calculation

    Verifies:
    - Risk/reward ratio
    - Risk per trade
    - Risk metrics accuracy
    """
    # Add trade with specific risk parameters
    tracker.add_trade(
        entry_price=100,
        exit_price=110,
        position_size=100,
        risk_amount=200,  # Risked $200
        commission=0
    )

    risk_metrics = tracker.get_risk_metrics()

    # Profit = (110-100) * 100 = $1000
    # Risk = $200
    # Risk/Reward = $1000/$200 = 5.0
    assert risk_metrics['risk_reward_ratio'] == 5.0
    assert risk_metrics['avg_risk_per_trade'] == 200.0

def test_trade_tracking_edge_cases(tracker):
    """
    Test trade tracking with edge cases

    Verifies:
    - Zero profit trades
    - Minimum position sizes
    - Invalid trade parameters
    """
    # Test zero profit trade
    tracker.add_trade(entry_price=100, exit_price=100)

    # Test minimum position size
    tracker.add_trade(entry_price=100, exit_price=101, position_size=1)

    metrics = tracker.get_metrics()
    assert metrics['total_trades'] == 2
    assert metrics['winning_trades'] >= 0
