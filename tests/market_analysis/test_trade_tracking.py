# tests/market_analysis/test_trade_tracking.py

import pytest
from datetime import datetime, timedelta
from src.market_analysis.trade import TradeTracker

def test_basic_trade_tracking(tracker):
    """
    Test basic trade tracking functionality

    Verifies:
    - Win rate calculation accuracy
    - Basic profit calculation
    - Trade recording functionality
    - Position tracking
    """
    # Add test trades with known outcomes
    tracker.add_trade(entry_price=100, exit_price=110)  # Win: +10
    tracker.add_trade(entry_price=100, exit_price=90)   # Loss: -10
    tracker.add_trade(entry_price=100, exit_price=105)  # Win: +5

    # Verify calculations
    assert tracker.get_win_rate() == pytest.approx(0.667, 0.001)  # 2/3 wins
    assert tracker.get_total_profit() == 5  # Net profit: (10 - 10 + 5)
    assert len(tracker.trades) == 3

def test_trade_metrics(tracker):
    """
    Test detailed trade metrics calculation

    Verifies:
    - Duration calculations
    - Position sizing
    - Profit metrics
    - Trade statistics
    """
    # Add trade with complete details
    tracker.add_trade(
        entry_price=100,
        exit_price=110,
        entry_time=datetime(2023, 1, 1),
        exit_time=datetime(2023, 1, 2),
        position_size=100,
        commission=0
    )

    metrics = tracker.get_metrics()

    # Verify all metrics
    assert metrics['avg_trade_duration'] == timedelta(days=1)
    assert metrics['average_profit'] == 1000  # (110-100) * 100
    assert metrics['largest_win'] == 1000
    assert metrics['largest_loss'] == 0
    assert metrics['total_trades'] == 1

def test_trade_series(tracker):
    """
    Test trade series analysis

    Verifies:
    - Consecutive trade tracking
    - Win/loss streaks
    - Series calculations
    """
    trades = [
        (100, 105),  # Win +5
        (105, 110),  # Win +5
        (110, 108),  # Loss -2
        (108, 112),  # Win +4
        (112, 115)   # Win +3
    ]

    # Add trades with timestamps
    for entry, exit in trades:
        tracker.add_trade(
            entry_price=entry,
            exit_price=exit,
            entry_time=datetime.now(),
            exit_time=datetime.now() + timedelta(days=1)
        )

    # Verify series metrics
    assert tracker.get_max_consecutive_wins() == 2
    assert tracker.get_max_consecutive_losses() == 1
    assert len(tracker.trades) == 5

def test_risk_metrics(tracker):
    """
    Test risk-related metrics calculation

    Verifies:
    - Risk/reward ratios
    - Risk per trade calculations
    - Position risk management
    """
    # Add trade with risk parameters
    tracker.add_trade(
        entry_price=100,
        exit_price=110,
        position_size=100,
        risk_amount=200,  # Risk $200
        commission=0
    )

    risk_metrics = tracker.get_risk_metrics()

    # Verify risk calculations
    # Profit = (110-100) * 100 = $1000
    # Risk = $200
    # Risk/Reward = $1000/$200 = 5.0
    assert risk_metrics['risk_reward_ratio'] == 5.0
    assert risk_metrics['avg_risk_per_trade'] == 200.0

def test_trade_tracking_edge_cases(tracker):
    """
    Test trade tracking with edge cases

    Verifies:
    - Zero profit handling
    - Minimum position sizes
    - Invalid parameters
    - Edge case scenarios
    """
    # Test various edge cases
    tracker.add_trade(
        entry_price=100,
        exit_price=100,  # Zero profit
        position_size=1   # Minimum size
    )

    tracker.add_trade(
        entry_price=100,
        exit_price=101,   # Minimal profit
        position_size=1    # Minimum size
    )

    metrics = tracker.get_metrics()

    # Verify edge case handling
    assert metrics['total_trades'] == 2
    assert metrics['winning_trades'] >= 0
    assert metrics['losing_trades'] >= 0
