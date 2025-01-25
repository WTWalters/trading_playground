# tests/market_analysis/test_trade_tracking.py

import pytest
from datetime import datetime
from src.market_analysis.trade import TradeTracker

@pytest.fixture
def tracker():
    """Create trade tracker instance for testing"""
    return TradeTracker()

def test_basic_trade_tracking():
    """Test basic trade tracking functionality"""
    tracker = TradeTracker()

    # Add some test trades
    tracker.add_trade(entry_price=100, exit_price=110)  # Win
    tracker.add_trade(entry_price=100, exit_price=90)   # Loss
    tracker.add_trade(entry_price=100, exit_price=105)  # Win

    # Test win rate calculation
    assert tracker.get_win_rate() == pytest.approx(0.667, 0.001)  # 2/3 wins

    # Test profit calculation
    total_profit = tracker.get_total_profit()
    assert total_profit == 5  # (10 - 10 + 5) = 5

def test_trade_metrics():
    """Test detailed trade metrics"""
    tracker = TradeTracker()

    # Add trades with more details
    tracker.add_trade(
        entry_price=100,
        exit_price=110,
        entry_time=datetime(2023, 1, 1),
        exit_time=datetime(2023, 1, 2),
        position_size=100
    )

    metrics = tracker.get_metrics()
    assert metrics['avg_trade_duration'] == timedelta(days=1)
    assert metrics['avg_profit_per_trade'] == 1000  # (110-100) * 100 shares
    assert metrics['largest_win'] == 1000
    assert metrics['largest_loss'] == 0

def test_trade_series():
    """Test trade series analysis"""
    tracker = TradeTracker()

    # Add a series of trades
    trades = [
        (100, 110),  # Win
        (110, 108),  # Loss
        (108, 115),  # Win
        (115, 113),  # Loss
        (113, 120)   # Win
    ]

    for entry, exit in trades:
        tracker.add_trade(entry_price=entry, exit_price=exit)

    # Test consecutive wins/losses
    assert tracker.get_max_consecutive_wins() == 2
    assert tracker.get_max_consecutive_losses() == 1

def test_risk_metrics():
    """Test risk-related metrics"""
    tracker = TradeTracker()

    # Add trades with risk information
    tracker.add_trade(
        entry_price=100,
        exit_price=110,
        risk_amount=200,  # Risked $200
        position_size=100
    )

    risk_metrics = tracker.get_risk_metrics()
    assert risk_metrics['risk_reward_ratio'] == 0.5  # Made $1000, risked $200
    assert risk_metrics['avg_risk_per_trade'] == 200
