# tests/market_analysis/test_risk_management.py

import pytest
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from src.market_analysis.risk import RiskManager
from decimal import Decimal

@pytest.fixture
def risk_manager():
    """Create risk manager instance for testing"""
    return RiskManager()

def test_basic_position_sizing():
    """Test basic position size calculation"""
    risk_manager = RiskManager()

    # Test case: $10,000 account, risking 2%, $1 stop loss
    position_size = risk_manager.calculate_trade_size(
        account_size=10000,
        risk_amount=2,
        stop_loss=1
    )
    assert position_size == 200  # Should be able to buy 200 shares

    # Test case: $5,000 account, risking 1%, $0.50 stop loss
    position_size = risk_manager.calculate_trade_size(
        account_size=5000,
        risk_amount=1,
        stop_loss=0.50
    )
    assert position_size == 100  # Should be able to buy 100 shares

def test_position_sizing_limits():
    """Test position sizing with extreme values"""
    risk_manager = RiskManager()

    # Test minimum position size
    position_size = risk_manager.calculate_trade_size(
        account_size=1000,
        risk_amount=1,
        stop_loss=10
    )
    assert position_size >= 1  # Should never be less than 1 share

    # Test maximum position size limit
    position_size = risk_manager.calculate_trade_size(
        account_size=1000000,
        risk_amount=2,
        stop_loss=0.01
    )
    assert position_size <= 100000  # Should respect max position size

def test_invalid_inputs():
    """Test handling of invalid inputs"""
    risk_manager = RiskManager()

    # Test negative values
    with pytest.raises(ValueError):
        risk_manager.calculate_trade_size(
            account_size=-1000,
            risk_amount=2,
            stop_loss=1
        )

    # Test zero stop loss
    with pytest.raises(ValueError):
        risk_manager.calculate_trade_size(
            account_size=1000,
            risk_amount=2,
            stop_loss=0
        )
