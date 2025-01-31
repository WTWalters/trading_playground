"""
Test suite for order book analysis.

Tests the OrderBook implementation with various market scenarios:
- Normal market conditions
- Illiquid markets
- Order book imbalances
- Extreme price movements
"""

import pytest
import pandas as pd
import numpy as np
from datetime import datetime
from market_analysis.microstructure.orderbook import OrderBook, OrderBookLevel


@pytest.fixture
def sample_order_book():
    """Create a sample order book with realistic bid-ask structure."""
    book = OrderBook(symbol='AAPL', levels=5)

    # Add bids (buys)
    book.update('bid', price=100.00, size=100)
    book.update('bid', price=99.95, size=200)
    book.update('bid', price=99.90, size=300)
    book.update('bid', price=99.85, size=400)
    book.update('bid', price=99.80, size=500)

    # Add asks (sells)
    book.update('ask', price=100.05, size=150)
    book.update('ask', price=100.10, size=250)
    book.update('ask', price=100.15, size=350)
    book.update('ask', price=100.20, size=450)
    book.update('ask', price=100.25, size=550)

    return book


@pytest.fixture
def illiquid_order_book():
    """Create an order book with wide spreads and low liquidity."""
    book = OrderBook(symbol='MICRO', levels=5)

    # Sparse bids
    book.update('bid', price=10.00, size=10)
    book.update('bid', price=9.90, size=15)

    # Sparse asks with wide spread
    book.update('ask', price=10.50, size=12)
    book.update('ask', price=10.75, size=18)

    return book


@pytest.fixture
def imbalanced_order_book():
    """Create an order book with significant buy/sell imbalance."""
    book = OrderBook(symbol='IMBAL', levels=5)

    # Light bids
    book.update('bid', price=50.00, size=100)
    book.update('bid', price=49.95, size=100)

    # Heavy asks
    book.update('ask', price=50.05, size=1000)
    book.update('ask', price=50.10, size=1500)

    return book


def test_order_book_initialization():
    """Test basic order book initialization."""
    book = OrderBook(symbol='TEST', levels=5)

    assert book.symbol == 'TEST'
    assert book.levels == 5
    assert len(book.bids) == 0
    assert len(book.asks) == 0
    assert book.mid_price is None


def test_order_book_updates(sample_order_book):
    """Test order book update functionality."""
    # Test basic properties
    assert len(sample_order_book.bids) == 5
    assert len(sample_order_book.asks) == 5

    # Test price levels
    best_bid = max(sample_order_book.bids.keys())
    best_ask = min(sample_order_book.asks.keys())
    assert best_bid == 100.00
    assert best_ask == 100.05

    # Test size at levels
    assert sample_order_book.bids[best_bid].size == 100
    assert sample_order_book.asks[best_ask].size == 150


def test_spread_calculation(sample_order_book, illiquid_order_book):
    """Test spread calculations under different market conditions."""
    # Test normal market spread
    normal_spread = sample_order_book.get_spread()
    np.testing.assert_almost_equal(normal_spread, 0.05, decimal=4)  # Added missing parenthesis

    # Test illiquid market spread
    wide_spread = illiquid_order_book.get_spread()
    assert wide_spread == 0.50

    # Test relative spreads
    normal_relative = sample_order_book.get_relative_spread()
    illiquid_relative = illiquid_order_book.get_relative_spread()
    assert normal_relative < illiquid_relative


def test_book_imbalance(imbalanced_order_book, sample_order_book):
    """Test order book imbalance calculations."""
    # Test imbalanced book
    imbal = imbalanced_order_book.get_book_imbalance()
    assert imbal < -0.5  # Heavy ask side

    # Test balanced book
    balanced_imbal = sample_order_book.get_book_imbalance()
    assert abs(balanced_imbal) < 0.2  # Relatively balanced


def test_weighted_price_calculation(sample_order_book):
    """Test VWAP calculations at different quantity levels."""
    # Small order (within best level)
    small_buy = sample_order_book.get_weighted_price('ask', 50)
    assert small_buy == 100.05

    # Large order (spans multiple levels)
    large_buy = sample_order_book.get_weighted_price('ask', 500)
    assert large_buy > 100.05  # Higher due to walking the book

    # Test insufficient liquidity
    too_large = sample_order_book.get_weighted_price('ask', 10000)
    assert too_large is None


def test_market_impact_estimation(sample_order_book):
    """Test market impact calculations."""
    # Small order impact
    small_impact = sample_order_book.estimate_market_impact(50, 'buy')

    # Large order impact
    large_impact = sample_order_book.estimate_market_impact(500, 'buy')

    assert large_impact[0] > small_impact[0]  # Price impact
    assert large_impact[1] > small_impact[1]  # BPS impact


def test_liquidity_metrics(sample_order_book, illiquid_order_book):
    """Test comprehensive liquidity metrics."""
    # Normal market metrics
    normal_metrics = sample_order_book.get_liquidity_metrics()

    # Illiquid market metrics
    illiquid_metrics = illiquid_order_book.get_liquidity_metrics()

    # Compare metrics
    assert normal_metrics['spread'] < illiquid_metrics['spread']
    assert normal_metrics['total_liquidity'] > illiquid_metrics['total_liquidity']


def test_book_dynamics():
    """Test dynamic order book updates and price formation."""
    book = OrderBook(symbol='TEST', levels=5)

    # Initial state
    book.update('bid', price=100.00, size=100)
    book.update('ask', price=100.10, size=100)
    initial_spread = book.get_spread()

    # Aggressive buy
    book.update('bid', price=100.05, size=200)
    assert book.get_spread() < initial_spread

    # Crossing orders
    book.update('bid', price=100.15, size=300)
    assert len(book.bids) > 0
    assert book.get_book_imbalance() > 0  # Buy pressure


def test_order_removal():
    """Test order removal and book maintenance."""
    book = OrderBook(symbol='TEST', levels=5)

    # Add and remove orders
    book.update('bid', price=100.00, size=100)
    book.update('bid', price=100.00, size=0)  # Remove

    assert 100.00 not in book.bids
    assert len(book.bids) == 0


def test_level_management():
    """Test order book level management."""
    book = OrderBook(symbol='TEST', levels=3)

    # Add more levels than limit
    prices = [100.00, 99.95, 99.90, 99.85, 99.80]
    for price in prices:
        book.update('bid', price=price, size=100)

    assert len(book.bids) == 3  # Should maintain only top 3 levels
    assert max(book.bids.keys()) == 100.00  # Should keep best prices


def test_empty_book_handling():
    """Test empty order book edge cases."""
    book = OrderBook(symbol='TEST', levels=5)

    assert book.get_spread() is None
    assert book.get_book_imbalance() is None
    assert book.get_weighted_price('bid', 100) is None

    metrics = book.get_liquidity_metrics()
    assert isinstance(metrics, dict)
    assert 'spread' in metrics


def test_price_level_ordering():
    """Test price level ordering and access."""
    book = OrderBook(symbol='TEST', levels=5)

    # Add orders out of order
    book.update('bid', price=99.90, size=100)
    book.update('bid', price=100.00, size=100)
    book.update('bid', price=99.95, size=100)

    bid_prices = list(book.bids.keys())
    assert bid_prices == sorted(bid_prices, reverse=True)


def test_book_state_representation(sample_order_book):
    """Test order book state representation."""
    state = sample_order_book.get_book_state()

    assert isinstance(state, pd.DataFrame)
    assert 'bid_price' in state.columns
    assert 'ask_price' in state.columns
    assert len(state) <= sample_order_book.levels
