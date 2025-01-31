"""
Integration tests for market microstructure analysis.

Tests interactions between:
- Order book analysis
- Liquidity measures
- Market impact analysis
- Price discovery processes
"""

import pytest
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from market_analysis.microstructure.orderbook import OrderBook
from market_analysis.microstructure.liquidity import LiquidityAnalyzer
from market_analysis.microstructure.liquidity_measures import AdvancedLiquidityMeasures
from market_analysis.microstructure.impact import MarketImpact


@pytest.fixture
def simulation_data():
    """Generate realistic market data for integration testing."""
    dates = pd.date_range(start='2024-01-01', periods=1000, freq='5min')
    np.random.seed(42)

    trades = pd.DataFrame({
        'timestamp': dates,
        'price': 100 * np.exp(np.random.randn(1000) * 0.001),
        'volume': np.random.lognormal(4, 1, 1000),
        'direction': np.random.choice([-1, 1], 1000)
    })

    quotes = pd.DataFrame({
        'timestamp': dates,
        'bid': trades['price'] - np.random.gamma(2, 0.02, 1000)/2,
        'ask': trades['price'] + np.random.gamma(2, 0.02, 1000)/2
    })

    return {
        'trades': trades,
        'quotes': quotes,
        'book': pd.DataFrame({
            'timestamp': dates,
            'price': trades['price'],
            'bid_depth': np.random.lognormal(4, 0.5, 1000),
            'ask_depth': np.random.lognormal(4, 0.5, 1000),
            'spread': np.random.gamma(2, 0.02, 1000)
        })
    }

    return {
        'trades': trades,
        'quotes': quotes,
        'book': book,  # Adding book data that was missing
        'market_data': pd.DataFrame({
            'timestamp': dates,
            'rf': 0.0001 / np.sqrt(252),
            'market_return': np.random.normal(0.0005, 0.01, 1000)
        })
    }
    return data

    # Generate base price process
    T = 1000
    dt = 1.0 / T
    sigma = 0.2  # Volatility
    mu = 0.05    # Drift

    # Geometric Brownian Motion
    dW = np.random.normal(0, np.sqrt(dt), T)
    price_path = 100 * np.exp(np.cumsum((mu - 0.5 * sigma**2) * dt + sigma * dW))

    dates = pd.date_range(start='2024-01-01', periods=T, freq='5min')

    # Generate matching order book data
    spreads = np.random.gamma(2, 0.02, T)  # Random spreads
    depths = np.random.lognormal(4, 0.5, T)  # Random depths

    trades = pd.DataFrame({
        'timestamp': dates,
        'price': price_path,
        'volume': np.random.lognormal(8, 1, T),
        'direction': np.random.choice([-1, 1], T)
    })

    quotes = pd.DataFrame({
        'timestamp': dates,
        'bid': price_path - spreads/2,
        'ask': price_path + spreads/2
    })

    book_data = []
    for i in range(T):
        # Generate order book snapshot
        book_data.append({
            'timestamp': dates[i],
            'price': price_path[i],
            'bid_depth': depths[i],
            'ask_depth': depths[i] * (1 + np.random.randn() * 0.1),
            'spread': spreads[i]
        })

    book_df = pd.DataFrame(book_data)

    return {
        'trades': trades,
        'quotes': quotes,
        'book': book_df
    }


def test_liquidity_orderbook_consistency(simulation_data):
    """Test consistency between liquidity measures and order book metrics."""
    # Initialize analyzers
    book = OrderBook(symbol='TEST')
    liquidity = LiquidityAnalyzer(
        trades=simulation_data['trades'],
        quotes=simulation_data['quotes']
    )
    advanced = AdvancedLiquidityMeasures(
        trades=simulation_data['trades'],
        quotes=simulation_data['quotes']
    )

    # Update order book
    for _, row in simulation_data['book'].iterrows():
        book.update('bid', row['price'] - row['spread']/2, row['bid_depth'])
        book.update('ask', row['price'] + row['spread']/2, row['ask_depth'])

    # Compare spread measures
    book_spread = book.get_spread()
    roll_spread = advanced.calculate_roll_spread().mean()

    # Spreads should be of similar magnitude
    assert abs(book_spread - roll_spread) / book_spread < 0.5

    # Compare liquidity metrics
    book_metrics = book.get_liquidity_metrics()
    kyle_lambda = liquidity.calculate_kyle_lambda().mean()

    # Book imbalance should affect Kyle's lambda
    assert (book_metrics['book_imbalance'] != 0) == (kyle_lambda > 0)


def test_impact_measures_integration(simulation_data):
    """Test integration between impact measures and order book."""
    # Initialize analyzers
    book = OrderBook(symbol='TEST')
    impact = MarketImpact(
        trades=simulation_data['trades'],
        quotes=simulation_data['quotes']
    )

    # Update order book and calculate impacts
    impacts = []
    book_impacts = []

    for _, row in simulation_data['book'].iterrows():
        # Update book
        book.update('bid', row['price'] - row['spread']/2, row['bid_depth'])
        book.update('ask', row['price'] + row['spread']/2, row['ask_depth'])

        # Calculate impacts
        model_impact = impact.calculate_square_root_impact(row['volume'], 0.1)
        book_impact = book.estimate_market_impact(row['volume'], 'buy')[0]

        impacts.append(model_impact)
        book_impacts.append(book_impact)

    # Compare impact estimates
    correlation = np.corrcoef(impacts, book_impacts)[0,1]
    assert correlation > 0.5  # Should be positively correlated


def test_liquidity_measures_combination(simulation_data):
    """Test combination of different liquidity measures."""
    # Initialize analyzers
    advanced = AdvancedLiquidityMeasures(
        trades=simulation_data['trades'],
        quotes=simulation_data['quotes']
    )
    liquidity = LiquidityAnalyzer(
        trades=simulation_data['trades'],
        quotes=simulation_data['quotes']
    )

    # Calculate various measures
    roll = advanced.calculate_roll_spread()
    kyle = liquidity.calculate_kyle_lambda()
    lot = advanced.calculate_lot_measure()

    # Combine measures into composite score
    measures = pd.DataFrame({
        'roll_spread': roll,
        'kyle_lambda': kyle,
        'lot_measure': lot['lot_measure']
    })

    # Measures should be correlated in illiquid periods
    high_spread_mask = measures['roll_spread'] > measures['roll_spread'].median()
    high_spread_corr = measures.loc[high_spread_mask].corr()

    assert high_spread_corr.loc['roll_spread', 'kyle_lambda'] > 0
    assert high_spread_corr.loc['roll_spread', 'lot_measure'] > 0


def test_price_discovery_integration(simulation_data):
    """Test price discovery measures across components."""
    # Initialize analyzers
    book = OrderBook(symbol='TEST')
    advanced = AdvancedLiquidityMeasures(
        trades=simulation_data['trades'],
        quotes=simulation_data['quotes'],
        benchmark=simulation_data['trades'][['timestamp', 'price']].rename(columns={'price': 'benchmark'})
    )

    # Calculate Hasbrouck information share
    info_share = advanced.calculate_hasbrouck_measure()

    # Update book and track price discovery
    midpoint_changes = []
    trade_impacts = []

    for _, row in simulation_data['book'].iterrows():
        # Pre-update midpoint
        old_mid = book.mid_price

        # Update book
        book.update('bid', row['price'] - row['spread']/2, row['bid_depth'])
        book.update('ask', row['price'] + row['spread']/2, row['ask_depth'])

        # Post-update midpoint
        new_mid = book.mid_price

        if old_mid is not None and new_mid is not None:
            midpoint_changes.append(new_mid - old_mid)
            trade_impacts.append(abs(row['price'] - old_mid))

    # Compare price discovery measures
    mid_vol = np.std(midpoint_changes)
    impact_vol = np.std(trade_impacts)

    # Information share should be related to relative volatilities
    assert abs(info_share['info_share'].mean() - (impact_vol / mid_vol)) < 0.5


def test_spread_decomposition_integration(simulation_data):
    """Test spread decomposition across different measures."""
    # Initialize analyzers
    book = OrderBook(symbol='TEST')
    advanced = AdvancedLiquidityMeasures(
        trades=simulation_data['trades'],
        quotes=simulation_data['quotes']
    )

    # Calculate Huang-Stoll components
    hs_components = advanced.calculate_huang_stoll()

    # Calculate order book based components
    book_spreads = []
    book_imbalances = []

    for _, row in simulation_data['book'].iterrows():
        book.update('bid', row['price'] - row['spread']/2, row['bid_depth'])
        book.update('ask', row['price'] + row['spread']/2, row['ask_depth'])

        book_spreads.append(book.get_spread())
        book_imbalances.append(book.get_book_imbalance())

    # Compare decompositions
    # Order processing should be higher when imbalances are low
    low_imbal_mask = np.abs(book_imbalances) < np.median(np.abs(book_imbalances))
    assert hs_components.loc[low_imbal_mask, 'order_processing'].mean() > \
           hs_components.loc[~low_imbal_mask, 'order_processing'].mean()

    # Adverse selection should be higher when spreads are high
    high_spread_mask = book_spreads > np.median(book_spreads)
    assert hs_components.loc[high_spread_mask, 'adverse_selection'].mean() > \
           hs_components.loc[~high_spread_mask, 'adverse_selection'].mean()
