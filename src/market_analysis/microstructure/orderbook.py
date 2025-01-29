"""
Order book analysis and market microstructure module.

This module implements tools for analyzing market microstructure:
- Order book reconstruction and analysis
- Bid-ask spread analytics
- Order flow imbalance
- Market impact modeling
- Liquidity analysis
"""

from typing import Dict, List, Optional, Tuple
import pandas as pd
import numpy as np
from dataclasses import dataclass, field


@dataclass
class OrderBookLevel:
    """
    Single level in the order book.

    Attributes:
        price: Price level
        size: Quantity at this price
        orders: Number of orders at this level
        timestamp: Last update time
    """
    price: float
    size: float
    orders: int = 1
    timestamp: pd.Timestamp = None

    def __post_init__(self):
        if self.timestamp is None:
            self.timestamp = pd.Timestamp.now()


@dataclass
class OrderBook:
    """
    Order book representation and analysis.

    Maintains bid and ask sides of the book with methods for:
    - Book updates and maintenance
    - Liquidity analysis
    - Price impact estimation
    - Order book imbalance
    """
    symbol: str
    timestamp: pd.Timestamp = field(default_factory=pd.Timestamp.now)
    levels: int = 10  # Number of levels to maintain
    
    def __post_init__(self):
        """Initialize order book structures."""
        self.bids: Dict[float, OrderBookLevel] = {}
        self.asks: Dict[float, OrderBookLevel] = {}
        self.mid_price: Optional[float] = None
        self._last_update = self.timestamp
        
    def update(self, side: str, price: float, size: float,
               orders: int = 1, timestamp: pd.Timestamp = None) -> None:
        """
        Update order book with new level.

        Args:
            side: 'bid' or 'ask'
            price: Price level
            size: Quantity at price
            orders: Number of orders
            timestamp: Update timestamp
        """
        book = self.bids if side.lower() == 'bid' else self.asks
        
        if size > 0:
            book[price] = OrderBookLevel(
                price=price,
                size=size,
                orders=orders,
                timestamp=timestamp or pd.Timestamp.now()
            )
        else:
            book.pop(price, None)
            
        # Maintain only top levels
        self._trim_book()
        
        # Update mid price
        self._update_mid_price()
        
    def _trim_book(self) -> None:
        """Maintain only specified number of levels."""
        self.bids = dict(sorted(self.bids.items(), reverse=True)[:self.levels])
        self.asks = dict(sorted(self.asks.items())[:self.levels])
        
    def _update_mid_price(self) -> None:
        """Update mid price calculation."""
        if self.bids and self.asks:
            best_bid = max(self.bids.keys())
            best_ask = min(self.asks.keys())
            self.mid_price = (best_bid + best_ask) / 2
            
    def get_spread(self) -> Optional[float]:
        """
        Calculate current bid-ask spread.

        Returns:
            float: Current spread or None if not enough data
        """
        if not (self.bids and self.asks):
            return None
            
        best_bid = max(self.bids.keys())
        best_ask = min(self.asks.keys())
        return best_ask - best_bid
        
    def get_relative_spread(self) -> Optional[float]:
        """
        Calculate relative spread (spread / mid price).

        Returns:
            float: Relative spread or None if not enough data
        """
        spread = self.get_spread()
        if spread is None or self.mid_price is None:
            return None
            
        return spread / self.mid_price
        
    def get_book_imbalance(self, levels: int = None) -> Optional[float]:
        """
        Calculate order book imbalance.

        Args:
            levels: Number of levels to consider (default: all)

        Returns:
            float: Order book imbalance [-1, 1] or None if not enough data
        """
        if not (self.bids and self.asks):
            return None
            
        levels = levels or self.levels
        bid_sizes = [level.size for level in list(self.bids.values())[:levels]]
        ask_sizes = [level.size for level in list(self.asks.values())[:levels]]
        
        total_bid_size = sum(bid_sizes)
        total_ask_size = sum(ask_sizes)
        total_size = total_bid_size + total_ask_size
        
        if total_size == 0:
            return 0.0
            
        return (total_bid_size - total_ask_size) / total_size
        
    def get_weighted_price(self, side: str, quantity: float) -> Optional[float]:
        """
        Calculate volume-weighted average price for specified quantity.

        Args:
            side: 'bid' or 'ask'
            quantity: Quantity to execute

        Returns:
            float: VWAP for specified quantity or None if not enough liquidity
        """
        book = self.bids if side.lower() == 'bid' else self.asks
        levels = sorted(book.values(), key=lambda x: x.price, reverse=(side.lower() == 'bid'))
        
        remaining_qty = quantity
        vwap_numerator = 0.0
        
        for level in levels:
            executed_qty = min(remaining_qty, level.size)
            vwap_numerator += executed_qty * level.price
            remaining_qty -= executed_qty
            
            if remaining_qty <= 0:
                break
                
        if remaining_qty > 0:
            return None  # Not enough liquidity
            
        return vwap_numerator / quantity
        
    def estimate_market_impact(self, quantity: float, side: str) -> Optional[Tuple[float, float]]:
        """
        Estimate market impact for specified quantity.

        Args:
            quantity: Quantity to execute
            side: 'buy' or 'sell'

        Returns:
            Tuple[float, float]: (Estimated impact in price, Estimated impact in basis points)
        """
        if not (self.bids and self.asks) or self.mid_price is None:
            return None
            
        vwap = self.get_weighted_price(side, quantity)
        if vwap is None:
            return None
            
        impact_price = abs(vwap - self.mid_price)
        impact_bps = (impact_price / self.mid_price) * 10000
        
        return impact_price, impact_bps
        
    def get_liquidity_metrics(self) -> Dict[str, float]:
        """
        Calculate comprehensive liquidity metrics.

        Returns:
            Dict with liquidity metrics:
            - Spread and relative spread
            - Book imbalance
            - Available liquidity at different levels
            - Price impact estimates
        """
        metrics = {}
        
        # Basic metrics
        metrics['spread'] = self.get_spread() or float('nan')
        metrics['relative_spread_bps'] = (self.get_relative_spread() or 0) * 10000
        metrics['book_imbalance'] = self.get_book_imbalance() or 0
        
        # Liquidity metrics
        if self.bids and self.asks:
            metrics['bid_liquidity'] = sum(level.size for level in self.bids.values())
            metrics['ask_liquidity'] = sum(level.size for level in self.asks.values())
            metrics['total_liquidity'] = metrics['bid_liquidity'] + metrics['ask_liquidity']
            
            # Concentration metrics
            best_bid_size = max(self.bids.values(), key=lambda x: x.price).size
            best_ask_size = min(self.asks.values(), key=lambda x: x.price).size
            metrics['top_level_concentration'] = (best_bid_size + best_ask_size) / metrics['total_liquidity']
        
        # Market impact estimates
        standard_size = metrics.get('total_liquidity', 0) * 0.01  # 1% of total liquidity
        if standard_size > 0:
            buy_impact = self.estimate_market_impact(standard_size, 'buy')
            sell_impact = self.estimate_market_impact(standard_size, 'sell')
            if buy_impact and sell_impact:
                metrics['standard_size_impact_buy_bps'] = buy_impact[1]
                metrics['standard_size_impact_sell_bps'] = sell_impact[1]
        
        return metrics
        
    def get_book_state(self) -> pd.DataFrame:
        """
        Get current order book state as DataFrame.

        Returns:
            DataFrame with bid and ask levels
        """
        max_levels = max(len(self.bids), len(self.asks))
        data = []
        
        for i in range(max_levels):
            bid_level = list(self.bids.values())[i] if i < len(self.bids) else None
            ask_level = list(self.asks.values())[i] if i < len(self.asks) else None
            
            data.append({
                'bid_price': bid_level.price if bid_level else None,
                'bid_size': bid_level.size if bid_level else None,
                'bid_orders': bid_level.orders if bid_level else None,
                'ask_price': ask_level.price if ask_level else None,
                'ask_size': ask_level.size if ask_level else None,
                'ask_orders': ask_level.orders if ask_level else None
            })
            
        return pd.DataFrame(data)