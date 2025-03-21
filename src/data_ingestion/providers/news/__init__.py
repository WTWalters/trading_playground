"""
News data provider implementations package.

This package provides data providers for financial news and market events.
"""

from .base import NewsDataProvider
from .alpha_vantage import AlphaVantageNewsProvider

__all__ = ['NewsDataProvider', 'AlphaVantageNewsProvider']
