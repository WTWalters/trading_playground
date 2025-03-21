"""
Market analysis package for quantitative trading.

This package provides tools for analyzing financial markets,
including time series analysis, market microstructure,
and statistical validation.
"""

from . import time_series
from .simple_backtest import SimpleBacktest

__all__ = [
    'time_series',
    'SimpleBacktest'
]