"""
Parameter Management Module

This module provides functionality for adaptive parameter management based on market regimes.
It includes methods for parameter optimization, storage, and smooth transitions between
different parameter sets as market conditions change.
"""

from .parameters import AdaptiveParameterManager

__all__ = ["AdaptiveParameterManager"]
