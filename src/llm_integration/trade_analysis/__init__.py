"""
Trade analysis package for LLM-enhanced trade post-mortem analysis.

This package provides components for collecting trade context,
performing post-mortem analysis, and identifying trading patterns.
"""

from .context_collector import TradeContextCollector

__all__ = ['TradeContextCollector']
