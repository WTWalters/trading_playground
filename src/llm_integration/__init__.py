"""
LLM integration package for the TITAN trading platform.

This package provides components for integrating Large Language Models (LLMs)
into the trading platform for enhanced market regime analysis and trade post-mortem analysis.
"""

from .config import LLMConfig, load_llm_config

__all__ = ['LLMConfig', 'load_llm_config']
