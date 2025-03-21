"""
This module contains methods used by the RiskManager class that couldn't fit in the
original file due to size limitations.
"""

from typing import Dict, Any


def adjust_metrics_for_scenario(metrics: Dict[str, Any], scenario: Dict[str, Any]) -> Dict[str, Any]:
    """
    Adjust strategy metrics based on stress scenario.
    
    Args:
        metrics: Original strategy metrics
        scenario: Stress test scenario parameters
        
    Returns:
        Adjusted metrics
    """
    # Create a copy to avoid modifying the original
    adjusted_metrics = metrics.copy()
    
    # Apply scenario-specific adjustments
    if scenario['name'] == "Market Crash":
        # Reduce win rate and win/loss ratio in a crash
        adjusted_metrics['win_rate'] = metrics.get('win_rate', 0.5) * 0.7
        adjusted_metrics['win_loss_ratio'] = metrics.get('win_loss_ratio', 1.5) * 0.6
        adjusted_metrics['volatility'] = metrics.get('volatility', 0.2) * 3.0
        adjusted_metrics['sharpe'] = max(0.1, metrics.get('sharpe', 1.0) * 0.4)
        
    elif scenario['name'] == "Liquidity Crisis":
        # Reduced win rate but higher win/loss ratio due to wider spreads
        adjusted_metrics['win_rate'] = metrics.get('win_rate', 0.5) * 0.8
        adjusted_metrics['win_loss_ratio'] = metrics.get('win_loss_ratio', 1.5) * 0.7
        adjusted_metrics['volatility'] = metrics.get('volatility', 0.2) * 2.0
        adjusted_metrics['sharpe'] = max(0.3, metrics.get('sharpe', 1.0) * 0.6)
        
    elif scenario['name'] == "Mean Reversion Breakdown":
        # Significantly reduced win rate for mean reversion strategies
        adjusted_metrics['win_rate'] = metrics.get('win_rate', 0.5) * 0.6
        adjusted_metrics['win_loss_ratio'] = metrics.get('win_loss_ratio', 1.5) * 0.9
        adjusted_metrics['volatility'] = metrics.get('volatility', 0.2) * 1.5
        adjusted_metrics['sharpe'] = max(0.2, metrics.get('sharpe', 1.0) * 0.5)
        
    # Apply correlation multiplier if available
    if 'correlation_multiplier' in scenario and 'correlation' in metrics:
        multiplier = scenario['correlation_multiplier']
        correlation = metrics['correlation']
        # Ensure correlation stays in [-1, 1] range
        adjusted_metrics['correlation'] = max(-1.0, min(1.0, correlation * multiplier))
        
    return adjusted_metrics
