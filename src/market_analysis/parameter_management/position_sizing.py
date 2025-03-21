"""
Kelly-based Position Sizing Module

This module implements position sizing strategies based on the Kelly criterion,
following Edward Thorp's recommendations for optimal capital allocation.

Key features:
1. Kelly criterion calculation for optimal position sizes
2. Position size caps (2% max as recommended by Thorp)
3. Regime-specific Kelly fraction adjustments
4. Expected value and edge calculations
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Union, Any
from dataclasses import dataclass, field
from datetime import datetime

from src.market_analysis.regime_detection.detector import RegimeType

@dataclass
class PositionSizeResult:
    """Container for position sizing recommendations and justification."""
    position_size: float
    position_pct: float
    raw_kelly: float
    adjusted_kelly: float
    edge: float
    expected_value: float
    kelly_fraction: float
    regime: RegimeType
    max_position_pct: float
    reasoning: str
    timestamp: datetime = field(default_factory=datetime.now)


class KellyPositionSizer:
    """
    Implements position sizing strategies based on the Kelly criterion.
    
    This class provides functionality for:
    1. Calculating optimal position sizes using the Kelly criterion
    2. Adjusting Kelly fractions based on market regimes
    3. Implementing position size caps and floors
    4. Computing expected value and edge for trading opportunities
    """
    
    def __init__(self,
                 default_kelly_fraction: float = 0.5,
                 max_position_pct: float = 0.02,
                 min_position_pct: float = 0.005,
                 min_edge_required: float = 0.55,
                 regime_adjustments: Optional[Dict[RegimeType, float]] = None,
                 default_fraction: float = None,  # For backward compatibility
                 max_kelly_fraction: float = None,  # For backward compatibility
                 min_kelly_fraction: float = None): # For backward compatibility
        """
        Initialize the Kelly position sizer.
        
        Args:
            default_kelly_fraction: Default Kelly fraction (typically 0.5 for Half-Kelly)
            max_position_pct: Maximum position size as percent of portfolio (e.g., 0.02 = 2%)
            min_position_pct: Minimum position size as percent of portfolio
            min_edge_required: Minimum edge required to take a position (win probability)
            regime_adjustments: Dict mapping regime types to Kelly fraction adjustments
        """
        # Handle backward compatibility parameters
        if default_fraction is not None:
            self.default_kelly_fraction = default_fraction
        else:
            self.default_kelly_fraction = default_kelly_fraction
            
        self.max_position_pct = max_position_pct
        self.min_position_pct = min_position_pct
        self.min_edge_required = min_edge_required
        
        # Store max/min kelly fractions for compatibility
        self.max_kelly_fraction = max_kelly_fraction if max_kelly_fraction is not None else 1.0
        self.min_kelly_fraction = min_kelly_fraction if min_kelly_fraction is not None else 0.1
        
        # Default regime adjustments if none provided
        if regime_adjustments is None:
            self.regime_adjustments = {
                RegimeType.HIGH_VOLATILITY: 0.3,    # More conservative in high volatility
                RegimeType.LOW_VOLATILITY: 0.7,     # More aggressive in low volatility
                RegimeType.TRENDING: 0.6,           # Moderately aggressive in trending markets
                RegimeType.MEAN_REVERTING: 0.5,     # Standard half-Kelly in mean-reverting markets
                RegimeType.RISK_ON: 0.7,            # More aggressive in risk-on environments
                RegimeType.RISK_OFF: 0.3,           # More conservative in risk-off environments
                RegimeType.HIGH_LIQUIDITY: 0.6,     # Moderately aggressive in high liquidity
                RegimeType.LOW_LIQUIDITY: 0.4,      # More conservative in low liquidity
                RegimeType.UNDEFINED: 0.5,          # Standard half-Kelly when regime is undefined
            }
        else:
            self.regime_adjustments = regime_adjustments
            
        # Track historical position sizing decisions for analysis
        self.sizing_history = []
    
    def calculate_position_size(self,
                               win_probability: float,
                               win_loss_ratio: float,
                               current_regime: RegimeType = RegimeType.UNDEFINED,
                               confidence: float = 1.0,
                               portfolio_value: float = 1.0) -> PositionSizeResult:
        """
        Calculate position size using Kelly criterion adjusted for the current market regime.
        
        Args:
            win_probability: Probability of a winning trade (0.0 to 1.0)
            win_loss_ratio: Ratio of average win to average loss (e.g., 2.0 means avg win is 2x avg loss)
            current_regime: Current market regime
            confidence: Confidence in the win probability estimate (0.0 to 1.0)
            portfolio_value: Total portfolio value (default 1.0 for percentage calculation)
            
        Returns:
            PositionSizeResult containing position size recommendation and justification
        """
        # Check if edge is sufficient
        edge = win_probability - (1 - win_probability)
        if win_probability < self.min_edge_required:
            reasoning = f"No position taken: edge ({edge:.2f}) below minimum threshold ({self.min_edge_required:.2f})"
            return PositionSizeResult(
                position_size=0.0,
                position_pct=0.0,
                raw_kelly=0.0,
                adjusted_kelly=0.0,
                edge=edge,
                expected_value=0.0,
                kelly_fraction=0.0,
                regime=current_regime,
                max_position_pct=self.max_position_pct,
                reasoning=reasoning
            )
            
        # Calculate raw Kelly percentage
        # Kelly formula: f* = p - (1-p)/R where p is win probability and R is win/loss ratio
        if win_loss_ratio <= 0:
            reasoning = "No position taken: invalid win/loss ratio (must be positive)"
            return PositionSizeResult(
                position_size=0.0,
                position_pct=0.0,
                raw_kelly=0.0,
                adjusted_kelly=0.0,
                edge=edge,
                expected_value=0.0,
                kelly_fraction=0.0,
                regime=current_regime,
                max_position_pct=self.max_position_pct,
                reasoning=reasoning
            )
            
        raw_kelly = win_probability - (1 - win_probability) / win_loss_ratio
        
        # Adjust Kelly fraction based on regime
        kelly_fraction = self.regime_adjustments.get(current_regime, self.default_kelly_fraction)
        
        # Further adjust based on confidence in the probability estimate
        kelly_fraction *= confidence
        
        # Calculate the adjusted Kelly percentage
        adjusted_kelly = raw_kelly * kelly_fraction
        
        # Apply position size constraints
        position_pct = min(adjusted_kelly, self.max_position_pct)
        position_pct = max(position_pct, 0.0)  # Ensure non-negative
        
        # Implement minimum position size if above zero
        if 0 < position_pct < self.min_position_pct:
            position_pct = self.min_position_pct
            
        # Calculate expected value (EV) of the trade
        expected_value = (win_probability * win_loss_ratio) - (1 - win_probability)
        
        # Calculate absolute position size
        position_size = position_pct * portfolio_value
        
        # Determine reasoning for the position size
        if position_pct == 0.0:
            reasoning = "No position taken: calculated Kelly is negative"
        elif position_pct == self.max_position_pct:
            reasoning = f"Position capped at maximum {self.max_position_pct:.1%} (Thorp's recommendation)"
        elif position_pct == self.min_position_pct:
            reasoning = f"Position floor applied at minimum {self.min_position_pct:.1%}"
        else:
            reasoning = f"Optimal Kelly position: {position_pct:.1%} (raw Kelly {raw_kelly:.1%} × fraction {kelly_fraction:.1f})"
        
        # Create result
        result = PositionSizeResult(
            position_size=position_size,
            position_pct=position_pct,
            raw_kelly=raw_kelly,
            adjusted_kelly=adjusted_kelly,
            edge=edge,
            expected_value=expected_value,
            kelly_fraction=kelly_fraction,
            regime=current_regime,
            max_position_pct=self.max_position_pct,
            reasoning=reasoning
        )
        
        # Add to history
        self.sizing_history.append(result)
        
        return result
    
    def calculate_kelly_fraction(self, 
                               current_regime: RegimeType, 
                               volatility: float = None,
                               correlation: float = None,
                               regime_stability: float = 1.0) -> float:
        """
        Calculate appropriate Kelly fraction based on market conditions.
        
        Args:
            current_regime: Current market regime
            volatility: Optional current market volatility
            correlation: Optional correlation with other portfolio positions
            regime_stability: Stability of the current regime (0.0 to 1.0)
            
        Returns:
            Recommended Kelly fraction
        """
        # Start with the regime-based adjustment
        kelly_fraction = self.regime_adjustments.get(current_regime, self.default_kelly_fraction)
        
        # Adjust for volatility if provided
        if volatility is not None:
            vol_adjustment = 1.0 - min(0.5, max(0.0, (volatility - 0.2) / 0.4))
            kelly_fraction *= vol_adjustment
            
        # Adjust for correlation if provided
        if correlation is not None:
            # Higher correlation = lower Kelly fraction (less diversification benefit)
            corr_adjustment = 1.0 - min(0.5, max(0.0, correlation))
            kelly_fraction *= corr_adjustment
            
        # Adjust for regime stability
        kelly_fraction *= regime_stability
        
        # Ensure reasonable bounds
        kelly_fraction = min(max(0.1, kelly_fraction), 1.0)
        
        return kelly_fraction
    
    def calculate_win_probability(self,
                                signal_strength: float,
                                historical_win_rate: float,
                                market_condition_adjustment: float = 0.0) -> float:
        """
        Calculate win probability based on signal strength and historical performance.
        
        Args:
            signal_strength: Strength of the current trading signal (0.0 to 1.0)
            historical_win_rate: Historical win rate for this strategy
            market_condition_adjustment: Adjustment based on current market conditions (-0.2 to 0.2)
            
        Returns:
            Estimated win probability
        """
        # Base probability on historical win rate
        base_probability = historical_win_rate
        
        # Adjust based on signal strength
        # If signal_strength = 0.5 (neutral), no adjustment
        # If signal_strength > 0.5, increase probability
        # If signal_strength < 0.5, decrease probability
        signal_adjustment = (signal_strength - 0.5) * 0.4  # Scale to -0.2 to 0.2 range
        
        # Combine adjustments
        win_probability = base_probability + signal_adjustment + market_condition_adjustment
        
        # Ensure probability is within valid range
        win_probability = min(max(0.0, win_probability), 1.0)
        
        return win_probability
    
    def calculate_position_for_strategy(self,
                                      strategy_parameters: Dict[str, Any],
                                      historical_performance: Dict[str, float],
                                      signal_strength: float,
                                      current_regime: RegimeType,
                                      portfolio_value: float,
                                      regime_stability: float = 1.0) -> PositionSizeResult:
        """
        Calculate position size for a specific strategy with its parameters.
        
        Args:
            strategy_parameters: Dictionary of strategy parameters
            historical_performance: Dictionary with historical performance metrics
            signal_strength: Strength of the current trading signal (0.0 to 1.0)
            current_regime: Current market regime
            portfolio_value: Total portfolio value
            regime_stability: Stability of the current regime (0.0 to 1.0)
            
        Returns:
            PositionSizeResult containing position size recommendation
        """
        # Extract historical performance metrics
        historical_win_rate = historical_performance.get('win_rate', 0.5)
        historical_win_loss_ratio = historical_performance.get('win_loss_ratio', 1.0)
        
        # Adjust win rate based on current conditions and signal strength
        market_adjustment = strategy_parameters.get('market_adjustment', 0.0)
        win_probability = self.calculate_win_probability(
            signal_strength, 
            historical_win_rate, 
            market_adjustment
        )
        
        # Calculate appropriate Kelly fraction
        kelly_fraction = self.calculate_kelly_fraction(
            current_regime,
            volatility=historical_performance.get('volatility'),
            correlation=historical_performance.get('correlation'),
            regime_stability=regime_stability
        )
        
        # Override default Kelly fraction
        original_default = self.default_kelly_fraction
        self.default_kelly_fraction = kelly_fraction
        
        # Calculate position size
        result = self.calculate_position_size(
            win_probability=win_probability,
            win_loss_ratio=historical_win_loss_ratio,
            current_regime=current_regime,
            confidence=signal_strength,
            portfolio_value=portfolio_value
        )
        
        # Restore default Kelly fraction
        self.default_kelly_fraction = original_default
        
        return result
    
    def analyze_position_history(self, 
                               window: int = 30,
                               by_regime: bool = True) -> pd.DataFrame:
        """
        Analyze historical position sizing decisions.
        
        Args:
            window: Number of recent decisions to analyze
            by_regime: Whether to group analysis by regime
            
        Returns:
            DataFrame with position sizing analysis
        """
        if not self.sizing_history:
            return pd.DataFrame()
            
        # Convert history to DataFrame
        df = pd.DataFrame([{
            'timestamp': result.timestamp,
            'regime': result.regime.value,
            'position_pct': result.position_pct,
            'raw_kelly': result.raw_kelly,
            'adjusted_kelly': result.adjusted_kelly,
            'edge': result.edge,
            'expected_value': result.expected_value,
            'kelly_fraction': result.kelly_fraction
        } for result in self.sizing_history[-window:]])
        
        if by_regime and not df.empty:
            # Group by regime
            grouped = df.groupby('regime').agg({
                'position_pct': ['mean', 'min', 'max'],
                'raw_kelly': ['mean', 'min', 'max'],
                'adjusted_kelly': ['mean'],
                'edge': ['mean'],
                'expected_value': ['mean'],
                'kelly_fraction': ['mean']
            })
            return grouped
        else:
            return df
    
    def get_portfolio_allocations(self, 
                                signals: Dict[str, Dict[str, Any]], 
                                current_regime: RegimeType,
                                portfolio_value: float,
                                max_allocation: float = 0.9) -> Dict[str, Dict[str, Any]]:
        """
        Calculate position sizes for multiple signals within portfolio constraints.
        
        Args:
            signals: Dictionary mapping signal IDs to signal info dictionaries
            current_regime: Current market regime
            portfolio_value: Total portfolio value
            max_allocation: Maximum total portfolio allocation (0.0 to 1.0)
            
        Returns:
            Dictionary mapping signal IDs to allocation recommendations
        """
        allocations = {}
        total_ev = 0
        total_allocation = 0
        
        # First pass: calculate EVs and raw allocations
        for signal_id, signal_info in signals.items():
            # Extract signal parameters
            win_probability = signal_info.get('win_probability', 0.5)
            win_loss_ratio = signal_info.get('win_loss_ratio', 1.0)
            signal_strength = signal_info.get('strength', 0.5)
            
            # Calculate position size
            result = self.calculate_position_size(
                win_probability=win_probability,
                win_loss_ratio=win_loss_ratio,
                current_regime=current_regime,
                confidence=signal_strength,
                portfolio_value=portfolio_value
            )
            
            allocations[signal_id] = {
                'result': result,
                'expected_value': result.expected_value,
                'allocation': result.position_pct
            }
            
            if result.expected_value > 0:
                total_ev += result.expected_value
                total_allocation += result.position_pct
                
        # Second pass: normalize allocations if over max
        if total_allocation > max_allocation and total_allocation > 0:
            scale_factor = max_allocation / total_allocation
            
            for signal_id, allocation_info in allocations.items():
                result = allocation_info['result']
                original_pct = result.position_pct
                
                # Scale down the allocation
                new_pct = original_pct * scale_factor
                new_size = new_pct * portfolio_value
                
                # Update the allocation info
                allocations[signal_id]['allocation'] = new_pct
                allocations[signal_id]['scaled'] = True
                allocations[signal_id]['original_allocation'] = original_pct
                allocations[signal_id]['position_size'] = new_size
                allocations[signal_id]['reasoning'] = (
                    f"Scaled from {original_pct:.1%} to {new_pct:.1%} due to portfolio constraints"
                )
        else:
            # No scaling needed
            for signal_id, allocation_info in allocations.items():
                result = allocation_info['result']
                allocations[signal_id]['scaled'] = False
                allocations[signal_id]['position_size'] = result.position_size
                allocations[signal_id]['reasoning'] = result.reasoning
        
        return allocations
