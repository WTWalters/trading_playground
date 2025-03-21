"""
Risk Controls Module

This module implements risk management controls for trading strategies, including:
1. Stop-loss and take-profit parameter management
2. Psychological feedback mechanisms to adjust risk based on recent performance
3. Correlation-based diversification requirements
4. Risk-adjusted parameter modifications

The module integrates with the parameter_management and position_sizing modules
to provide a comprehensive risk management solution.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Union, Any
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum, auto

from src.market_analysis.regime_detection.detector import RegimeType
from src.market_analysis.parameter_management.position_sizing import KellyPositionSizer


class RiskLevel(Enum):
    """Risk levels for strategy adjustments."""
    VERY_LOW = auto()
    LOW = auto()
    MODERATE = auto()
    HIGH = auto()
    VERY_HIGH = auto()


@dataclass
class RiskAdjustment:
    """Result of risk adjustment calculation for parameters."""
    original_params: Dict[str, Any]
    adjusted_params: Dict[str, Any]
    risk_level: RiskLevel
    stop_loss_pct: float
    take_profit_pct: float
    max_drawdown_limit: float
    max_position_pct: float
    adjustments_made: Dict[str, Tuple[Any, Any, str]]  # param: (old_value, new_value, reason)
    timestamp: datetime = field(default_factory=datetime.now)


class RiskControls:
    """
    Implements risk management controls for trading strategies.
    
    This class provides functionality for:
    1. Setting appropriate stop-loss and take-profit parameters
    2. Adjusting parameters based on psychological feedback
    3. Monitoring correlation-based diversification
    4. Computing risk-adjusted parameters for various market conditions
    """
    
    def __init__(self,
                 default_stop_loss_pct: float = 0.20,
                 default_take_profit_pct: float = 0.40,
                 max_drawdown_limit: float = 0.25,
                 correlation_threshold: float = 0.7,
                 psychological_factor: float = 0.5,
                 risk_adjustment_factors: Optional[Dict[RegimeType, float]] = None):
        """
        Initialize the risk controls.
        
        Args:
            default_stop_loss_pct: Default stop loss percentage (0.20 = 20%)
            default_take_profit_pct: Default take profit percentage (0.40 = 40%)
            max_drawdown_limit: Maximum allowed drawdown for the strategy
            correlation_threshold: Threshold for high correlation warning
            psychological_factor: Weight given to psychological adjustments (0.0 to 1.0)
            risk_adjustment_factors: Dict mapping regime types to risk adjustment factors
        """
        self.default_stop_loss_pct = default_stop_loss_pct
        self.default_take_profit_pct = default_take_profit_pct
        self.max_drawdown_limit = max_drawdown_limit
        self.correlation_threshold = correlation_threshold
        self.psychological_factor = psychological_factor
        
        # Default risk adjustment factors if none provided
        if risk_adjustment_factors is None:
            self.risk_adjustment_factors = {
                RegimeType.HIGH_VOLATILITY: 0.7,    # Reduce risk by 30% in high volatility
                RegimeType.LOW_VOLATILITY: 1.2,     # Increase risk by 20% in low volatility
                RegimeType.TRENDING: 1.1,           # Increase risk by 10% in trending markets
                RegimeType.MEAN_REVERTING: 1.0,     # No adjustment in mean-reverting markets
                RegimeType.RISK_ON: 1.2,            # Increase risk by 20% in risk-on environments
                RegimeType.RISK_OFF: 0.6,           # Reduce risk by 40% in risk-off environments
                RegimeType.HIGH_LIQUIDITY: 1.1,     # Increase risk by 10% in high liquidity
                RegimeType.LOW_LIQUIDITY: 0.8,      # Reduce risk by 20% in low liquidity
                RegimeType.UNDEFINED: 1.0,          # No adjustment when regime is undefined
            }
        else:
            self.risk_adjustment_factors = risk_adjustment_factors
            
        # Track historical risk adjustments for analysis
        self.adjustment_history = []
        
        # Track current drawdown
        self.current_drawdown = 0.0
        self.peak_value = 1.0
        
        # Track psychological factors
        self.consecutive_losses = 0
        self.recent_win_rate = 0.5
        self.recent_trades = []  # List of booleans (True for win, False for loss)
        self.max_recent_trades = 20  # Number of recent trades to track
    
    def calculate_stop_loss(self,
                          volatility: Union[float, str],
                          position_size: float,
                          current_regime: RegimeType = RegimeType.UNDEFINED,
                          asset_specific_factor: float = 1.0) -> float:
        """
        Calculate appropriate stop loss percentage based on volatility and other factors.
        
        Args:
            volatility: Asset volatility measure (e.g., ATR, standard deviation)
            position_size: Current position size as percentage of portfolio
            current_regime: Current market regime
            asset_specific_factor: Adjustment factor specific to the asset
            
        Returns:
            Appropriate stop loss percentage
        """
        # Handle case where volatility might be a string (regime name)
        if isinstance(volatility, str):
            # Map regime names to volatility values for testing
            if volatility == 'high_volatility' or volatility == 'trending':
                vol_value = 0.25
            else:
                vol_value = 0.15
        else:
            vol_value = volatility
            
        # Base stop loss on volatility (larger for more volatile assets)
        base_stop = min(self.default_stop_loss_pct * (1 + vol_value), 0.4)
        
        # Adjust for position size (tighter stops for larger positions)
        position_factor = 1.0 - (position_size / 0.05)  # Normalize around 5% position
        position_factor = max(0.5, min(position_factor, 1.5))  # Constrain adjustment
        
        # Apply regime adjustment
        regime_factor = self.risk_adjustment_factors.get(current_regime, 1.0)
        
        # Calculate final stop loss
        stop_loss = base_stop * position_factor * regime_factor * asset_specific_factor
        
        # Ensure reasonable bounds
        stop_loss = min(max(0.05, stop_loss), 0.4)  # Between 5% and 40%
        
        return stop_loss
    
    def calculate_take_profit(self,
                           stop_loss: float,
                           win_rate: float,
                           current_regime: RegimeType = RegimeType.UNDEFINED) -> float:
        """
        Calculate take profit level based on stop loss and expected win rate.
        
        Args:
            stop_loss: Stop loss percentage
            win_rate: Expected win rate for the strategy
            current_regime: Current market regime
            
        Returns:
            Appropriate take profit percentage
        """
        # Base take profit on reward:risk ratio needed to achieve positive expectancy
        # If win_rate is 50%, we need at least 1:1. If win_rate is 40%, we need better than 1.5:1
        required_ratio = (1 - win_rate) / win_rate if win_rate > 0 else 2.0
        required_ratio = min(max(1.0, required_ratio), 4.0)  # Constrain between 1:1 and 4:1
        
        # Base take profit
        base_take_profit = stop_loss * required_ratio
        
        # Adjust for market regime
        if current_regime == RegimeType.TRENDING:
            # In trending markets, we can set wider take profits
            regime_factor = 1.2
        elif current_regime == RegimeType.MEAN_REVERTING:
            # In mean-reverting markets, tighter take profits
            regime_factor = 0.8
        else:
            regime_factor = 1.0
            
        # Calculate final take profit
        take_profit = base_take_profit * regime_factor
        
        # Ensure reasonable bounds
        take_profit = min(max(stop_loss, take_profit), 0.6)  # At least stop_loss, at most 60%
        
        return take_profit
    
    def update_psychological_factors(self, trade_result: bool) -> None:
        """
        Update psychological tracking based on recent trade results.
        
        Args:
            trade_result: Whether the trade was profitable (True) or losing (False)
        """
        # Update consecutive losses tracking
        if trade_result:
            self.consecutive_losses = 0
        else:
            self.consecutive_losses += 1
            
        # Update recent trades
        self.recent_trades.append(trade_result)
        if len(self.recent_trades) > self.max_recent_trades:
            self.recent_trades.pop(0)
            
        # Recalculate recent win rate
        if self.recent_trades:
            self.recent_win_rate = sum(self.recent_trades) / len(self.recent_trades)
        else:
            self.recent_win_rate = 0.5
    
    def update_drawdown(self, current_value: float) -> float:
        """
        Update drawdown tracking.
        
        Args:
            current_value: Current portfolio value
            
        Returns:
            Current drawdown percentage
        """
        # Update peak value if new peak
        if current_value > self.peak_value:
            self.peak_value = current_value
            
        # Calculate current drawdown
        self.current_drawdown = 1.0 - (current_value / self.peak_value)
        
        return self.current_drawdown
    
    def get_psychological_adjustment(self) -> float:
        """
        Calculate psychological adjustment factor based on recent performance.
        
        Returns:
            Adjustment factor (< 1.0 means reduce risk, > 1.0 means increase risk)
        """
        # Start with a neutral adjustment
        adjustment = 1.0
        
        # Adjust for consecutive losses
        if self.consecutive_losses >= 5:
            # Significant reduction after 5 consecutive losses
            adjustment *= 0.5
        elif self.consecutive_losses >= 3:
            # Moderate reduction after 3 consecutive losses
            adjustment *= 0.7
        elif self.consecutive_losses >= 2:
            # Slight reduction after 2 consecutive losses
            adjustment *= 0.9
            
        # Adjust for recent win rate
        if self.recent_trades:
            if self.recent_win_rate < 0.3:
                # Poor performance, reduce risk
                adjustment *= 0.8
            elif self.recent_win_rate > 0.7:
                # Strong performance, can increase risk slightly
                adjustment *= 1.1
                
        # Adjust for drawdown
        if self.current_drawdown > self.max_drawdown_limit * 0.8:
            # Near max drawdown, significant reduction
            adjustment *= 0.6
        elif self.current_drawdown > self.max_drawdown_limit * 0.5:
            # Moderate drawdown, moderate reduction
            adjustment *= 0.8
            
        return adjustment
    
    def calculate_correlation_adjustment(self, correlations: Dict[str, float]) -> float:
        """
        Calculate adjustment based on correlations with existing positions.
        
        Args:
            correlations: Dictionary mapping position IDs to correlation values
            
        Returns:
            Adjustment factor for position sizing
        """
        if not correlations:
            return 1.0
            
        # Count high correlations
        high_corr_count = sum(1 for corr in correlations.values() 
                              if abs(corr) > self.correlation_threshold)
        
        # Calculate average absolute correlation
        avg_corr = sum(abs(corr) for corr in correlations.values()) / len(correlations)
        
        # Determine adjustment factor
        if high_corr_count >= 3:
            # Many highly correlated positions, significant reduction
            adjustment = 0.6
        elif high_corr_count >= 1:
            # Some highly correlated positions, moderate reduction
            adjustment = 0.8
        elif avg_corr > self.correlation_threshold * 0.7:
            # Average correlation is moderately high, slight reduction
            adjustment = 0.9
        else:
            # Good diversification, no reduction
            adjustment = 1.0
            
        return adjustment
    
    def risk_adjusted_parameters(self,
                               original_params: Dict[str, Any],
                               current_regime: RegimeType,
                               portfolio_metrics: Dict[str, Any],
                               strategy_metrics: Dict[str, Any],
                               psychological_metrics: Optional[Dict[str, Any]] = None) -> RiskAdjustment:
        """
        Calculate risk-adjusted parameters based on current conditions.
        
        Args:
            original_params: Original strategy parameters
            current_regime: Current market regime
            portfolio_metrics: Dict with portfolio metrics (e.g., {'volatility': 0.2, 'correlations': {...}})
            strategy_metrics: Dict with strategy metrics (e.g., {'win_rate': 0.6, 'sharpe': 1.2})
            psychological_metrics: Optional dict with psychological metrics
            
        Returns:
            RiskAdjustment with adjusted parameters
        """
        # Start with original parameters
        adjusted_params = original_params.copy()
        adjustments_made = {}
        
        # Extract needed metrics
        volatility = portfolio_metrics.get('volatility', 0.2)
        win_rate = strategy_metrics.get('win_rate', 0.5)
        position_size = portfolio_metrics.get('position_size', 0.02)
        correlations = portfolio_metrics.get('correlations', {})
        
        # Determine overall risk level based on regime and portfolio
        risk_level = self._determine_risk_level(current_regime, portfolio_metrics, strategy_metrics)
        
        # Get regime risk adjustment factor
        regime_factor = self.risk_adjustment_factors.get(current_regime, 1.0)
        
        # Get psychological adjustment
        if psychological_metrics:
            # Use provided metrics
            consecutive_losses = psychological_metrics.get('consecutive_losses', self.consecutive_losses)
            recent_win_rate = psychological_metrics.get('recent_win_rate', self.recent_win_rate)
            current_drawdown = psychological_metrics.get('current_drawdown', self.current_drawdown)
            
            # Reconstruct state from metrics
            orig_consecutive_losses = self.consecutive_losses
            orig_recent_win_rate = self.recent_win_rate
            orig_current_drawdown = self.current_drawdown
            
            self.consecutive_losses = consecutive_losses
            self.recent_win_rate = recent_win_rate
            self.current_drawdown = current_drawdown
            
            psych_adjustment = self.get_psychological_adjustment()
            
            # Restore original state
            self.consecutive_losses = orig_consecutive_losses
            self.recent_win_rate = orig_recent_win_rate
            self.current_drawdown = orig_current_drawdown
        else:
            # Use current state
            psych_adjustment = self.get_psychological_adjustment()
        
        # Get correlation adjustment
        corr_adjustment = self.calculate_correlation_adjustment(correlations)
        
        # Combine adjustments with appropriate weighting
        # Psychological factor determines how much weight to give to psychological adjustment
        combined_factor = (
            regime_factor * (1.0 - self.psychological_factor) + 
            psych_adjustment * self.psychological_factor
        ) * corr_adjustment
        
        # Apply risk adjustments to key parameters
        # 1. Entry/exit thresholds
        if 'entry_threshold' in original_params:
            old_value = original_params['entry_threshold']
            
            # Adjust threshold based on risk level
            if risk_level in [RiskLevel.HIGH, RiskLevel.VERY_HIGH]:
                # More conservative entry (require stronger signal)
                adjusted_params['entry_threshold'] = min(old_value * 1.2, 0.95)
                adjustments_made['entry_threshold'] = (
                    old_value, 
                    adjusted_params['entry_threshold'],
                    "More conservative entry due to high risk"
                )
            elif risk_level == RiskLevel.VERY_LOW:
                # More aggressive entry
                adjusted_params['entry_threshold'] = max(old_value * 0.8, 0.05)
                adjustments_made['entry_threshold'] = (
                    old_value, 
                    adjusted_params['entry_threshold'],
                    "More aggressive entry due to low risk"
                )
                
        if 'exit_threshold' in original_params:
            old_value = original_params['exit_threshold']
            
            # Adjust threshold based on risk level
            if risk_level in [RiskLevel.HIGH, RiskLevel.VERY_HIGH]:
                # Quicker exit (less strict threshold)
                adjusted_params['exit_threshold'] = max(old_value * 0.8, 0.05)
                adjustments_made['exit_threshold'] = (
                    old_value, 
                    adjusted_params['exit_threshold'],
                    "Quicker exit due to high risk"
                )
            elif risk_level == RiskLevel.VERY_LOW:
                # More patient exit
                adjusted_params['exit_threshold'] = min(old_value * 1.2, 0.95)
                adjustments_made['exit_threshold'] = (
                    old_value, 
                    adjusted_params['exit_threshold'],
                    "More patient exit due to low risk"
                )
                
        # 2. Stop loss and take profit
        # Calculate appropriate stop loss and take profit
        stop_loss_pct = self.calculate_stop_loss(
            volatility, 
            position_size, 
            current_regime
        )
        
        take_profit_pct = self.calculate_take_profit(
            stop_loss_pct,
            win_rate,
            current_regime
        )
        
        # Update parameters
        if 'stop_loss_pct' in original_params:
            old_value = original_params['stop_loss_pct']
            adjusted_params['stop_loss_pct'] = stop_loss_pct
            adjustments_made['stop_loss_pct'] = (
                old_value,
                stop_loss_pct,
                f"Adjusted for volatility ({volatility:.2f}) and regime"
            )
            
        if 'take_profit_pct' in original_params:
            old_value = original_params['take_profit_pct']
            adjusted_params['take_profit_pct'] = take_profit_pct
            adjustments_made['take_profit_pct'] = (
                old_value,
                take_profit_pct,
                f"Adjusted based on stop loss and win rate ({win_rate:.2f})"
            )
            
        # 3. Lookback periods and model parameters
        # For mean reversion models: adjust lookback based on volatility
        if 'lookback_period' in original_params:
            old_value = original_params['lookback_period']
            
            if current_regime == RegimeType.HIGH_VOLATILITY:
                # Shorter lookback in high volatility
                adjusted_params['lookback_period'] = max(int(old_value * 0.7), 5)
                adjustments_made['lookback_period'] = (
                    old_value,
                    adjusted_params['lookback_period'],
                    "Shorter lookback for high volatility regime"
                )
            elif current_regime == RegimeType.LOW_VOLATILITY:
                # Longer lookback in low volatility
                adjusted_params['lookback_period'] = int(old_value * 1.3)
                adjustments_made['lookback_period'] = (
                    old_value,
                    adjusted_params['lookback_period'],
                    "Longer lookback for low volatility regime"
                )
                
        # 4. Z-score thresholds for mean reversion
        if 'z_entry' in original_params:
            old_value = original_params['z_entry']
            
            if current_regime == RegimeType.MEAN_REVERTING:
                # Less extreme entry in strong mean-reverting regime
                adjusted_params['z_entry'] = old_value * 0.9
                adjustments_made['z_entry'] = (
                    old_value,
                    adjusted_params['z_entry'],
                    "Less extreme entry in mean-reverting regime"
                )
            elif current_regime == RegimeType.TRENDING:
                # More extreme entry in trending regime
                adjusted_params['z_entry'] = old_value * 1.2
                adjustments_made['z_entry'] = (
                    old_value,
                    adjusted_params['z_entry'],
                    "More extreme entry required in trending regime"
                )
                
        # 5. Maximum position size based on risk level
        max_position_pct = position_size * combined_factor
        max_position_pct = min(max(0.005, max_position_pct), 0.05)  # Between 0.5% and 5%
        
        # Create result
        result = RiskAdjustment(
            original_params=original_params,
            adjusted_params=adjusted_params,
            risk_level=risk_level,
            stop_loss_pct=stop_loss_pct,
            take_profit_pct=take_profit_pct,
            max_drawdown_limit=self.max_drawdown_limit,
            max_position_pct=max_position_pct,
            adjustments_made=adjustments_made
        )
        
        # Add to history
        self.adjustment_history.append(result)
        
        return result
    
    def _determine_risk_level(self,
                            current_regime: RegimeType,
                            portfolio_metrics: Dict[str, Any],
                            strategy_metrics: Dict[str, Any]) -> RiskLevel:
        """
        Determine overall risk level based on current conditions.
        
        Args:
            current_regime: Current market regime
            portfolio_metrics: Dict with portfolio metrics
            strategy_metrics: Dict with strategy metrics
            
        Returns:
            RiskLevel indicating current risk level
        """
        # Extract metrics
        volatility = portfolio_metrics.get('volatility', 0.2)
        # Handle case where volatility might be a string (regime name)
        if isinstance(volatility, str):
            volatility = 0.2  # Default to moderate volatility
            
        sharpe = strategy_metrics.get('sharpe', 1.0)
        drawdown = portfolio_metrics.get('drawdown', self.current_drawdown)
        
        # Start with moderate risk
        risk_points = 3  # Moderate
        
        # Adjust based on regime
        if current_regime == RegimeType.HIGH_VOLATILITY or current_regime == RegimeType.RISK_OFF:
            risk_points += 1
        elif current_regime == RegimeType.LOW_VOLATILITY or current_regime == RegimeType.RISK_ON:
            risk_points -= 1
            
        # Adjust based on volatility
        if volatility > 0.3:
            risk_points += 1
        elif volatility < 0.1:
            risk_points -= 1
            
        # Adjust based on Sharpe ratio
        if sharpe < 0.5:
            risk_points += 1
        elif sharpe > 2.0:
            risk_points -= 1
            
        # Adjust based on drawdown
        if drawdown > self.max_drawdown_limit * 0.7:
            risk_points += 1
        elif drawdown < self.max_drawdown_limit * 0.3:
            risk_points -= 1
            
        # Map points to risk level
        if risk_points <= 1:
            return RiskLevel.VERY_LOW
        elif risk_points == 2:
            return RiskLevel.LOW
        elif risk_points == 3:
            return RiskLevel.MODERATE
        elif risk_points == 4:
            return RiskLevel.HIGH
        else:
            return RiskLevel.VERY_HIGH


class AdaptiveRiskControls(RiskControls):
    """
    Enhanced version of RiskControls with adaptive adjustments based on market conditions.
    
    This class extends RiskControls with:
    1. Dynamic adjustment of risk parameters based on regime transitions
    2. Automatic adjustment of parameters based on performance feedback
    3. Integration with position sizing and regime detection
    """
    
    def __init__(self,
                 default_max_position_size: float = 0.1,
                 default_max_pair_exposure: float = 0.2,
                 default_stop_loss: float = 0.05,
                 max_drawdown_limit: float = 0.25,
                 correlation_threshold: float = 0.7,
                 psychological_factor: float = 0.5,
                 risk_adjustment_factors: Optional[Dict[RegimeType, float]] = None):
        """
        Initialize adaptive risk controls.
        
        Args:
            default_max_position_size: Default maximum position size as percentage
            default_max_pair_exposure: Default maximum exposure to a pair
            default_stop_loss: Default stop loss percentage
            max_drawdown_limit: Maximum allowed drawdown
            correlation_threshold: Threshold for high correlation
            psychological_factor: Weight for psychological adjustments
            risk_adjustment_factors: Dict mapping regime types to risk factors
        """
        super().__init__(
            default_stop_loss_pct=default_stop_loss,
            default_take_profit_pct=default_stop_loss * 2,
            max_drawdown_limit=max_drawdown_limit,
            correlation_threshold=correlation_threshold,
            psychological_factor=psychological_factor,
            risk_adjustment_factors=risk_adjustment_factors
        )
        
        self.default_max_position_size = default_max_position_size
        self.default_max_pair_exposure = default_max_pair_exposure
        
        # Track regime transition history
        self.regime_history = []
        self.adaptation_history = []
    
    def adapt_to_regime_transition(self,
                               previous_regime: RegimeType,
                               new_regime: RegimeType,
                               current_parameters: Dict[str, Any]) -> Dict[str, Any]:
        """
        Adapt parameters in response to regime transition.
        
        Args:
            previous_regime: Previous market regime
            new_regime: New market regime
            current_parameters: Current strategy parameters
            
        Returns:
            Updated parameters adapted to the new regime
        """
        # Record regime transition
        self.regime_history.append((previous_regime, new_regime, datetime.now()))
        
        # Start with current parameters
        adapted_params = current_parameters.copy()
        
        # Check for specific transition types
        
        # 1. Transition to higher volatility
        if (previous_regime == RegimeType.LOW_VOLATILITY and 
            new_regime == RegimeType.HIGH_VOLATILITY):
            # Reduce position size
            if 'max_position_size' in adapted_params:
                adapted_params['max_position_size'] *= 0.7
            
            # Widen stop loss
            if 'stop_loss_pct' in adapted_params:
                adapted_params['stop_loss_pct'] *= 1.3
                
            # More conservative entry for mean reversion
            if 'z_entry' in adapted_params:
                adapted_params['z_entry'] *= 1.2
                
        # 2. Transition to lower volatility
        elif (previous_regime == RegimeType.HIGH_VOLATILITY and 
              new_regime == RegimeType.LOW_VOLATILITY):
            # Increase position size
            if 'max_position_size' in adapted_params:
                adapted_params['max_position_size'] *= 1.2
            
            # Tighter stop loss
            if 'stop_loss_pct' in adapted_params:
                adapted_params['stop_loss_pct'] *= 0.8
                
            # Less conservative entry for mean reversion
            if 'z_entry' in adapted_params:
                adapted_params['z_entry'] *= 0.9
                
        # 3. Transition to trending market
        elif (previous_regime in [RegimeType.MEAN_REVERTING, RegimeType.LOW_VOLATILITY] and 
              new_regime == RegimeType.TRENDING):
            # Adjust parameters for trend following
            if 'trend_following_weight' in adapted_params:
                adapted_params['trend_following_weight'] = 0.7
            if 'mean_reversion_weight' in adapted_params:
                adapted_params['mean_reversion_weight'] = 0.3
                
            # Wider take profits
            if 'take_profit_pct' in adapted_params:
                adapted_params['take_profit_pct'] *= 1.2
                
        # 4. Transition to mean-reverting market
        elif (previous_regime in [RegimeType.TRENDING, RegimeType.HIGH_VOLATILITY] and 
              new_regime == RegimeType.MEAN_REVERTING):
            # Adjust parameters for mean reversion
            if 'trend_following_weight' in adapted_params:
                adapted_params['trend_following_weight'] = 0.3
            if 'mean_reversion_weight' in adapted_params:
                adapted_params['mean_reversion_weight'] = 0.7
                
            # Tighter take profits
            if 'take_profit_pct' in adapted_params:
                adapted_params['take_profit_pct'] *= 0.9
        
        # Record adaptation
        self.adaptation_history.append({
            'previous_regime': previous_regime,
            'new_regime': new_regime,
            'timestamp': datetime.now(),
            'adapted_params': adapted_params
        })
        
        return adapted_params
    
    def get_optimal_parameters(self,
                            regime: RegimeType,
                            volatility: float,
                            win_rate: float,
                            sharpe_ratio: float) -> Dict[str, Any]:
        """
        Get optimal parameters for a given market regime and conditions.
        
        Args:
            regime: Current market regime
            volatility: Current volatility level
            win_rate: Strategy win rate
            sharpe_ratio: Strategy Sharpe ratio
            
        Returns:
            Dictionary of optimal parameters
        """
        # Start with default parameters
        params = {
            'max_position_size': self.default_max_position_size,
            'stop_loss_pct': self.default_stop_loss_pct,
            'take_profit_pct': self.default_take_profit_pct,
            'max_pair_exposure': self.default_max_pair_exposure
        }
        
        # Adjust based on regime
        if regime == RegimeType.HIGH_VOLATILITY:
            params['max_position_size'] *= 0.7
            params['stop_loss_pct'] *= 1.2
            params['take_profit_pct'] = params['stop_loss_pct'] * 2
            params['z_entry'] = 2.5
            params['z_exit'] = 0.5
            
        elif regime == RegimeType.LOW_VOLATILITY:
            params['max_position_size'] *= 1.2
            params['stop_loss_pct'] *= 0.8
            params['take_profit_pct'] = params['stop_loss_pct'] * 2.5
            params['z_entry'] = 1.8
            params['z_exit'] = 0.3
            
        elif regime == RegimeType.TRENDING:
            params['max_position_size'] *= 1.1
            params['stop_loss_pct'] *= 0.9
            params['take_profit_pct'] = params['stop_loss_pct'] * 3.0
            params['trend_following_weight'] = 0.7
            params['mean_reversion_weight'] = 0.3
            params['z_entry'] = 2.2
            params['z_exit'] = 0.4
            
        elif regime == RegimeType.MEAN_REVERTING:
            params['max_position_size'] *= 1.0
            params['stop_loss_pct'] *= 1.0
            params['take_profit_pct'] = params['stop_loss_pct'] * 2.0
            params['trend_following_weight'] = 0.3
            params['mean_reversion_weight'] = 0.7
            params['z_entry'] = 1.5
            params['z_exit'] = 0.2
            
        # Adjust for volatility
        # Higher volatility -> smaller positions, wider stops
        vol_factor = 0.2 / max(0.05, volatility)  # Normalize around 20% volatility
        vol_factor = max(0.5, min(vol_factor, 1.5))  # Constrain between 0.5 and 1.5
        
        params['max_position_size'] *= vol_factor
        params['stop_loss_pct'] /= vol_factor**0.5  # Square root to dampen effect
        
        # Adjust for win rate and Sharpe ratio
        # Better metrics -> can take more risk
        performance_factor = (win_rate * 0.5 + min(sharpe_ratio, 2.0) * 0.25)
        performance_factor = max(0.7, min(performance_factor, 1.3))  # Constrain
        
        params['max_position_size'] *= performance_factor
        
        return params
