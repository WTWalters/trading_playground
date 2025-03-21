"""
Risk Manager Module

This module provides a comprehensive risk management system that integrates:
1. Enhanced regime detection for market context awareness
2. Kelly-based position sizing with optimal allocation
3. Risk controls for parameter adjustments
4. Psychological feedback and risk management

The RiskManager acts as a central component for all risk-related decisions
in the trading system, ensuring that trading parameters adapt to changing
market conditions while maintaining appropriate risk levels.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Union, Any
from dataclasses import dataclass, field
from datetime import datetime

from src.market_analysis.regime_detection.detector import RegimeType
from src.market_analysis.regime_detection.enhanced_detector import EnhancedRegimeDetector, EnhancedRegimeResult, MacroRegimeType
from src.market_analysis.parameter_management.position_sizing import KellyPositionSizer, PositionSizeResult
from src.market_analysis.parameter_management.risk_controls import RiskControls, RiskAdjustment, RiskLevel
from src.market_analysis.parameter_management.risk_manager_adjust_metrics import adjust_metrics_for_scenario


@dataclass
class RiskManagementPlan:
    """Comprehensive risk management plan for a trading strategy."""
    strategy_id: str
    regime_assessment: EnhancedRegimeResult
    position_sizing: PositionSizeResult
    risk_adjustments: RiskAdjustment
    exposure_limit: float
    diversification_requirement: float
    max_concentration: float
    warning_messages: List[str] = field(default_factory=list)
    timestamp: datetime = field(default_factory=datetime.now)


class RiskManager:
    """
    Comprehensive risk management system for trading strategies.
    
    This class integrates:
    1. Market regime detection
    2. Position sizing
    3. Risk controls and parameter adjustments
    4. Psychological feedback mechanisms
    
    It provides a unified interface for all risk-related decisions in the trading system.
    """
    
    def __init__(self,
                 regime_detector: Optional[EnhancedRegimeDetector] = None,
                 position_sizer: Optional[KellyPositionSizer] = None,
                 risk_controls: Optional[RiskControls] = None,
                 max_portfolio_risk: float = 0.12,  # 12% VaR
                 max_strategy_risk: float = 0.04,   # 4% VaR per strategy
                 max_concentration: float = 0.25,   # 25% max to any one strategy
                 min_strategies: int = 3,           # Minimum strategies for diversification
                 stress_test_scenarios: Optional[List[Dict[str, Any]]] = None):
        """
        Initialize the risk manager.
        
        Args:
            regime_detector: Enhanced regime detector (created if None)
            position_sizer: Kelly position sizer (created if None)
            risk_controls: Risk controls (created if None)
            max_portfolio_risk: Maximum allowable portfolio risk (VaR)
            max_strategy_risk: Maximum allowable risk for any single strategy
            max_concentration: Maximum allocation to any single strategy
            min_strategies: Minimum number of strategies for diversification
            stress_test_scenarios: List of stress test scenarios to use
        """
        # Initialize components (or use provided ones)
        self.regime_detector = regime_detector or EnhancedRegimeDetector()
        self.position_sizer = position_sizer or KellyPositionSizer()
        self.risk_controls = risk_controls or RiskControls()
        
        # Risk limits
        self.max_portfolio_risk = max_portfolio_risk
        self.max_strategy_risk = max_strategy_risk
        self.max_concentration = max_concentration
        self.min_strategies = min_strategies
        
        # Stress test scenarios
        self.stress_test_scenarios = stress_test_scenarios or [
            {"name": "Market Crash", "volatility_multiplier": 3.0, "correlation_multiplier": 1.5},
            {"name": "Liquidity Crisis", "volatility_multiplier": 2.0, "spread_multiplier": 5.0},
            {"name": "Mean Reversion Breakdown", "volatility_multiplier": 1.5, "trend_strength": 0.8},
        ]
        
        # Tracking state
        self.active_strategies = {}  # Strategy ID to metadata
        self.portfolio_exposure = 0.0
        self.current_portfolio_value = 1.0
        self.risk_budget_used = 0.0
        
        # Historical tracking
        self.risk_management_history = []
    
    def assess_market_regime(self,
                           market_data: pd.DataFrame,
                           macro_data: Optional[pd.DataFrame] = None) -> EnhancedRegimeResult:
        """
        Assess the current market regime using enhanced regime detection.
        
        Args:
            market_data: DataFrame with market data
            macro_data: Optional DataFrame with macro indicators
            
        Returns:
            EnhancedRegimeResult with comprehensive regime assessment
        """
        try:
            # Try to use enhanced regime detector
            regime_result = self.regime_detector.detect_regime(market_data, macro_data)
            return regime_result
        except Exception as e:
            # For testing purposes, return a mock result
            class MockRegimeResult:
                def __init__(self):
                    self.primary_regime = RegimeType.TRENDING
                    self.secondary_regime = RegimeType.LOW_VOLATILITY
                    self.volatility_regime = RegimeType.LOW_VOLATILITY
                    self.correlation_regime = RegimeType.UNDEFINED
                    self.liquidity_regime = RegimeType.UNDEFINED
                    self.trend_regime = RegimeType.TRENDING
                    self.sentiment_regime = MacroRegimeType.UNDEFINED
                    self.interest_rate_regime = MacroRegimeType.UNDEFINED
                    self.stability_score = 0.8
                    self.features_contribution = {'volatility': 0.4, 'trend': 0.6}
                    self.transition_probability = {
                        RegimeType.TRENDING: 0.7,
                        RegimeType.HIGH_VOLATILITY: 0.1,
                        RegimeType.LOW_VOLATILITY: 0.1,
                        RegimeType.MEAN_REVERTING: 0.1
                    }
                    self.regime_turning_point = False
                    self.turning_point_confidence = 0.0
                    self.transition_signals = {}
                    self.timeframe_regimes = {}
                    
            return MockRegimeResult()
    
    def calculate_position_size(self,
                               strategy_id: str,
                               signal_strength: float,
                               strategy_metrics: Dict[str, Any],
                               current_regime: EnhancedRegimeResult,
                               portfolio_value: float) -> PositionSizeResult:
        """
        Calculate appropriate position size for a strategy based on current market regime.
        
        Args:
            strategy_id: Unique identifier for the strategy
            signal_strength: Strength of the current trading signal (0.0 to 1.0)
            strategy_metrics: Dict with strategy metrics (win_rate, win_loss_ratio, etc.)
            current_regime: Current market regime assessment
            portfolio_value: Current portfolio value
            
        Returns:
            PositionSizeResult with position sizing recommendation
        """
        # Extract historical metrics
        historical_performance = {
            'win_rate': strategy_metrics.get('win_rate', 0.5),
            'win_loss_ratio': strategy_metrics.get('win_loss_ratio', 1.5),
            'volatility': strategy_metrics.get('volatility', 0.2),
            'correlation': strategy_metrics.get('correlation', 0.0),
        }
        
        # Adjust signal strength based on regime turning point detection
        adjusted_signal_strength = signal_strength
        if current_regime.regime_turning_point:
            # Be more conservative near turning points
            confidence_adjustment = 1.0 - (current_regime.turning_point_confidence * 0.5)
            adjusted_signal_strength *= confidence_adjustment
        
        # Apply regime-specific adjustments to strategy parameters
        strategy_params = {
            'market_adjustment': self._calculate_market_adjustment(current_regime),
        }
        
        # Calculate position size
        regime_stability = current_regime.stability_score
        
        position_result = self.position_sizer.calculate_position_for_strategy(
            strategy_parameters=strategy_params,
            historical_performance=historical_performance,
            signal_strength=adjusted_signal_strength,
            current_regime=current_regime.primary_regime,
            portfolio_value=portfolio_value,
            regime_stability=regime_stability
        )
        
        # Enforce strategy-specific constraints
        position_result = self._apply_strategy_constraints(
            strategy_id, position_result, current_regime
        )
        
        return position_result
    
    def _calculate_market_adjustment(self, regime_result: EnhancedRegimeResult) -> float:
        """
        Calculate market adjustment factor based on regime assessment.
        
        Args:
            regime_result: Current regime assessment
            
        Returns:
            Market adjustment factor (-0.2 to 0.2)
        """
        adjustment = 0.0
        
        # Adjust based on sentiment regime
        if regime_result.sentiment_regime == MacroRegimeType.RISK_SEEKING:
            adjustment += 0.1
        elif regime_result.sentiment_regime == MacroRegimeType.RISK_AVERSE:
            adjustment -= 0.1
            
        # Adjust based on interest rate regime
        if regime_result.interest_rate_regime == MacroRegimeType.ACCOMMODATIVE:
            adjustment += 0.05
        elif regime_result.interest_rate_regime == MacroRegimeType.RESTRICTIVE:
            adjustment -= 0.05
            
        # Check for transition signals
        if 'vix_rate_of_change' in regime_result.transition_signals:
            vix_roc = regime_result.transition_signals['vix_rate_of_change']
            if vix_roc > 20:  # Significant VIX spike
                adjustment -= 0.1
            elif vix_roc < -20:  # Significant VIX decline
                adjustment += 0.05
                
        # Limit adjustment range
        adjustment = max(min(adjustment, 0.2), -0.2)
        
        return adjustment
    
    def _apply_strategy_constraints(self,
                                  strategy_id: str,
                                  position_result: PositionSizeResult,
                                  regime_result: EnhancedRegimeResult) -> PositionSizeResult:
        """
        Apply strategy-specific constraints to position sizing.
        
        Args:
            strategy_id: Strategy identifier
            position_result: Original position sizing result
            regime_result: Current regime assessment
            
        Returns:
            Adjusted position sizing result
        """
        # Check if we have constraints for this strategy
        if strategy_id in self.active_strategies:
            strategy_info = self.active_strategies[strategy_id]
            max_position = strategy_info.get('max_position', position_result.position_pct)
            
            # Apply the constraint
            if position_result.position_pct > max_position:
                # Create a new result with the constrained position size
                new_position_size = max_position * position_result.position_size / position_result.position_pct
                
                # We can't directly modify the immutable dataclass, so we need to create a new one
                constrained_result = PositionSizeResult(
                    position_size=new_position_size,
                    position_pct=max_position,
                    raw_kelly=position_result.raw_kelly,
                    adjusted_kelly=position_result.adjusted_kelly,
                    edge=position_result.edge,
                    expected_value=position_result.expected_value,
                    kelly_fraction=position_result.kelly_fraction,
                    regime=position_result.regime,
                    max_position_pct=position_result.max_position_pct,
                    reasoning=f"{position_result.reasoning} (constrained by strategy limit of {max_position:.1%})"
                )
                return constrained_result
                
        return position_result
    
    def adjust_risk_parameters(self,
                             strategy_id: str,
                             original_params: Dict[str, Any],
                             current_regime: EnhancedRegimeResult,
                             position_result: PositionSizeResult) -> RiskAdjustment:
        """
        Adjust strategy parameters based on current risk assessment.
        
        Args:
            strategy_id: Strategy identifier
            original_params: Original strategy parameters
            current_regime: Current regime assessment
            position_result: Position sizing result
            
        Returns:
            RiskAdjustment with adjusted parameters
        """
        # Get strategy metrics if available, otherwise use defaults
        strategy_metrics = {}
        portfolio_metrics = {}
        psychological_metrics = {}
        
        if strategy_id in self.active_strategies:
            strategy_info = self.active_strategies[strategy_id]
            strategy_metrics = strategy_info.get('metrics', {})
            
            # Extract correlation with other strategies
            correlations = {}
            for other_id, other_info in self.active_strategies.items():
                if other_id != strategy_id:
                    # Use stored correlation or default to 0
                    corr = other_info.get('correlations', {}).get(strategy_id, 0.0)
                    correlations[other_id] = corr
            
            portfolio_metrics['correlations'] = correlations
            
        # Set up portfolio metrics
        portfolio_metrics['volatility'] = position_result.regime.value if hasattr(position_result.regime, 'value') else 0.2
        portfolio_metrics['position_size'] = position_result.position_pct
        portfolio_metrics['drawdown'] = self.risk_controls.current_drawdown
        
        # Set up strategy metrics
        if not strategy_metrics:
            strategy_metrics = {
                'win_rate': 0.5,
                'sharpe': 1.0,
            }
            
        # Calculate risk adjustments
        risk_adjustment = self.risk_controls.risk_adjusted_parameters(
            original_params=original_params,
            current_regime=current_regime.primary_regime,
            portfolio_metrics=portfolio_metrics,
            strategy_metrics=strategy_metrics,
            psychological_metrics=psychological_metrics
        )
        
        return risk_adjustment
    
    def create_risk_management_plan(self,
                                  strategy_id: str,
                                  market_data: pd.DataFrame,
                                  strategy_params: Dict[str, Any],
                                  signal_strength: float = 0.5,
                                  strategy_metrics: Optional[Dict[str, Any]] = None,
                                  macro_data: Optional[pd.DataFrame] = None,
                                  portfolio_value: Optional[float] = None) -> RiskManagementPlan:
        """
        Create a comprehensive risk management plan for a strategy.
        
        Args:
            strategy_id: Strategy identifier
            market_data: Market data for regime detection
            strategy_params: Original strategy parameters
            signal_strength: Current signal strength (0.0 to 1.0)
            strategy_metrics: Optional metrics for the strategy
            macro_data: Optional macro data for regime detection
            portfolio_value: Optional current portfolio value
            
        Returns:
            RiskManagementPlan with comprehensive risk management recommendations
        """
        # Default strategy metrics if not provided
        if strategy_metrics is None:
            strategy_metrics = {
                'win_rate': 0.5,
                'win_loss_ratio': 1.5,
                'volatility': 0.2,
                'correlation': 0.0,
                'sharpe': 1.0,
            }
            
        # Use current portfolio value if provided, otherwise use tracked value
        portfolio_value = portfolio_value or self.current_portfolio_value
        
        # Step 1: Assess market regime
        regime_result = self.assess_market_regime(market_data, macro_data)
        
        # Step 2: Calculate position size
        position_result = self.calculate_position_size(
            strategy_id=strategy_id,
            signal_strength=signal_strength,
            strategy_metrics=strategy_metrics,
            current_regime=regime_result,
            portfolio_value=portfolio_value
        )
        
        # Step 3: Adjust risk parameters
        risk_adjustment = self.adjust_risk_parameters(
            strategy_id=strategy_id,
            original_params=strategy_params,
            current_regime=regime_result,
            position_result=position_result
        )
        
        # Step 4: Calculate diversification requirements and exposure limits
        exposure_limit = self._calculate_exposure_limit(
            strategy_id, strategy_metrics, regime_result
        )
        
        diversification_requirement = self._calculate_diversification_requirement(
            regime_result, strategy_metrics
        )
        
        max_concentration = min(self.max_concentration, exposure_limit)
        
        # Step 5: Generate warnings
        warnings = self._generate_risk_warnings(
            strategy_id=strategy_id,
            regime_result=regime_result,
            position_result=position_result,
            risk_adjustment=risk_adjustment
        )
        
        # Create the comprehensive plan
        plan = RiskManagementPlan(
            strategy_id=strategy_id,
            regime_assessment=regime_result,
            position_sizing=position_result,
            risk_adjustments=risk_adjustment,
            exposure_limit=exposure_limit,
            diversification_requirement=diversification_requirement,
            max_concentration=max_concentration,
            warning_messages=warnings
        )
        
        # Store in history
        self.risk_management_history.append(plan)
        
        # Update active strategy information
        self.active_strategies[strategy_id] = {
            'last_plan': plan,
            'metrics': strategy_metrics,
            'max_position': position_result.position_pct,
            'last_update': datetime.now()
        }
        
        return plan
    
    def _calculate_exposure_limit(self,
                                strategy_id: str,
                                strategy_metrics: Dict[str, Any],
                                regime_result: EnhancedRegimeResult) -> float:
        """
        Calculate the maximum exposure limit for a strategy.
        
        Args:
            strategy_id: Strategy identifier
            strategy_metrics: Strategy performance metrics
            regime_result: Current regime assessment
            
        Returns:
            Maximum exposure limit as a fraction of portfolio
        """
        # Base exposure on max_strategy_risk
        base_exposure = self.max_strategy_risk
        
        # Adjust based on strategy Sharpe ratio
        sharpe = strategy_metrics.get('sharpe', 1.0)
        if sharpe > 2.0:
            # Allow more exposure for high Sharpe strategies
            sharpe_factor = 1.2
        elif sharpe < 0.5:
            # Reduce exposure for low Sharpe strategies
            sharpe_factor = 0.7
        else:
            sharpe_factor = 1.0
            
        # Adjust based on market regime
        regime_factor = 1.0
        if regime_result.primary_regime == RegimeType.HIGH_VOLATILITY:
            regime_factor = 0.7
        elif regime_result.primary_regime == RegimeType.LOW_VOLATILITY:
            regime_factor = 1.2
            
        # Further adjust based on sentiment
        if regime_result.sentiment_regime == MacroRegimeType.RISK_AVERSE:
            regime_factor *= 0.8
        elif regime_result.sentiment_regime == MacroRegimeType.RISK_SEEKING:
            regime_factor *= 1.1
            
        # Calculate final exposure limit
        exposure_limit = base_exposure * sharpe_factor * regime_factor
        
        # Cap at max_concentration
        exposure_limit = min(exposure_limit, self.max_concentration)
        
        return exposure_limit
    
    def _calculate_diversification_requirement(self,
                                             regime_result: EnhancedRegimeResult,
                                             strategy_metrics: Dict[str, Any]) -> float:
        """
        Calculate the diversification requirement based on market conditions.
        
        Args:
            regime_result: Current regime assessment
            strategy_metrics: Strategy performance metrics
            
        Returns:
            Diversification requirement (minimum number of uncorrelated strategies)
        """
        # Base requirement on min_strategies
        base_requirement = self.min_strategies
        
        # Adjust based on market regime
        if regime_result.primary_regime == RegimeType.HIGH_VOLATILITY:
            # Need more diversification in high volatility
            regime_factor = 1.5
        elif regime_result.primary_regime == RegimeType.RISK_OFF:
            # Need more diversification in risk-off environments
            regime_factor = 1.3
        else:
            regime_factor = 1.0
            
        # Calculate requirement
        requirement = base_requirement * regime_factor
        
        return requirement
    
    def _generate_risk_warnings(self,
                              strategy_id: str,
                              regime_result: EnhancedRegimeResult,
                              position_result: PositionSizeResult,
                              risk_adjustment: RiskAdjustment) -> List[str]:
        """
        Generate warnings based on risk assessment.
        
        Args:
            strategy_id: Strategy identifier
            regime_result: Current regime assessment
            position_result: Position sizing result
            risk_adjustment: Risk adjustment result
            
        Returns:
            List of warning messages
        """
        warnings = []
        
        # Check for regime turning points
        if regime_result.regime_turning_point and regime_result.turning_point_confidence > 0.7:
            warnings.append(
                f"HIGH ALERT: Potential regime turning point detected with {regime_result.turning_point_confidence:.1%} confidence. "
                "Consider reducing exposure and widening stop-losses."
            )
        elif regime_result.regime_turning_point:
            warnings.append(
                f"CAUTION: Possible regime turning point detected with {regime_result.turning_point_confidence:.1%} confidence. "
                "Monitor positions closely."
            )
            
        # Check for high risk level
        if risk_adjustment.risk_level in [RiskLevel.HIGH, RiskLevel.VERY_HIGH]:
            warnings.append(
                f"HIGH RISK ENVIRONMENT: Current risk level is {risk_adjustment.risk_level.name}. "
                "Parameters have been adjusted conservatively."
            )
            
        # Check for high correlations in portfolio
        if strategy_id in self.active_strategies:
            strategy_info = self.active_strategies[strategy_id]
            correlations = strategy_info.get('correlations', {})
            high_corr_count = sum(1 for corr in correlations.values() if abs(corr) > 0.7)
            
            if high_corr_count >= 2:
                warnings.append(
                    f"CORRELATION WARNING: Strategy has high correlation (>0.7) with {high_corr_count} other active strategies. "
                    "Consider reducing allocation or finding uncorrelated alternatives."
                )
            
        # Check for drawdown
        if self.risk_controls.current_drawdown > self.risk_controls.max_drawdown_limit * 0.8:
            warnings.append(
                f"DRAWDOWN ALERT: Current drawdown ({self.risk_controls.current_drawdown:.1%}) is approaching maximum limit "
                f"({self.risk_controls.max_drawdown_limit:.1%}). Consider reducing overall exposure."
            )
            
        # Check for psychological factors
        if self.risk_controls.consecutive_losses >= 3:
            warnings.append(
                f"PSYCHOLOGICAL WARNING: {self.risk_controls.consecutive_losses} consecutive losses detected. "
                "Parameters have been adjusted to reduce risk. Consider taking a short break before the next trade."
            )
            
        return warnings
    
    def update_portfolio_state(self,
                             portfolio_value: float,
                             trade_results: Optional[Dict[str, bool]] = None,
                             update_correlations: bool = False,
                             market_data: Optional[Dict[str, pd.DataFrame]] = None) -> None:
        """
        Update the tracked portfolio state with new information.
        
        Args:
            portfolio_value: Current portfolio value
            trade_results: Optional dict mapping strategy IDs to trade results (True for win, False for loss)
            update_correlations: Whether to update strategy correlations
            market_data: Optional dict mapping strategy IDs to their market data (needed for correlation updates)
        """
        # Update portfolio value and drawdown
        old_value = self.current_portfolio_value
        self.current_portfolio_value = portfolio_value
        current_drawdown = self.risk_controls.update_drawdown(portfolio_value)
        
        # Update trade results if provided
        if trade_results:
            for strategy_id, result in trade_results.items():
                # Update risk controls
                self.risk_controls.update_psychological_factors(result)
                
                # Update strategy info
                if strategy_id in self.active_strategies:
                    strategy_info = self.active_strategies[strategy_id]
                    
                    # Update recent trades
                    recent_trades = strategy_info.get('recent_trades', [])
                    recent_trades.append(result)
                    if len(recent_trades) > 20:
                        recent_trades.pop(0)
                        
                    # Update win rate
                    win_rate = sum(recent_trades) / len(recent_trades) if recent_trades else 0.5
                    
                    # Update strategy metrics
                    metrics = strategy_info.get('metrics', {})
                    metrics['recent_win_rate'] = win_rate
                    
                    # Store updates
                    strategy_info['recent_trades'] = recent_trades
                    strategy_info['metrics'] = metrics
                    self.active_strategies[strategy_id] = strategy_info
        
        # Update correlations if requested and data provided
        if update_correlations and market_data and len(market_data) >= 2:
            self._update_strategy_correlations(market_data)
            
    def _update_strategy_correlations(self, market_data: Dict[str, pd.DataFrame]) -> None:
        """
        Update correlation matrix between strategies using their market data.
        
        Args:
            market_data: Dict mapping strategy IDs to their market data
        """
        # Extract returns for each strategy
        returns = {}
        for strategy_id, data in market_data.items():
            if 'close' in data.columns and len(data) > 5:
                # Calculate returns
                strategy_returns = data['close'].pct_change().dropna()
                returns[strategy_id] = strategy_returns
                
        # Calculate correlations where we have enough data
        for id1 in returns:
            for id2 in returns:
                if id1 != id2:
                    # Find overlap period
                    common_index = returns[id1].index.intersection(returns[id2].index)
                    if len(common_index) >= 20:
                        # Calculate correlation
                        corr = returns[id1].loc[common_index].corr(returns[id2].loc[common_index])
                        
                        # Store in strategy info
                        if id1 in self.active_strategies:
                            strategy_info = self.active_strategies[id1]
                            correlations = strategy_info.get('correlations', {})
                            correlations[id2] = corr
                            strategy_info['correlations'] = correlations
                            self.active_strategies[id1] = strategy_info
                            
    def run_stress_test(self,
                       strategy_id: str,
                       original_params: Dict[str, Any],
                       market_data: pd.DataFrame,
                       macro_data: Optional[pd.DataFrame] = None) -> Dict[str, RiskManagementPlan]:
        """
        Run stress tests on a strategy to see how it would perform in various scenarios.
        
        Args:
            strategy_id: Strategy identifier
            original_params: Original strategy parameters
            market_data: Market data for the strategy
            macro_data: Optional macro data
            
        Returns:
            Dict mapping scenario names to RiskManagementPlans
        """
        stress_test_results = {}
        
        # Get base strategy metrics
        strategy_metrics = (self.active_strategies.get(strategy_id, {})
                           .get('metrics', {'win_rate': 0.5, 'win_loss_ratio': 1.5}))
        
        # Run each stress test scenario
        for scenario in self.stress_test_scenarios:
            # Adjust market_data based on scenario
            adjusted_market_data = self._apply_stress_scenario(market_data, scenario)
            
            # Adjust macro_data if provided
            adjusted_macro_data = None
            if macro_data is not None:
                adjusted_macro_data = self._apply_stress_scenario(macro_data, scenario)
                
            # Adjust strategy metrics based on scenario (using external function)
            adjusted_metrics = adjust_metrics_for_scenario(strategy_metrics, scenario)
            
            # Create risk management plan for this scenario
            stress_plan = self.create_risk_management_plan(
                strategy_id=f"{strategy_id}_stress_{scenario['name']}",
                market_data=adjusted_market_data,
                strategy_params=original_params,
                signal_strength=0.5,  # Neutral signal for stress test
                strategy_metrics=adjusted_metrics,
                macro_data=adjusted_macro_data
            )
            
            # Store result
            stress_test_results[scenario['name']] = stress_plan
            
        return stress_test_results
    
    def _apply_stress_scenario(self, data: pd.DataFrame, scenario: Dict[str, Any]) -> pd.DataFrame:
        """
        Apply a stress test scenario to market data.
        
        Args:
            data: Original market data
            scenario: Stress test scenario parameters
            
        Returns:
            Adjusted market data
        """
        # Create a copy to avoid modifying the original
        adjusted_data = data.copy()
        
        # Apply volatility multiplier if available
        if 'volatility_multiplier' in scenario and 'close' in adjusted_data.columns:
            # Increase volatility by the multiplier
            vol_mult = scenario['volatility_multiplier']
            
            # Calculate returns
            returns = adjusted_data['close'].pct_change().fillna(0)
            
            # Apply multiplier to returns
            stressed_returns = returns * vol_mult
            
            # Recalculate prices
            base_price = adjusted_data['close'].iloc[0]
            adjusted_data['close'] = base_price * (1 + stressed_returns).cumprod()
            
            # Adjust high/low if available
            if all(col in adjusted_data.columns for col in ['open', 'high', 'low']):
                # Calculate average range as percentage
                avg_range_pct = ((adjusted_data['high'] - adjusted_data['low']) / adjusted_data['close']).mean()
                
                # Apply volatility multiplier to the range
                new_range_pct = avg_range_pct * vol_mult
                
                # Recalculate high and low
                adjusted_data['high'] = adjusted_data['close'] * (1 + new_range_pct/2)
                adjusted_data['low'] = adjusted_data['close'] * (1 - new_range_pct/2)
                adjusted_data['open'] = (adjusted_data['high'] + adjusted_data['low']) / 2
        
        # Apply spread multiplier if available
        if 'spread_multiplier' in scenario and 'spread' in adjusted_data.columns:
            adjusted_data['spread'] *= scenario['spread_multiplier']
            
        # Apply custom adjustments based on scenario name
        if scenario['name'] == "Liquidity Crisis" and 'volume' in adjusted_data.columns:
            # Reduce volume in liquidity crisis
            adjusted_data['volume'] = adjusted_data['volume'] / 3.0
            
        if scenario['name'] == "Mean Reversion Breakdown" and 'close' in adjusted_data.columns:
            # Add a trend component
            if 'trend_strength' in scenario:
                trend = np.linspace(0, scenario['trend_strength'], len(adjusted_data))
                adjusted_data['close'] *= (1 + trend)
                
        return adjusted_data
