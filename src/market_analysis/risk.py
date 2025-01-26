# src/market_analysis/risk.py

from typing import Dict, Optional, Union, Tuple, List, Any
from dataclasses import dataclass
from enum import Enum
import logging
import numpy as np

class RiskLevel(Enum):
    """Risk level classifications"""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    EXTREME = "extreme"

@dataclass
class RiskParameters:
    """Risk management configuration parameters"""
    max_position_size: float = 100000.0  # Maximum position size in base currency
    max_risk_percent: float = 2.0  # Maximum risk per trade (%)
    min_risk_reward: float = 1.5  # Minimum risk/reward ratio
    max_account_risk: float = 5.0  # Maximum total account risk (%)
    position_limit: int = 5  # Maximum number of concurrent positions
    max_correlation: float = 0.7  # Maximum position correlation allowed
    atr_multiplier: float = 2.0  # ATR multiplier for stop calculation
    profit_target_multiplier: float = 2.0  # Multiplier for take profit levels
    max_drawdown: float = 20.0  # Maximum allowed drawdown (%)
    min_position_size: float = 100.0  # Minimum position size in base currency

class RiskManager:
    """
    Risk management system for trading operations.

    Handles:
    - Position sizing
    - Stop loss calculation
    - Risk/reward validation
    - Portfolio risk management
    - Drawdown monitoring
    """

    def __init__(
        self,
        parameters: Optional[RiskParameters] = None,
        initial_capital: float = 10000.0
    ):
        """
        Initialize risk manager.

        Args:
            parameters: Risk management parameters
            initial_capital: Starting capital amount
        """
        self.params = parameters or RiskParameters()
        self.capital = initial_capital
        self.logger = logging.getLogger(__name__)
        self.open_positions: Dict[str, Dict] = {}
        self.position_history: List[Dict] = []
        self.max_capital = initial_capital
        self.current_drawdown = 0.0

    def calculate_position_size(
        self,
        capital: float,
        risk_amount: float,
        entry_price: float,
        stop_loss: float,
        volatility_factor: float = 1.0
    ) -> float:
        """
        Calculate safe position size based on risk parameters.

        Args:
            capital: Available capital
            risk_amount: Amount willing to risk
            entry_price: Trade entry price
            stop_loss: Stop loss price
            volatility_factor: Volatility adjustment (0-1)

        Returns:
            Position size in base currency units

        Raises:
            ValueError: If inputs are invalid
        """
        self._validate_position_inputs(capital, risk_amount, entry_price, stop_loss)

        try:
            # Calculate stop distance
            stop_distance = abs(entry_price - stop_loss)
            if stop_distance == 0:
                raise ValueError("Stop distance cannot be zero")

            # Calculate raw position size
            risk_adjusted = risk_amount * volatility_factor
            position_size = risk_adjusted / stop_distance

            # Apply limits
            position_size = min(
                position_size,
                self.params.max_position_size,
                capital * 0.5  # Max 50% of capital
            )
            position_size = max(
                position_size,
                self.params.min_position_size
            )

            return round(position_size, 2)

        except Exception as e:
            self.logger.error(f"Position size calculation failed: {str(e)}")
            raise

    def calculate_stop_levels(
        self,
        entry_price: float,
        atr: float,
        direction: str,
        custom_risk_reward: Optional[float] = None
    ) -> Dict[str, float]:
        """
        Calculate stop loss and take profit levels.

        Args:
            entry_price: Trade entry price
            atr: Average True Range value
            direction: Trade direction ('LONG' or 'SHORT')
            custom_risk_reward: Optional custom R:R ratio

        Returns:
            Dictionary with stop loss and take profit prices
        """
        try:
            # Calculate base stop distance
            stop_distance = atr * self.params.atr_multiplier
            risk_reward = custom_risk_reward or self.params.min_risk_reward

            if direction.upper() == 'LONG':
                stop_loss = entry_price - stop_distance
                take_profit = entry_price + (stop_distance * risk_reward)
            elif direction.upper() == 'SHORT':
                stop_loss = entry_price + stop_distance
                take_profit = entry_price - (stop_distance * risk_reward)
            else:
                raise ValueError(f"Invalid direction: {direction}")

            return {
                'stop_loss': round(stop_loss, 4),
                'take_profit': round(take_profit, 4),
                'risk_reward_ratio': risk_reward,
                'stop_distance': stop_distance
            }

        except Exception as e:
            self.logger.error(f"Stop level calculation failed: {str(e)}")
            raise

    def validate_trade(
        self,
        position_size: float,
        risk_amount: float,
        correlation: Optional[float] = None
    ) -> Tuple[bool, str]:
        """
        Validate trade against risk parameters.

        Args:
            position_size: Proposed position size
            risk_amount: Amount to risk
            correlation: Optional correlation with existing positions

        Returns:
            Tuple of (is_valid: bool, reason: str)
        """
        try:
            # Check position size limits
            if position_size > self.params.max_position_size:
                return False, "Position size exceeds maximum"
            if position_size < self.params.min_position_size:
                return False, "Position size below minimum"

            # Check risk percentage
            risk_percent = (risk_amount / self.capital) * 100
            if risk_percent > self.params.max_risk_percent:
                return False, f"Risk exceeds maximum ({risk_percent:.1f}%)"

            # Check portfolio risk
            total_risk = self._calculate_portfolio_risk(risk_amount)
            if total_risk > self.params.max_account_risk:
                return False, f"Portfolio risk too high ({total_risk:.1f}%)"

            # Check correlation if provided
            if correlation is not None:
                if abs(correlation) > self.params.max_correlation:
                    return False, f"Position correlation too high ({correlation:.2f})"

            # Check position limit
            if len(self.open_positions) >= self.params.position_limit:
                return False, "Maximum positions reached"

            # Check drawdown
            if self.current_drawdown > self.params.max_drawdown:
                return False, f"Maximum drawdown exceeded ({self.current_drawdown:.1f}%)"

            return True, "Trade validated"

        except Exception as e:
            self.logger.error(f"Trade validation failed: {str(e)}")
            return False, str(e)

    def update_capital(self, new_capital: float) -> None:
        """
        Update capital and drawdown metrics.

        Args:
            new_capital: Current capital amount
        """
        try:
            self.capital = new_capital
            self.max_capital = max(self.max_capital, new_capital)
            self.current_drawdown = ((self.max_capital - new_capital) / self.max_capital) * 100

        except Exception as e:
            self.logger.error(f"Capital update failed: {str(e)}")

    def get_risk_metrics(self) -> Dict[str, Union[float, str]]:
        """
        Get current risk metrics.

        Returns:
            Dictionary containing risk metrics
        """
        try:
            total_risk = sum(pos.get('risk_amount', 0) for pos in self.open_positions.values())
            risk_percent = (total_risk / self.capital) * 100 if self.capital > 0 else 0

            return {
                'total_risk_percent': round(risk_percent, 2),
                'current_drawdown': round(self.current_drawdown, 2),
                'position_count': len(self.open_positions),
                'risk_level': self._classify_risk_level(risk_percent).value,
                'available_risk': round(self.params.max_account_risk - risk_percent, 2),
                'capital': round(self.capital, 2),
                'max_capital': round(self.max_capital, 2)
            }

        except Exception as e:
            self.logger.error(f"Risk metrics calculation failed: {str(e)}")
            return self._get_empty_metrics()

    def _calculate_portfolio_risk(self, new_risk: float) -> float:
        """Calculate total portfolio risk including new position"""
        current_risk = sum(pos.get('risk_amount', 0) for pos in self.open_positions.values())
        return ((current_risk + new_risk) / self.capital) * 100 if self.capital > 0 else 0

    def _classify_risk_level(self, risk_percent: float) -> RiskLevel:
        """Classify current risk level"""
        if risk_percent <= self.params.max_account_risk * 0.3:
            return RiskLevel.LOW
        elif risk_percent <= self.params.max_account_risk * 0.6:
            return RiskLevel.MEDIUM
        elif risk_percent <= self.params.max_account_risk:
            return RiskLevel.HIGH
        return RiskLevel.EXTREME

    @staticmethod
    def _validate_position_inputs(
        capital: float,
        risk_amount: float,
        entry_price: float,
        stop_loss: float
    ) -> None:
        """Validate position calculation inputs"""
        if capital <= 0:
            raise ValueError("Capital must be positive")
        if risk_amount <= 0:
            raise ValueError("Risk amount must be positive")
        if entry_price <= 0:
            raise ValueError("Entry price must be positive")
        if stop_loss <= 0:
            raise ValueError("Stop loss must be positive")

    def _get_empty_metrics(self) -> Dict[str, Union[float, str]]:
        """Return empty risk metrics structure"""
        return {
            'total_risk_percent': 0.0,
            'current_drawdown': 0.0,
            'position_count': 0,
            'risk_level': RiskLevel.LOW.value,
            'available_risk': self.params.max_account_risk,
            'capital': self.capital,
            'max_capital': self.max_capital
        }
