# src/market_analysis/risk.py

from typing import Dict, Optional, Union, Tuple
from dataclasses import dataclass
import logging
import numpy as np

@dataclass
class RiskParameters:
    """
    Container for risk management parameters

    Attributes:
        max_position_size: Maximum allowed position size
        max_risk_percent: Maximum risk per trade (%)
        min_risk_reward: Minimum risk/reward ratio
        max_account_risk: Maximum total account risk (%)
        position_limit: Maximum positions allowed
    """
    max_position_size: int = 100000
    max_risk_percent: float = 2.0
    min_risk_reward: float = 1.5
    max_account_risk: float = 5.0
    position_limit: int = 5

class RiskManager:
    """
    Comprehensive risk management system

    Features:
    - Position sizing
    - Stop loss calculation
    - Risk validation
    - Account risk management
    - Position limits
    """

    def __init__(self, parameters: Optional[RiskParameters] = None):
        """
        Initialize risk manager

        Args:
            parameters: Optional risk parameters (uses defaults if None)
        """
        self.params = parameters or RiskParameters()
        self.logger = logging.getLogger(__name__)
        self.open_positions: Dict[str, float] = {}

    def calculate_position_size(
        self,
        capital: float,
        risk_amount: float,
        entry_price: float,
        stop_loss: float
    ) -> int:
        """
        Calculate optimal position size based on risk parameters

        Args:
            capital: Available trading capital
            risk_amount: Amount willing to risk (in currency)
            entry_price: Planned entry price
            stop_loss: Stop loss price level

        Returns:
            Integer position size

        Raises:
            ValueError: If inputs are invalid
        """
        try:
            # Validate inputs
            self._validate_inputs(capital, risk_amount, entry_price, stop_loss)

            # Calculate risk per unit
            risk_per_unit = abs(entry_price - stop_loss)
            if risk_per_unit == 0:
                raise ValueError("Zero risk per unit")

            # Calculate raw position size
            position_size = risk_amount / risk_per_unit

            # Apply position limits
            position_size = min(
                position_size,
                self.params.max_position_size,
                (capital * self.params.max_risk_percent / 100) / entry_price
            )

            return max(1, int(position_size))  # Ensure at least 1 unit

        except Exception as e:
            self.logger.error(f"Position size calculation failed: {str(e)}")
            return 0

    def calculate_stop_levels(
        self,
        entry_price: float,
        volatility: float,
        direction: str = 'LONG',
        risk_multiple: float = 1.0,
        atr_multiple: float = 2.0
    ) -> Dict[str, float]:
        """
        Calculate stop loss and take profit levels

        Args:
            entry_price: Trade entry price
            volatility: Current volatility (e.g., ATR)
            direction: Trade direction ('LONG' or 'SHORT')
            risk_multiple: Risk multiplier
            atr_multiple: ATR multiplier for stops

        Returns:
            Dictionary with stop loss and take profit prices
        """
        try:
            # Calculate base stop distance
            stop_distance = volatility * atr_multiple * risk_multiple

            if direction == 'LONG':
                stop_loss = entry_price - stop_distance
                take_profit = entry_price + (stop_distance * self.params.min_risk_reward)
            else:
                stop_loss = entry_price + stop_distance
                take_profit = entry_price - (stop_distance * self.params.min_risk_reward)

            return {
                'stop_loss': round(stop_loss, 4),
                'take_profit': round(take_profit, 4)
            }

        except Exception as e:
            self.logger.error(f"Stop level calculation failed: {str(e)}")
            return {'stop_loss': entry_price, 'take_profit': entry_price}

    def validate_trade(
        self,
        position_size: int,
        entry_price: float,
        stop_loss: float,
        capital: float,
        direction: str = 'LONG'
    ) -> Tuple[bool, str]:
        """
        Validate if trade meets all risk criteria

        Args:
            position_size: Number of units
            entry_price: Entry price
            stop_loss: Stop loss price
            capital: Available capital
            direction: Trade direction

        Returns:
            Tuple of (valid: bool, reason: str)
        """
        try:
            # Check position size limits
            if position_size > self.params.max_position_size:
                return False, "Position size exceeds maximum"

            # Calculate risk amount
            risk_amount = abs(entry_price - stop_loss) * position_size
            risk_percent = (risk_amount / capital) * 100

            # Check risk percentage
            if risk_percent > self.params.max_risk_percent:
                return False, f"Risk exceeds maximum ({risk_percent:.1f}%)"

            # Check total account risk
            total_risk = self._calculate_total_risk(risk_amount)
            if total_risk > self.params.max_account_risk:
                return False, f"Total account risk too high ({total_risk:.1f}%)"

            # Check position limits
            if len(self.open_positions) >= self.params.position_limit:
                return False, "Maximum positions reached"

            return True, "Trade validated"

        except Exception as e:
            self.logger.error(f"Trade validation failed: {str(e)}")
            return False, f"Validation error: {str(e)}"

    def _validate_inputs(
        self,
        capital: float,
        risk_amount: float,
        entry_price: float,
        stop_loss: float
    ) -> None:
        """Validate input parameters"""
        if capital <= 0:
            raise ValueError("Capital must be positive")
        if risk_amount <= 0 or risk_amount > capital:
            raise ValueError("Invalid risk amount")
        if entry_price <= 0:
            raise ValueError("Entry price must be positive")
        if stop_loss <= 0:
            raise ValueError("Stop loss must be positive")
        if entry_price == stop_loss:
            raise ValueError("Entry price equals stop loss")

    def _calculate_total_risk(self, new_risk: float) -> float:
        """Calculate total account risk including new position"""
        current_risk = sum(self.open_positions.values())
        return ((current_risk + new_risk) / self.capital) * 100

    def add_position(self, position_id: str, risk_amount: float) -> None:
        """Track new position risk"""
        self.open_positions[position_id] = risk_amount

    def remove_position(self, position_id: str) -> None:
        """Remove closed position"""
        self.open_positions.pop(position_id, None)

    def get_risk_metrics(self) -> Dict[str, float]:
        """
        Get current risk metrics

        Returns:
            Dictionary containing:
            - total_risk_percent: Total account risk
            - largest_position_risk: Largest single position risk
            - average_position_risk: Average position risk
            - position_count: Number of open positions
        """
        if not self.open_positions:
            return {
                'total_risk_percent': 0.0,
                'largest_position_risk': 0.0,
                'average_position_risk': 0.0,
                'position_count': 0
            }

        risks = list(self.open_positions.values())
        return {
            'total_risk_percent': sum(risks) / self.capital * 100,
            'largest_position_risk': max(risks),
            'average_position_risk': np.mean(risks),
            'position_count': len(risks)
        }
