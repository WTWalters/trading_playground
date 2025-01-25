# src/market_analysis/risk.py

from typing import Dict, Optional
import logging

class RiskManager:
    """Handle position sizing and risk management"""

    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.max_position_size = 100000  # Maximum allowed position size

    def calculate_trade_size(
        self,
        account_size: float,
        risk_amount: float,
        stop_loss: float
    ) -> int:
        """
        Calculate proper position size based on risk parameters.

        Args:
            account_size: Total account value
            risk_amount: Percentage of account willing to risk (1 = 1%)
            stop_loss: Distance to stop loss in price units

        Returns:
            Number of shares/contracts to trade

        Raises:
            ValueError: If inputs are invalid
        """
        try:
            # Validate inputs
            if account_size <= 0:
                raise ValueError("Account size must be positive")
            if risk_amount <= 0 or risk_amount > 100:
                raise ValueError("Risk amount must be between 0 and 100")
            if stop_loss <= 0:
                raise ValueError("Stop loss must be positive")

            # Calculate maximum loss amount
            max_loss_amount = account_size * (risk_amount / 100)

            # Calculate position size
            position_size = int(max_loss_amount / stop_loss)

            # Apply position limits
            position_size = max(1, min(position_size, self.max_position_size))

            return position_size

        except Exception as e:
            self.logger.error(f"Position size calculation failed: {str(e)}")
            raise

    def calculate_stop_levels(
        self,
        entry_price: float,
        volatility: float,
        risk_multiple: float = 1.0
    ) -> Dict[str, float]:
        """
        Calculate stop loss and take profit levels based on volatility.

        Args:
            entry_price: Entry price of the trade
            volatility: Current volatility measure (e.g., ATR)
            risk_multiple: Risk:Reward ratio multiplier

        Returns:
            Dictionary containing stop loss and take profit levels
        """
        try:
            stop_distance = volatility * risk_multiple

            return {
                'stop_loss': entry_price - stop_distance,
                'take_profit': entry_price + (stop_distance * 2)  # 1:2 risk:reward
            }

        except Exception as e:
            self.logger.error(f"Stop level calculation failed: {str(e)}")
            return {'stop_loss': 0, 'take_profit': 0}

    def validate_risk_levels(
        self,
        position_size: int,
        entry_price: float,
        stop_loss: float,
        account_size: float,
        max_risk_percent: float = 2.0
    ) -> bool:
        """
        Validate if the trade meets risk management criteria.

        Args:
            position_size: Number of shares/contracts
            entry_price: Entry price of the trade
            stop_loss: Stop loss price
            account_size: Total account value
            max_risk_percent: Maximum allowed risk percentage

        Returns:
            Boolean indicating if risk levels are acceptable
        """
        try:
            risk_amount = abs(entry_price - stop_loss) * position_size
            risk_percent = (risk_amount / account_size) * 100

            return risk_percent <= max_risk_percent

        except Exception as e:
            self.logger.error(f"Risk validation failed: {str(e)}")
            return False
