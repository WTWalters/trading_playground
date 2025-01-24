from typing import Dict, Any
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import logging

class DataValidator:
    """Validates market data quality and integrity"""

    def __init__(self, validation_rules: Dict[str, Any]):
        self.rules = validation_rules
        self.logger = logging.getLogger(__name__)

    def validate_market_data(self, data: pd.DataFrame) -> bool:
        """
        Validate market data against defined rules
        Returns True if data passes all validation checks
        """
        try:
            validations = [
                (self._validate_basic_structure(data), "basic structure"),
                (self._validate_price_consistency(data), "price consistency"),
                (self._validate_timestamps(data), "timestamps"),
                (self._validate_gaps(data), "time gaps"),
                (self._validate_outliers(data), "outliers")
            ]

            for validation, name in validations:
                if not validation:
                    self.logger.warning(f"Validation failed: {name}")
                    return False
            return True
            
        except Exception as e:
            self.logger.error(f"Validation error: {str(e)}")
            return False

    def _validate_basic_structure(self, data: pd.DataFrame) -> bool:
        """Validate basic data structure and completeness"""
        required_columns = ['open', 'high', 'low', 'close', 'volume']

        # Check required columns
        if not all(col in data.columns for col in required_columns):
            self.logger.warning("Missing required columns")
            return False

        # Check for null values
        null_threshold = self.rules.get('null_threshold', 0.01)
        null_percentages = data[required_columns].isnull().mean()
        if (null_percentages > null_threshold).any():
            self.logger.warning(f"Null values exceed threshold: {null_percentages}")
            return False

        # Check data types
        numeric_cols = required_columns[:-1]  # All except volume
        if not all(data[col].dtype.kind in 'fc' for col in numeric_cols):
            self.logger.warning("Invalid data types detected")
            return False

        return True

    def _validate_price_consistency(self, data: pd.DataFrame) -> bool:
        """Validate price relationship consistency"""
        try:
            # Add small tolerance for floating point comparisons
            tolerance = 1e-10

            # High should be highest price
            high_valid = (
                (data['high'] >= (data['open'] - tolerance)) &
                (data['high'] >= (data['close'] - tolerance)) &
                (data['high'] >= (data['low'] - tolerance))
            ).all()

            if not high_valid:
                self.logger.warning("High price validation failed")
                return False

            # Low should be lowest price
            low_valid = (
                (data['low'] <= (data['open'] + tolerance)) &
                (data['low'] <= (data['close'] + tolerance)) &
                (data['low'] <= (data['high'] + tolerance))
            ).all()

            if not low_valid:
                self.logger.warning("Low price validation failed")
                return False

            # Volume should be non-negative
            if not (data['volume'] >= 0).all():
                self.logger.warning("Negative volume detected")
                return False

            return True

        except Exception as e:
            self.logger.error(f"Price consistency validation error: {str(e)}")
            return False

    def _validate_timestamps(self, data: pd.DataFrame) -> bool:
        """Validate timestamp consistency and ordering"""
        try:
            if not isinstance(data.index, pd.DatetimeIndex):
                self.logger.warning("Index is not DatetimeIndex")
                return False

            # Check timestamp ordering
            if not data.index.is_monotonic_increasing:
                self.logger.warning("Timestamps not monotonically increasing")
                return False

            # Check for duplicates
            if data.index.has_duplicates:
                self.logger.warning("Duplicate timestamps detected")
                return False

            return True

        except Exception as e:
            self.logger.error(f"Timestamp validation error: {str(e)}")
            return False

    def _validate_gaps(self, data: pd.DataFrame) -> bool:
        """Validate for excessive gaps in data"""
        try:
            max_gap = self.rules.get('max_time_gap', pd.Timedelta(days=5))

            # Calculate time differences
            time_diffs = data.index[1:] - data.index[:-1]

            # Check for gaps exceeding threshold
            if (time_diffs > max_gap).any():
                self.logger.warning(f"Time gaps exceed maximum: {max_gap}")
                return False

            return True

        except Exception as e:
            self.logger.error(f"Gap validation error: {str(e)}")
            return False

    def _validate_outliers(self, data: pd.DataFrame) -> bool:
        """Validate for price and volume outliers"""
        try:
            price_z_threshold = self.rules.get('price_z_threshold', 4.0)
            volume_z_threshold = self.rules.get('volume_z_threshold', 5.0)
            max_outlier_percentage = 0.01  # Maximum 1% outliers allowed

            # Check price outliers
            for col in ['open', 'high', 'low', 'close']:
                if len(data[col].unique()) <= 1:  # Skip if all values are the same
                    continue
                    
                z_scores = np.abs((data[col] - data[col].mean()) / data[col].std())
                outlier_percentage = (z_scores > price_z_threshold).mean()
                
                if outlier_percentage > max_outlier_percentage:
                    self.logger.warning(f"Excessive price outliers in {col}: {outlier_percentage:.2%}")
                    return False

            # Volume outliers
            if len(data['volume'].unique()) > 1:  # Skip if all volumes are the same
                volume_z_scores = np.abs(
                    (data['volume'] - data['volume'].mean()) / data['volume'].std()
                )
                volume_outlier_percentage = (volume_z_scores > volume_z_threshold).mean()
                
                if volume_outlier_percentage > max_outlier_percentage:
                    self.logger.warning(f"Excessive volume outliers: {volume_outlier_percentage:.2%}")
                    return False

            return True

        except Exception as e:
            self.logger.error(f"Outlier validation error: {str(e)}")
            return False

    def validate_corporate_actions(
        self,
        data: pd.DataFrame,
        actions: pd.DataFrame
    ) -> bool:
        """Validate data against corporate actions"""
        # Implementation for corporate actions validation
        return True  # Placeholder for future implementation