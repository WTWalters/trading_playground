from typing import Dict, Any
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

class DataValidator:
    """Validates market data quality and integrity"""

    def __init__(self, validation_rules: Dict[str, Any]):
        self.rules = validation_rules

    def validate_market_data(self, data: pd.DataFrame) -> bool:
        """
        Validate market data against defined rules
        Returns True if data passes all validation checks
        """
        try:
            return all([
                self._validate_basic_structure(data),
                self._validate_price_consistency(data),
                self._validate_timestamps(data),
                self._validate_gaps(data),
                self._validate_outliers(data)
            ])
        except Exception:
            return False

    def _validate_basic_structure(self, data: pd.DataFrame) -> bool:
        """Validate basic data structure and completeness"""
        required_columns = ['open', 'high', 'low', 'close', 'volume']

        # Check required columns
        if not all(col in data.columns for col in required_columns):
            return False

        # Check for null values
        null_threshold = self.rules.get('null_threshold', 0.01)
        null_percentages = data[required_columns].isnull().mean()
        if (null_percentages > null_threshold).any():
            return False

        # Check data types
        numeric_cols = required_columns[:-1]  # All except volume
        if not all(data[col].dtype.kind in 'fc' for col in numeric_cols):
            return False

        return True

    def _validate_price_consistency(self, data: pd.DataFrame) -> bool:
        """Validate price relationship consistency"""
        # High should be highest price
        if not (
            (data['high'] >= data['open']) &
            (data['high'] >= data['close']) &
            (data['high'] >= data['low'])
        ).all():
            return False

        # Low should be lowest price
        if not (
            (data['low'] <= data['open']) &
            (data['low'] <= data['close']) &
            (data['low'] <= data['high'])
        ).all():
            return False

        # Volume should be non-negative
        if not (data['volume'] >= 0).all():
            return False

        return True

    def _validate_timestamps(self, data: pd.DataFrame) -> bool:
        """Validate timestamp consistency and ordering"""
        if not isinstance(data.index, pd.DatetimeIndex):
            return False

        # Check timestamp ordering
        if not data.index.is_monotonic_increasing:
            return False

        # Check for duplicates
        if data.index.has_duplicates:
            return False

        return True

    def _validate_gaps(self, data: pd.DataFrame) -> bool:
        """Validate for excessive gaps in data"""
        max_gap = self.rules.get('max_time_gap', pd.Timedelta(days=5))

        # Calculate time differences
        time_diffs = data.index[1:] - data.index[:-1]

        # Check for gaps exceeding threshold
        if (time_diffs > max_gap).any():
            return False

        return True

    def _validate_outliers(self, data: pd.DataFrame) -> bool:
        """Validate for price and volume outliers"""
        price_z_threshold = self.rules.get('price_z_threshold', 4.0)
        volume_z_threshold = self.rules.get('volume_z_threshold', 5.0)

        for col in ['open', 'high', 'low', 'close']:
            z_scores = np.abs(
                (data[col] - data[col].mean()) / data[col].std()
            )
            if (z_scores > price_z_threshold).sum() > len(data) * 0.01:
                return False

        # Volume outliers
        volume_z_scores = np.abs(
            (data['volume'] - data['volume'].mean()) / data['volume'].std()
        )
        if (volume_z_scores > volume_z_threshold).sum() > len(data) * 0.01:
            return False

        return True

    def validate_corporate_actions(
        self,
        data: pd.DataFrame,
        actions: pd.DataFrame
    ) -> bool:
        """Validate data against corporate actions"""
        # Implementation for corporate actions validation
        return True  # Placeholder for future implementation
