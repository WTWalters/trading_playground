"""
Data validation module for market data.

This module provides utilities for validating and correcting market data,
including checking for:
- Missing or invalid values
- Price relationship violations
- Outliers
- Gaps in time series
"""

from typing import Dict, Any, List, Optional, Tuple, Union
import pandas as pd
import numpy as np
import logging
from datetime import datetime, timedelta
from dataclasses import dataclass


@dataclass
class ValidationResult:
    """Container for data validation results."""
    
    valid: bool
    quality_score: float
    issues: Dict[str, Any]
    df: pd.DataFrame  # The validated (and potentially corrected) DataFrame
    
    def __post_init__(self):
        """Initialize computed attributes."""
        self.has_issues = len(self.issues) > 0
        
    def __str__(self) -> str:
        """String representation of validation result."""
        status = "VALID" if self.valid else "INVALID"
        return f"{status} (Score: {self.quality_score:.1f}) - Issues: {list(self.issues.keys())}"


class DataValidator:
    """
    Validates and corrects market data.
    
    Provides methods to:
    1. Check for common data quality issues
    2. Calculate a quality score
    3. Correct issues where possible
    4. Log serious issues for review
    """
    
    def __init__(self, auto_correct: bool = False):
        """
        Initialize the data validator.
        
        Args:
            auto_correct: Whether to automatically correct issues when possible
        """
        self.auto_correct = auto_correct
        self.logger = logging.getLogger(__name__)
        
    def validate_ohlcv(
        self, 
        data: pd.DataFrame,
        symbol: str,
        min_quality_score: float = 0.0
    ) -> ValidationResult:
        """
        Validate OHLCV market data for common issues.
        
        Args:
            data: DataFrame with OHLCV data
            symbol: The symbol being validated
            min_quality_score: Minimum quality score to be considered valid
            
        Returns:
            ValidationResult object with validation details
        """
        if data.empty:
            return ValidationResult(
                valid=False,
                quality_score=0.0,
                issues={"empty_data": "Data contains no rows"},
                df=data
            )
            
        # Check for required columns
        required_cols = ['open', 'high', 'low', 'close', 'volume']
        missing_cols = [col for col in required_cols if col not in data.columns]
        
        if missing_cols:
            return ValidationResult(
                valid=False,
                quality_score=0.0,
                issues={"missing_columns": missing_cols},
                df=data
            )
            
        # Create a copy to avoid modifying original
        df = data.copy()
        issues = {}
        deductions = []
        
        # Convert columns to appropriate numeric types if needed
        for col in required_cols:
            if not pd.api.types.is_numeric_dtype(df[col]):
                try:
                    df[col] = pd.to_numeric(df[col], errors='coerce')
                    issues["non_numeric_converted"] = f"Converted non-numeric values in {col}"
                    deductions.append(("non_numeric_conversion", 5))
                except:
                    issues["non_numeric_error"] = f"Could not convert column {col} to numeric"
                    deductions.append(("non_numeric_error", 40))
                    return ValidationResult(
                        valid=False,
                        quality_score=0.0,
                        issues=issues,
                        df=data
                    )
        
        # Check for NaN values
        nan_counts = df[required_cols].isna().sum()
        total_nan_count = nan_counts.sum()
        
        if total_nan_count > 0:
            nan_percent = (total_nan_count / (len(df) * len(required_cols))) * 100
            issues["nan_values"] = {
                "count": int(total_nan_count),
                "percentage": f"{nan_percent:.2f}%",
                "by_column": nan_counts.to_dict()
            }
            
            # Scale deduction by percentage of NaNs
            nan_deduction = min(40, nan_percent * 2)
            deductions.append(("nan_values", nan_deduction))
            
            # Attempt to fix NaNs if auto-correct is enabled
            if self.auto_correct:
                # Use forward fill then backward fill for prices
                for col in ['open', 'high', 'low', 'close']:
                    df[col] = df[col].ffill().bfill()
                
                # Use zero for volume NaNs
                df['volume'] = df['volume'].fillna(0)
                
                issues["nan_values_corrected"] = "NaN values were filled"
        
        # Check for negative prices
        neg_prices = (df[['open', 'high', 'low', 'close']] <= 0).any()
        neg_price_cols = neg_prices[neg_prices].index.tolist()
        
        if neg_price_cols:
            issues["negative_prices"] = {
                "columns": neg_price_cols,
                "count": (df[neg_price_cols] <= 0).sum().to_dict()
            }
            deductions.append(("negative_prices", 30))
            
            # Correct negative prices if auto-correct is enabled
            if self.auto_correct:
                for col in neg_price_cols:
                    neg_mask = df[col] <= 0
                    if neg_mask.any():
                        # Replace with absolute value or previous valid value
                        # Use abs for small negative values, use previous for large negatives
                        small_neg = (df[col] > -1) & (df[col] <= 0)
                        large_neg = df[col] <= -1
                        
                        if small_neg.any():
                            df.loc[small_neg, col] = df.loc[small_neg, col].abs()
                            
                        if large_neg.any():
                            df.loc[large_neg, col] = df[col].ffill().bfill()
                            
                        issues["negative_prices_corrected"] = "Negative prices were corrected"
        
        # Check for negative volumes
        neg_volumes = (df['volume'] < 0).any()
        if neg_volumes:
            neg_count = (df['volume'] < 0).sum()
            issues["negative_volumes"] = {
                "count": int(neg_count),
                "percentage": f"{(neg_count / len(df)) * 100:.2f}%"
            }
            deductions.append(("negative_volumes", 20))
            
            # Correct negative volumes if auto-correct is enabled
            if self.auto_correct:
                df['volume'] = df['volume'].clip(lower=0)
                issues["negative_volumes_corrected"] = "Negative volumes were corrected"
        
        # Check price relationship (high >= open, high >= close, low <= open, low <= close)
        price_violations = pd.DataFrame({
            "high_lt_open": df['high'] < df['open'],
            "high_lt_close": df['high'] < df['close'],
            "low_gt_open": df['low'] > df['open'],
            "low_gt_close": df['low'] > df['close']
        })
        
        price_violation_rows = price_violations.any(axis=1)
        price_violation_count = price_violation_rows.sum()
        
        if price_violation_count > 0:
            violation_percent = (price_violation_count / len(df)) * 100
            
            issues["price_violations"] = {
                "count": int(price_violation_count),
                "percentage": f"{violation_percent:.2f}%",
                "details": {
                    "high_lt_open": int(price_violations['high_lt_open'].sum()),
                    "high_lt_close": int(price_violations['high_lt_close'].sum()),
                    "low_gt_open": int(price_violations['low_gt_open'].sum()),
                    "low_gt_close": int(price_violations['low_gt_close'].sum())
                }
            }
            
            # Scale deduction by percentage of violations
            violation_deduction = min(40, violation_percent * 2)
            deductions.append(("price_violations", violation_deduction))
            
            # Correct price violations if auto-correct is enabled
            if self.auto_correct:
                for idx in df.index[price_violation_rows]:
                    row = df.loc[idx]
                    
                    # Find the min and max of open and close
                    min_price = min(row['open'], row['close'])
                    max_price = max(row['open'], row['close'])
                    
                    # Correct high to be at least the max of open and close
                    if row['high'] < max_price:
                        df.loc[idx, 'high'] = max_price
                        
                    # Correct low to be at most the min of open and close
                    if row['low'] > min_price:
                        df.loc[idx, 'low'] = min_price
                        
                issues["price_violations_corrected"] = "Price violations were corrected"
        
        # Check for zero volumes
        zero_volumes = (df['volume'] == 0).any()
        if zero_volumes:
            zero_count = (df['volume'] == 0).sum()
            zero_percent = (zero_count / len(df)) * 100
            
            issues["zero_volumes"] = {
                "count": int(zero_count),
                "percentage": f"{zero_percent:.2f}%"
            }
            
            if zero_percent > 20:
                deductions.append(("zero_volumes", min(15, zero_percent / 2)))
        
        # Check for duplicate indices
        duplicated = df.index.duplicated()
        if duplicated.any():
            dup_count = duplicated.sum()
            
            issues["duplicate_indices"] = {
                "count": int(dup_count),
                "percentage": f"{(dup_count / len(df)) * 100:.2f}%"
            }
            
            deductions.append(("duplicate_indices", 15))
            
            # Correct duplicates if auto-correct is enabled
            if self.auto_correct:
                df = df[~df.index.duplicated(keep='first')]
                issues["duplicate_indices_corrected"] = "Duplicate indices were removed"
        
        # Check for time gaps (if index is DatetimeIndex)
        if isinstance(df.index, pd.DatetimeIndex) and len(df) > 1:
            # Determine expected frequency
            time_diffs = df.index.to_series().diff().dropna()
            
            if len(time_diffs) > 0:
                # Try to infer frequency
                most_common_diff = time_diffs.mode()[0]
                
                # Look for gaps larger than 3x the most common diff
                large_gaps = time_diffs > (most_common_diff * 3)
                large_gap_count = large_gaps.sum()
                
                if large_gap_count > 0:
                    gap_percent = (large_gap_count / len(df)) * 100
                    max_gap = time_diffs.max()
                    
                    issues["time_gaps"] = {
                        "count": int(large_gap_count),
                        "percentage": f"{gap_percent:.2f}%",
                        "max_gap": f"{max_gap}",
                        "expected_interval": f"{most_common_diff}"
                    }
                    
                    # Scale deduction by percentage of gaps
                    gap_deduction = min(15, gap_percent)
                    deductions.append(("time_gaps", gap_deduction))
        
        # Check for outliers using z-score
        z_scores = df[required_cols].sub(df[required_cols].mean()).div(df[required_cols].std())
        outliers = (z_scores.abs() > 3).any(axis=1)
        outlier_count = outliers.sum()
        
        if outlier_count > 0:
            outlier_percent = (outlier_count / len(df)) * 100
            
            issues["outliers"] = {
                "count": int(outlier_count),
                "percentage": f"{outlier_percent:.2f}%",
                "details": {
                    col: int((z_scores[col].abs() > 3).sum()) 
                    for col in required_cols
                }
            }
            
            # Scale deduction by percentage of outliers
            outlier_deduction = min(20, outlier_percent)
            deductions.append(("outliers", outlier_deduction))
            
            # Correct outliers if auto-correct is enabled
            if self.auto_correct and outlier_percent < 20:  # Only auto-correct if not too many outliers
                for col in ['open', 'high', 'low', 'close']:
                    col_outliers = z_scores[col].abs() > 3
                    
                    if col_outliers.any():
                        # Replace with median of surrounding values
                        window = 5  # Use 5-point window for replacement
                        
                        # Create rolling median for replacement
                        rolling_med = df[col].rolling(window=window, center=True, min_periods=2).median()
                        
                        # Replace outliers with rolling median
                        df.loc[col_outliers, col] = rolling_med.loc[col_outliers]
                        
                issues["outliers_corrected"] = "Price outliers were corrected"
        
        # Check for stale prices (no change over multiple periods)
        price_changes = df['close'].pct_change().fillna(0)
        stale_prices = (price_changes == 0)
        
        # Look for runs of 5 or more unchanged prices
        if len(stale_prices) >= 5:
            stale_runs = []
            run_length = 0
            
            for i, is_stale in enumerate(stale_prices):
                if is_stale:
                    run_length += 1
                else:
                    if run_length >= 5:
                        stale_runs.append((i - run_length, i - 1, run_length))
                    run_length = 0
            
            # Check the last run
            if run_length >= 5:
                stale_runs.append((len(stale_prices) - run_length, len(stale_prices) - 1, run_length))
            
            if stale_runs:
                max_run = max(run_length for _, _, run_length in stale_runs)
                stale_percent = sum(run_length for _, _, run_length in stale_runs) / len(df) * 100
                
                issues["stale_prices"] = {
                    "count": len(stale_runs),
                    "max_run_length": max_run,
                    "percentage": f"{stale_percent:.2f}%"
                }
                
                if stale_percent > 30:
                    deductions.append(("stale_prices", min(10, stale_percent / 5)))
        
        # Calculate final quality score
        base_score = 100.0
        total_deduction = sum(deduction for _, deduction in deductions)
        quality_score = max(0, base_score - total_deduction)
        
        # Log issues for debugging
        if issues:
            issue_keys = list(issues.keys())
            self.logger.debug(
                f"Data validation for {symbol}: Quality score {quality_score:.1f}, " 
                f"Issues: {issue_keys}"
            )
        
        # Determine if data meets minimum quality threshold
        is_valid = quality_score >= min_quality_score and total_nan_count == 0
        
        return ValidationResult(
            valid=is_valid,
            quality_score=quality_score,
            issues=issues,
            df=df
        )
    
    def detect_and_correct_outliers(
        self, 
        data: pd.DataFrame,
        columns: Optional[List[str]] = None,
        z_threshold: float = 3.0,
        window_size: int = 5
    ) -> Tuple[pd.DataFrame, Dict[str, Any]]:
        """
        Detect and correct outliers in the data.
        
        Args:
            data: DataFrame to process
            columns: Columns to check for outliers (default: all numeric columns)
            z_threshold: Z-score threshold for outlier detection
            window_size: Window size for rolling median calculation
            
        Returns:
            Tuple of (corrected DataFrame, outlier info)
        """
        if data.empty:
            return data.copy(), {"outlier_count": 0}
            
        df = data.copy()
        
        # Use all numeric columns if none specified
        if columns is None:
            columns = df.select_dtypes(include=np.number).columns.tolist()
        else:
            # Filter to columns that exist and are numeric
            columns = [
                col for col in columns 
                if col in df.columns and pd.api.types.is_numeric_dtype(df[col])
            ]
            
        if not columns:
            return df, {"outlier_count": 0}
            
        # Calculate z-scores for each column
        z_scores = df[columns].sub(df[columns].mean()).div(df[columns].std())
        
        # Find outliers
        outliers = (z_scores.abs() > z_threshold)
        outlier_count = outliers.sum().sum()
        
        if outlier_count == 0:
            return df, {"outlier_count": 0}
            
        # Calculate outlier details
        outlier_details = {
            "outlier_count": int(outlier_count),
            "percentage": float((outlier_count / (len(df) * len(columns))) * 100),
            "by_column": {
                col: int(outliers[col].sum()) for col in columns if outliers[col].sum() > 0
            }
        }
        
        # Correct outliers if auto-correct is enabled
        if self.auto_correct:
            for col in columns:
                col_outliers = outliers[col]
                
                if col_outliers.sum() > 0:
                    # Calculate rolling median
                    rolling_med = df[col].rolling(
                        window=window_size, 
                        center=True, 
                        min_periods=min(2, window_size//2)
                    ).median()
                    
                    # Replace outliers with rolling median
                    df.loc[col_outliers, col] = rolling_med.loc[col_outliers]
            
            outlier_details["corrected"] = True
            
        return df, outlier_details
        
    def interpolate_missing_values(
        self, 
        data: pd.DataFrame,
        columns: Optional[List[str]] = None,
        method: str = 'linear'
    ) -> Tuple[pd.DataFrame, Dict[str, Any]]:
        """
        Interpolate missing values in the data.
        
        Args:
            data: DataFrame to process
            columns: Columns to interpolate (default: all numeric columns)
            method: Interpolation method (linear, time, nearest, etc.)
            
        Returns:
            Tuple of (interpolated DataFrame, missing value info)
        """
        if data.empty:
            return data.copy(), {"nan_count": 0}
            
        df = data.copy()
        
        # Use all numeric columns if none specified
        if columns is None:
            columns = df.select_dtypes(include=np.number).columns.tolist()
        else:
            # Filter to columns that exist and are numeric
            columns = [
                col for col in columns 
                if col in df.columns and pd.api.types.is_numeric_dtype(df[col])
            ]
            
        if not columns:
            return df, {"nan_count": 0}
            
        # Count missing values
        nan_counts = df[columns].isna().sum()
        total_nan_count = nan_counts.sum()
        
        if total_nan_count == 0:
            return df, {"nan_count": 0}
            
        # Calculate missing value details
        missing_details = {
            "nan_count": int(total_nan_count),
            "percentage": float((total_nan_count / (len(df) * len(columns))) * 100),
            "by_column": {
                col: int(count) for col, count in nan_counts.items() if count > 0
            }
        }
        
        # Interpolate missing values
        df[columns] = df[columns].interpolate(method=method)
        
        # Handle edge cases (first/last rows) with forward/backward fill
        df[columns] = df[columns].ffill().bfill()
        
        missing_details["interpolated"] = True
        
        return df, missing_details
        
    def ensure_price_consistency(
        self, 
        data: pd.DataFrame
    ) -> Tuple[pd.DataFrame, Dict[str, Any]]:
        """
        Ensure price consistency (high >= open, high >= close, low <= open, low <= close).
        
        Args:
            data: DataFrame with OHLC data
            
        Returns:
            Tuple of (corrected DataFrame, inconsistency info)
        """
        if data.empty:
            return data.copy(), {"inconsistency_count": 0}
            
        df = data.copy()
        
        # Check for required columns
        required_cols = ['open', 'high', 'low', 'close']
        if not all(col in df.columns for col in required_cols):
            missing = [col for col in required_cols if col not in df.columns]
            return df, {
                "error": f"Missing required columns: {missing}",
                "inconsistency_count": 0
            }
            
        # Find price inconsistencies
        inconsistencies = pd.DataFrame({
            "high_lt_open": df['high'] < df['open'],
            "high_lt_close": df['high'] < df['close'],
            "low_gt_open": df['low'] > df['open'],
            "low_gt_close": df['low'] > df['close']
        })
        
        # Check for any inconsistency
        has_inconsistency = inconsistencies.any(axis=1)
        inconsistency_count = has_inconsistency.sum()
        
        if inconsistency_count == 0:
            return df, {"inconsistency_count": 0}
            
        # Calculate inconsistency details
        inconsistency_details = {
            "inconsistency_count": int(inconsistency_count),
            "percentage": float((inconsistency_count / len(df)) * 100),
            "by_type": {
                "high_lt_open": int(inconsistencies['high_lt_open'].sum()),
                "high_lt_close": int(inconsistencies['high_lt_close'].sum()),
                "low_gt_open": int(inconsistencies['low_gt_open'].sum()),
                "low_gt_close": int(inconsistencies['low_gt_close'].sum())
            }
        }
        
        # Correct inconsistencies if auto-correct is enabled
        if self.auto_correct:
            for idx in df.index[has_inconsistency]:
                row = df.loc[idx]
                
                # Find the min and max of open and close
                min_price = min(row['open'], row['close'])
                max_price = max(row['open'], row['close'])
                
                # Correct high and low
                if row['high'] < max_price:
                    df.loc[idx, 'high'] = max_price
                    
                if row['low'] > min_price:
                    df.loc[idx, 'low'] = min_price
            
            inconsistency_details["corrected"] = True
            
        return df, inconsistency_details
