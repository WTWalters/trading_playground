from typing import List, Optional, Union
import pandas as pd


def validate_ohlcv_data(
    df: pd.DataFrame,
    required_columns: Optional[List[str]] = None
) -> bool:
    """Validate OHLCV data meets requirements.
    
    Args:
        df: DataFrame to validate
        required_columns: Optional list of required columns
        
    Returns:
        True if valid, raises ValueError otherwise
    """
    if required_columns is None:
        required_columns = ['timestamp', 'open', 'high', 'low', 'close', 'volume']
        
    # Check required columns
    missing_cols = set(required_columns) - set(df.columns)
    if missing_cols:
        raise ValueError(f"Missing required columns: {missing_cols}")
        
    # Validate price relationships
    if not (
        (df['high'] >= df['low']).all() and
        (df['high'] >= df['open']).all() and
        (df['high'] >= df['close']).all() and
        (df['low'] <= df['open']).all() and
        (df['low'] <= df['close']).all()
    ):
        raise ValueError("Invalid price relationships in OHLCV data")
        
    return True
