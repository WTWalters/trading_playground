from datetime import datetime, timedelta
from typing import Optional, Tuple

import pytz


def parse_timeframe(timeframe: str) -> Tuple[int, str]:
    """Parse timeframe string into value and unit.
    
    Args:
        timeframe: String like '1m', '5m', '1h', '1d'
        
    Returns:
        Tuple of (value, unit)
    """
    units = {
        'm': 'minutes',
        'h': 'hours',
        'd': 'days'
    }
    value = int(timeframe[:-1])
    unit = timeframe[-1]
    if unit not in units:
        raise ValueError(f"Invalid timeframe unit: {unit}")
    return value, units[unit]


def convert_timestamp(timestamp: any, timezone: Optional[str] = None) -> datetime:
    """Convert various timestamp formats to datetime.
    
    Args:
        timestamp: Input timestamp (string, int, or datetime)
        timezone: Optional timezone string (e.g., 'UTC', 'America/New_York')
        
    Returns:
        datetime object
    """
    if isinstance(timestamp, datetime):
        dt = timestamp
    elif isinstance(timestamp, (int, float)):
        dt = datetime.fromtimestamp(timestamp)
    elif isinstance(timestamp, str):
        dt = datetime.fromisoformat(timestamp.replace('Z', '+00:00'))
    else:
        raise ValueError(f"Unsupported timestamp format: {type(timestamp)}")
        
    if timezone:
        dt = dt.astimezone(pytz.timezone(timezone))
    return dt
