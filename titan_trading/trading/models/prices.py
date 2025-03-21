"""
Price models for TITAN Trading System.

Defines time-series price data models using TimescaleDB.
"""
from django.db import models
from common.models import TimescaleModel
from .symbols import Symbol


class Price(TimescaleModel):
    """
    Time-series price data for securities.
    
    Stores OHLCV (Open, High, Low, Close, Volume) data for symbols
    with timestamp indexing. Uses TimescaleDB for efficient time-series storage.
    """
    symbol = models.ForeignKey(
        Symbol, 
        on_delete=models.CASCADE, 
        related_name='prices',
        help_text="Symbol this price data belongs to"
    )
    
    timestamp = models.DateTimeField(
        db_index=True,
        help_text="Timestamp for this price point"
    )
    
    open = models.DecimalField(
        max_digits=19, 
        decimal_places=6,
        help_text="Opening price"
    )
    
    high = models.DecimalField(
        max_digits=19, 
        decimal_places=6,
        help_text="Highest price during the period"
    )
    
    low = models.DecimalField(
        max_digits=19, 
        decimal_places=6,
        help_text="Lowest price during the period"
    )
    
    close = models.DecimalField(
        max_digits=19, 
        decimal_places=6,
        help_text="Closing price"
    )
    
    volume = models.BigIntegerField(
        help_text="Trading volume during the period"
    )
    
    adjusted_close = models.DecimalField(
        max_digits=19, 
        decimal_places=6, 
        null=True, 
        blank=True,
        help_text="Adjusted closing price (accounting for splits, dividends, etc.)"
    )
    
    source = models.CharField(
        max_length=50, 
        default='yahoo',
        help_text="Data source (e.g., 'yahoo', 'polygon', 'alpha_vantage')"
    )
    
    timeframe = models.CharField(
        max_length=10,
        default='1d',
        help_text="Time period (e.g., '1m', '5m', '1h', '1d')"
    )
    
    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)
    
    class Meta:
        indexes = [
            models.Index(fields=['symbol', 'timestamp']),
            models.Index(fields=['source']),
            models.Index(fields=['timeframe']),
        ]
        constraints = [
            models.UniqueConstraint(
                fields=['symbol', 'timestamp', 'source', 'timeframe'], 
                name='unique_price_point'
            )
        ]
        ordering = ['timestamp']
        
    def __str__(self):
        return f"{self.symbol.ticker} - {self.timestamp} - {self.close}"


class AggregatedPrice(TimescaleModel):
    """
    Aggregated time-series price data.
    
    Stores pre-calculated aggregations of price data at various timeframes.
    Uses TimescaleDB continuous aggregates for efficient storage and retrieval.
    """
    symbol = models.ForeignKey(
        Symbol, 
        on_delete=models.CASCADE, 
        related_name='aggregated_prices',
        help_text="Symbol this aggregated data belongs to"
    )
    
    time_bucket = models.DateTimeField(
        db_index=True,
        help_text="Start of the time bucket for this aggregation"
    )
    
    timeframe = models.CharField(
        max_length=20,
        help_text="Aggregation period (e.g., '1h', '1d', '1w')"
    )
    
    open = models.DecimalField(
        max_digits=19, 
        decimal_places=6,
        help_text="Opening price in the period"
    )
    
    high = models.DecimalField(
        max_digits=19, 
        decimal_places=6,
        help_text="Highest price during the period"
    )
    
    low = models.DecimalField(
        max_digits=19, 
        decimal_places=6,
        help_text="Lowest price during the period"
    )
    
    close = models.DecimalField(
        max_digits=19, 
        decimal_places=6,
        help_text="Closing price"
    )
    
    volume = models.BigIntegerField(
        help_text="Total volume during the period"
    )
    
    vwap = models.DecimalField(
        max_digits=19, 
        decimal_places=6, 
        null=True, 
        blank=True,
        help_text="Volume-weighted average price"
    )
    
    count = models.IntegerField(
        help_text="Number of data points in this bucket"
    )
    
    class Meta:
        indexes = [
            models.Index(fields=['symbol', 'time_bucket']),
            models.Index(fields=['timeframe']),
        ]
        constraints = [
            models.UniqueConstraint(
                fields=['symbol', 'time_bucket', 'timeframe'], 
                name='unique_aggregated_price'
            )
        ]
        ordering = ['time_bucket']
        
    def __str__(self):
        return f"{self.symbol.ticker} - {self.timeframe} - {self.time_bucket}"
