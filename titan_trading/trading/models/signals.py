"""
Trading signal models for TITAN Trading System.

Defines models for trading signals generated by the system.
"""
from django.db import models
from .pairs import TradingPair


class Signal(models.Model):
    """
    Generated trading signals.
    
    Represents entry, exit, and risk management signals for trading pairs.
    """
    SIGNAL_TYPES = [
        ('ENTRY_LONG', 'Entry Long'),
        ('ENTRY_SHORT', 'Entry Short'),
        ('EXIT_LONG', 'Exit Long'),
        ('EXIT_SHORT', 'Exit Short'),
        ('STOP_LONG', 'Stop Loss Long'),
        ('STOP_SHORT', 'Stop Loss Short'),
    ]
    
    pair = models.ForeignKey(
        TradingPair, 
        on_delete=models.CASCADE, 
        related_name='signals',
        help_text="Trading pair this signal is for"
    )
    
    timestamp = models.DateTimeField(
        db_index=True,
        help_text="Timestamp when the signal was generated"
    )
    
    signal_type = models.CharField(
        max_length=15, 
        choices=SIGNAL_TYPES,
        help_text="Type of trading signal"
    )
    
    z_score = models.FloatField(
        help_text="Z-score that triggered the signal"
    )
    
    spread_value = models.FloatField(
        help_text="Spread value at signal generation"
    )
    
    confidence_score = models.FloatField(
        null=True, 
        blank=True,
        help_text="Confidence score for the signal (higher is more confident)"
    )
    
    processed = models.BooleanField(
        default=False,
        help_text="Whether this signal has been processed by trading systems"
    )
    
    regime = models.ForeignKey(
        'trading.MarketRegime', 
        on_delete=models.SET_NULL, 
        null=True, 
        blank=True,
        help_text="Market regime when signal was generated"
    )
    
    created_at = models.DateTimeField(auto_now_add=True)
    
    notes = models.TextField(
        blank=True,
        help_text="Additional notes about the signal"
    )
    
    parameters = models.JSONField(
        default=dict,
        help_text="Parameter values used to generate this signal"
    )
    
    class Meta:
        indexes = [
            models.Index(fields=['pair', 'timestamp']),
            models.Index(fields=['signal_type']),
            models.Index(fields=['processed']),
            models.Index(fields=['regime']),
        ]
        ordering = ['-timestamp']
        
    def __str__(self):
        return f"{self.pair} - {self.signal_type} - {self.timestamp}"
