"""
Symbol models for TITAN Trading System.

Defines models for securities and their metadata.
"""
from django.db import models


class Symbol(models.Model):
    """
    Security identifiers and metadata.
    
    Represents tradable securities with their identifying information
    and classification data.
    """
    ticker = models.CharField(
        max_length=10, 
        unique=True,
        help_text="Trading symbol (e.g., 'AAPL')"
    )
    
    name = models.CharField(
        max_length=255,
        help_text="Full name of the security"
    )
    
    sector = models.CharField(
        max_length=100, 
        null=True, 
        blank=True,
        help_text="Industry sector"
    )
    
    exchange = models.CharField(
        max_length=50, 
        null=True, 
        blank=True,
        help_text="Exchange where the security is traded"
    )
    
    asset_type = models.CharField(
        max_length=50, 
        null=True, 
        blank=True,
        help_text="Type of asset (e.g., 'EQUITY', 'ETF', 'CRYPTO')"
    )
    
    is_active = models.BooleanField(
        default=True,
        help_text="Whether the symbol is currently active for trading"
    )
    
    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)
    
    class Meta:
        indexes = [
            models.Index(fields=['ticker']),
            models.Index(fields=['sector']),
            models.Index(fields=['is_active']),
        ]
        verbose_name = 'Symbol'
        verbose_name_plural = 'Symbols'
        
    def __str__(self):
        return f"{self.ticker} ({self.name})"
