"""
Pair trading models for TITAN Trading System.

Defines models for cointegrated pairs and related data.
"""
from django.db import models
from .symbols import Symbol


class TradingPair(models.Model):
    """
    Identified cointegrated pairs for statistical arbitrage.
    
    Stores pairs of securities that have been identified as cointegrated,
    along with their statistical properties and trading parameters.
    """
    symbol_1 = models.ForeignKey(
        Symbol, 
        on_delete=models.CASCADE, 
        related_name='pairs_as_first',
        help_text="First symbol in the pair"
    )
    
    symbol_2 = models.ForeignKey(
        Symbol, 
        on_delete=models.CASCADE, 
        related_name='pairs_as_second',
        help_text="Second symbol in the pair"
    )
    
    cointegration_pvalue = models.FloatField(
        help_text="P-value from cointegration test"
    )
    
    half_life = models.FloatField(
        help_text="Half-life of mean reversion in days"
    )
    
    correlation = models.FloatField(
        help_text="Pearson correlation coefficient between the securities"
    )
    
    hedge_ratio = models.FloatField(
        help_text="Hedge ratio for creating the spread"
    )
    
    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)
    
    is_active = models.BooleanField(
        default=True,
        help_text="Whether this pair is currently active for trading"
    )
    
    lookback_days = models.IntegerField(
        default=252,
        help_text="Trading days used in calculation"
    )
    
    stability_score = models.FloatField(
        null=True, 
        blank=True,
        help_text="Stability score of the cointegration relationship (higher is more stable)"
    )
    
    last_spread_value = models.FloatField(
        null=True, 
        blank=True,
        help_text="Most recent spread value"
    )
    
    last_zscore = models.FloatField(
        null=True, 
        blank=True,
        help_text="Most recent Z-score of the spread"
    )
    
    last_updated_spread = models.DateTimeField(
        null=True, 
        blank=True,
        help_text="When the spread was last updated"
    )
    
    class Meta:
        indexes = [
            models.Index(fields=['symbol_1', 'symbol_2']),
            models.Index(fields=['cointegration_pvalue']),
            models.Index(fields=['is_active']),
            models.Index(fields=['stability_score']),
        ]
        constraints = [
            models.UniqueConstraint(
                fields=['symbol_1', 'symbol_2'], 
                name='unique_trading_pair'
            )
        ]
        verbose_name = 'Trading Pair'
        verbose_name_plural = 'Trading Pairs'
        
    def __str__(self):
        return f"{self.symbol_1.ticker}/{self.symbol_2.ticker} (p={self.cointegration_pvalue:.4f})"


class PairSpread(models.Model):
    """
    Historical spread values for trading pairs.
    
    Tracks the spread between securities in a pair over time,
    which is used for mean reversion trading signals.
    """
    pair = models.ForeignKey(
        TradingPair, 
        on_delete=models.CASCADE, 
        related_name='spreads',
        help_text="Trading pair this spread belongs to"
    )
    
    timestamp = models.DateTimeField(
        help_text="Timestamp for this spread value"
    )
    
    spread_value = models.FloatField(
        help_text="Raw spread value"
    )
    
    z_score = models.FloatField(
        help_text="Z-score of the spread"
    )
    
    mean = models.FloatField(
        help_text="Rolling mean used for normalization"
    )
    
    std_dev = models.FloatField(
        help_text="Rolling standard deviation used for normalization"
    )
    
    lookback_window = models.IntegerField(
        default=20,
        help_text="Window size for rolling statistics"
    )
    
    class Meta:
        indexes = [
            models.Index(fields=['pair', 'timestamp']),
            models.Index(fields=['z_score']),
        ]
        constraints = [
            models.UniqueConstraint(
                fields=['pair', 'timestamp'], 
                name='unique_pair_spread'
            )
        ]
        ordering = ['-timestamp']
        
    def __str__(self):
        return f"{self.pair} - {self.timestamp} - Z: {self.z_score:.2f}"
