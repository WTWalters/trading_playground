"""
Market regime models for TITAN Trading System.

Defines models for market regimes and regime detection.
"""
from django.db import models


class MarketRegime(models.Model):
    """
    Identified market regimes.
    
    Represents different market conditions that affect trading strategies.
    Regimes are time periods with distinct characteristics like volatility and trend.
    """
    REGIME_TYPES = [
        ('CRISIS', 'Crisis'),
        ('STABLE', 'Stable'),
        ('BULL', 'Bull'),
        ('BEAR', 'Bear'),
        ('VOLATILE', 'Volatile'),
        ('SIDEWAYS', 'Sideways'),
        ('RECOVERY', 'Recovery'),
    ]
    
    start_date = models.DateTimeField(
        help_text="Start date of the regime period"
    )
    
    end_date = models.DateTimeField(
        null=True, 
        blank=True,
        help_text="End date of the regime period (null if current)"
    )
    
    regime_type = models.CharField(
        max_length=15, 
        choices=REGIME_TYPES,
        help_text="Type of market regime"
    )
    
    vix_average = models.FloatField(
        help_text="Average VIX value during this regime"
    )
    
    volatility_score = models.FloatField(
        help_text="Volatility score for this regime"
    )
    
    macro_score = models.FloatField(
        help_text="Macroeconomic conditions score"
    )
    
    created_at = models.DateTimeField(auto_now_add=True)
    
    description = models.TextField(
        blank=True,
        help_text="Description of this market regime"
    )
    
    key_events = models.JSONField(
        default=list,
        help_text="List of key market events during this regime"
    )
    
    data_summary = models.JSONField(
        default=dict,
        help_text="Summary statistics for this regime"
    )
    
    class Meta:
        indexes = [
            models.Index(fields=['start_date', 'end_date']),
            models.Index(fields=['regime_type']),
        ]
        ordering = ['-start_date']
        
    def __str__(self):
        end_str = self.end_date.strftime('%Y-%m-%d') if self.end_date else 'current'
        return f"{self.regime_type} ({self.start_date.strftime('%Y-%m-%d')} to {end_str})"
    
    @property
    def is_active(self):
        """Check if this is the currently active regime."""
        return self.end_date is None


class RegimeTransition(models.Model):
    """
    Transitions between market regimes.
    
    Captures information about changes from one regime to another,
    which helps in analyzing regime shift patterns.
    """
    from_regime = models.ForeignKey(
        MarketRegime, 
        on_delete=models.CASCADE, 
        related_name='transitions_from',
        help_text="Source regime"
    )
    
    to_regime = models.ForeignKey(
        MarketRegime, 
        on_delete=models.CASCADE, 
        related_name='transitions_to',
        help_text="Target regime"
    )
    
    transition_date = models.DateTimeField(
        help_text="Date of the regime transition"
    )
    
    transition_score = models.FloatField(
        help_text="Score indicating the clarity/confidence of transition"
    )
    
    notes = models.TextField(
        blank=True,
        help_text="Notes about this transition"
    )
    
    class Meta:
        indexes = [
            models.Index(fields=['transition_date']),
            models.Index(fields=['from_regime', 'to_regime']),
        ]
        ordering = ['-transition_date']
        
    def __str__(self):
        return f"{self.from_regime.regime_type} -> {self.to_regime.regime_type} ({self.transition_date.strftime('%Y-%m-%d')})"
