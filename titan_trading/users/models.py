"""
User models for TITAN Trading System.

Extended user model and related models for trading preferences.
"""
from django.db import models
from django.contrib.auth.models import AbstractUser


class User(AbstractUser):
    """
    Extended user model with trading preferences.
    """
    # Trading preferences
    default_position_size = models.DecimalField(
        max_digits=5, 
        decimal_places=2, 
        default=2.00,
        help_text="Default position size as percentage of portfolio"
    )
    
    risk_tolerance = models.CharField(
        max_length=10, 
        choices=[
            ('LOW', 'Low'),
            ('MEDIUM', 'Medium'),
            ('HIGH', 'High'),
        ], 
        default='MEDIUM',
        help_text="User's risk tolerance level"
    )
    
    notification_preferences = models.JSONField(
        default=dict,
        help_text="User's notification preferences"
    )
    
    class Meta:
        verbose_name = 'User'
        verbose_name_plural = 'Users'
        
    def __str__(self):
        return self.username


class TradingJournal(models.Model):
    """
    User trading journal entries.
    
    Allows users to record notes and insights about trades.
    """
    user = models.ForeignKey(
        User, 
        on_delete=models.CASCADE, 
        related_name='journal_entries'
    )
    
    entry_date = models.DateTimeField(auto_now_add=True)
    title = models.CharField(max_length=255)
    content = models.TextField()
    
    # Will link to BacktestTrade once it's defined
    trade = models.ForeignKey(
        'trading.BacktestTrade', 
        on_delete=models.SET_NULL, 
        null=True, 
        blank=True
    )
    
    tags = models.CharField(max_length=255, blank=True)
    
    class Meta:
        ordering = ['-entry_date']
        
    def __str__(self):
        return f"{self.title} ({self.entry_date.strftime('%Y-%m-%d')})"
