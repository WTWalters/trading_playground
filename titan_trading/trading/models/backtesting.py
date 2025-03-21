"""
Backtesting models for TITAN Trading System.

Defines models for backtesting configurations, results, and trades.
"""
from django.db import models
from django.contrib.auth import get_user_model
from .pairs import TradingPair
from .regimes import MarketRegime

User = get_user_model()


class BacktestRun(models.Model):
    """
    Backtest execution record.
    
    Stores configuration and metadata for a backtest run.
    """
    STATUS_CHOICES = [
        ('PENDING', 'Pending'),
        ('RUNNING', 'Running'),
        ('COMPLETED', 'Completed'),
        ('FAILED', 'Failed'),
    ]
    
    name = models.CharField(
        max_length=255,
        help_text="Name of this backtest run"
    )
    
    user = models.ForeignKey(
        User, 
        on_delete=models.CASCADE, 
        related_name='backtests',
        help_text="User who created this backtest"
    )
    
    start_date = models.DateTimeField(
        help_text="Start date for backtest period"
    )
    
    end_date = models.DateTimeField(
        help_text="End date for backtest period"
    )
    
    pairs = models.ManyToManyField(
        TradingPair, 
        related_name='backtests',
        help_text="Trading pairs included in this backtest"
    )
    
    parameters = models.JSONField(
        help_text="Strategy parameters used for this backtest"
    )
    
    created_at = models.DateTimeField(auto_now_add=True)
    
    status = models.CharField(
        max_length=10, 
        choices=STATUS_CHOICES, 
        default='PENDING',
        help_text="Current status of this backtest"
    )
    
    status_message = models.TextField(
        null=True, 
        blank=True,
        help_text="Status message or error details"
    )
    
    description = models.TextField(
        blank=True,
        help_text="Description of this backtest run"
    )
    
    regime_aware = models.BooleanField(
        default=False,
        help_text="Whether this backtest uses regime-aware parameters"
    )
    
    regimes = models.ManyToManyField(
        MarketRegime,
        related_name='backtests',
        blank=True,
        help_text="Market regimes detected during this backtest"
    )
    
    class Meta:
        indexes = [
            models.Index(fields=['user', 'created_at']),
            models.Index(fields=['status']),
        ]
        ordering = ['-created_at']
        
    def __str__(self):
        return f"{self.name} ({self.created_at.strftime('%Y-%m-%d')})"


class BacktestResult(models.Model):
    """
    Backtest performance metrics.
    
    Stores aggregate performance results from a backtest run.
    """
    backtest = models.OneToOneField(
        BacktestRun, 
        on_delete=models.CASCADE, 
        related_name='result',
        help_text="Backtest this result belongs to"
    )
    
    total_return = models.FloatField(
        help_text="Total return percentage"
    )
    
    annualized_return = models.FloatField(
        help_text="Annualized return percentage"
    )
    
    sharpe_ratio = models.FloatField(
        help_text="Sharpe ratio (risk-adjusted return)"
    )
    
    sortino_ratio = models.FloatField(
        null=True, 
        blank=True,
        help_text="Sortino ratio (downside risk-adjusted return)"
    )
    
    max_drawdown = models.FloatField(
        help_text="Maximum drawdown percentage"
    )
    
    win_rate = models.FloatField(
        help_text="Percentage of winning trades"
    )
    
    profit_factor = models.FloatField(
        help_text="Ratio of gross profits to gross losses"
    )
    
    trade_count = models.IntegerField(
        help_text="Total number of trades"
    )
    
    avg_holding_period = models.FloatField(
        null=True, 
        blank=True,
        help_text="Average holding period in days"
    )
    
    detailed_metrics = models.JSONField(
        help_text="Detailed performance metrics"
    )
    
    created_at = models.DateTimeField(auto_now_add=True)
    
    equity_curve = models.JSONField(
        null=True,
        blank=True,
        help_text="Equity curve data points"
    )
    
    monthly_returns = models.JSONField(
        null=True,
        blank=True,
        help_text="Monthly return breakdown"
    )
    
    class Meta:
        ordering = ['-created_at']
        
    def __str__(self):
        return f"Results for {self.backtest.name} ({self.sharpe_ratio:.2f} Sharpe)"


class BacktestTrade(models.Model):
    """
    Individual trades from a backtest.
    
    Records details of each trade executed during a backtest.
    """
    TRADE_TYPES = [
        ('LONG', 'Long'),
        ('SHORT', 'Short'),
    ]
    
    EXIT_REASONS = [
        ('TARGET', 'Price Target'),
        ('STOP', 'Stop Loss'),
        ('SIGNAL', 'Signal'),
        ('TIMEOUT', 'Time Limit'),
    ]
    
    backtest = models.ForeignKey(
        BacktestRun, 
        on_delete=models.CASCADE, 
        related_name='trades',
        help_text="Backtest this trade belongs to"
    )
    
    pair = models.ForeignKey(
        TradingPair, 
        on_delete=models.CASCADE, 
        related_name='backtest_trades',
        help_text="Trading pair for this trade"
    )
    
    trade_type = models.CharField(
        max_length=5, 
        choices=TRADE_TYPES,
        help_text="Type of trade (long or short)"
    )
    
    entry_date = models.DateTimeField(
        help_text="Entry date and time"
    )
    
    exit_date = models.DateTimeField(
        null=True, 
        blank=True,
        help_text="Exit date and time"
    )
    
    entry_price = models.FloatField(
        help_text="Entry price"
    )
    
    exit_price = models.FloatField(
        null=True, 
        blank=True,
        help_text="Exit price"
    )
    
    position_size = models.FloatField(
        help_text="Position size as percentage of portfolio"
    )
    
    pnl = models.FloatField(
        null=True, 
        blank=True,
        help_text="Profit/loss in currency units"
    )
    
    pnl_percent = models.FloatField(
        null=True, 
        blank=True,
        help_text="Profit/loss as percentage"
    )
    
    exit_reason = models.CharField(
        max_length=10, 
        choices=EXIT_REASONS, 
        null=True, 
        blank=True,
        help_text="Reason for exiting the trade"
    )
    
    entry_signal = models.ForeignKey(
        'trading.Signal',
        on_delete=models.SET_NULL,
        null=True,
        blank=True,
        related_name='entry_trades',
        help_text="Signal that triggered entry"
    )
    
    exit_signal = models.ForeignKey(
        'trading.Signal',
        on_delete=models.SET_NULL,
        null=True,
        blank=True,
        related_name='exit_trades',
        help_text="Signal that triggered exit"
    )
    
    regime = models.ForeignKey(
        MarketRegime,
        on_delete=models.SET_NULL,
        null=True,
        blank=True,
        related_name='trades',
        help_text="Market regime during this trade"
    )
    
    notes = models.TextField(
        blank=True,
        help_text="Notes about this trade"
    )
    
    class Meta:
        indexes = [
            models.Index(fields=['backtest', 'pair']),
            models.Index(fields=['entry_date']),
            models.Index(fields=['trade_type']),
            models.Index(fields=['exit_reason']),
        ]
        ordering = ['-entry_date']
        
    def __str__(self):
        return f"{self.trade_type} {self.pair} ({self.entry_date.strftime('%Y-%m-%d')})"
    
    @property
    def duration(self):
        """Calculate trade duration in days."""
        if self.exit_date and self.entry_date:
            return (self.exit_date - self.entry_date).total_seconds() / (60 * 60 * 24)
        return None
    
    @property
    def is_winner(self):
        """Determine if this trade was profitable."""
        if self.pnl is not None:
            return self.pnl > 0
        return None


class WalkForwardTest(models.Model):
    """
    Walk-forward test configuration and results.
    
    Manages walk-forward testing which involves multiple backtest windows.
    """
    STATUS_CHOICES = [
        ('PENDING', 'Pending'),
        ('RUNNING', 'Running'),
        ('COMPLETED', 'Completed'),
        ('FAILED', 'Failed'),
    ]
    
    name = models.CharField(
        max_length=255,
        help_text="Name of this walk-forward test"
    )
    
    user = models.ForeignKey(
        User, 
        on_delete=models.CASCADE, 
        related_name='walk_forward_tests',
        help_text="User who created this test"
    )
    
    start_date = models.DateTimeField(
        help_text="Start date for test period"
    )
    
    end_date = models.DateTimeField(
        help_text="End date for test period"
    )
    
    pairs = models.ManyToManyField(
        TradingPair, 
        related_name='walk_forward_tests',
        help_text="Trading pairs included in this test"
    )
    
    in_sample_size = models.IntegerField(
        help_text="Size of in-sample window in days"
    )
    
    out_of_sample_size = models.IntegerField(
        help_text="Size of out-of-sample window in days"
    )
    
    parameter_ranges = models.JSONField(
        help_text="Parameter ranges for optimization"
    )
    
    optimization_metric = models.CharField(
        max_length=50,
        default='sharpe_ratio',
        help_text="Metric to optimize (e.g., 'sharpe_ratio', 'total_return')"
    )
    
    status = models.CharField(
        max_length=10, 
        choices=STATUS_CHOICES, 
        default='PENDING',
        help_text="Current status of this test"
    )
    
    created_at = models.DateTimeField(auto_now_add=True)
    
    completed_at = models.DateTimeField(
        null=True, 
        blank=True,
        help_text="When the test completed"
    )
    
    results_summary = models.JSONField(
        null=True,
        blank=True,
        help_text="Summary of test results"
    )
    
    class Meta:
        indexes = [
            models.Index(fields=['user', 'created_at']),
            models.Index(fields=['status']),
        ]
        ordering = ['-created_at']
        
    def __str__(self):
        return f"{self.name} ({self.created_at.strftime('%Y-%m-%d')})"


class WalkForwardWindow(models.Model):
    """
    Individual window in a walk-forward test.
    
    Represents a single optimization+validation window in walk-forward testing.
    """
    walk_forward_test = models.ForeignKey(
        WalkForwardTest, 
        on_delete=models.CASCADE, 
        related_name='windows',
        help_text="Walk-forward test this window belongs to"
    )
    
    in_sample_start = models.DateTimeField(
        help_text="Start date of in-sample period"
    )
    
    in_sample_end = models.DateTimeField(
        help_text="End date of in-sample period"
    )
    
    out_of_sample_start = models.DateTimeField(
        help_text="Start date of out-of-sample period"
    )
    
    out_of_sample_end = models.DateTimeField(
        help_text="End date of out-of-sample period"
    )
    
    optimized_parameters = models.JSONField(
        null=True,
        blank=True,
        help_text="Optimized parameters from in-sample period"
    )
    
    is_metrics = models.JSONField(
        null=True,
        blank=True,
        help_text="Performance metrics for in-sample period"
    )
    
    oos_metrics = models.JSONField(
        null=True,
        blank=True,
        help_text="Performance metrics for out-of-sample period"
    )
    
    class Meta:
        indexes = [
            models.Index(fields=['walk_forward_test', 'out_of_sample_start']),
        ]
        ordering = ['out_of_sample_start']
        
    def __str__(self):
        return f"Window {self.in_sample_start.strftime('%Y-%m-%d')} - {self.out_of_sample_end.strftime('%Y-%m-%d')}"
