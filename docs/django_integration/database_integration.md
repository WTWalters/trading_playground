# Database Integration

## Overview

The TITAN Trading System uses a dual database approach:

1. **PostgreSQL**: Standard relational database for user data, configurations, and metadata
2. **TimescaleDB**: Specialized extension for time-series market data

This architecture optimizes both regular relational queries and high-performance time-series operations.

## Database Router Configuration

Django will use a custom database router to direct queries to the appropriate database:

```python
# titan/db_router.py

class TimescaleRouter:
    """
    Database router for TimescaleDB integration.
    Routes time-series models to TimescaleDB and everything else to default.
    """
    
    def db_for_read(self, model, **hints):
        if hasattr(model, 'is_timescale_model') and model.is_timescale_model:
            return 'timescale'
        return 'default'
    
    def db_for_write(self, model, **hints):
        if hasattr(model, 'is_timescale_model') and model.is_timescale_model:
            return 'timescale'
        return 'default'
    
    def allow_relation(self, obj1, obj2, **hints):
        # Allow relations between models in the same database
        # or if either model doesn't have the is_timescale_model attribute
        db1 = 'timescale' if hasattr(obj1, 'is_timescale_model') and obj1.is_timescale_model else 'default'
        db2 = 'timescale' if hasattr(obj2, 'is_timescale_model') and obj2.is_timescale_model else 'default'
        
        return db1 == db2 or db1 == 'default' or db2 == 'default'
    
    def allow_migrate(self, db, app_label, model_name=None, **hints):
        # Only allow TimescaleDB models to migrate to the timescale database
        if db == 'timescale':
            model = hints.get('model')
            return model and hasattr(model, 'is_timescale_model') and model.is_timescale_model
        
        # Allow all non-TimescaleDB models to migrate to default database
        if db == 'default':
            model = hints.get('model')
            return not model or not hasattr(model, 'is_timescale_model') or not model.is_timescale_model
        
        return False
```

## TimescaleDB Integration

TimescaleDB models will use a custom base class:

```python
# common/models.py

from django.db import models

class TimescaleModel(models.Model):
    """
    Base class for TimescaleDB models.
    Provides common functionality and marks models for the database router.
    """
    is_timescale_model = True
    
    class Meta:
        abstract = True
```

## Key Database Models

### Market Data Models

```python
# trading/models/symbols.py

from django.db import models

class Symbol(models.Model):
    """Security identifiers and metadata"""
    ticker = models.CharField(max_length=10, unique=True)
    name = models.CharField(max_length=255)
    sector = models.CharField(max_length=100, null=True, blank=True)
    exchange = models.CharField(max_length=50, null=True, blank=True)
    asset_type = models.CharField(max_length=50, null=True, blank=True)
    is_active = models.BooleanField(default=True)
    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)
    
    class Meta:
        indexes = [
            models.Index(fields=['ticker']),
            models.Index(fields=['sector']),
            models.Index(fields=['is_active']),
        ]
        
    def __str__(self):
        return f"{self.ticker} ({self.name})"
```

```python
# trading/models/prices.py

from django.db import models
from common.models import TimescaleModel
from trading.models.symbols import Symbol

class Price(TimescaleModel):
    """Time-series price data for securities"""
    symbol = models.ForeignKey(Symbol, on_delete=models.CASCADE, related_name='prices')
    timestamp = models.DateTimeField(db_index=True)
    open = models.DecimalField(max_digits=19, decimal_places=6)
    high = models.DecimalField(max_digits=19, decimal_places=6)
    low = models.DecimalField(max_digits=19, decimal_places=6)
    close = models.DecimalField(max_digits=19, decimal_places=6)
    volume = models.BigIntegerField()
    adjusted_close = models.DecimalField(max_digits=19, decimal_places=6, null=True, blank=True)
    source = models.CharField(max_length=50, default='yahoo')
    
    class Meta:
        indexes = [
            models.Index(fields=['symbol', 'timestamp']),
        ]
        constraints = [
            models.UniqueConstraint(fields=['symbol', 'timestamp', 'source'], name='unique_price_point')
        ]
        ordering = ['timestamp']
        
    def __str__(self):
        return f"{self.symbol.ticker} - {self.timestamp} - {self.close}"
```

### Trading Models

```python
# trading/models/pairs.py

from django.db import models
from trading.models.symbols import Symbol

class TradingPair(models.Model):
    """Identified cointegrated pairs"""
    symbol_1 = models.ForeignKey(Symbol, on_delete=models.CASCADE, related_name='pairs_as_first')
    symbol_2 = models.ForeignKey(Symbol, on_delete=models.CASCADE, related_name='pairs_as_second')
    cointegration_pvalue = models.FloatField()
    half_life = models.FloatField()
    correlation = models.FloatField()
    hedge_ratio = models.FloatField()
    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)
    is_active = models.BooleanField(default=True)
    lookback_days = models.IntegerField(default=252)  # Trading days used in calculation
    stability_score = models.FloatField(null=True, blank=True)  # Higher is more stable
    
    class Meta:
        indexes = [
            models.Index(fields=['symbol_1', 'symbol_2']),
            models.Index(fields=['cointegration_pvalue']),
            models.Index(fields=['is_active']),
            models.Index(fields=['stability_score']),
        ]
        constraints = [
            models.UniqueConstraint(fields=['symbol_1', 'symbol_2'], name='unique_trading_pair')
        ]
        
    def __str__(self):
        return f"{self.symbol_1.ticker}/{self.symbol_2.ticker} (p={self.cointegration_pvalue:.4f})"
```

```python
# trading/models/signals.py

from django.db import models
from trading.models.pairs import TradingPair

class Signal(models.Model):
    """Generated trading signals"""
    SIGNAL_TYPES = [
        ('ENTRY_LONG', 'Entry Long'),
        ('ENTRY_SHORT', 'Entry Short'),
        ('EXIT_LONG', 'Exit Long'),
        ('EXIT_SHORT', 'Exit Short'),
        ('STOP_LONG', 'Stop Loss Long'),
        ('STOP_SHORT', 'Stop Loss Short'),
    ]
    
    pair = models.ForeignKey(TradingPair, on_delete=models.CASCADE, related_name='signals')
    timestamp = models.DateTimeField(db_index=True)
    signal_type = models.CharField(max_length=15, choices=SIGNAL_TYPES)
    z_score = models.FloatField()
    spread_value = models.FloatField()
    confidence_score = models.FloatField(null=True, blank=True)
    processed = models.BooleanField(default=False)
    regime = models.ForeignKey('trading.MarketRegime', on_delete=models.SET_NULL, null=True, blank=True)
    created_at = models.DateTimeField(auto_now_add=True)
    
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
```

```python
# trading/models/regimes.py

from django.db import models

class MarketRegime(models.Model):
    """Identified market regimes"""
    REGIME_TYPES = [
        ('CRISIS', 'Crisis'),
        ('STABLE', 'Stable'),
        ('BULL', 'Bull'),
    ]
    
    start_date = models.DateTimeField()
    end_date = models.DateTimeField(null=True, blank=True)
    regime_type = models.CharField(max_length=10, choices=REGIME_TYPES)
    vix_average = models.FloatField()
    volatility_score = models.FloatField()
    macro_score = models.FloatField()
    created_at = models.DateTimeField(auto_now_add=True)
    
    class Meta:
        indexes = [
            models.Index(fields=['start_date', 'end_date']),
            models.Index(fields=['regime_type']),
        ]
        
    def __str__(self):
        end_str = self.end_date.strftime('%Y-%m-%d') if self.end_date else 'current'
        return f"{self.regime_type} ({self.start_date.strftime('%Y-%m-%d')} to {end_str})"
```

### Backtesting Models

```python
# trading/models/backtesting.py

from django.db import models
from django.contrib.auth import get_user_model
from trading.models.pairs import TradingPair

User = get_user_model()

class BacktestRun(models.Model):
    """Backtest execution record"""
    STATUS_CHOICES = [
        ('PENDING', 'Pending'),
        ('RUNNING', 'Running'),
        ('COMPLETED', 'Completed'),
        ('FAILED', 'Failed'),
    ]
    
    name = models.CharField(max_length=255)
    user = models.ForeignKey(User, on_delete=models.CASCADE, related_name='backtests')
    start_date = models.DateTimeField()
    end_date = models.DateTimeField()
    pairs = models.ManyToManyField(TradingPair, related_name='backtests')
    parameters = models.JSONField()
    created_at = models.DateTimeField(auto_now_add=True)
    status = models.CharField(max_length=10, choices=STATUS_CHOICES, default='PENDING')
    status_message = models.TextField(null=True, blank=True)
    
    class Meta:
        indexes = [
            models.Index(fields=['user', 'created_at']),
            models.Index(fields=['status']),
        ]
        
    def __str__(self):
        return f"{self.name} ({self.created_at.strftime('%Y-%m-%d')})"

class BacktestResult(models.Model):
    """Backtest performance metrics"""
    backtest = models.OneToOneField(BacktestRun, on_delete=models.CASCADE, related_name='result')
    total_return = models.FloatField()
    annualized_return = models.FloatField()
    sharpe_ratio = models.FloatField()
    sortino_ratio = models.FloatField(null=True, blank=True)
    max_drawdown = models.FloatField()
    win_rate = models.FloatField()
    profit_factor = models.FloatField()
    trade_count = models.IntegerField()
    avg_holding_period = models.FloatField(null=True, blank=True)
    detailed_metrics = models.JSONField()
    created_at = models.DateTimeField(auto_now_add=True)
    
    def __str__(self):
        return f"Results for {self.backtest.name} ({self.sharpe_ratio:.2f} Sharpe)"

class BacktestTrade(models.Model):
    """Individual trades from a backtest"""
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
    
    backtest = models.ForeignKey(BacktestRun, on_delete=models.CASCADE, related_name='trades')
    pair = models.ForeignKey(TradingPair, on_delete=models.CASCADE, related_name='backtest_trades')
    trade_type = models.CharField(max_length=5, choices=TRADE_TYPES)
    entry_date = models.DateTimeField()
    exit_date = models.DateTimeField(null=True, blank=True)
    entry_price = models.FloatField()
    exit_price = models.FloatField(null=True, blank=True)
    position_size = models.FloatField()
    pnl = models.FloatField(null=True, blank=True)
    pnl_percent = models.FloatField(null=True, blank=True)
    exit_reason = models.CharField(max_length=10, choices=EXIT_REASONS, null=True, blank=True)
    
    class Meta:
        indexes = [
            models.Index(fields=['backtest', 'pair']),
            models.Index(fields=['entry_date']),
            models.Index(fields=['trade_type']),
        ]
        
    def __str__(self):
        return f"{self.trade_type} {self.pair} ({self.entry_date.strftime('%Y-%m-%d')})"
```

### User Models

```python
# users/models.py

from django.db import models
from django.contrib.auth.models import AbstractUser

class User(AbstractUser):
    """Extended user model with trading preferences"""
    
    # Trading preferences
    default_position_size = models.DecimalField(max_digits=5, decimal_places=2, default=2.00)  # Default 2% of portfolio
    risk_tolerance = models.CharField(max_length=10, choices=[
        ('LOW', 'Low'),
        ('MEDIUM', 'Medium'),
        ('HIGH', 'High'),
    ], default='MEDIUM')
    notification_preferences = models.JSONField(default=dict)
    
    class Meta:
        verbose_name = 'User'
        verbose_name_plural = 'Users'

class TradingJournal(models.Model):
    """User trading journal entries"""
    user = models.ForeignKey(User, on_delete=models.CASCADE, related_name='journal_entries')
    entry_date = models.DateTimeField(auto_now_add=True)
    title = models.CharField(max_length=255)
    content = models.TextField()
    trade = models.ForeignKey('trading.BacktestTrade', on_delete=models.SET_NULL, null=True, blank=True)
    tags = models.CharField(max_length=255, blank=True)
    
    class Meta:
        ordering = ['-entry_date']
        
    def __str__(self):
        return f"{self.title} ({self.entry_date.strftime('%Y-%m-%d')})"
```

## Database Migrations

Django migrations will be managed carefully to ensure proper database setup:

1. **Initial Migrations**: Set up the base schema
2. **TimescaleDB Hypertables**: Custom migrations for TimescaleDB
3. **Indexing Strategy**: Optimize for common query patterns
4. **Data Retention Policies**: Configure TimescaleDB's data retention

## TimescaleDB Custom Migration

```python
# trading/migrations/XXXX_create_timescale_hypertable.py

from django.db import migrations

class Migration(migrations.Migration):
    dependencies = [
        ('trading', 'XXXX_previous_migration'),
    ]

    operations = [
        migrations.RunSQL(
            sql="""
            SELECT create_hypertable('trading_price', 'timestamp', 
                                     chunk_time_interval => interval '1 day');
            """,
            reverse_sql="SELECT 1;"  # No easy way to reverse this
        ),
        migrations.RunSQL(
            sql="""
            SELECT add_compression_policy('trading_price', interval '7 days');
            """,
            reverse_sql="SELECT 1;"
        )
    ]
```

## Performance Considerations

1. **Indexing Strategy**: Optimized for common queries on time-series data
2. **Query Optimization**: Use TimescaleDB's specialized functions for time-series queries
3. **Compression Policies**: Configure TimescaleDB compression for historical data
4. **Cache Configuration**: Use Django's cache framework for frequently accessed data
5. **Batch Processing**: Implement batch processing for bulk operations
6. **Connection Pooling**: Configure Postgres connection pooling for optimal database connections

## Data Migration Strategy

1. **Initial Data Import**: ETL process for importing historical market data
2. **Incremental Updates**: Daily/hourly update processes for new market data
3. **Data Validation**: Validation checks during migration to ensure data integrity
4. **Downtime Minimization**: Migration strategies to minimize application downtime

## Database Settings

```python
# titan/settings/base.py

DATABASES = {
    'default': {
        'ENGINE': 'django.db.backends.postgresql',
        'NAME': os.getenv('DB_NAME', 'titan_django'),
        'USER': os.getenv('DB_USER', 'postgres'),
        'PASSWORD': os.getenv('DB_PASSWORD', ''),
        'HOST': os.getenv('DB_HOST', 'localhost'),
        'PORT': os.getenv('DB_PORT', '5432'),
        'CONN_MAX_AGE': 60,
        'OPTIONS': {
            'connect_timeout': 10,
        }
    },
    'timescale': {
        'ENGINE': 'timescale.db.backends.postgresql',
        'NAME': os.getenv('TIMESCALE_NAME', 'titan_timescale'),
        'USER': os.getenv('TIMESCALE_USER', 'postgres'),
        'PASSWORD': os.getenv('TIMESCALE_PASSWORD', ''),
        'HOST': os.getenv('TIMESCALE_HOST', 'localhost'),
        'PORT': os.getenv('TIMESCALE_PORT', '5432'),
        'CONN_MAX_AGE': 60,
        'OPTIONS': {
            'connect_timeout': 10,
        }
    }
}

DATABASE_ROUTERS = ['titan.db_router.TimescaleRouter']

# Cache configuration
CACHES = {
    'default': {
        'BACKEND': 'django_redis.cache.RedisCache',
        'LOCATION': os.getenv('REDIS_URL', 'redis://localhost:6379/1'),
        'OPTIONS': {
            'CLIENT_CLASS': 'django_redis.client.DefaultClient',
        }
    }
}
```
