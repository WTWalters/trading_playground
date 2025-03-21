# Extending the TITAN Trading System Django Models

This guide explains how to extend the Django models in the TITAN Trading System to add new functionality.

## Understanding the Model Structure

The system uses a dual database approach:

1. **Regular PostgreSQL** for standard models
2. **TimescaleDB** for time-series data models

### Regular Models

Regular models are stored in the default PostgreSQL database and follow standard Django model patterns:

```python
from django.db import models

class MyRegularModel(models.Model):
    name = models.CharField(max_length=100)
    created_at = models.DateTimeField(auto_now_add=True)
    
    class Meta:
        indexes = [
            models.Index(fields=['name']),
        ]
```

### TimescaleDB Models

Time-series models should inherit from `TimescaleModel` to ensure they're routed to TimescaleDB:

```python
from django.db import models
from common.models import TimescaleModel

class MyTimeSeriesModel(TimescaleModel):
    timestamp = models.DateTimeField(db_index=True)
    value = models.FloatField()
    
    class Meta:
        indexes = [
            models.Index(fields=['timestamp']),
        ]
```

## Adding a New Model

### Step 1: Create the Model Class

Create a new Python file in the appropriate `models` directory or add to an existing file:

```python
# trading/models/my_new_model.py
from django.db import models
from .symbols import Symbol

class MyNewModel(models.Model):
    name = models.CharField(max_length=100)
    symbol = models.ForeignKey(Symbol, on_delete=models.CASCADE, related_name='my_new_models')
    value = models.FloatField()
    created_at = models.DateTimeField(auto_now_add=True)
    
    class Meta:
        indexes = [
            models.Index(fields=['name']),
            models.Index(fields=['symbol', 'created_at']),
        ]
        
    def __str__(self):
        return f"{self.name} - {self.symbol.ticker}"
```

### Step 2: Update __init__.py

Add your model to the `__init__.py` file to make it importable:

```python
# trading/models/__init__.py
from .symbols import Symbol
from .prices import Price
# ... other imports
from .my_new_model import MyNewModel

__all__ = [
    'Symbol',
    'Price',
    # ... other models
    'MyNewModel',
]
```

### Step 3: Create and Run Migrations

Create and apply migrations to add the model to the database:

```bash
python manage.py makemigrations
python manage.py migrate
```

### Step 4: Register with Admin (Optional)

Register your model with the Django admin site:

```python
# trading/admin.py
from django.contrib import admin
from .models import MyNewModel

@admin.register(MyNewModel)
class MyNewModelAdmin(admin.ModelAdmin):
    list_display = ('name', 'symbol', 'value', 'created_at')
    list_filter = ('symbol',)
    search_fields = ('name', 'symbol__ticker')
```

## Creating a TimescaleDB Model

### Step 1: Create the Model Class

Inherit from TimescaleModel and define your time-series fields:

```python
# trading/models/my_timeseries.py
from django.db import models
from common.models import TimescaleModel
from .symbols import Symbol

class MyTimeSeries(TimescaleModel):
    symbol = models.ForeignKey(Symbol, on_delete=models.CASCADE, related_name='my_timeseries')
    timestamp = models.DateTimeField(db_index=True)
    value = models.FloatField()
    source = models.CharField(max_length=50, default='system')
    
    class Meta:
        indexes = [
            models.Index(fields=['symbol', 'timestamp']),
        ]
        constraints = [
            models.UniqueConstraint(
                fields=['symbol', 'timestamp', 'source'], 
                name='unique_my_timeseries'
            )
        ]
        ordering = ['timestamp']
        
    def __str__(self):
        return f"{self.symbol.ticker} - {self.timestamp} - {self.value}"
```

### Step 2: Create Migration for TimescaleDB Hypertable

After creating the initial migration, create a custom migration to convert the table to a TimescaleDB hypertable:

```python
# trading/migrations/XXXX_create_my_timeseries_hypertable.py
from django.db import migrations

class Migration(migrations.Migration):
    dependencies = [
        ('trading', 'XXXX_previous_migration'),  # Update this to match your migration
    ]

    operations = [
        migrations.RunSQL(
            sql="""
            SELECT create_hypertable('trading_mytimeseries', 'timestamp', 
                                     chunk_time_interval => interval '1 day',
                                     if_not_exists => TRUE);
            """,
            reverse_sql="SELECT 1;"  # No easy way to reverse this
        ),
    ]
```

### Step 3: Update Admin

Register the time-series model with admin:

```python
@admin.register(MyTimeSeries)
class MyTimeSeriesAdmin(admin.ModelAdmin):
    list_display = ('symbol', 'timestamp', 'value', 'source')
    list_filter = ('source',)
    search_fields = ('symbol__ticker',)
    date_hierarchy = 'timestamp'
    raw_id_fields = ('symbol',)
```

## Best Practices

1. **Indexing**: Always add appropriate indexes for fields commonly used in queries
2. **ForeignKey Relationships**: Use descriptive `related_name` for reverse relationships
3. **Constraints**: Use unique constraints to prevent duplicate data
4. **String Representations**: Implement meaningful `__str__` methods
5. **TimescaleDB**: For time-series data with timestamps, always use TimescaleModel
6. **Field Choices**: Use choice fields for fields with a limited set of possible values
7. **Help Text**: Add help_text to fields to document their purpose
8. **JSONField**: Use JSONField for structured data that doesn't need relational queries
