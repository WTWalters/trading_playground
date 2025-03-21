"""
Common models for TITAN Trading System.

Contains base model classes and shared functionality.
"""
from django.db import models


class TimescaleModel(models.Model):
    """
    Base class for TimescaleDB models.
    Provides common functionality and marks models for the database router.
    """
    is_timescale_model = True
    
    class Meta:
        abstract = True
