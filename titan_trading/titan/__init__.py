"""
TITAN Trading System Django project.

This module contains the main Django project for the TITAN Trading System.
"""

# Configure Celery
from .celery import app as celery_app

__all__ = ('celery_app',)
