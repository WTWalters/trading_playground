"""
Django settings module for TITAN Trading System.

Imports from base, development, or production settings based on environment.
"""
import os

# Default to development settings
from .base import *

# Switch to production settings if DJANGO_ENV is set to 'production'
if os.environ.get('DJANGO_ENV') == 'production':
    from .production import *
else:
    from .development import *
