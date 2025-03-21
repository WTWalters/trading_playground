"""
Celery application configuration for TITAN Trading System.

This module initializes the Celery application and configures it for use with Django.
"""
import os
from celery import Celery

# Set the default Django settings module
os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'titan.settings')

# Create the Celery application
app = Celery('titan')

# Configure Celery using Django settings
app.config_from_object('django.conf:settings', namespace='CELERY')

# Auto-discover tasks in all installed apps
app.autodiscover_tasks()


@app.task(bind=True)
def debug_task(self):
    """
    Debug task for Celery.
    
    Used to test that Celery is working properly.
    """
    print(f'Request: {self.request!r}')
