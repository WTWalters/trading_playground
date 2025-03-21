"""
App configuration for TITAN Trading System users app.
"""
from django.apps import AppConfig


class UsersConfig(AppConfig):
    """
    Configuration for the users app.
    """
    default_auto_field = 'django.db.models.BigAutoField'
    name = 'users'
    
    def ready(self):
        """
        Initialize app when Django starts.
        """
        pass  # Import signal handlers if needed
