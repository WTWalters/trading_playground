"""
Database router for TimescaleDB integration.

Routes time-series models to TimescaleDB and everything else to default database.
"""


class TimescaleRouter:
    """
    Database router for TimescaleDB integration.
    Routes time-series models to TimescaleDB and everything else to default.
    """
    
    def db_for_read(self, model, **hints):
        """
        Determine which database to use for read operations.
        
        Args:
            model: The model class
            hints: Router hints
            
        Returns:
            Database alias to use
        """
        if hasattr(model, 'is_timescale_model') and model.is_timescale_model:
            return 'timescale'
        return 'default'
    
    def db_for_write(self, model, **hints):
        """
        Determine which database to use for write operations.
        
        Args:
            model: The model class
            hints: Router hints
            
        Returns:
            Database alias to use
        """
        if hasattr(model, 'is_timescale_model') and model.is_timescale_model:
            return 'timescale'
        return 'default'
    
    def allow_relation(self, obj1, obj2, **hints):
        """
        Determine if a relation between obj1 and obj2 is allowed.
        
        Args:
            obj1: First object
            obj2: Second object
            hints: Router hints
            
        Returns:
            True if relation is allowed, False otherwise
        """
        # Allow relations between models in the same database
        # or if either model doesn't have the is_timescale_model attribute
        db1 = 'timescale' if hasattr(obj1, 'is_timescale_model') and obj1.is_timescale_model else 'default'
        db2 = 'timescale' if hasattr(obj2, 'is_timescale_model') and obj2.is_timescale_model else 'default'
        
        return db1 == db2 or db1 == 'default' or db2 == 'default'
    
    def allow_migrate(self, db, app_label, model_name=None, **hints):
        """
        Determine if a migration operation is allowed to run on a database.
        
        Args:
            db: Database alias
            app_label: Application label
            model_name: Model name
            hints: Router hints
            
        Returns:
            True if migration is allowed, False otherwise
        """
        # Only allow TimescaleDB models to migrate to the timescale database
        if db == 'timescale':
            model = hints.get('model')
            return model and hasattr(model, 'is_timescale_model') and model.is_timescale_model
        
        # Allow all non-TimescaleDB models to migrate to default database
        if db == 'default':
            model = hints.get('model')
            return not model or not hasattr(model, 'is_timescale_model') or not model.is_timescale_model
        
        return False
