"""
Admin configuration for users app.
"""
from django.contrib import admin
from django.contrib.auth.admin import UserAdmin
from .models import User, TradingJournal


@admin.register(User)
class CustomUserAdmin(UserAdmin):
    """Admin configuration for User model."""
    list_display = ('username', 'email', 'first_name', 'last_name', 'risk_tolerance')
    list_filter = ('is_staff', 'is_superuser', 'is_active', 'risk_tolerance')
    fieldsets = UserAdmin.fieldsets + (
        ('Trading Preferences', {
            'fields': ('default_position_size', 'risk_tolerance', 'notification_preferences'),
        }),
    )


@admin.register(TradingJournal)
class TradingJournalAdmin(admin.ModelAdmin):
    """Admin configuration for TradingJournal model."""
    list_display = ('title', 'user', 'entry_date', 'trade')
    list_filter = ('user', 'entry_date')
    search_fields = ('title', 'content', 'tags')
    date_hierarchy = 'entry_date'
