"""
Admin interface for ticket management.
"""

from django.contrib import admin
from .models import Ticket, Category


@admin.register(Category)
class CategoryAdmin(admin.ModelAdmin):
    """Admin interface for categories."""
    list_display = ['name', 'department', 'created_at']
    search_fields = ['name', 'department']
    ordering = ['name']


@admin.register(Ticket)
class TicketAdmin(admin.ModelAdmin):
    """Admin interface for tickets."""
    list_display = [
        'ticket_id', 'subject', 'predicted_category',
        'confidence_score', 'priority', 'status', 'created_at'
    ]
    list_filter = ['status', 'priority', 'predicted_category', 'created_at']
    search_fields = ['subject', 'description', 'customer_name', 'customer_email']
    readonly_fields = [
        'ticket_id', 'predicted_category', 'confidence_score',
        'department', 'sla_hours', 'created_at', 'updated_at'
    ]
    ordering = ['-created_at']
    
    fieldsets = (
        ('Customer Information', {
            'fields': ('customer_name', 'customer_email')
        }),
        ('Ticket Details', {
            'fields': ('subject', 'description', 'status', 'notes')
        }),
        ('Classification', {
            'fields': ('predicted_category', 'confidence_score')
        }),
        ('Routing', {
            'fields': ('department', 'priority', 'sla_hours')
        }),
        ('Metadata', {
            'fields': ('ticket_id', 'created_at', 'updated_at'),
            'classes': ('collapse',)
        }),
    )
