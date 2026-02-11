"""
Django REST framework serializers for tickets.
"""

from rest_framework import serializers
from .models import Ticket, Category


class CategorySerializer(serializers.ModelSerializer):
    """Serializer for Category model."""
    
    class Meta:
        model = Category
        fields = ['id', 'name', 'department', 'description', 'created_at']
        read_only_fields = ['id', 'created_at']


class TicketSerializer(serializers.ModelSerializer):
    """Serializer for Ticket model."""
    
    class Meta:
        model = Ticket
        fields = [
            'ticket_id', 'customer_name', 'customer_email',
            'subject', 'description', 'predicted_category',
            'confidence_score', 'department', 'priority',
            'sla_hours', 'status', 'created_at', 'updated_at', 'notes'
        ]
        read_only_fields = [
            'ticket_id', 'predicted_category', 'confidence_score',
            'department', 'sla_hours', 'created_at', 'updated_at'
        ]


class TicketSubmissionSerializer(serializers.Serializer):
    """Serializer for ticket submission (input only)."""
    customer_name = serializers.CharField(max_length=200)
    customer_email = serializers.EmailField(required=False, allow_blank=True)
    subject = serializers.CharField(max_length=300)
    description = serializers.CharField()
