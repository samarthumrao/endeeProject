"""
Django models for ticket classification system.
"""

from django.db import models
from django.utils import timezone
import uuid


class Category(models.Model):
    """Ticket category model."""
    name = models.CharField(max_length=100, unique=True)
    department = models.CharField(max_length=100)
    description = models.TextField(blank=True)
    created_at = models.DateTimeField(auto_now_add=True)
    
    class Meta:
        verbose_name_plural = "Categories"
        ordering = ['name']
    
    def __str__(self):
        return self.name


class Ticket(models.Model):
    """Support ticket model."""
    
    STATUS_CHOICES = [
        ('new', 'New'),
        ('in_progress', 'In Progress'),
        ('resolved', 'Resolved'),
        ('closed', 'Closed'),
    ]
    
    PRIORITY_CHOICES = [
        ('low', 'Low'),
        ('medium', 'Medium'),
        ('high', 'High'),
    ]
    
    # Basic fields
    ticket_id = models.UUIDField(primary_key=True, default=uuid.uuid4, editable=False)
    customer_name = models.CharField(max_length=200)
    customer_email = models.EmailField(blank=True)
    subject = models.CharField(max_length=300)
    description = models.TextField()
    
    # Classification fields
    predicted_category = models.CharField(max_length=100)
    confidence_score = models.FloatField(default=0.0)
    
    # Routing fields
    department = models.CharField(max_length=100, blank=True)
    priority = models.CharField(max_length=20, choices=PRIORITY_CHOICES, default='medium')
    sla_hours = models.IntegerField(default=24)
    
    # Status fields
    status = models.CharField(max_length=20, choices=STATUS_CHOICES, default='new')
    created_at = models.DateTimeField(default=timezone.now)
    updated_at = models.DateTimeField(auto_now=True)
    
    # Additional metadata
    notes = models.TextField(blank=True)
    
    class Meta:
        ordering = ['-created_at']
    
    def __str__(self):
        return f"{self.subject} - {self.predicted_category}"
