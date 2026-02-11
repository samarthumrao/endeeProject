"""
Django views for ticket classification system.
"""

import sys
import os

# Add project root to Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from rest_framework import viewsets, status
from rest_framework.decorators import action
from rest_framework.response import Response
from django.shortcuts import render

from .models import Ticket, Category
from .serializers import TicketSerializer, CategorySerializer, TicketSubmissionSerializer

try:
    from core.routing_engine import get_routing_engine
except ImportError:
    get_routing_engine = None


class TicketViewSet(viewsets.ModelViewSet):
    """ViewSet for managing tickets."""
    queryset = Ticket.objects.all()
    serializer_class = TicketSerializer
    
    @action(detail=False, methods=['post'])
    def submit(self, request):
        """
        Submit a new ticket for classification and routing.
        """
        # Validate input
        submission_serializer = TicketSubmissionSerializer(data=request.data)
        if not submission_serializer.is_valid():
            return Response(
                submission_serializer.errors,
                status=status.HTTP_400_BAD_REQUEST
            )
        
        data = submission_serializer.validated_data
        
        # Combine subject and description for classification
        ticket_text = f"{data['subject']} | {data['description']}"
        
        # Classify and route the ticket
        try:
            if get_routing_engine:
                routing_engine = get_routing_engine()
                routing_result = routing_engine.route_ticket(ticket_text)
                
                # Create ticket with classification results
                ticket = Ticket.objects.create(
                    customer_name=data['customer_name'],
                    customer_email=data.get('customer_email', ''),
                    subject=data['subject'],
                    description=data['description'],
                    predicted_category=routing_result.get('category', 'Unknown'),
                    confidence_score=routing_result.get('confidence', 0.0),
                    department=routing_result.get('department', 'General Support'),
                    priority=routing_result.get('priority', 'medium'),
                    sla_hours=routing_result.get('sla_hours', 24)
                )
                
                # Prepare response
                response_data = {
                    'ticket': TicketSerializer(ticket).data,
                    'classification': {
                        'category': routing_result.get('category'),
                        'confidence': routing_result.get('confidence'),
                        'department': routing_result.get('department'),
                        'priority': routing_result.get('priority'),
                        'sla_hours': routing_result.get('sla_hours'),
                    }
                }
            else:
                # Fallback if classifier not available
                ticket = Ticket.objects.create(
                    customer_name=data['customer_name'],
                    customer_email=data.get('customer_email', ''),
                    subject=data['subject'],
                    description=data['description'],
                    predicted_category='General',
                    confidence_score=0.0,
                    department='General Support',
                    priority='medium',
                    sla_hours=24
                )
                response_data = {
                    'ticket': TicketSerializer(ticket).data,
                    'classification': {
                        'category': 'General',
                        'confidence': 0.0,
                        'department': 'General Support',
                        'note': 'Classifier not configured'
                    }
                }
            
            return Response(response_data, status=status.HTTP_201_CREATED)
            
        except Exception as e:
            return Response(
                {'error': f'Classification failed: {str(e)}'},
                status=status.HTTP_500_INTERNAL_SERVER_ERROR
            )


class CategoryViewSet(viewsets.ReadOnlyModelViewSet):
    """ViewSet for viewing categories."""
    queryset = Category.objects.all()
    serializer_class = CategorySerializer


def submit_ticket_view(request):
    """HTML form for submitting tickets."""
    return render(request, 'tickets/submit.html')


def dashboard_view(request):
    """Dashboard view for monitoring tickets."""
    tickets = Ticket.objects.all()[:10]  # Latest 10 tickets
    context = {
        'tickets': tickets,
        'total_tickets': Ticket.objects.count(),
    }
    return render(request, 'tickets/dashboard.html', context)
