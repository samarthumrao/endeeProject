"""
Routing engine for support tickets.
Determines which team/department should handle a ticket.
"""

from typing import Dict
from core.classifier import get_classifier


class RoutingEngine:
    """Engine for routing tickets to appropriate teams."""
    
    def __init__(self):
        """Initialize routing engine."""
        self.classifier = get_classifier()
        
        # Define routing rules
        self.routing_rules = {
            'Technical': {
                'department': 'Technical Support',
                'priority': 'high',
                'sla_hours': 4
            },
            'Account': {
                'department': 'Account Services',
                'priority': 'high',
                'sla_hours': 2
            },
            'Billing': {
                'department': 'Billing Team',
                'priority': 'medium',
                'sla_hours': 24
            },
            'Product': {
                'department': 'Sales Team',
                'priority': 'medium',
                'sla_hours': 8
            },
            'Bug': {
                'department': 'Engineering',
                'priority': 'high',
                'sla_hours': 8
            },
            'General': {
                'department': 'General Support',
                'priority': 'low',
                'sla_hours': 48
            }
        }
    
    def route_ticket(self, ticket_text: str) -> Dict:
        """
        Route a ticket to the appropriate department.
        
        Args:
            ticket_text: The ticket text
            
        Returns:
            Routing decision with department, priority, etc.
        """
        # Classify the ticket
        classification = self.classifier.classify_ticket(ticket_text)
        
        # Get category
        category = classification.get('category', 'General')
        confidence = classification.get('confidence', 0.0)
        
        # Find matching routing rule (partial match)
        routing_info = self.routing_rules.get('General')  # Default
        for key, rule in self.routing_rules.items():
            if key.lower() in category.lower():
                routing_info = rule
                break
        
        return {
            'category': category,
            'confidence': confidence,
            'department': routing_info['department'],
            'priority': routing_info['priority'],
            'sla_hours': routing_info['sla_hours'],
            'similar_tickets': classification.get('similar_tickets', [])
        }


# Create singleton instance
_routing_engine = None

def get_routing_engine() -> RoutingEngine:
    """Get or create the routing engine singleton."""
    global _routing_engine
    if _routing_engine is None:
        _routing_engine = RoutingEngine()
    return _routing_engine
