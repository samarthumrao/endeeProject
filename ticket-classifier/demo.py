"""
Demo script to test the ticket classification system.
"""

import requests
import json
import time

# Configuration
BASE_URL = "http://localhost:8000"

# Sample tickets to test
test_tickets = [
    {
        "customer_name": "Alice Johnson",
        "customer_email": "alice@example.com",
        "subject": "Cannot log into my account",
        "description": "I forgot my password and the password reset link isn't working. I've tried multiple times."
    },
    {
        "customer_name": "Bob Smith",
        "customer_email": "bob@example.com",
        "subject": "Billing issue with my last payment",
        "description": "I was charged twice for my monthly subscription. Can you please refund the duplicate charge?"
    },
    {
        "customer_name": "Carol Davis",
        "customer_email": "carol@example.com",
        "subject": "Software crashes when I open files",
        "description": "Every time I try to open a PDF file, the application crashes immediately. This started after the latest update."
    },
    {
        "customer_name": "David Wilson",
        "customer_email": "david@example.com",
        "subject": "Question about your premium features",
        "description": "I'm interested in upgrading to the premium plan. Can you tell me more about the additional features included?"
    },
    {
        "customer_name": "Eve Martinez",
        "customer_email": "eve@example.com",
        "subject": "Bug: Export function not working",
        "description": "When I try to export my data to CSV format, I get an error message. The export button is completely unresponsive."
    }
]


def test_ticket_submission():
    """Test submitting tickets through the API."""
    print("=" * 80)
    print("TESTING TICKET CLASSIFICATION SYSTEM")
    print("=" * 80)
    print()
    
    # Check if server is running
    try:
        response = requests.get(f"{BASE_URL}/api/tickets/")
        print("βœ… Server is running")
    except requests.ConnectionError:
        print("❌ Error: Server not running. Please start with: python manage.py runserver")
        return
    
    print()
    print("Submitting test tickets...")
    print("-" * 80)
    
    for i, ticket in enumerate(test_tickets, 1):
        print(f"\n[Test {i}/{len(test_tickets)}]")
        print(f"Subject: {ticket['subject']}")
        print(f"Customer: {ticket['customer_name']}")
        
        try:
            # Submit ticket
            response = requests.post(
                f"{BASE_URL}/api/tickets/submit/",
                json=ticket,
                headers={"Content-Type": "application/json"}
            )
            
            if response.status_code == 201:
                data = response.json()
                classification = data.get('classification', {})
                
                print(f"\nβœ… Ticket classified successfully!")
                print(f"   Category: {classification.get('category', 'Unknown')}")
                print(f"   Confidence: {(classification.get('confidence', 0) * 100):.1f}%")
                print(f"   Department: {classification.get('department', 'N/A')}")
                print(f"   Priority: {classification.get('priority', 'N/A')}")
                print(f"   SLA: {classification.get('sla_hours', 'N/A')} hours")
            else:
                print(f"❌ Error: {response.status_code}")
                print(f"   Response: {response.text[:200]}")
        
        except Exception as e:
            print(f"❌ Error: {e}")
        
        # Small delay between requests
        time.sleep(0.5)
    
    print()
    print("=" * 80)
    print("TEST COMPLETE")
    print("=" * 80)
    print()
    print("View results:")
    print(f"  Dashboard: {BASE_URL}/dashboard/")
    print(f"  Admin: {BASE_URL}/admin/")
    print(f"  Submit more: {BASE_URL}/submit/")


if __name__ == "__main__":
    test_ticket_submission()
