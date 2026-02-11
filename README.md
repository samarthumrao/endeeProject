Customer Support Ticket Classifier
AI-Powered Support Ticket Classification and Routing System
Automatically categorize and route support tickets using semantic similarity search with Endee vector database and transformer models.

Features


Automatic Classification – Uses sentence-transformers to generate embeddings and classify tickets by semantic similarity


Smart Routing – Routes tickets to appropriate departments based on category


Confidence Scoring – Provides confidence scores for classifications


REST API – Full REST API for ticket submission and management


Admin Dashboard – Django admin interface for monitoring and management


Modern UI – Beautiful, responsive web interface for ticket submission



Tech Stack


Backend: Python, Django, Django REST Framework


Vector Database: Endee (running in Docker)


ML Models: Sentence-Transformers (all-MiniLM-L6-v2)


Database: SQLite (development)


Frontend: HTML, CSS, JavaScript



Prerequisites


Python 3.8+


Docker (for Endee vector database)


Windows/Linux/Mac



Quick Start
1. Ensure Endee is Running
Check if Endee is running:
docker ps --filter name=endee-server

If not running:
docker start endee-server


2. Activate Virtual Environment
.\venv\Scripts\Activate.ps1


3. Run Development Server
python manage.py runserver


4. Access the Application


Submit Ticket: http://localhost:8000/submit/


Dashboard: http://localhost:8000/dashboard/


Admin Panel: http://localhost:8000/admin/ (create superuser first)


API Root: http://localhost:8000/api/



API Endpoints
Submit Ticket
POST /api/tickets/submit/
Content-Type: application/json
{
  "customer_name": "John Doe",
  "customer_email": "john@example.com",
  "subject": "Cannot access my account",
  "description": "I forgot my password and the reset email isn't arriving"
}

Response:
{
  "ticket": {
    "ticket_id": "uuid",
    "subject": "Cannot access my account",
    "predicted_category": "Account Access",
    "confidence_score": 0.85,
    "department": "Account Services",
    "priority": "high",
    "sla_hours": 2,
    "status": "new"
  },
  "classification": {
    "category": "Account Access",
    "confidence": 0.85,
    "department": "Account Services",
    "priority": "high",
    "sla_hours": 2
  }
}


List Tickets
GET /api/tickets/

Get Ticket Details
GET /api/tickets/{ticket_id}/

List Categories
GET /api/categories/

Project Structure
ticket-classifier/
│
├── core/                      # Core ML modules
│   ├── classifier.py          # Ticket classifier
│   ├── embedding_service.py   # Embedding generation
│   ├── endee_client.py        # Endee API client
│   └── routing_engine.py      # Routing logic
│
├── tickets/                   # Django app
│   ├── models.py              # Ticket & Category models
│   ├── views.py               # API views
│   ├── serializers.py         # DRF serializers
│   ├── admin.py               # Admin interface
│   └── urls.py                # URL routing
│
├── templates/tickets/         # HTML templates
│
├── data/                      # Dataset storage
│   ├── raw/                   # Original dataset
│   └── processed/             # Train/test splits
│
├── scripts/                   # Utility scripts
│   ├── preprocess_data.py     # Data preprocessing
│   ├── load_to_endee.py       # Load vectors to Endee
│   └── explore_dataset.py     # Dataset exploration
│
├── config.py                  # Configuration
├── manage.py                  # Django management
└── requirements.txt           # Python dependencies


Configuration
Edit .env file:
ENDEE_BASE_URL=http://localhost:8080
ENDEE_AUTH_TOKEN=""
EMBEDDING_MODEL=sentence-transformers/all-MiniLM-L6-v2
DJANGO_SECRET_KEY=your-secret-key
DEBUG=True


Creating Admin User
python manage.py createsuperuser


How Classification Works


Ticket Submission – User submits ticket through web form or API


Embedding Generation – Ticket text (subject + description) is converted to a 384-dimensional vector using sentence-transformers


Similarity Search – Vector is compared against historical tickets in Endee database


Weighted Voting – Top-k similar tickets vote for category, weighted by similarity score


Routing – Category is mapped to department with priority and SLA


Storage – Ticket saved to database with classification metadata



Routing Rules
CategoryDepartmentPrioritySLA (hours)TechnicalTechnical SupportHigh4AccountAccount ServicesHigh2BillingBilling TeamMedium24ProductSales TeamMedium8BugEngineeringHigh8GeneralGeneral SupportLow48

Dataset


Dataset location: data/raw/customer_support_tickets.csv


Preprocessed and split into train (80%) / test (20%)


Training data used to populate Endee vector database


Each ticket embedded and stored with metadata



Development Commands
# Run development server
python manage.py runserver

# Make migrations
python manage.py makemigrations

# Apply migrations
python manage.py migrate

# Preprocess dataset
python scripts/preprocess_data.py

# Load data to Endee (when Endee index creation is configured)
python scripts/load_to_endee.py

# Create superuser
python manage.py createsuperuser


Current Limitations


Vector Loading – Index creation API endpoint needs configuration (Endee API documentation required)


Mock Classification – System currently works with fallback routing if vector database is not populated


Dataset – Uses pre-downloaded dataset; flexible to any customer support ticket CSV



Future Enhancements


Complete Endee vector database population


Add real-time classification accuracy monitoring


Implement feedback loop for model improvement


Add multi-language support


Enhanced analytics dashboard


Email notifications for ticket updates


Customer portal for ticket tracking



License
Apache License 2.0

Author
Built as a demonstration of AI-powered ticket classification using vector similarity search.

Support
For questions or issues, submit a ticket through the system! 😊