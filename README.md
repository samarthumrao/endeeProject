# Customer Support Ticket Classifier

**AI-Powered Support Ticket Classification and Routing System**

Automatically categorize and route support tickets using semantic similarity search with Endee vector database and transformer models.

---

## Features

- **Automatic Classification**: Uses sentence-transformers to generate embeddings and classify tickets by semantic similarity
- **Smart Routing**: Routes tickets to appropriate departments based on category
- **Confidence Scoring**: Provides confidence scores for classifications
- **REST API**: Full REST API for ticket submission and management
- **Admin Dashboard**: Django admin interface for monitoring and management
- **Modern UI**: Beautiful, responsive web interface for ticket submission

---

## Tech Stack

- **Backend**: Python, Django, Django REST Framework
- **Vector Database**: Endee (running in Docker)
- **ML Models**: Sentence-Transformers (all-MiniLM-L6-v2)
- **Database**: SQLite (development)
- **Frontend**: HTML, CSS, JavaScript

---

## Prerequisites

- Python 3.8+
- Docker (for Endee vector database)
- Windows/Linux/Mac

---

## Quick Start

### 1. Ensure Endee is Running

The Endee vector database should already be running:
```powershell
docker ps --filter name=endee-server
```

If not running, start it:
```powershell
docker start endee-server
```

### 2. Activate Virtual Environment

```powershell
.\venv\Scripts\Activate.ps1
```

### 3. Run Development Server

```powershell
python manage.py runserver
```

### 4. Access the Application

- **Submit Ticket**: http://localhost:8000/submit/
- **Dashboard**: http://localhost:8000/dashboard/
- **Admin Panel**: http://localhost:8000/admin/ (create superuser first)
- **API Root**: http://localhost:8000/api/

---

## πŸ"š API Endpoints

### Submit Ticket
```http
POST /api/tickets/submit/
Content-Type: application/json

{
  "customer_name": "John Doe",
  "customer_email": "john@example.com",
  "subject": "Cannot access my account",
  "description": "I forgot my password and the reset email isn't arriving"
}
```

**Response:**
```json
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
```

### List Tickets
```http
GET /api/tickets/
```

### Get Ticket Details
```http
GET /api/tickets/{ticket_id}/
```

### List Categories
```http
GET /api/categories/
```

---

## πŸ"‚ Project Structure

```
ticket-classifier/
β"œβ"€β"€ core/                      # Core ML modules
β"‚   β"œβ"€β"€ classifier.py          # Ticket classifier
β"‚   β"œβ"€β"€ embedding_service.py   # Embedding generation
β"‚   β"œβ"€β"€ endee_client.py        # Endee API client
β"‚   └── routing_engine.py     # Routing logic
β"œβ"€β"€ tickets/                  # Django app
β"‚   β"œβ"€β"€ models.py              # Ticket & Category models
β"‚   β"œβ"€β"€ views.py               # API views
β"‚   β"œβ"€β"€ serializers.py         # DRF serializers
β"‚   β"œβ"€β"€ admin.py               # Admin interface
β"‚   └── urls.py               # URL routing
β"œβ"€β"€ templates/tickets/       # HTML templates
β"œβ"€β"€ data/                     # Dataset storage
β"‚   β"œβ"€β"€ raw/                   # Original dataset
β"‚   └── processed/            # Train/test splits
β"œβ"€β"€ scripts/                  # Utility scripts
β"‚   β"œβ"€β"€ preprocess_data.py     # Data preprocessing
β"‚   β"œβ"€β"€ load_to_endee.py       # Load vectors to Endee
β"‚   └── explore_dataset.py    # Dataset exploration
β"œβ"€β"€ config.py                 # Configuration
β"œβ"€β"€ manage.py                 # Django management
└── requirements.txt          # Python dependencies
```

---

## πŸ"§ Configuration

Edit `.env` file:

```env
ENDEE_BASE_URL=http://localhost:8080
ENDEE_AUTH_TOKEN=""
EMBEDDING_MODEL=sentence-transformers/all-MiniLM-L6-v2
DJANGO_SECRET_KEY=your-secret-key
DEBUG=True
```

---

## πŸ'₯ Creating Admin User

```powershell
python manage.py createsuperuser
```

---

## πŸ"Š How Classification Works

1. **Ticket Submission**: User submits ticket through web form or API
2. **Embedding Generation**: Ticket text (subject + description) is converted to 384-dimensional vector using sentence-transformers
3. **Similarity Search**: Vector is compared against historical tickets in Endee database
4. **Weighted Voting**: Top-k similar tickets vote for category, weighted by similarity score
5. **Routing**: Category is mapped to department with priority and SLA
6. **Storage**: Ticket saved to database with classification metadata

---

## 🎯 Routing Rules

| Category | Department | Priority | SLA (hours) |
|----------|------------|----------|-------------|
| Technical | Technical Support | High | 4 |
| Account | Account Services | High | 2 |
| Billing | Billing Team | Medium | 24 |
| Product | Sales Team | Medium | 8 |
| Bug | Engineering | High | 8 |
| General | General Support | Low | 48 |

---

## πŸ" Dataset

Dataset: `data/raw/customer_support_tickets.csv`
- Preprocessed and split into train (80%) / test (20%)
- Training data used to populate Endee vector database
- Each ticket embedded and stored with metadata

---

## πŸ› οΈ Development Commands

```powershell
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
```

---

## 🚧 Current Limitations

- **Vector Loading**: Index creation API endpoint needs configuration (Endee API documentation required)
- **Mock Classification**: System currently works with fallback routing if vector database isn't populated
- **Dataset**: Uses pre-downloaded dataset; flexible to any customer support ticket CSV

---

## πŸ"ˆ Future Enhancements

- [ ] Complete Endee vector database population
- [ ] Add real-time classification accuracy monitoring
- [ ] Implement feedback loop for model improvement
- [ ] Add multi-language support
- [ ] Enhanced analytics dashboard
- [ ] Email notifications for ticket updates
- [ ] Customer portal for ticket tracking

---

## πŸ"„ License

Apache License 2.0

---

## πŸ'€ Author

Built as a demonstration of AI-powered ticket classification using vector similarity search.

---

## πŸ"ž Support

For questions or issues, submit a ticket through the system! 😊
