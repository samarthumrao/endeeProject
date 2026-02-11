```markdown
# 🎫 Customer Support Ticket Classifier

An AI-powered classification and routing system that utilizes **Semantic Similarity Search** to automatically categorize support tickets and direct them to the appropriate departments.



## 🚀 Overview
Traditional ticket systems rely on rigid, keyword-based rules that often fail to understand context. This system uses **Transformer-based embeddings** to understand the *meaning* behind a customer's inquiry. By comparing new tickets against historical data in the **Endee Vector Database**, it provides high-accuracy routing, priority assignment, and SLA tracking.

### Key Features
* **🤖 Automatic Classification**: Uses `sentence-transformers` to generate 384-dimensional semantic embeddings.
* **🎯 Smart Routing**: Automatically maps tickets to departments (Technical, Billing, Engineering, etc.).
* **📈 Confidence Scoring**: Provides a reliability score for every classification to ensure high-quality routing.
* **🔌 REST API**: Fully documented endpoints for ticket submission and management.
* **🖥️ Admin Dashboard**: Integrated Django admin interface for monitoring and manual overrides.
* **📱 Modern UI**: Responsive web interface for seamless ticket submission.

---

## 🛠️ Tech Stack
| Component | Technology |
| :--- | :--- |
| **Backend** | Python 3.8+, Django, Django REST Framework |
| **Vector Database** | [Endee](https://github.com) (Running via Docker) |
| **ML Models** | Sentence-Transformers (`all-MiniLM-L6-v2`) |
| **Primary Database** | SQLite (Development) |
| **Frontend** | HTML5, CSS3, JavaScript |

---

## 🏗️ Project Structure
```text
ticket-classifier/
├── core/                      # ML & Vector Logic
│   ├── classifier.py          # Ticket classification engine
│   ├── embedding_service.py   # Vector generation
│   └── endee_client.py        # Endee API integration
│
├── tickets/                   # Django Application
│   ├── models.py              # Ticket & Category schemas
│   ├── views.py               # API & Web views
│   └── serializers.py         # DRF Serializers
│
├── scripts/                   # Data Pipeline
│   ├── preprocess_data.py     # CSV Cleaning & Splitting
│   └── load_to_endee.py       # Vector Ingestion
│
└── config.py                  # Environment Configuration

```

---

## 🚦 Quick Start

### 1. Prerequisites

* Python 3.8+
* Docker (Required to run the Endee Vector Database)

### 2. Start Endee Vector Engine

```bash
# Start the container
docker start endee-server

# Verify status
docker ps --filter name=endee-server

```

### 3. Setup Environment

```bash
# Clone the repository
git clone [https://github.com/your-username/ticket-classifier.git](https://github.com/your-username/ticket-classifier.git)
cd ticket-classifier

# Create and activate virtual environment
python -m venv venv
.\venv\Scripts\Activate.ps1  # Windows
source venv/bin/activate     # Mac/Linux

# Install dependencies
pip install -r requirements.txt

```

### 4. Initialize & Run

```bash
# Apply migrations
python manage.py migrate

# Create admin user
python manage.py createsuperuser

# Start server
python manage.py runserver

```

---

## 🔌 API Reference

### Submit a Ticket

`POST /api/tickets/submit/`

**Request:**

```json
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
    "ticket_id": "uuid-12345",
    "predicted_category": "Account Access",
    "confidence_score": 0.85,
    "department": "Account Services",
    "priority": "high",
    "sla_hours": 2,
    "status": "new"
  }
}

```

---

## 🧠 How It Works

1. **Vectorization**: The system combines `subject` and `description` and converts the text into a numerical vector using `all-MiniLM-L6-v2`.
2. **Semantic Search**: The vector is queried against the **Endee Vector Database** to find the  most similar historical tickets.
3. **Weighted Voting**: The system predicts the category based on the labels of the nearest neighbors, weighted by their similarity score.
4. **Routing Engine**: Based on the predicted category, the system assigns a Department, Priority level, and SLA.

---

## 📊 Routing Matrix

| Category | Department | Priority | SLA (Hours) |
| --- | --- | --- | --- |
| **Technical** | Technical Support | High | 4 |
| **Account** | Account Services | High | 2 |
| **Billing** | Billing Team | Medium | 24 |
| **Bug** | Engineering | High | 8 |
| **Product** | Sales Team | Medium | 8 |

---

## 📝 License

This project is licensed under the **Apache License 2.0**.

---

**Author**:Samarth Umrao

*Built to demonstrate the power of Vector Databases in modern Customer Service workflows.*

```

Would you like me to generate a `requirements.txt` file or a `docker-compose.yml` to help others set up your project even faster?

```
