"""
Configuration management for the ticket classifier application.
Loads environment variables and provides access to settings.
"""

import os
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()


class Config:
    """Application configuration."""
    
    # Endee Vector Database Settings
    ENDEE_BASE_URL = os.getenv('ENDEE_BASE_URL', 'http://localhost:8080')
    ENDEE_AUTH_TOKEN = os.getenv('ENDEE_AUTH_TOKEN', '')
    
    # Embedding Model Settings
    EMBEDDING_MODEL = os.getenv('EMBEDDING_MODEL', 'sentence-transformers/all-MiniLM-L6-v2')
    EMBEDDING_DIMENSION = 384  # Dimension for all-MiniLM-L6-v2
    
    # Django Settings
    DJANGO_SECRET_KEY = os.getenv('DJANGO_SECRET_KEY', 'change-me-in-production')
    DEBUG = os.getenv('DEBUG', 'True').lower() == 'true'
    
    # Classification Settings
    TOP_K_SIMILAR = 5  # Number of similar tickets to retrieve
    CONFIDENCE_THRESHOLD = 0.6  # Minimum confidence for classification
    
    # Index Settings
    INDEX_NAME = 'support_tickets'
    
    @classmethod
    def validate(cls):
        """Validate required configuration."""
        if not cls.ENDEE_BASE_URL:
            raise ValueError("ENDEE_BASE_URL is required")
        
        return True


# Create a singleton instance
config = Config()
