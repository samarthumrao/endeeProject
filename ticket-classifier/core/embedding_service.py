"""
Embedding Service.
Handles generation of vector embeddings from text using sentence-transformers.
"""

from typing import List, Union
from sentence_transformers import SentenceTransformer
from config import config


class EmbeddingService:
    """Service for generating text embeddings."""
    
    def __init__(self, model_name: str = None):
        """
        Initialize embedding service.
        
        Args:
            model_name: Name of the sentence-transformer model
        """
        self.model_name = model_name or config.EMBEDDING_MODEL
        print(f"Loading embedding model: {self.model_name}")
        self.model = SentenceTransformer(self.model_name)
        print(f"Model loaded successfully. Embedding dimension: {self.model.get_sentence_embedding_dimension()}")
    
    def generate_embedding(self, text: str) -> List[float]:
        """
        Generate embedding for a single text.
        
        Args:
            text: Input text
            
        Returns:
            Embedding vector as list of floats
        """
        if not text or not text.strip():
            raise ValueError("Text cannot be empty")
        
        embedding = self.model.encode(text, convert_to_numpy=True)
        return embedding.tolist()
    
    def batch_generate(self, texts: List[str], batch_size: int = 32, show_progress: bool = True) -> List[List[float]]:
        """
        Generate embeddings for multiple texts.
        
        Args:
            texts: List of input texts
            batch_size: Batch size for encoding
            show_progress: Whether to show progress bar
            
        Returns:
            List of embedding vectors
        """
        if not texts:
            return []
        
        embeddings = self.model.encode(
            texts,
            batch_size=batch_size,
            show_progress_bar=show_progress,
            convert_to_numpy=True
        )
        
        return [emb.tolist() for emb in embeddings]
    
    def get_dimension(self) -> int:
        """
        Get the dimension of embeddings produced by this model.
        
        Returns:
            Embedding dimension
        """
        return self.model.get_sentence_embedding_dimension()


# Create singleton instance
_embedding_service = None

def get_embedding_service() -> EmbeddingService:
    """Get or create the embedding service singleton."""
    global _embedding_service
    if _embedding_service is None:
        _embedding_service = EmbeddingService()
    return _embedding_service
