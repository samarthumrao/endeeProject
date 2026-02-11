"""
Ticket Classifier - Main classification logic.
Uses Endee vector database for semantic similarity search.
"""

from typing import List, Dict, Tuple
import pandas as pd
import os
from collections import Counter

from core.endee_client import EndeeClient
from core.embedding_service import get_embedding_service
from config import config


class TicketClassifier:
    """Classifier for support tickets using vector similarity."""
    
    def __init__(self, index_name: str = None):
        """
        Initialize classifier.
        
        Args:
            index_name: Name of the Endee index to use
        """
        self.index_name = index_name or config.INDEX_NAME
        self.endee_client = EndeeClient()
        self.embedding_service = get_embedding_service()
        self.top_k = config.TOP_K_SIMILAR
        self.confidence_threshold = config.CONFIDENCE_THRESHOLD
        self._load_ticket_cache()
        
    def _load_ticket_cache(self):
        """Load training data into memory for metadata lookup (workaround for Endee metadata issue)."""
        try:
            
            base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
            csv_path = os.path.join(base_dir, 'data', 'processed', 'train.csv')
            
            self.ticket_cache = {}
            if os.path.exists(csv_path):
                df = pd.read_csv(csv_path)
                for idx, row in df.iterrows():
                    # Extract category (check multiple possible column names)
                    category = 'Unknown'
                    for col in ['Ticket Type', 'Category', 'Type', 'Issue Type', 'Topic']:
                        if col in row and pd.notna(row[col]):
                            category = str(row[col])
                            break
                    
                    # Combine text for display
                    text_parts = []
                    for col in ['Ticket Subject', 'Subject', 'Ticket Description', 'Description']:
                        if col in row and pd.notna(row[col]):
                            text_parts.append(str(row[col]))
                    combined_text = " | ".join(text_parts)
                    
                    self.ticket_cache[f"ticket_{idx}"] = {
                        "category": category,
                        "text": combined_text
                    }
                print(f"Loaded {len(self.ticket_cache)} tickets into metadata cache")
            else:
                print(f"Warning: Training data not found at {csv_path}")
        except Exception as e:
            self.ticket_cache = {}
            print(f"Error loading ticket cache: {e}")

    def classify_ticket(self, ticket_text: str) -> Dict:
        """
        Classify a support ticket.
        
        Args:
            ticket_text: The ticket text to classify
            
        Returns:
            Dictionary with classification results
        """
        # Generate embedding for the input ticket
        embedding = self.embedding_service.generate_embedding(ticket_text)
        
        # Search for similar tickets in Endee
        try:
            results = self.endee_client.search(
                self.index_name,
                embedding,
                top_k=self.top_k
            )
        except Exception as e:
            print(f"Error searching Endee: {e}")
            return {
                'category': 'Unknown',
                'confidence': 0.0,
                'similar_tickets': [],
                'error': str(e)
            }
        
        if not results:
            return {
                'category': 'Unknown',
                'confidence': 0.0,
                'similar_tickets': [],
                'error': 'No similar tickets found'
            }
        
        # Extract categories from similar tickets
        # Endee MessagePack format: [distance, id, metadata_field1, metadata_field2, ...]
        # Metadata is not returned by Endee, so we look it up in our cache
        categories = []
        similarities = []
        similar_tickets_info = []
        
        for result in results:
            # Endee MessagePack format: [distance, id, metadata_field1, metadata_field2, ...]
            # Based on our load script: [distance, id, combined_text, category, priority, sparse]
            if len(result) < 2:
                continue
                
            distance = result[0]  # Lower distance = more similar
            ticket_id = result[1]
            
            # Lookup metadata from cache
            cached_data = self.ticket_cache.get(ticket_id, {})
            category = cached_data.get('category')
            combined_text = cached_data.get('text', "")
            
            # Convert distance to similarity score (inverse)
            # Cosine distance: 0 = identical, 2 = opposite
            # Convert to similarity: 1 = identical, 0 = opposite
            score = 1.0 - (distance / 2.0) if distance <= 2.0 else 0.0
            
            if category:
                categories.append(category)
                similarities.append(score)
                similar_tickets_info.append({
                    'category': category,
                    'score': score,
                    'distance': distance,
                    'text': combined_text[:200]
                })
        
        # Determine final category by weighted voting
        if not categories:
            return {
                'category': 'Uncategorized',
                'confidence': 0.0,
                'similar_tickets': similar_tickets_info
            }
        
        # Weighted voting: each category gets votes weighted by similarity score
        category_scores = {}
        for cat, sim in zip(categories, similarities):
            if cat not in category_scores:
                category_scores[cat] = 0.0
            category_scores[cat] += sim
        
        # Find category with highest weighted score
        best_category = max(category_scores.items(), key=lambda x: x[1])
        predicted_category = best_category[0]
        
        # Calculate confidence as the proportion of the best category's score
        total_score = sum(category_scores.values())
        confidence = best_category[1] / total_score if total_score > 0 else 0.0
        
        return {
            'category': predicted_category,
            'confidence': confidence,
            'similar_tickets': similar_tickets_info,
            'category_scores': category_scores
        }
    
    def get_routing_suggestion(self, category: str) -> str:
        """
        Get routing suggestion based on category.
        
        Args:
            category: Predicted category
            
        Returns:
            Department or team to route to
        """
        # Simple routing rules (customize based on your organization)
        routing_map = {
            'Technical': 'Technical Support Team',
            'Account Access': 'Account Services Team',
            'Billing': 'Billing Department',
            'Product Inquiry': 'Sales Team',
            'Bug Report': 'Engineering Team',
            'Feature Request': 'Product Management',
            'General': 'General Support',
        }
        
        return routing_map.get(category, 'General Support')


# Create singleton instance
_classifier = None

def get_classifier() -> TicketClassifier:
    """Get or create the classifier singleton."""
    global _classifier
    if _classifier is None:
        _classifier = TicketClassifier()
    return _classifier
