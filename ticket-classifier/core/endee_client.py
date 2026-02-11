"""
Endee Vector Database Client.
Provides a Python interface for interacting with the Endee vector database.
"""

import requests
import json
from typing import List, Dict, Any, Optional
from config import config


class EndeeClient:
    """Client for Endee vector database operations."""
    
    def __init__(self, base_url: str = None, auth_token: str = None):
        """
        Initialize Endee client.
        
        Args:
            base_url: Endee server base URL (default from config)
            auth_token: Authentication token (default from config)
        """
        self.base_url = (base_url or config.ENDEE_BASE_URL).rstrip('/')
        self.auth_token = auth_token or config.ENDEE_AUTH_TOKEN
        self.headers = {}
        
        if self.auth_token:
            self.headers['Authorization'] = self.auth_token
    
    def _make_request(self, method: str, endpoint: str, **kwargs) -> requests.Response:
        """
        Make HTTP request to Endee server.
        
        Args:
            method: HTTP method (GET, POST, DELETE, etc.)
            endpoint: API endpoint
            **kwargs: Additional arguments for requests
            
        Returns:
            Response object
            
        Raises:
            Exception: If request fails
        """
        url = f"{self.base_url}{endpoint}"
        kwargs.setdefault('headers', {}).update(self.headers)
        
        try:
            response = requests.request(method, url, **kwargs)
            response.raise_for_status()
            return response
        except requests.exceptions.RequestException as e:
            print(f"Error making request to {url}: {e}")
            raise
    
    def create_index(self, index_name: str, dimension: int, metric: str = "cosine") -> Dict[str, Any]:
        """
        Create a new vector index using correct Endee API format.
        
        Args:
            index_name: Name of the index
            dimension: Dimension of vectors
            metric: Distance metric (cosine, euclidean, dot)
            
        Returns:
            Response data
        """
        # Convert metric to space_type (Endee's terminology)
        space_type_map = {
            "cosine": "cosine",
            "euclidean": "l2",  
            "dot": "ip",
            "dot_product": "ip"
        }
        space_type = space_type_map.get(metric, "cosine")
        
        payload = {
            "index_name": index_name,
            "dim": dimension,
            "space_type": space_type
        }
        
        endpoint = "/api/v1/index/create"
        response = self._make_request('POST', endpoint, json=payload)
        print(f"Created index '{index_name}' with dimension {dimension}")
        return {"status": "success", "message": response.text}
    
    def delete_index(self, index_name: str) -> Dict[str, Any]:
        """
        Delete an index.
        
        Args:
            index_name: Name of the index to delete
            
        Returns:
            Response data
        """
        endpoint = f"/api/v1/index/{index_name}"
        response = self._make_request('DELETE', endpoint)
        print(f"Deleted index '{index_name}'")
        return response.json() if response.text else {}
    
    def insert_vectors(self, index_name: str, vectors: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Insert vectors into an index.
        
        Args:
            index_name: Name of the index
            vectors: List of vector objects with 'id', 'vector', and optional 'metadata'
            
        Returns:
            Response data
            
        Example:
            vectors = [
                {
                    "id": "ticket_1",
                    "vector": [0.1, 0.2, ...],
                    "metadata": {"category": "Technical", "priority": "high"}
                }
            ]
        """
        endpoint = f"/api/v1/index/{index_name}/vector/insert"
        payload = vectors  # Send array directly, not wrapped
        
        response = self._make_request('POST', endpoint, json=payload)
        print(f"Inserted {len(vectors)} vectors into '{index_name}'")
        return response.json() if response.text else {}
    
    def search(self, index_name: str, query_vector: List[float], top_k: int = 5) -> List[Dict[str, Any]]:
        """
        Search for similar vectors.
        
        Args:
            index_name: Name of the index
            query_vector: Query vector
            top_k: Number of results to return
            
        Returns:
            List of search results with id, score, and metadata
        """
        import msgpack
        
        endpoint = f"/api/v1/index/{index_name}/search"
        payload = {
            "vector": query_vector,
            "k": top_k
        }
        
        response = self._make_request('POST', endpoint, json=payload)
        
        # Endee returns MessagePack format
        content_type = response.headers.get('Content-Type', '')
        if 'msgpack' in content_type:
            results = msgpack.unpackb(response.content)
            return results if isinstance(results, list) else []
        else:
            # Fallback to JSON
            results = response.json() if response.text else {"results": []}
            return results.get('results', [])
    
    def list_indexes(self) -> List[str]:
        """
        List all indexes.
        
        Returns:
            List of index names
        """
        endpoint = "/api/v1/index/list"
        response = self._make_request('GET', endpoint)
        data = response.json() if response.text else {"indexes": []}
        return data.get('indexes', [])
    
    def index_stats(self, index_name: str) -> Dict[str, Any]:
        """
        Get statistics for an index.
        
        Args:
            index_name: Name of the index
            
        Returns:
            Index statistics
        """
        import msgpack
        endpoint = f"/api/v1/index/{index_name}/stats"
        response = self._make_request('GET', endpoint)
        
        # Endee returns MessagePack format
        content_type = response.headers.get('Content-Type', '')
        if 'msgpack' in content_type:
            try:
                stats = msgpack.unpackb(response.content)
                return stats if isinstance(stats, dict) else {}
            except Exception as e:
                print(f"Error unpacking stats: {e}")
                return {}
        else:
            return response.json() if response.text else {}
