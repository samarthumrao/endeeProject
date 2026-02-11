"""
Load support tickets into Endee vector database.
Generates embeddings for each ticket and inserts them into the index.
"""

import pandas as pd
import json
import os
import sys
from typing import List, Dict

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.endee_client import EndeeClient
from core.embedding_service import get_embedding_service
from config import config


def combine_ticket_text(row: pd.Series) -> str:
    """
    Combine relevant fields into a single text for embedding.
    
    Args:
        row: DataFrame row
        
    Returns:
        Combined text string
    """
    # Identify which columns to use (you might want to adjust this based on your dataset)
    text_parts = []
    
    # Common column names to combine
    possible_columns = [
        'Ticket Subject', 'Subject', 'Ticket Description', 'Description',
        'Ticket Type', 'Type', 'Issue', 'Problem', 'Question',
        'Product Purchased', 'Product', 'Category'
    ]
    
    for col in possible_columns:
        if col in row.index and pd.notna(row[col]) and str(row[col]).strip():
            text_parts.append(str(row[col]))
    
    # Fallback: use all text columns if no specific ones found
    if not text_parts:
        for col, val in row.items():
            if isinstance(val, str) and len(val) > 10:
                text_parts.append(val)
    
    combined = " | ".join(text_parts)
    return combined if combined else "No description"


def load_tickets_to_endee(csv_path: str, index_name: str):
    """
    Load tickets from CSV into Endee vector database.
    
    Args:
        csv_path: Path to CSV file
        index_name: Name of the index to create/use
    """
    print("=" * 80)
    print("LOADING TICKETS TO ENDEE VECTOR DATABASE")
    print("=" * 80)
    
    # Initialize services
    print("\nInitializing services...")
    endee_client = EndeeClient()
    embedding_service = get_embedding_service()
    
    # Load data
    print(f"\nLoading tickets from: {csv_path}")
    df = pd.read_csv(csv_path)
    print(f"Loaded {len(df)} tickets")
    
    # Check if index exists, if so delete it
    print(f"\nChecking for existing index '{index_name}'...")
    try:
        existing_indexes = endee_client.list_indexes()
        if index_name in existing_indexes:
            print(f"Index '{index_name}' already exists. Deleting...")
            endee_client.delete_index(index_name)
    except Exception as e:
        print(f"Could not check existing indexes: {e}")
    
    # Create index
    print(f"\nCreating index '{index_name}'...")
    dimension = embedding_service.get_dimension()
    endee_client.create_index(index_name, dimension=dimension, metric="cosine")
    
    # Generate embeddings and prepare vectors
    print("\nGenerating embeddings...")
    texts = []
    metadata_list = []
    
    for idx, row in df.iterrows():
        # Combine text for embedding
        text = combine_ticket_text(row)
        texts.append(text)
        
        # Prepare metadata (all columns except the text we're embedding)
        metadata = {}
        for col in df.columns:
            val = row[col]
            if pd.notna(val):
                # Convert to string for JSON serialization
                metadata[col] = str(val)
        
        metadata['combined_text'] = text
        metadata_list.append(metadata)
    
    print(f"Generating embeddings for {len(texts)} tickets...")
    embeddings = embedding_service.batch_generate(texts, batch_size=32, show_progress=True)
    
    # Prepare vectors for insertion
    print("\nPreparing vectors for insertion...")
    vectors = []
    for idx, (embedding, metadata) in enumerate(zip(embeddings, metadata_list)):
        vector_obj = {
            "id": f"ticket_{idx}",
            "vector": embedding,
            "metadata": metadata
        }
        vectors.append(vector_obj)
    
    # Insert in batches
    print(f"\nInserting {len(vectors)} vectors into Endee...")
    batch_size = 100
    for i in range(0, len(vectors), batch_size):
        batch = vectors[i:i+batch_size]
        endee_client.insert_vectors(index_name, batch)
        print(f"  Inserted batch {i//batch_size + 1}/{(len(vectors)-1)//batch_size + 1}")
    
    # Verify
    print("\nVerifying insertion...")
    try:
        stats = endee_client.index_stats(index_name)
        print(f"Index stats: {stats}")
    except Exception as e:
        print(f"Could not get stats: {e}")
    
    print("\n" + "=" * 80)
    print("LOADING COMPLETE")
    print("=" * 80)
    print(f"\n✓ Successfully loaded {len(vectors)} tickets into index '{index_name}'")


def main():
    """Main function."""
    # Load from processed training data
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_dir = os.path.dirname(script_dir)
    train_path = os.path.join(project_dir, 'data', 'processed', 'train.csv')
    
    if not os.path.exists(train_path):
        print(f"Error: Training data not found at {train_path}")
        print("Please run preprocess_data.py first")
        return
    
    load_tickets_to_endee(train_path, config.INDEX_NAME)


if __name__ == "__main__":
    main()
