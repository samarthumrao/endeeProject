"""
Preprocess customer support tickets and prepare for vector database loading.
"""

import pandas as pd
import os
import json
from typing import List, Dict, Tuple

def load_and_clean_data():
    """Load and clean the support tickets dataset."""
    # Load dataset
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_dir = os.path.dirname(script_dir)
    data_path = os.path.join(project_dir, 'data', 'raw', 'customer_support_tickets.csv')
    
    print(f"Loading dataset from: {data_path}")
    df = pd.read_csv(data_path)
    
    print(f"Loaded {len(df)} tickets")
    print(f"Columns: {list(df.columns)}")
    
    # Clean the data
    print("\nCleaning data...")
    
    # Remove rows with missing critical fields
    initial_count = len(df)
    df = df.dropna(subset=['Ticket ID'])
    print(f"Removed {initial_count - len(df)} tickets with missing Ticket ID")
    
    # Fill missing values in text fields with empty string
    text_columns = df.select_dtypes(include=['object']).columns
    for col in text_columns:
        df[col] = df[col].fillna('')
    
    print(f"\nFinal dataset size: {len(df)} tickets")
    
    return df


def prepare_training_data(df: pd.DataFrame, train_ratio: float = 0.8) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Split dataset into training and test sets.
    
    Args:
        df: Input dataframe
        train_ratio: Proportion for training set
        
    Returns:
        Tuple of (train_df, test_df)
    """
    print(f"\nSplitting data: {train_ratio*100:.0f}% train, {(1-train_ratio)*100:.0f}% test")
    
   # Shuffle the data
    df = df.sample(frac=1, random_state=42).reset_index(drop=True)
    
    # Split
    split_idx = int(len(df) * train_ratio)
    train_df = df.iloc[:split_idx]
    test_df = df.iloc[split_idx:]
    
    print(f"Train set: {len(train_df)} tickets")
    print(f"Test set: {len(test_df)} tickets")
    
    return train_df, test_df


def save_processed_data(train_df: pd.DataFrame, test_df: pd.DataFrame):
    """Save processed data to CSV files."""
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_dir = os.path.dirname(script_dir)
    processed_dir = os.path.join(project_dir, 'data', 'processed')
    
    os.makedirs(processed_dir, exist_ok=True)
    
    train_path = os.path.join(processed_dir, 'train.csv')
    test_path = os.path.join(processed_dir, 'test.csv')
    
    train_df.to_csv(train_path, index=False)
    test_df.to_csv(test_path, index=False)
    
    print(f"\nSaved processed data:")
    print(f"  Train: {train_path}")
    print(f"  Test: {test_path}")
    
    # Save dataset info
    info = {
        'total_tickets': len(train_df) + len(test_df),
        'train_count': len(train_df),
        'test_count': len(test_df),
        'columns': list(train_df.columns),
    }
    
    info_path = os.path.join(processed_dir, 'dataset_info.json')
    with open(info_path, 'w') as f:
        json.dump(info, f, indent=2)
    
    print(f"  Info: {info_path}")


def main():
    """Main preprocessing pipeline."""
    print("=" * 80)
    print("DATA PREPROCESSING PIPELINE")
    print("=" * 80)
    
    # Load and clean
    df = load_and_clean_data()
    
    # Split into train/test
    train_df, test_df = prepare_training_data(df, train_ratio=0.8)
    
    # Save
    save_processed_data(train_df, test_df)
    
    print("\n" + "=" * 80)
    print("PREPROCESSING COMPLETE")
    print("=" * 80)


if __name__ == "__main__":
    main()
