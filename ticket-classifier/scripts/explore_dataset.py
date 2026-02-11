"""
Explore the customer support tickets dataset.
Analyze structure, columns, categories, and data quality.
"""

import pandas as pd
import os

def explore_dataset():
    """Explore and analyze the support tickets dataset."""
    # Load the dataset
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_dir = os.path.dirname(script_dir)
    data_path = os.path.join(project_dir, 'data', 'raw', 'customer_support_tickets.csv')
    
    print("=" * 80)
    print("CUSTOMER SUPPORT TICKETS DATASET EXPLORATION")
    print("=" * 80)
    print(f"\nLoading dataset from: {data_path}\n")
    
    # Read the CSV
    df = pd.read_csv(data_path)
    
    # Basic information
    print("=" * 80)
    print("BASIC INFORMATION")
    print("=" * 80)
    print(f"Total Tickets: {len(df)}")
    print(f"Total Columns: {len(df.columns)}")
    print(f"\nColumns: {list(df.columns)}")
    
    # Data types
    print("\n" + "=" * 80)
    print("DATA TYPES")
    print("=" * 80)
    print(df.dtypes)
    
    # First few rows
    print("\n"  + "=" * 80)
    print("SAMPLE DATA (First 5 rows)")
    print("=" * 80)
    print(df.head())
    
    # Missing values
    print("\n" + "=" * 80)
    print("MISSING VALUES")
    print("=" * 80)
    missing = df.isnull().sum()
    missing_pct = (missing / len(df)) * 100
    missing_df = pd.DataFrame({
        'Missing Count': missing,
        'Percentage': missing_pct
    })
    print(missing_df[missing_df['Missing Count'] > 0])
    
    # Identify potential category columns
    print("\n" + "=" * 80)
    print("CATEGORICAL COLUMNS ANALYSIS")
    print("=" * 80)
    
    for col in df.columns:
        unique_count = df[col].nunique()
        if unique_count < 50:  # Likely categorical
            print(f"\n{col}:")
            print(f"  Unique values: {unique_count}")
            print(f"  Distribution:")
            print(df[col].value_counts().head(10))
    
    # Identify text columns (likely ticket content)
    print("\n" + "=" * 80)
    print("TEXT COLUMNS (Potential ticket content)")
    print("=" * 80)
    
    for col in df.columns:
        if df[col].dtype == 'object':
            avg_length = df[col].astype(str).str.len().mean()
            if avg_length > 50:  # Likely text content
                print(f"\n{col}:")
                print(f"  Average length: {avg_length:.2f} characters")
                print(f"  Sample text:")
                print(f"  \"{df[col].iloc[0][:200]}...\"")
    
    # Summary statistics for numerical columns
    if df.select_dtypes(include=['int64', 'float64']).shape[1] > 0:
        print("\n" + "=" * 80)
        print("NUMERICAL COLUMNS STATISTICS")
        print("=" * 80)
        print(df.describe())
    
    print("\n" + "=" * 80)
    print("EXPLORATION COMPLETE")
    print("=" * 80)
    
    return df


if __name__ == "__main__":
    df = explore_dataset()
