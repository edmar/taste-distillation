import pandas as pd
import json
from datetime import datetime
from collections import Counter

def analyze_csv_data(csv_path):
    """Analyze the export.csv data to understand structure and distribution"""
    
    # Read the CSV
    df = pd.read_csv(csv_path)
    
    print(f"Dataset shape: {df.shape}")
    print(f"Columns: {list(df.columns)}")
    print("\nFirst few rows:")
    print(df.head())
    
    # Analyze tags column
    print("\n=== TAG ANALYSIS ===")
    tags_series = df['Document tags'].dropna()
    print(f"Articles with tags: {len(tags_series)} / {len(df)}")
    
    # Count favorites
    favorite_count = 0
    all_tags = []
    
    for tags_str in tags_series:
        if pd.isna(tags_str) or tags_str == '':
            continue
        try:
            # Handle different tag formats
            if tags_str.startswith('[') and tags_str.endswith(']'):
                # Parse as list string
                tags = eval(tags_str)  # Note: eval is risky but needed for this format
                all_tags.extend(tags)
                if 'favorite' in tags:
                    favorite_count += 1
        except:
            # Handle other formats
            if 'favorite' in str(tags_str).lower():
                favorite_count += 1
    
    print(f"Articles with 'favorite' tag: {favorite_count}")
    print(f"Percentage of favorites: {favorite_count/len(df)*100:.2f}%")
    
    # Show tag distribution
    tag_counts = Counter(all_tags)
    print(f"\nTop 10 tags:")
    for tag, count in tag_counts.most_common(10):
        print(f"  {tag}: {count}")
    
    # Analyze dates
    print("\n=== DATE ANALYSIS ===")
    df['Saved date'] = pd.to_datetime(df['Saved date'], format='mixed')
    print(f"Date range: {df['Saved date'].min()} to {df['Saved date'].max()}")
    print(f"Articles by year:")
    print(df['Saved date'].dt.year.value_counts().sort_index())
    
    # Analyze reading progress
    print("\n=== READING PROGRESS ===")
    progress = df['Reading progress'].dropna()
    print(f"Average reading progress: {progress.mean():.3f}")
    print(f"Articles fully read (1.0): {(progress == 1.0).sum()}")
    print(f"Articles not started (0.0): {(progress == 0.0).sum()}")
    
    return df, favorite_count

if __name__ == "__main__":
    df, fav_count = analyze_csv_data("/Users/edmar/Code/taste/export.csv")