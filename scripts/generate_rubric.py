#!/usr/bin/env python3
"""
Generate a taste rubric from real favorite articles data.
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from lib.taste_rubric import TasteRubricGenerator
import pandas as pd

def main():
    """Generate rubric from real data."""
    
    # Check if processed data exists
    processed_data_path = "data/processed/train/favorite_classifier.csv"
    if os.path.exists(processed_data_path):
        print(f"Loading processed data from {processed_data_path}")
        df = pd.read_csv(processed_data_path)
    else:
        # Check if raw data exists
        raw_data_path = "data/raw/export.csv"
        if os.path.exists(raw_data_path):
            print(f"Loading raw data from {raw_data_path}")
            df = pd.read_csv(raw_data_path)
        else:
            print("No data found. Please ensure data/export.csv exists.")
            print("Or run: python scripts/favorite_classifier/prepare.py")
            return
    
    # Get favorites - check column names
    if 'label' in df.columns:
        favorites_df = df[df['label'] == 1]  # Processed data
        favorite_titles = favorites_df['title'].tolist()
    elif 'has_favorite' in df.columns:
        favorites_df = df[df['has_favorite'] == True]  # Raw data
        favorite_titles = favorites_df['title_clean'].tolist()
    elif 'Document tags' in df.columns:
        # Handle export.csv format where favorites are in Document tags column
        favorites_df = df[df['Document tags'].str.contains('favorite', na=False)]
        favorite_titles = favorites_df['Title'].tolist()
    else:
        print("Could not find favorite column. Available columns:", df.columns.tolist())
        return
    
    print(f"Found {len(favorite_titles)} favorite articles")
    print("\nSample favorites:")
    for i, title in enumerate(favorite_titles[:5]):
        print(f"{i+1}. {title}")
    
    # Generate rubric
    generator = TasteRubricGenerator(model_name="gpt-4.1")
    print(f"\nGenerating taste rubric from {len(favorite_titles)} favorites...")
    
    rubric = generator.generate_rubric(favorite_titles)
    
    print("\n" + "="*70)
    print("GENERATED TASTE RUBRIC FROM REAL DATA:")
    print("="*70)
    print(rubric)
    
    # Save to a standard location for other scripts to use
    rubric_path = "saved/rubrics/personal_taste_rubric.txt"
    with open(rubric_path, "w") as f:
        f.write(rubric)
    
    print(f"\n✓ Rubric saved to {rubric_path}")
    print("This rubric can now be used by other taste classification scripts.")

if __name__ == "__main__":
    main()