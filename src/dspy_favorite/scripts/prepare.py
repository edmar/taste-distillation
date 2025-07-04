#!/usr/bin/env python3
"""
DSPy Data preparation script for favorite classifier.

This script loads the raw export.csv file and creates train/valid/test splits
specifically for the dspy_favorite model following DSPy standards.
"""

import os
import sys
import json
import pandas as pd
import numpy as np
from pathlib import Path
import argparse
import ast
from typing import List, Dict, Tuple
import dspy


def parse_tags(tag_string: str) -> List[str]:
    """Parse the tag string into a list of tags."""
    if pd.isna(tag_string) or tag_string == '' or tag_string == 'nan':
        return []
    
    try:
        # Remove outer quotes if present
        tag_string = tag_string.strip()
        if tag_string.startswith('"') and tag_string.endswith('"'):
            tag_string = tag_string[1:-1]
        
        # Parse as Python literal
        tags = ast.literal_eval(tag_string)
        if isinstance(tags, list):
            return tags
        else:
            return []
    except:
        # Try alternative parsing methods
        try:
            # Handle cases like "['tag1', 'tag2']"
            if tag_string.startswith("['") and tag_string.endswith("']"):
                tags = tag_string[2:-2].split("', '")
                return tags
        except:
            pass
        return []


def has_favorite_tag(tags: List[str]) -> bool:
    """Check if 'favorite' is in the tags list."""
    return 'favorite' in tags


def load_and_process_data(raw_csv_path: str) -> pd.DataFrame:
    """Load and process the raw data."""
    print(f"Loading data from {raw_csv_path}...")
    
    # Load CSV
    df = pd.read_csv(raw_csv_path)
    
    # Parse tags
    df['parsed_tags'] = df['Document tags'].apply(parse_tags)
    df['has_favorite'] = df['parsed_tags'].apply(has_favorite_tag)
    
    # Filter out empty titles
    df = df[df['Title'].notna() & (df['Title'].str.strip() != '')]
    
    print(f"Before deduplication: {len(df)} entries")
    print(f"Unique titles: {df['Title'].nunique()}")
    print(f"Duplicate titles: {len(df) - df['Title'].nunique()}")
    
    # Remove duplicates by title, keeping the first occurrence
    # This preserves the chronological order of when articles were first seen
    df = df.drop_duplicates(subset=['Title'], keep='first')
    
    print(f"After deduplication: {len(df)} entries")
    
    # Add additional features that might be useful
    df['reading_progress'] = pd.to_numeric(df['Reading progress'], errors='coerce').fillna(0.0)
    df['is_seen'] = df['Seen'].apply(lambda x: str(x).lower() == 'true')
    
    print(f"Final dataset: {len(df)} unique entries")
    print(f"Favorites: {df['has_favorite'].sum()} ({df['has_favorite'].mean():.2%})")
    
    return df


def create_dspy_examples(df: pd.DataFrame) -> List[dspy.Example]:
    """Convert DataFrame rows to DSPy examples."""
    examples = []
    
    for _, row in df.iterrows():
        # Create DSPy example with inputs and outputs marked
        example = dspy.Example(
            title=row['Title'],
            is_favorite=row['has_favorite']
        ).with_inputs('title')
        
        examples.append(example)
    
    return examples


def split_data(df: pd.DataFrame, train_ratio: float = 0.7, val_ratio: float = 0.15, 
               test_ratio: float = 0.15, seed: int = 42, balanced: bool = True) -> Dict[str, pd.DataFrame]:
    """Split data into train/validation/test sets with optional balancing."""
    assert abs(train_ratio + val_ratio + test_ratio - 1.0) < 0.001, "Ratios must sum to 1"
    
    if balanced:
        # Create balanced dataset by undersampling majority class
        favorites = df[df['has_favorite'] == True]
        non_favorites = df[df['has_favorite'] == False]
        
        # Use the smaller class size as the target
        min_class_size = min(len(favorites), len(non_favorites))
        
        print(f"Balancing dataset: {len(favorites)} favorites, {len(non_favorites)} non-favorites")
        print(f"Using {min_class_size} samples per class")
        
        # Sample equal numbers from each class
        balanced_favorites = favorites.sample(n=min_class_size, random_state=seed)
        balanced_non_favorites = non_favorites.sample(n=min_class_size, random_state=seed)
        
        # Combine and shuffle
        balanced_df = pd.concat([balanced_favorites, balanced_non_favorites])
        df = balanced_df.sample(frac=1, random_state=seed).reset_index(drop=True)
        
        print(f"Balanced dataset: {len(df)} total samples, {df['has_favorite'].mean():.2%} favorites")
    else:
        # Keep original distribution
        df = df.sample(frac=1, random_state=seed).reset_index(drop=True)
    
    # Calculate split indices
    n = len(df)
    train_end = int(n * train_ratio)
    val_end = train_end + int(n * val_ratio)
    
    # Create splits
    splits = {
        'train': df[:train_end],
        'val': df[train_end:val_end],
        'test': df[val_end:]
    }
    
    # Verify splits
    for split_name, split_df in splits.items():
        fav_ratio = split_df['has_favorite'].mean()
        print(f"{split_name}: {len(split_df)} samples, {fav_ratio:.2%} favorites")
    
    return splits


def save_splits(splits: Dict[str, pd.DataFrame], output_dir: Path):
    """Save data splits in multiple formats."""
    
    for split_name, split_df in splits.items():
        split_dir = output_dir / split_name
        split_dir.mkdir(parents=True, exist_ok=True)
        
        # Save as CSV (for compatibility with existing code)
        csv_path = split_dir / 'favorite_classifier.csv'
        split_df.to_csv(csv_path, index=False)
        print(f"Saved {split_name} CSV: {len(split_df)} samples -> {csv_path}")
        
        # Convert to DSPy examples
        examples = create_dspy_examples(split_df)
        
        # Save as JSON (DSPy format)
        json_path = split_dir / 'dspy_examples.json'
        examples_dict = []
        for ex in examples:
            examples_dict.append({
                'title': ex.title,
                'is_favorite': ex.is_favorite
            })
        
        with open(json_path, 'w') as f:
            json.dump(examples_dict, f, indent=2)
        print(f"Saved {split_name} DSPy examples: {len(examples)} -> {json_path}")


def create_summary(splits: Dict[str, pd.DataFrame], output_dir: Path):
    """Create a summary of the prepared data."""
    summary = {
        'task': 'DSPy Favorite Classification',
        'description': 'Binary classification to predict if an article will be marked as favorite',
        'data_source': 'data/raw/export.csv',
        'splits': {}
    }
    
    for split_name, split_df in splits.items():
        summary['splits'][split_name] = {
            'total_samples': len(split_df),
            'positive_samples': int(split_df['has_favorite'].sum()),
            'negative_samples': int((~split_df['has_favorite']).sum()),
            'positive_ratio': float(split_df['has_favorite'].mean()),
            'files': {
                'csv': f"{split_name}/favorite_classifier.csv",
                'dspy_examples': f"{split_name}/dspy_examples.json"
            }
        }
    
    # Save summary
    summary_path = output_dir / 'dspy_favorite_summary.json'
    with open(summary_path, 'w') as f:
        json.dump(summary, f, indent=2)
    
    print(f"\nSummary saved to: {summary_path}")
    return summary


def main():
    """Main data preparation function."""
    parser = argparse.ArgumentParser(description="Prepare data for DSPy favorite classifier")
    parser.add_argument('--data', type=str, default='data/raw/export.csv',
                        help='Path to raw data file')
    parser.add_argument('--output', type=str, default='data/processed/reader_favorite',
                        help='Output directory for processed datasets')
    parser.add_argument('--train-ratio', type=float, default=0.7,
                        help='Training set ratio')
    parser.add_argument('--val-ratio', type=float, default=0.15,
                        help='Validation set ratio')
    parser.add_argument('--test-ratio', type=float, default=0.15,
                        help='Test set ratio')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed for reproducibility')
    parser.add_argument('--balanced', action='store_true', default=True,
                        help='Create balanced dataset by undersampling majority class')
    parser.add_argument('--unbalanced', action='store_true',
                        help='Keep original class distribution')
    
    args = parser.parse_args()
    
    # Determine if we want balanced data
    balanced = not args.unbalanced  # Default to balanced unless --unbalanced is specified
    
    # Set up paths (scripts is 4 levels deep from project root)
    project_root = Path(__file__).parent.parent.parent.parent.parent
    raw_csv_path = project_root / args.data
    output_dir = project_root / args.output
    
    # Verify raw data exists
    if not raw_csv_path.exists():
        print(f"ERROR: Raw data file not found: {raw_csv_path}")
        sys.exit(1)
    
    print("="*60)
    print("DSPy FAVORITE CLASSIFIER DATA PREPARATION")
    print("="*60)
    
    # Load and process data
    df = load_and_process_data(str(raw_csv_path))
    
    # Create splits
    splits = split_data(df, args.train_ratio, args.val_ratio, args.test_ratio, args.seed, balanced)
    
    # Save splits
    save_splits(splits, output_dir)
    
    # Create summary
    summary = create_summary(splits, output_dir)
    
    print("\n" + "="*60)
    print("DATA PREPARATION COMPLETE")
    print("="*60)
    print(f"\nDataset Statistics:")
    for split_name, info in summary['splits'].items():
        print(f"\n{split_name.upper()}:")
        print(f"  Total: {info['total_samples']} samples")
        print(f"  Positive: {info['positive_samples']} ({info['positive_ratio']:.2%})")
        print(f"  Negative: {info['negative_samples']} ({100-info['positive_ratio']*100:.2%})")
    
    print("\n✅ Data preparation completed successfully!")
    print("\nNext steps:")
    print("1. Review the data summary in:", output_dir / 'dspy_favorite_summary.json')
    print("2. Run evaluation script to establish baseline:", 
          "python scripts/in_context_taste_models/dspy_favorite/evaluate.py")
    print("3. Train the model with:", 
          "python scripts/in_context_taste_models/dspy_favorite/train.py")


if __name__ == "__main__":
    main()