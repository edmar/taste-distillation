#!/usr/bin/env python3
"""
DSPy Data preparation script for pairwise classifier.

This script loads the raw export.csv file and creates train/valid/test splits
specifically for the dspy_pairwise model following DSPy standards.
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
import random


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


def generate_pairwise_examples(df: pd.DataFrame, max_pairs: int = 2000, 
                              seed: int = 42) -> List[Dict]:
    """Generate pairwise comparison examples from the dataset using mixed strategy."""
    print(f"Generating pairwise examples using 'mixed' strategy...")
    
    random.seed(seed)
    np.random.seed(seed)
    
    favorites = df[df['has_favorite'] == True]
    non_favorites = df[df['has_favorite'] == False]
    
    print(f"Available: {len(favorites)} favorites, {len(non_favorites)} non-favorites")
    
    pairs = []
    
    # Create clear signal pairs: favorite vs non-favorite
    for i, fav_row in favorites.iterrows():
        # Sample random non-favorites to pair with this favorite
        sample_size = min(5, len(non_favorites))  # Limit to avoid too many pairs
        sampled_non_favs = non_favorites.sample(n=sample_size, random_state=seed+i)
        
        for j, non_fav_row in sampled_non_favs.iterrows():
            # Randomly assign A/B positions
            if random.random() < 0.5:
                pair = {
                    'title_a': fav_row['Title'],
                    'title_b': non_fav_row['Title'],
                    'preference': 'A',  # Favorite is A
                    'confidence': 'high'
                }
            else:
                pair = {
                    'title_a': non_fav_row['Title'],
                    'title_b': fav_row['Title'],
                    'preference': 'B',  # Favorite is B
                    'confidence': 'high'
                }
            
            pairs.append(pair)
            
            # Limit total pairs
            if len(pairs) >= max_pairs:
                break
        
        if len(pairs) >= max_pairs:
            break
    
    print(f"Generated {len(pairs)} pairwise examples")
    return pairs


def create_dspy_examples(pairs: List[Dict]) -> List[dspy.Example]:
    """Convert pairwise examples to DSPy examples."""
    examples = []
    
    for pair in pairs:
        # Create DSPy example with inputs and outputs marked
        example = dspy.Example(
            title_a=pair['title_a'],
            title_b=pair['title_b'],
            preferred_title=pair['preference'],
            confidence=pair['confidence']
        ).with_inputs('title_a', 'title_b')
        
        examples.append(example)
    
    return examples


def split_pairwise_data(pairs: List[Dict], train_ratio: float = 0.7, val_ratio: float = 0.15, 
                       test_ratio: float = 0.15, seed: int = 42) -> Dict[str, List[Dict]]:
    """Split pairwise data into train/validation/test sets."""
    assert abs(train_ratio + val_ratio + test_ratio - 1.0) < 0.001, "Ratios must sum to 1"
    
    # Shuffle pairs
    random.seed(seed)
    shuffled_pairs = pairs.copy()
    random.shuffle(shuffled_pairs)
    
    # Calculate split indices
    n = len(shuffled_pairs)
    train_end = int(n * train_ratio)
    val_end = train_end + int(n * val_ratio)
    
    # Create splits
    splits = {
        'train': shuffled_pairs[:train_end],
        'val': shuffled_pairs[train_end:val_end],
        'test': shuffled_pairs[val_end:]
    }
    
    # Verify splits
    for split_name, split_pairs in splits.items():
        print(f"{split_name}: {len(split_pairs)} pairs")
    
    return splits


def save_splits(splits: Dict[str, List[Dict]], output_dir: Path):
    """Save data splits in multiple formats."""
    
    for split_name, split_pairs in splits.items():
        split_dir = output_dir / split_name
        split_dir.mkdir(parents=True, exist_ok=True)
        
        # Save as JSON (for pairwise comparison format)
        json_path = split_dir / 'pairwise_classifier.json'
        with open(json_path, 'w') as f:
            json.dump(split_pairs, f, indent=2)
        print(f"Saved {split_name} JSON: {len(split_pairs)} pairs -> {json_path}")
        
        # Convert to DSPy examples
        examples = create_dspy_examples(split_pairs)
        
        # Save as DSPy examples JSON
        dspy_json_path = split_dir / 'dspy_examples.json'
        examples_dict = []
        for ex in examples:
            examples_dict.append({
                'title_a': ex.title_a,
                'title_b': ex.title_b,
                'preferred_title': ex.preferred_title,
                'confidence': ex.confidence
            })
        
        with open(dspy_json_path, 'w') as f:
            json.dump(examples_dict, f, indent=2)
        print(f"Saved {split_name} DSPy examples: {len(examples)} -> {dspy_json_path}")


def create_summary(splits: Dict[str, List[Dict]], output_dir: Path):
    """Create a summary of the prepared data."""
    summary = {
        'task': 'DSPy Pairwise Classification',
        'description': 'Pairwise comparison to predict which article is more likely to be marked as favorite',
        'data_source': 'data/raw/export.csv',
        'splits': {}
    }
    
    for split_name, split_pairs in splits.items():
        # Count confidence levels
        confidence_counts = {}
        for pair in split_pairs:
            conf = pair['confidence']
            confidence_counts[conf] = confidence_counts.get(conf, 0) + 1
        
        # Count preferences
        pref_counts = {'A': 0, 'B': 0}
        for pair in split_pairs:
            pref = pair['preference']
            pref_counts[pref] = pref_counts.get(pref, 0) + 1
        
        summary['splits'][split_name] = {
            'total_pairs': len(split_pairs),
            'preference_distribution': pref_counts,
            'confidence_distribution': confidence_counts,
            'files': {
                'json': f"{split_name}/pairwise_classifier.json",
                'dspy_examples': f"{split_name}/dspy_examples.json"
            }
        }
    
    # Save summary
    summary_path = output_dir / 'dspy_pairwise_summary.json'
    with open(summary_path, 'w') as f:
        json.dump(summary, f, indent=2)
    
    print(f"\nSummary saved to: {summary_path}")
    return summary


def main():
    """Main data preparation function."""
    parser = argparse.ArgumentParser(description="Prepare data for DSPy pairwise classifier")
    parser.add_argument('--data', type=str, default='data/raw/export.csv',
                        help='Path to raw data file')
    parser.add_argument('--output', type=str, default='data/processed/dspy_pairwise',
                        help='Output directory for processed datasets')
    parser.add_argument('--max-pairs', type=int, default=2000,
                        help='Maximum number of pairwise examples to generate')
    parser.add_argument('--train-ratio', type=float, default=0.7,
                        help='Training set ratio')
    parser.add_argument('--val-ratio', type=float, default=0.15,
                        help='Validation set ratio')
    parser.add_argument('--test-ratio', type=float, default=0.15,
                        help='Test set ratio')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed for reproducibility')
    
    args = parser.parse_args()
    
    # Set up paths (scripts is 4 levels deep from project root)
    project_root = Path(__file__).parent.parent.parent.parent.parent
    raw_csv_path = project_root / args.data
    output_dir = project_root / args.output
    
    # Verify raw data exists
    if not raw_csv_path.exists():
        print(f"ERROR: Raw data file not found: {raw_csv_path}")
        sys.exit(1)
    
    print("="*60)
    print("DSPy PAIRWISE CLASSIFIER DATA PREPARATION")
    print("="*60)
    
    # Load and process data
    df = load_and_process_data(str(raw_csv_path))
    
    # Generate pairwise examples
    pairs = generate_pairwise_examples(df, args.max_pairs, args.seed)
    
    # Create splits
    splits = split_pairwise_data(pairs, args.train_ratio, args.val_ratio, args.test_ratio, args.seed)
    
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
        print(f"  Total: {info['total_pairs']} pairs")
        print(f"  Preference A: {info['preference_distribution']['A']}")
        print(f"  Preference B: {info['preference_distribution']['B']}")
        print(f"  Confidence: {info['confidence_distribution']}")
    
    print("\n✅ Data preparation completed successfully!")
    print("\nNext steps:")
    print("1. Review the data summary in:", output_dir / 'dspy_pairwise_summary.json')
    print("2. Train the model with:", 
          "python src/in_context_taste_models/dspy_pairwise/scripts/train.py")
    print("3. Evaluate the trained model with:", 
          "python src/in_context_taste_models/dspy_pairwise/scripts/evaluate.py")


if __name__ == "__main__":
    main()