#!/usr/bin/env python3
"""
Prepare pairwise dataset from HN scraper output.

This script converts the HN dataset (voted/non-voted posts) into pairwise comparisons
for training preference models.
"""

import os
import sys
import json
import pandas as pd
import numpy as np
from pathlib import Path
import argparse
import random
from typing import List, Dict, Tuple
from datetime import datetime


def load_hn_data(csv_path: str) -> pd.DataFrame:
    """Load and validate HN dataset."""
    print(f"Loading HN data from {csv_path}...")
    
    # Load CSV
    df = pd.read_csv(csv_path)
    
    # Validate required columns
    required_cols = ['id', 'title', 'your_vote', 'source', 'domain', 'score']
    missing_cols = [col for col in required_cols if col not in df.columns]
    if missing_cols:
        raise ValueError(f"Missing required columns: {missing_cols}")
    
    # Filter out empty titles
    df = df[df['title'].notna() & (df['title'].str.strip() != '')]
    
    # Remove duplicates by ID
    df = df.drop_duplicates(subset=['id'], keep='first')
    
    print(f"Loaded {len(df)} unique posts")
    print(f"Upvoted: {df['your_vote'].sum()} ({df['your_vote'].mean():.2%})")
    print(f"Sources: {df['source'].value_counts().to_dict()}")
    
    return df


def generate_pairwise_examples(df: pd.DataFrame, max_pairs: int = 5000, 
                              strategy: str = 'balanced', seed: int = 42) -> List[Dict]:
    """
    Generate pairwise comparison examples from the dataset.
    
    Strategies:
    - balanced: Equal mix of clear (voted vs non-voted) and subtle (same class) pairs
    - clear: Only voted vs non-voted pairs
    - all: Include all possible pair types
    """
    print(f"\nGenerating pairwise examples using '{strategy}' strategy...")
    
    random.seed(seed)
    np.random.seed(seed)
    
    voted = df[df['your_vote'] == 1]
    non_voted = df[df['your_vote'] == 0]
    
    print(f"Available: {len(voted)} voted, {len(non_voted)} non-voted posts")
    
    pairs = []
    
    if strategy in ['balanced', 'clear', 'all']:
        # Generate clear signal pairs: voted vs non-voted
        print("Generating clear preference pairs (voted vs non-voted)...")
        
        for i, voted_row in voted.iterrows():
            # Sample random non-voted posts to pair with this voted post
            sample_size = min(5, len(non_voted))  # Limit to avoid too many pairs
            
            if sample_size > 0:
                sampled_non_voted = non_voted.sample(n=sample_size, random_state=seed+i)
                
                for j, non_voted_row in sampled_non_voted.iterrows():
                    # Create pair with metadata
                    pair = create_pair(voted_row, non_voted_row, preference='voted')
                    pairs.append(pair)
                    
                    if len(pairs) >= max_pairs and strategy == 'clear':
                        break
            
            if len(pairs) >= max_pairs and strategy == 'clear':
                break
    
    if strategy in ['balanced', 'all'] and len(pairs) < max_pairs:
        # Generate subtle pairs: voted vs voted
        print("Generating subtle preference pairs (voted vs voted)...")
        
        # Use score as a proxy for preference among voted posts
        voted_sorted = voted.sort_values('score', ascending=False)
        
        for i in range(len(voted_sorted) - 1):
            for j in range(i + 1, min(i + 6, len(voted_sorted))):  # Limit comparisons
                high_score = voted_sorted.iloc[i]
                low_score = voted_sorted.iloc[j]
                
                # Only create pair if score difference is significant
                if high_score['score'] > low_score['score'] * 1.5:
                    pair = create_pair(high_score, low_score, preference='higher_score', 
                                     confidence='medium')
                    pairs.append(pair)
                    
                    if len(pairs) >= max_pairs:
                        break
            
            if len(pairs) >= max_pairs:
                break
    
    if strategy == 'all' and len(pairs) < max_pairs:
        # Generate hard pairs: non-voted vs non-voted (no clear preference)
        print("Generating neutral pairs (non-voted vs non-voted)...")
        
        for i in range(min(100, len(non_voted) - 1)):  # Limit these pairs
            j = i + 1
            if j < len(non_voted):
                pair = create_pair(non_voted.iloc[i], non_voted.iloc[j], 
                                 preference='neutral', confidence='low')
                pairs.append(pair)
                
                if len(pairs) >= max_pairs:
                    break
    
    # Shuffle pairs to mix different types
    random.shuffle(pairs)
    
    print(f"Generated {len(pairs)} pairwise examples")
    
    # Print statistics
    pref_counts = {}
    conf_counts = {}
    for pair in pairs:
        pref = pair['preference_reason']
        conf = pair['confidence']
        pref_counts[pref] = pref_counts.get(pref, 0) + 1
        conf_counts[conf] = conf_counts.get(conf, 0) + 1
    
    print(f"Preference types: {pref_counts}")
    print(f"Confidence levels: {conf_counts}")
    
    return pairs[:max_pairs]


def create_pair(row_a: pd.Series, row_b: pd.Series, preference: str, 
                confidence: str = 'high') -> Dict:
    """Create a pairwise comparison example."""
    
    # Determine which post is preferred
    if preference == 'voted':
        # A is voted, B is not voted
        if row_a['your_vote'] == 1:
            preferred = 'A'
        else:
            # Swap so voted is always preferred
            row_a, row_b = row_b, row_a
            preferred = 'A'
    elif preference == 'higher_score':
        # Higher score is preferred
        if row_a['score'] >= row_b['score']:
            preferred = 'A'
        else:
            row_a, row_b = row_b, row_a
            preferred = 'A'
    else:
        # Neutral or random
        if random.random() < 0.5:
            preferred = 'A'
        else:
            preferred = 'B'
            confidence = 'low'
    
    # Randomly swap positions to avoid position bias
    if random.random() < 0.5:
        row_a, row_b = row_b, row_a
        preferred = 'B' if preferred == 'A' else 'A'
    
    return {
        'title_a': row_a['title'],
        'title_b': row_b['title'],
        'url_a': row_a.get('url', ''),
        'url_b': row_b.get('url', ''),
        'domain_a': row_a.get('domain', ''),
        'domain_b': row_b.get('domain', ''),
        'score_a': int(row_a.get('score', 0)),
        'score_b': int(row_b.get('score', 0)),
        'voted_a': int(row_a['your_vote']),
        'voted_b': int(row_b['your_vote']),
        'preference': preferred,
        'confidence': confidence,
        'preference_reason': preference
    }


def split_pairwise_data(pairs: List[Dict], train_ratio: float = 0.7, 
                       val_ratio: float = 0.15, test_ratio: float = 0.15, 
                       seed: int = 42) -> Dict[str, List[Dict]]:
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
        
        # Save full pairwise data as JSON
        json_path = split_dir / 'hn_pairwise.json'
        with open(json_path, 'w') as f:
            json.dump(split_pairs, f, indent=2)
        print(f"Saved {split_name} JSON: {len(split_pairs)} pairs -> {json_path}")
        
        # Save minimal format for DSPy compatibility
        minimal_pairs = []
        for pair in split_pairs:
            minimal_pairs.append({
                'title_a': pair['title_a'],
                'title_b': pair['title_b'],
                'preferred_title': pair['preference'],
                'confidence': pair['confidence']
            })
        
        dspy_json_path = split_dir / 'dspy_examples.json'
        with open(dspy_json_path, 'w') as f:
            json.dump(minimal_pairs, f, indent=2)
        print(f"Saved {split_name} DSPy examples: {len(minimal_pairs)} -> {dspy_json_path}")


def create_summary(splits: Dict[str, List[Dict]], output_dir: Path, source_file: str):
    """Create a summary of the prepared data."""
    summary = {
        'task': 'HN Pairwise Preference Learning',
        'description': 'Pairwise comparison to predict which HN post user would upvote',
        'data_source': source_file,
        'created_at': datetime.now().isoformat(),
        'splits': {}
    }
    
    for split_name, split_pairs in splits.items():
        # Count various statistics
        confidence_counts = {}
        preference_counts = {'A': 0, 'B': 0}
        preference_reason_counts = {}
        voted_vs_not = 0
        both_voted = 0
        neither_voted = 0
        
        for pair in split_pairs:
            # Confidence
            conf = pair['confidence']
            confidence_counts[conf] = confidence_counts.get(conf, 0) + 1
            
            # Preference
            pref = pair['preference']
            preference_counts[pref] += 1
            
            # Preference reason
            reason = pair['preference_reason']
            preference_reason_counts[reason] = preference_reason_counts.get(reason, 0) + 1
            
            # Vote patterns
            votes_a = pair['voted_a']
            votes_b = pair['voted_b']
            if votes_a == 1 and votes_b == 0:
                voted_vs_not += 1
            elif votes_a == 0 and votes_b == 1:
                voted_vs_not += 1
            elif votes_a == 1 and votes_b == 1:
                both_voted += 1
            else:
                neither_voted += 1
        
        summary['splits'][split_name] = {
            'total_pairs': len(split_pairs),
            'preference_distribution': preference_counts,
            'confidence_distribution': confidence_counts,
            'preference_reasons': preference_reason_counts,
            'vote_patterns': {
                'voted_vs_not_voted': voted_vs_not,
                'both_voted': both_voted,
                'neither_voted': neither_voted
            },
            'files': {
                'full_data': f"{split_name}/hn_pairwise.json",
                'dspy_format': f"{split_name}/dspy_examples.json"
            }
        }
    
    # Save summary
    summary_path = output_dir / 'hn_pairwise_summary.json'
    with open(summary_path, 'w') as f:
        json.dump(summary, f, indent=2)
    
    print(f"\nSummary saved to: {summary_path}")
    return summary


def main():
    """Main data preparation function."""
    parser = argparse.ArgumentParser(description="Prepare HN data for pairwise preference learning")
    parser.add_argument('--data', type=str, required=True,
                        help='Path to HN dataset CSV file')
    parser.add_argument('--output', type=str, default='data/processed/hn_pairwise',
                        help='Output directory for processed datasets')
    parser.add_argument('--max-pairs', type=int, default=5000,
                        help='Maximum number of pairwise examples to generate')
    parser.add_argument('--strategy', type=str, default='balanced',
                        choices=['balanced', 'clear', 'all'],
                        help='Pairing strategy: balanced, clear (only voted vs non-voted), or all')
    parser.add_argument('--train-ratio', type=float, default=0.7,
                        help='Training set ratio')
    parser.add_argument('--val-ratio', type=float, default=0.15,
                        help='Validation set ratio')
    parser.add_argument('--test-ratio', type=float, default=0.15,
                        help='Test set ratio')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed for reproducibility')
    
    args = parser.parse_args()
    
    # Set up paths
    if not os.path.isabs(args.data):
        # If relative path, assume from project root
        project_root = Path(__file__).parent.parent.parent
        csv_path = project_root / args.data
    else:
        csv_path = Path(args.data)
    
    if not os.path.isabs(args.output):
        project_root = Path(__file__).parent.parent.parent
        output_dir = project_root / args.output
    else:
        output_dir = Path(args.output)
    
    # Verify data exists
    if not csv_path.exists():
        print(f"ERROR: Data file not found: {csv_path}")
        sys.exit(1)
    
    print("="*60)
    print("HN PAIRWISE DATASET PREPARATION")
    print("="*60)
    
    # Load and process data
    df = load_hn_data(str(csv_path))
    
    # Generate pairwise examples
    pairs = generate_pairwise_examples(df, args.max_pairs, args.strategy, args.seed)
    
    # Create splits
    splits = split_pairwise_data(pairs, args.train_ratio, args.val_ratio, 
                                args.test_ratio, args.seed)
    
    # Save splits
    save_splits(splits, output_dir)
    
    # Create summary
    summary = create_summary(splits, output_dir, str(csv_path))
    
    print("\n" + "="*60)
    print("DATA PREPARATION COMPLETE")
    print("="*60)
    print(f"\nDataset Statistics:")
    for split_name, info in summary['splits'].items():
        print(f"\n{split_name.upper()}:")
        print(f"  Total pairs: {info['total_pairs']}")
        print(f"  Preferences: A={info['preference_distribution']['A']}, "
              f"B={info['preference_distribution']['B']}")
        print(f"  Confidence: {info['confidence_distribution']}")
        print(f"  Vote patterns: {info['vote_patterns']}")
    
    print("\n✅ Pairwise data preparation completed successfully!")
    print("\nNext steps:")
    print("1. Review the data summary in:", output_dir / 'hn_pairwise_summary.json')
    print("2. Use the data for training preference models")
    print("\nExample usage:")
    print(f"python lib/hn_scraper/prepare.py --data {args.data}")


if __name__ == "__main__":
    main()