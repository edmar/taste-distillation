#!/usr/bin/env python3
"""
Unified data preparation script for Readwise Reader export data.
Generates all classification and pairwise datasets.
"""

import os
import sys
from pathlib import Path
import argparse
import random

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from reader.base import (
    load_readwise_export, filter_by_date, filter_by_location, 
    deduplicate_by_title, split_data, balance_binary_data,
    generate_pairwise_examples, save_classification_splits,
    save_pairwise_splits, create_classification_summary
)
from reader.favorite import prepare_favorite_data, FAVORITE_CONFIG
from reader.shortlist import prepare_shortlist_data, SHORTLIST_CONFIG


def prepare_classification_dataset(df, config, output_base_dir, csv_path, days_override=None, seed=42):
    """Prepare a single classification dataset (favorite or shortlist)."""
    
    # Apply task-specific preparation
    if config['dataset_name'] == 'favorite':
        df = prepare_favorite_data(df)
    else:  # shortlist
        df = prepare_shortlist_data(df)
    
    # Apply date filter if specified (use override if provided)
    days_filter = days_override if days_override is not None else config['days_filter']
    if days_filter:
        df = filter_by_date(df, days_filter)
    
    # Apply location filter if specified
    if config['excluded_locations']:
        df = filter_by_location(df, config['excluded_locations'])
    
    # Deduplicate
    df = deduplicate_by_title(df)
    
    # Balance data if requested
    if config['balance_data']:
        df = balance_binary_data(df, config['label_column'], seed)
    
    # Create splits
    splits = split_data(df, seed=seed)
    
    # Save splits
    output_dir = output_base_dir / f"reader_{config['dataset_name']}"
    save_classification_splits(splits, output_dir, config['label_column'], config['dataset_name'])
    
    # Create summary
    days_desc = f" (last {days_filter} days)" if days_filter else ""
    description = config['task_description'] + days_desc
    
    summary = create_classification_summary(
        splits, output_dir, csv_path, config['label_column'],
        config['task_name'], description, days_filter
    )
    
    # Print statistics
    total_positive = sum(info[config['dataset_name']] for info in summary['splits'].values())
    total_examples = sum(info['total_examples'] for info in summary['splits'].values())
    
    print(f"\n{config['task_name']} Statistics:")
    print(f"  Total examples: {total_examples}")
    print(f"  {config['task_name']}: {total_positive} ({total_positive/total_examples:.1%})")
    print(f"  Non-{config['dataset_name']}: {total_examples - total_positive}")
    
    return splits


def prepare_pairwise_dataset(classification_splits, config, output_base_dir, csv_path, max_pairs=2000, seed=42):
    """Prepare a pairwise dataset from classification splits."""
    
    # Combine all splits to create comprehensive pairwise dataset
    import pandas as pd
    all_data = pd.concat(classification_splits.values(), ignore_index=True)
    
    # Generate pairwise examples
    pairs = generate_pairwise_examples(all_data, config['label_column'], max_pairs, seed)
    
    # Split pairwise data
    random.seed(seed)
    shuffled_pairs = pairs.copy()
    random.shuffle(shuffled_pairs)
    
    # Create splits
    n = len(shuffled_pairs)
    train_end = int(n * 0.7)
    val_end = train_end + int(n * 0.15)
    
    pairwise_splits = {
        'train': shuffled_pairs[:train_end],
        'val': shuffled_pairs[train_end:val_end],
        'test': shuffled_pairs[val_end:]
    }
    
    # Save splits
    output_dir = output_base_dir / f"reader_{config['dataset_name']}_pairwise"
    save_pairwise_splits(pairwise_splits, output_dir)
    
    # Create summary
    from datetime import datetime
    import json
    
    summary = {
        'task': f"{config['task_name']} Pairwise Classification",
        'description': f"Pairwise comparison to predict which article is more likely to be {config['dataset_name']}",
        'source_data': csv_path,
        'created_at': datetime.now().isoformat(),
        'splits': {}
    }
    
    for split_name, split_pairs in pairwise_splits.items():
        # Count preferences
        a_preferred = sum(1 for pair in split_pairs if pair['preferred_title'] == 'A')
        b_preferred = len(split_pairs) - a_preferred
        
        summary['splits'][split_name] = {
            'total_pairs': len(split_pairs),
            'a_preferred': a_preferred,
            'b_preferred': b_preferred,
            'preference_balance': a_preferred / len(split_pairs) if split_pairs else 0,
            'files': {
                'dspy_format': f"{split_name}/dspy_examples.json"
            }
        }
    
    # Save summary
    summary_path = output_dir / f'{config["dataset_name"]}_pairwise_summary.json'
    with open(summary_path, 'w') as f:
        json.dump(summary, f, indent=2)
    
    print(f"\n{config['task_name']} Pairwise Statistics:")
    print(f"  Total pairs: {len(pairs)}")
    for split_name, split_info in summary['splits'].items():
        balance = split_info['preference_balance']
        print(f"  {split_name}: {split_info['total_pairs']} pairs (A:{balance:.1%}, B:{1-balance:.1%})")
    
    return pairwise_splits


def main():
    """Main function to prepare all datasets."""
    parser = argparse.ArgumentParser(description="Prepare all Readwise Reader datasets")
    parser.add_argument('--data', type=str, required=True,
                        help='Path to Readwise Reader export CSV file')
    parser.add_argument('--output', type=str, default='data',
                        help='Output directory for processed data')
    parser.add_argument('--days', type=int, default=None,
                        help='Number of days to include (default: None for favorite, 90 for shortlist)')
    parser.add_argument('--max-pairs', type=int, default=2000,
                        help='Maximum number of pairwise comparisons per dataset')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed for reproducibility')
    parser.add_argument('--datasets', nargs='+', 
                        choices=['favorite', 'shortlist', 'all'],
                        default=['all'],
                        help='Which datasets to prepare')
    
    args = parser.parse_args()
    
    # Set up paths
    csv_path = Path(args.data)
    output_base_dir = Path(args.output)
    
    # Verify data exists
    if not csv_path.exists():
        print(f"ERROR: Data file not found: {csv_path}")
        sys.exit(1)
    
    print("="*60)
    print("READWISE READER DATA PREPARATION")
    print("="*60)
    print(f"Data source: {csv_path}")
    print(f"Output directory: {output_base_dir}")
    
    # Load data once
    df_all = load_readwise_export(str(csv_path))
    
    # Determine which datasets to prepare
    if 'all' in args.datasets:
        datasets = ['favorite', 'shortlist']
    else:
        datasets = args.datasets
    
    # Prepare each dataset
    for dataset_name in datasets:
        print(f"\n{'='*40}")
        print(f"Preparing {dataset_name.upper()} datasets...")
        print(f"{'='*40}")
        
        # Get config
        config = FAVORITE_CONFIG if dataset_name == 'favorite' else SHORTLIST_CONFIG
        
        # Prepare classification dataset
        splits = prepare_classification_dataset(
            df_all.copy(), config, output_base_dir, str(csv_path), args.days, args.seed
        )
        
        # Prepare pairwise dataset
        prepare_pairwise_dataset(
            splits, config, output_base_dir, str(csv_path), args.max_pairs, args.seed
        )
    
    print(f"\n{'='*60}")
    print(" All datasets prepared successfully!")
    print(f"Output directory: {output_base_dir}")
    
    # Print next steps
    print("\nGenerated datasets:")
    for dataset_name in datasets:
        print(f"\n{dataset_name.capitalize()}:")
        print(f"  - reader_{dataset_name}/           # Classification dataset")
        print(f"  - reader_{dataset_name}_pairwise/  # Pairwise comparison dataset")


if __name__ == "__main__":
    main()