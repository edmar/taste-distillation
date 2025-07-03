#!/usr/bin/env python3
"""
Data preparation script for pairwise classifier.

This script loads the raw export.csv file and creates train/valid/test splits
specifically for the pairwise_classifier model following DSPy standards.
"""

import os
import sys
import json
import pandas as pd
from pathlib import Path

# Add src directory to path to import data loader
project_root = Path(__file__).parent.parent.parent
sys.path.append(str(project_root / 'src'))

from pairwise_classifier.pairwise_data_loader import PairwiseDataLoader


def prepare_pairwise_classifier_data(raw_csv_path: str, output_dir: Path):
    """Prepare data for pairwise classification task"""
    print("\n" + "="*60)
    print("PREPARING PAIRWISE CLASSIFIER DATA")
    print("="*60)
    
    # Load and process data
    loader = PairwiseDataLoader(raw_csv_path)
    data = loader.load_and_process()
    
    # Generate pairwise examples - using mixed strategy for clear signal
    pairwise_examples = loader.generate_pairwise_examples(
        max_examples=2000,
        balanced=True,
        pair_strategy='mixed',  # Only clear fav vs non-fav signals
        augment_flip=False  # Don't augment to avoid overfitting
    )
    
    # Create splits
    splits = loader.create_splits(
        pairwise_examples,
        train_size=0.7,
        val_size=0.15,
        test_size=0.15,
        stratify_by='confidence'
    )
    
    # Save splits as JSON files
    for split_name, split_examples in splits.items():
        if len(split_examples) > 0:
            # Ensure directory exists
            split_dir = output_dir / split_name
            split_dir.mkdir(parents=True, exist_ok=True)
            
            output_path = split_dir / 'pairwise_classifier.json'
            with open(output_path, 'w') as f:
                json.dump(split_examples, f, indent=2)
            print(f"Saved {split_name} split: {len(split_examples)} pairs -> {output_path}")
    
    return splits


def create_data_summary(splits, output_dir: Path):
    """Create a summary of the prepared data"""
    summary = {
        'preparation_info': {
            'script': 'scripts/pairwise_classifier/prepare.py',
            'raw_data': 'data/raw/export.csv',
            'processed_data_dir': 'data/processed/',
        },
        'pairwise_classifier': {
            'task': 'Pairwise comparison - which article is more likely to be favorite',
            'train_pairs': len(splits['train']) if 'train' in splits else 0,
            'val_pairs': len(splits['val']) if 'val' in splits else 0,
            'test_pairs': len(splits['test']) if 'test' in splits else 0,
            'files': [
                'train/pairwise_classifier.json',
                'valid/pairwise_classifier.json', 
                'test/pairwise_classifier.json'
            ]
        }
    }
    
    # Save summary
    summary_path = output_dir / 'pairwise_classifier_summary.json'
    with open(summary_path, 'w') as f:
        json.dump(summary, f, indent=2)
    
    print("\n" + "="*60)
    print("PAIRWISE CLASSIFIER DATA PREPARATION SUMMARY")
    print("="*60)
    print(f"Raw data: data/raw/export.csv")
    print(f"Processed data: data/processed/")
    print(f"Summary saved: {summary_path}")
    
    print(f"\nDataset Statistics:")
    print(f"  Train: {summary['pairwise_classifier']['train_pairs']} pairs")
    print(f"  Valid: {summary['pairwise_classifier']['val_pairs']} pairs")
    print(f"  Test:  {summary['pairwise_classifier']['test_pairs']} pairs")


def main():
    """Main data preparation function"""
    print("Starting data preparation for pairwise classifier...")
    
    # Set up paths
    project_root = Path(__file__).parent.parent.parent
    raw_csv_path = project_root / 'data' / 'raw' / 'export.csv'
    processed_dir = project_root / 'data' / 'processed'
    
    # Verify raw data exists
    if not raw_csv_path.exists():
        print(f"ERROR: Raw data file not found: {raw_csv_path}")
        print("Please ensure export.csv is in data/raw/ directory")
        sys.exit(1)
    
    print(f"Raw data: {raw_csv_path}")
    print(f"Output directory: {processed_dir}")
    
    # Prepare data
    splits = prepare_pairwise_classifier_data(str(raw_csv_path), processed_dir)
    
    # Create summary
    create_data_summary(splits, processed_dir)
    
    print("\n✅ Pairwise classifier data preparation completed successfully!")
    print("\nNext steps:")
    print("1. Review data splits in data/processed/")
    print("2. Run training script: scripts/pairwise_classifier/train.py")


if __name__ == "__main__":
    main()