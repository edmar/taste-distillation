#!/usr/bin/env python3
"""
Data preparation script for favorite classifier.

This script loads the raw export.csv file and creates train/valid/test splits
specifically for the favorite_classifier model following DSPy standards.
"""

import os
import sys
import json
import pandas as pd
from pathlib import Path

# Add src directory to path to import data loader
project_root = Path(__file__).parent.parent.parent
sys.path.append(str(project_root / 'src'))

from favorite_classifier.data_loader import TasteDataLoader


def prepare_favorite_classifier_data(raw_csv_path: str, output_dir: Path):
    """Prepare data for favorite classification task"""
    print("\n" + "="*60)
    print("PREPARING FAVORITE CLASSIFIER DATA")
    print("="*60)
    
    # Load and process data
    loader = TasteDataLoader(raw_csv_path)
    data = loader.load_and_process()
    
    # Create splits - using random split for better generalization
    splits = loader.create_splits(
        split_method='random',
        train_size=0.7,
        val_size=0.15,
        test_size=0.15
    )
    
    # Save splits as CSV files
    for split_name, split_df in splits.items():
        if len(split_df) > 0:
            # Ensure directory exists
            split_dir = output_dir / split_name
            split_dir.mkdir(parents=True, exist_ok=True)
            
            output_path = split_dir / 'favorite_classifier.csv'
            split_df.to_csv(output_path, index=False)
            print(f"Saved {split_name} split: {len(split_df)} samples -> {output_path}")
    
    # Also save training examples in DSPy format
    training_examples = loader.get_training_examples(splits, balanced=False)
    training_examples_balanced = loader.get_training_examples(splits, balanced=True)
    
    # Save as JSON for DSPy
    train_dir = output_dir / 'train'
    train_dir.mkdir(parents=True, exist_ok=True)
    
    train_examples_path = train_dir / 'favorite_classifier_examples.json'
    train_examples_balanced_path = train_dir / 'favorite_classifier_examples_balanced.json'
    
    with open(train_examples_path, 'w') as f:
        json.dump(training_examples, f, indent=2)
    print(f"Saved training examples: {len(training_examples)} -> {train_examples_path}")
    
    with open(train_examples_balanced_path, 'w') as f:
        json.dump(training_examples_balanced, f, indent=2)
    print(f"Saved balanced training examples: {len(training_examples_balanced)} -> {train_examples_balanced_path}")
    
    return splits


def create_data_summary(splits, output_dir: Path):
    """Create a summary of the prepared data"""
    summary = {
        'preparation_info': {
            'script': 'scripts/favorite_classifier/prepare.py',
            'raw_data': 'data/raw/export.csv',
            'processed_data_dir': 'data/processed/',
        },
        'favorite_classifier': {
            'task': 'Binary classification - predict if article is favorite',
            'train_samples': len(splits['train']) if 'train' in splits else 0,
            'val_samples': len(splits['val']) if 'val' in splits else 0,
            'test_samples': len(splits['test']) if 'test' in splits else 0,
            'files': [
                'train/favorite_classifier.csv',
                'train/favorite_classifier_examples.json',
                'train/favorite_classifier_examples_balanced.json',
                'valid/favorite_classifier.csv',
                'test/favorite_classifier.csv'
            ]
        }
    }
    
    # Save summary
    summary_path = output_dir / 'favorite_classifier_summary.json'
    with open(summary_path, 'w') as f:
        json.dump(summary, f, indent=2)
    
    print("\n" + "="*60)
    print("FAVORITE CLASSIFIER DATA PREPARATION SUMMARY")
    print("="*60)
    print(f"Raw data: data/raw/export.csv")
    print(f"Processed data: data/processed/")
    print(f"Summary saved: {summary_path}")
    
    print(f"\nDataset Statistics:")
    print(f"  Train: {summary['favorite_classifier']['train_samples']} samples")
    print(f"  Valid: {summary['favorite_classifier']['val_samples']} samples") 
    print(f"  Test:  {summary['favorite_classifier']['test_samples']} samples")


def main():
    """Main data preparation function"""
    print("Starting data preparation for favorite classifier...")
    
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
    splits = prepare_favorite_classifier_data(str(raw_csv_path), processed_dir)
    
    # Create summary
    create_data_summary(splits, processed_dir)
    
    print("\n✅ Favorite classifier data preparation completed successfully!")
    print("\nNext steps:")
    print("1. Review data splits in data/processed/")
    print("2. Run training script: scripts/favorite_classifier/train.py")


if __name__ == "__main__":
    main()