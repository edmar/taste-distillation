"""
Base functionality for processing Readwise Reader export data.
"""

import pandas as pd
import numpy as np
import ast
from typing import List, Dict, Tuple, Optional
from datetime import datetime, timezone, timedelta
import random
import json
from pathlib import Path


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


def load_readwise_export(csv_path: str) -> pd.DataFrame:
    """Load and perform basic processing on Readwise export data."""
    print(f"Loading data from {csv_path}...")
    
    # Load CSV
    df = pd.read_csv(csv_path)
    
    # Parse tags
    df['parsed_tags'] = df['Document tags'].apply(parse_tags)
    
    # Filter out empty titles
    df = df[df['Title'].notna() & (df['Title'].str.strip() != '')]
    
    # Parse dates with timezone awareness
    df['Saved date'] = pd.to_datetime(df['Saved date'], format='mixed', utc=True)
    
    # Add additional features
    df['reading_progress'] = pd.to_numeric(df['Reading progress'], errors='coerce').fillna(0.0)
    df['is_seen'] = df['Seen'].apply(lambda x: str(x).lower() == 'true')
    
    print(f"Loaded {len(df)} entries with valid titles")
    
    return df


def filter_by_date(df: pd.DataFrame, days: Optional[int] = None) -> pd.DataFrame:
    """Filter dataframe to recent entries if days is specified."""
    if days is None:
        return df
    
    cutoff_date = datetime.now(timezone.utc) - timedelta(days=days)
    df_filtered = df[df['Saved date'] >= cutoff_date]
    
    print(f"Filtered to last {days} days: {len(df_filtered)} entries from {len(df)} total")
    
    return df_filtered


def filter_by_location(df: pd.DataFrame, excluded_locations: List[str] = ['new', 'inbox']) -> pd.DataFrame:
    """Filter out articles in specific locations (e.g., unprocessed articles)."""
    before_filter = len(df)
    df_filtered = df[~df['Location'].isin(excluded_locations)]
    
    print(f"Excluded {before_filter - len(df_filtered)} articles in locations: {excluded_locations}")
    print(f"Included locations: {sorted(df_filtered['Location'].unique())}")
    
    return df_filtered


def deduplicate_by_title(df: pd.DataFrame) -> pd.DataFrame:
    """Remove duplicate entries by title, keeping the first occurrence."""
    print(f"Before deduplication: {len(df)} entries")
    print(f"Unique titles: {df['Title'].nunique()}")
    print(f"Duplicate titles: {len(df) - df['Title'].nunique()}")
    
    df_dedup = df.drop_duplicates(subset=['Title'], keep='first')
    
    print(f"After deduplication: {len(df_dedup)} entries")
    
    return df_dedup


def split_data(df: pd.DataFrame, train_ratio: float = 0.7, val_ratio: float = 0.15, 
               test_ratio: float = 0.15, seed: int = 42) -> Dict[str, pd.DataFrame]:
    """Split data into train/validation/test sets."""
    assert abs(train_ratio + val_ratio + test_ratio - 1.0) < 0.001, "Ratios must sum to 1"
    
    # Shuffle data
    df_shuffled = df.sample(frac=1, random_state=seed).reset_index(drop=True)
    
    # Calculate split indices
    n = len(df_shuffled)
    train_end = int(n * train_ratio)
    val_end = train_end + int(n * val_ratio)
    
    # Create splits
    splits = {
        'train': df_shuffled[:train_end],
        'val': df_shuffled[train_end:val_end],
        'test': df_shuffled[val_end:]
    }
    
    return splits


def balance_binary_data(df: pd.DataFrame, label_column: str, seed: int = 42) -> pd.DataFrame:
    """Balance binary classification data by undersampling the majority class."""
    positive = df[df[label_column] == True]
    negative = df[df[label_column] == False]
    
    # Use the smaller class size as the target
    min_class_size = min(len(positive), len(negative))
    
    print(f"Balancing dataset: {len(positive)} positive, {len(negative)} negative")
    print(f"Using {min_class_size} samples per class")
    
    # Sample equal numbers from each class
    balanced_positive = positive.sample(n=min_class_size, random_state=seed)
    balanced_negative = negative.sample(n=min_class_size, random_state=seed)
    
    # Combine and shuffle
    balanced_df = pd.concat([balanced_positive, balanced_negative])
    balanced_df = balanced_df.sample(frac=1, random_state=seed).reset_index(drop=True)
    
    print(f"Balanced dataset: {len(balanced_df)} total samples, {balanced_df[label_column].mean():.2%} positive")
    
    return balanced_df


def generate_pairwise_examples(df: pd.DataFrame, label_column: str, max_pairs: int = 2000, 
                              seed: int = 42) -> List[Dict]:
    """Generate pairwise comparison examples from binary classification data."""
    random.seed(seed)
    np.random.seed(seed)
    
    positive = df[df[label_column] == True]
    negative = df[df[label_column] == False]
    
    print(f"Generating pairwise examples from {len(positive)} positive and {len(negative)} negative samples")
    
    pairs = []
    
    # Create clear signal pairs: positive vs negative
    for i, pos_row in positive.iterrows():
        # Sample random negatives to pair with this positive
        sample_size = min(5, len(negative))  # Limit to avoid too many pairs
        sampled_negatives = negative.sample(n=sample_size, random_state=seed+i)
        
        for j, neg_row in sampled_negatives.iterrows():
            # Randomly assign A/B positions to avoid position bias
            if random.random() < 0.5:
                pair = {
                    'title_a': pos_row['Title'],
                    'title_b': neg_row['Title'],
                    'preferred_title': 'A',  # Positive is A
                    'confidence': 'high'
                }
            else:
                pair = {
                    'title_a': neg_row['Title'],
                    'title_b': pos_row['Title'],
                    'preferred_title': 'B',  # Positive is B
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


def save_classification_splits(splits: Dict[str, pd.DataFrame], output_dir: Path, 
                              label_column: str, dataset_name: str):
    """Save classification data splits in multiple formats."""
    output_dir.mkdir(parents=True, exist_ok=True)
    
    for split_name, split_df in splits.items():
        split_dir = output_dir / split_name
        split_dir.mkdir(parents=True, exist_ok=True)
        
        # Save full data as CSV (for inspection)
        csv_path = split_dir / f'{dataset_name}_classifier.csv'
        columns_to_save = ['Title', label_column, 'parsed_tags', 'Saved date']
        split_df[columns_to_save].to_csv(csv_path, index=False)
        print(f"Saved {split_name} CSV: {len(split_df)} samples -> {csv_path}")
        
        # Save DSPy examples as JSON
        json_data = []
        # Use standardized field names for DSPy
        if label_column == 'has_favorite':
            dspy_field = 'is_favorite'
        elif label_column == 'has_shortlist':
            dspy_field = 'is_shortlist'
        else:
            dspy_field = label_column
            
        for _, row in split_df.iterrows():
            json_data.append({
                'title': row['Title'],
                dspy_field: bool(row[label_column])
            })
        
        json_path = split_dir / 'dspy_examples.json'
        with open(json_path, 'w') as f:
            json.dump(json_data, f, indent=2)
        print(f"Saved {split_name} DSPy examples: {len(json_data)} -> {json_path}")


def save_pairwise_splits(splits: Dict[str, List[Dict]], output_dir: Path):
    """Save pairwise data splits."""
    output_dir.mkdir(parents=True, exist_ok=True)
    
    for split_name, split_pairs in splits.items():
        split_dir = output_dir / split_name
        split_dir.mkdir(parents=True, exist_ok=True)
        
        # Save DSPy examples as JSON
        json_path = split_dir / 'dspy_examples.json'
        with open(json_path, 'w') as f:
            json.dump(split_pairs, f, indent=2)
        print(f"Saved {split_name} DSPy examples: {len(split_pairs)} -> {json_path}")


def create_classification_summary(splits: Dict[str, pd.DataFrame], output_dir: Path, 
                                 source_file: str, label_column: str, task_name: str, 
                                 task_description: str, days_filter: Optional[int] = None) -> Dict:
    """Create a summary of the prepared classification data."""
    summary = {
        'task': task_name,
        'description': task_description,
        'data_source': source_file,
        'created_at': datetime.now().isoformat(),
        'splits': {}
    }
    
    if days_filter:
        summary['days_filter'] = days_filter
    
    for split_name, split_df in splits.items():
        # Count statistics
        positive_count = split_df[label_column].sum()
        negative_count = len(split_df) - positive_count
        
        # Calculate tag distribution (top 10)
        all_tags = []
        for tags_list in split_df['parsed_tags']:
            all_tags.extend(tags_list)
        
        tag_counts = {}
        for tag in all_tags:
            tag_counts[tag] = tag_counts.get(tag, 0) + 1
        
        top_tags = dict(sorted(tag_counts.items(), key=lambda x: x[1], reverse=True)[:10])
        
        summary['splits'][split_name] = {
            'total_examples': int(len(split_df)),
            label_column.replace('has_', ''): int(positive_count),
            f'non_{label_column.replace("has_", "")}': int(negative_count),
            f'{label_column.replace("has_", "")}_ratio': float(positive_count / len(split_df) if len(split_df) > 0 else 0),
            'top_tags': {k: int(v) for k, v in top_tags.items()},
            'files': {
                'csv_data': f"{split_name}/{task_name.lower()}_classifier.csv",
                'dspy_format': f"{split_name}/dspy_examples.json"
            }
        }
    
    # Save summary
    summary_path = output_dir / f'{task_name.lower()}_summary.json'
    with open(summary_path, 'w') as f:
        json.dump(summary, f, indent=2)
    
    print(f"\nSummary saved to: {summary_path}")
    return summary