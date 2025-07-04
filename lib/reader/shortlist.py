"""
Shortlist-specific functionality for processing Readwise Reader data.
"""

from typing import List
import pandas as pd


def has_shortlist_tag(tags: List[str]) -> bool:
    """Check if 'shortlist' is in the tags list."""
    return 'shortlist' in tags


def prepare_shortlist_data(df: pd.DataFrame) -> pd.DataFrame:
    """Add shortlist-specific columns to the dataframe."""
    df['has_shortlist'] = df['parsed_tags'].apply(has_shortlist_tag)
    return df


# Configuration for shortlist dataset
SHORTLIST_CONFIG = {
    'task_name': 'Shortlist',
    'task_description': 'Binary classification to predict which articles get shortlisted',
    'label_column': 'has_shortlist',
    'dataset_name': 'shortlist',
    'days_filter': 90,  # Only use last 90 days of data
    'excluded_locations': ['new', 'inbox'],  # Exclude unprocessed articles
    'balance_data': True  # Balance positive/negative examples
}