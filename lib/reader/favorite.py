"""
Favorite-specific functionality for processing Readwise Reader data.
"""

from typing import List
import pandas as pd


def has_favorite_tag(tags: List[str]) -> bool:
    """Check if 'favorite' is in the tags list."""
    return 'favorite' in tags


def prepare_favorite_data(df: pd.DataFrame) -> pd.DataFrame:
    """Add favorite-specific columns to the dataframe."""
    df['has_favorite'] = df['parsed_tags'].apply(has_favorite_tag)
    return df


# Configuration for favorite dataset
FAVORITE_CONFIG = {
    'task_name': 'Favorite',
    'task_description': 'Binary classification to predict which articles get marked as favorite',
    'label_column': 'has_favorite',
    'dataset_name': 'favorite',
    'days_filter': None,  # Use all data
    'excluded_locations': [],  # Include all locations
    'balance_data': False  # Keep natural distribution
}