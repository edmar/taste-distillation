import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from typing import Tuple, List, Dict
import ast

class TasteDataLoader:
    def __init__(self, csv_path: str):
        self.csv_path = csv_path
        self.df = None
        self.processed_data = None
        
    def load_and_process(self) -> pd.DataFrame:
        """Load CSV and process into clean format"""
        print("Loading CSV data...")
        self.df = pd.read_csv(self.csv_path)
        
        # Process tags to extract favorites
        print("Processing tags...")
        self.df['has_favorite'] = self.df['Document tags'].apply(self._extract_favorite)
        
        # Clean titles
        self.df['title_clean'] = self.df['Title'].fillna('').str.strip()
        
        # Parse dates
        self.df['saved_date'] = pd.to_datetime(self.df['Saved date'], format='mixed')
        
        # Filter out rows with empty titles
        valid_data = self.df[self.df['title_clean'].str.len() > 0].copy()
        
        print(f"Processed {len(valid_data)} articles with valid titles")
        print(f"Favorites: {valid_data['has_favorite'].sum()} ({valid_data['has_favorite'].mean()*100:.1f}%)")
        
        self.processed_data = valid_data
        return valid_data
    
    def _extract_favorite(self, tags_str) -> bool:
        """Extract whether 'favorite' tag exists"""
        if pd.isna(tags_str) or tags_str == '':
            return False
        
        try:
            # Handle list format like "['favorite', 'writing']"
            if isinstance(tags_str, str) and tags_str.startswith('['):
                tags = ast.literal_eval(tags_str)
                return 'favorite' in tags
            # Handle other string formats
            else:
                return 'favorite' in str(tags_str).lower()
        except:
            # Fallback for any parsing issues
            return 'favorite' in str(tags_str).lower()
    
    def create_splits(self, 
                     split_method: str = 'random', 
                     train_size: float = 0.7,
                     val_size: float = 0.15,
                     test_size: float = 0.15,
                     temporal_split_year: int = 2023) -> Dict[str, pd.DataFrame]:
        """
        Create train/val/test splits
        
        Args:
            split_method: 'random' or 'temporal'
            train_size, val_size, test_size: split proportions (for random)
            temporal_split_year: year to split on (for temporal)
        """
        if self.processed_data is None:
            raise ValueError("Must call load_and_process() first")
        
        data = self.processed_data
        
        if split_method == 'temporal':
            return self._temporal_split(data, temporal_split_year)
        else:
            return self._random_split(data, train_size, val_size, test_size)
    
    def _random_split(self, data: pd.DataFrame, train_size: float, val_size: float, test_size: float) -> Dict[str, pd.DataFrame]:
        """Create stratified random splits"""
        print(f"Creating random splits: {train_size:.1f}/{val_size:.1f}/{test_size:.1f}")
        
        # First split: train vs temp
        X = data[['title_clean']]
        y = data['has_favorite']
        
        X_train, X_temp, y_train, y_temp = train_test_split(
            X, y, test_size=(val_size + test_size), 
            stratify=y, random_state=42
        )
        
        # Second split: val vs test
        val_ratio = val_size / (val_size + test_size)
        X_val, X_test, y_val, y_test = train_test_split(
            X_temp, y_temp, test_size=(1-val_ratio),
            stratify=y_temp, random_state=42
        )
        
        # Reconstruct full dataframes
        train_df = data.loc[X_train.index].copy()
        val_df = data.loc[X_val.index].copy()
        test_df = data.loc[X_test.index].copy()
        
        self._print_split_stats(train_df, val_df, test_df)
        
        return {
            'train': train_df,
            'val': val_df,
            'test': test_df
        }
    
    def _temporal_split(self, data: pd.DataFrame, split_year: int) -> Dict[str, pd.DataFrame]:
        """Create temporal splits: older data for train, newer for test"""
        print(f"Creating temporal split at year {split_year}")
        
        # Split by year
        train_data = data[data['saved_date'].dt.year < split_year].copy()
        recent_data = data[data['saved_date'].dt.year >= split_year].copy()
        
        # Split recent data into val/test
        if len(recent_data) > 0:
            X_recent = recent_data[['title_clean']]
            y_recent = recent_data['has_favorite']
            
            # Only split if we have enough data and both classes
            if len(recent_data) >= 10 and y_recent.nunique() > 1:
                X_val, X_test, y_val, y_test = train_test_split(
                    X_recent, y_recent, test_size=0.5, 
                    stratify=y_recent, random_state=42
                )
                val_df = recent_data.loc[X_val.index].copy()
                test_df = recent_data.loc[X_test.index].copy()
            else:
                # If not enough data, put all in test
                val_df = pd.DataFrame(columns=data.columns)
                test_df = recent_data.copy()
        else:
            val_df = pd.DataFrame(columns=data.columns)
            test_df = pd.DataFrame(columns=data.columns)
        
        self._print_split_stats(train_data, val_df, test_df)
        
        return {
            'train': train_data,
            'val': val_df,
            'test': test_df
        }
    
    def _print_split_stats(self, train_df: pd.DataFrame, val_df: pd.DataFrame, test_df: pd.DataFrame):
        """Print statistics about the splits"""
        print("\n=== SPLIT STATISTICS ===")
        for name, df in [('Train', train_df), ('Val', val_df), ('Test', test_df)]:
            if len(df) > 0:
                fav_count = df['has_favorite'].sum()
                fav_pct = df['has_favorite'].mean() * 100
                print(f"{name:5s}: {len(df):5d} articles, {fav_count:4d} favorites ({fav_pct:.1f}%)")
            else:
                print(f"{name:5s}: {len(df):5d} articles")
    
    def get_training_examples(self, split_data: Dict[str, pd.DataFrame], balanced: bool = False, max_examples: int = None) -> List[Dict]:
        """Convert to format suitable for DSPy training with optional balancing"""
        train_df = split_data['train']
        
        if balanced:
            return self._get_balanced_examples(train_df, max_examples)
        else:
            return self._get_all_examples(train_df, max_examples)
    
    def _get_all_examples(self, train_df: pd.DataFrame, max_examples: int = None) -> List[Dict]:
        """Get all training examples (original behavior)"""
        examples = []
        for _, row in train_df.iterrows():
            examples.append({
                'title': row['title_clean'],
                'is_favorite': row['has_favorite']
            })
        
        if max_examples and len(examples) > max_examples:
            # Proportional sampling to maintain ratio
            favorites = [ex for ex in examples if ex['is_favorite']]
            non_favorites = [ex for ex in examples if not ex['is_favorite']]
            
            fav_ratio = len(favorites) / len(examples)
            n_fav = min(int(max_examples * fav_ratio), len(favorites))
            n_non_fav = min(max_examples - n_fav, len(non_favorites))
            
            examples = favorites[:n_fav] + non_favorites[:n_non_fav]
        
        return examples
    
    def _get_balanced_examples(self, train_df: pd.DataFrame, max_examples: int = None) -> List[Dict]:
        """Get balanced training examples (equal favorites and non-favorites)"""
        favorites = []
        non_favorites = []
        
        for _, row in train_df.iterrows():
            example = {
                'title': row['title_clean'],
                'is_favorite': row['has_favorite']
            }
            if example['is_favorite']:
                favorites.append(example)
            else:
                non_favorites.append(example)
        
        # Determine balanced size
        min_class_size = min(len(favorites), len(non_favorites))
        
        if max_examples:
            # Split max_examples between classes
            per_class = max_examples // 2
            balanced_size = min(per_class, min_class_size)
        else:
            balanced_size = min_class_size
        
        print(f"\n=== BALANCED SAMPLING ===")
        print(f"Available: {len(favorites)} favorites, {len(non_favorites)} non-favorites")
        print(f"Using: {balanced_size} favorites, {balanced_size} non-favorites")
        print(f"Total balanced examples: {balanced_size * 2}")
        print(f"Class balance: 50.0% / 50.0% (vs original {len(favorites)/(len(favorites)+len(non_favorites))*100:.1f}% / {len(non_favorites)/(len(favorites)+len(non_favorites))*100:.1f}%)")
        
        # Sample equally from both classes
        balanced_examples = favorites[:balanced_size] + non_favorites[:balanced_size]
        
        # Shuffle to mix the classes
        import random
        random.shuffle(balanced_examples)
        
        return balanced_examples

if __name__ == "__main__":
    # Test the data loader
    loader = TasteDataLoader("/Users/edmar/Code/taste/export.csv")
    data = loader.load_and_process()
    
    print("\n" + "="*50)
    print("TESTING RANDOM SPLIT")
    random_splits = loader.create_splits(split_method='random')
    
    print("\n" + "="*50)  
    print("TESTING TEMPORAL SPLIT")
    temporal_splits = loader.create_splits(split_method='temporal', temporal_split_year=2024)
    
    # Show some examples
    print("\n=== SAMPLE TRAINING EXAMPLES ===")
    examples = loader.get_training_examples(random_splits)
    favorites = [ex for ex in examples if ex['is_favorite']][:3]
    non_favorites = [ex for ex in examples if not ex['is_favorite']][:3]
    
    print("Favorites:")
    for ex in favorites:
        print(f"  - {ex['title']}")
    
    print("\nNon-favorites:")  
    for ex in non_favorites:
        print(f"  - {ex['title']}")