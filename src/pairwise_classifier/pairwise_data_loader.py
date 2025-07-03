import pandas as pd
import numpy as np
import random
from typing import Tuple, List, Dict
import ast
from itertools import combinations

def train_test_split_simple(data, test_size=0.2, stratify=None, random_state=42):
    """Simple train_test_split implementation without sklearn dependency"""
    random.seed(random_state)
    np.random.seed(random_state)
    
    n_samples = len(data)
    n_test = int(n_samples * test_size)
    
    if stratify is not None:
        # Stratified split
        unique_labels = list(set(stratify))
        train_data, test_data = [], []
        
        for label in unique_labels:
            label_indices = [i for i, l in enumerate(stratify) if l == label]
            n_label_test = max(1, int(len(label_indices) * test_size))
            
            random.shuffle(label_indices)
            test_indices = label_indices[:n_label_test]
            train_indices = label_indices[n_label_test:]
            
            test_data.extend([data[i] for i in test_indices])
            train_data.extend([data[i] for i in train_indices])
        
        return train_data, test_data
    else:
        # Simple random split
        indices = list(range(n_samples))
        random.shuffle(indices)
        
        test_indices = indices[:n_test]
        train_indices = indices[n_test:]
        
        train_data = [data[i] for i in train_indices]
        test_data = [data[i] for i in test_indices]
        
        return train_data, test_data

class PairwiseDataLoader:
    def __init__(self, csv_path: str):
        self.csv_path = csv_path
        self.df = None
        self.processed_data = None
        self.favorites = None
        self.non_favorites = None
        
    def load_and_process(self) -> pd.DataFrame:
        """Load CSV and process into clean format, separating favorites and non-favorites"""
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
        
        # Separate favorites and non-favorites
        self.favorites = valid_data[valid_data['has_favorite'] == True]['title_clean'].tolist()
        self.non_favorites = valid_data[valid_data['has_favorite'] == False]['title_clean'].tolist()
        
        print(f"Processed {len(valid_data)} articles with valid titles")
        print(f"Favorites: {len(self.favorites)} ({len(self.favorites)/len(valid_data)*100:.1f}%)")
        print(f"Non-favorites: {len(self.non_favorites)} ({len(self.non_favorites)/len(valid_data)*100:.1f}%)")
        
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
    
    def generate_pairwise_examples(self, 
                                 max_examples: int = 1000,
                                 balanced: bool = True,
                                 pair_strategy: str = 'mixed',
                                 augment_flip: bool = True) -> List[Dict]:
        """
        Generate pairwise training examples from the loaded data
        
        Args:
            max_examples: Maximum number of pairs to generate
            balanced: Whether to balance different types of pairs
            pair_strategy: 'mixed', 'fav_vs_nonfav', 'all_combinations'
            augment_flip: Whether to include both (A,B) and (B,A) versions
        """
        if self.favorites is None or self.non_favorites is None:
            raise ValueError("Must call load_and_process() first")
        
        print(f"\n=== GENERATING PAIRWISE EXAMPLES ===")
        print(f"Strategy: {pair_strategy}")
        print(f"Max examples: {max_examples}")
        print(f"Balanced: {balanced}")
        print(f"Augment flip: {augment_flip}")
        
        all_pairs = []
        
        if pair_strategy == 'fav_vs_nonfav':
            # Only generate favorite vs non-favorite pairs
            all_pairs = self._generate_fav_vs_nonfav_pairs(max_examples, augment_flip)
        
        elif pair_strategy == 'all_combinations':
            # Generate all possible combinations
            all_pairs = self._generate_all_combination_pairs(max_examples, augment_flip)
        
        else:  # 'mixed' strategy
            # Generate a mix of different pair types
            all_pairs = self._generate_mixed_pairs(max_examples, balanced, augment_flip)
        
        # Shuffle the final pairs
        random.shuffle(all_pairs)
        
        # Limit to max_examples
        if len(all_pairs) > max_examples:
            all_pairs = all_pairs[:max_examples]
        
        print(f"Generated {len(all_pairs)} pairwise examples")
        self._print_pair_statistics(all_pairs)
        
        return all_pairs
    
    def _generate_fav_vs_nonfav_pairs(self, max_examples: int, augment_flip: bool) -> List[Dict]:
        """Generate only favorite vs non-favorite pairs (clearest signal)"""
        pairs = []
        
        # Calculate how many pairs we can generate
        max_possible = len(self.favorites) * len(self.non_favorites)
        if augment_flip:
            max_possible *= 2
        
        target_pairs = min(max_examples, max_possible)
        pairs_per_favorite = max(1, target_pairs // len(self.favorites))
        
        print(f"Generating favorite vs non-favorite pairs...")
        print(f"Target pairs: {target_pairs}, pairs per favorite: {pairs_per_favorite}")
        
        for fav_title in self.favorites:
            # Sample non-favorites for this favorite
            selected_nonfavs = random.sample(self.non_favorites, 
                                           min(pairs_per_favorite, len(self.non_favorites)))
            
            for nonfav_title in selected_nonfavs:
                # Favorite should win (A wins)
                pairs.append({
                    'title_a': fav_title,
                    'title_b': nonfav_title,
                    'preferred_title': 'A',
                    'confidence': 'high',
                    'pair_type': 'fav_vs_nonfav'
                })
                
                # Add flipped version if augmenting
                if augment_flip:
                    pairs.append({
                        'title_a': nonfav_title,
                        'title_b': fav_title,
                        'preferred_title': 'B',
                        'confidence': 'high',
                        'pair_type': 'nonfav_vs_fav'
                    })
                
                if len(pairs) >= max_examples:
                    break
            
            if len(pairs) >= max_examples:
                break
        
        return pairs
    
    def _generate_all_combination_pairs(self, max_examples: int, augment_flip: bool) -> List[Dict]:
        """Generate all possible combinations of pairs"""
        pairs = []
        all_titles = self.favorites + self.non_favorites
        
        print(f"Generating all combination pairs from {len(all_titles)} titles...")
        
        # Generate all combinations
        for title_a, title_b in combinations(all_titles, 2):
            fav_a = title_a in self.favorites
            fav_b = title_b in self.favorites
            
            if fav_a and not fav_b:
                # A is favorite, B is not -> A wins
                preferred, confidence = 'A', 'high'
                pair_type = 'fav_vs_nonfav'
            elif not fav_a and fav_b:
                # B is favorite, A is not -> B wins
                preferred, confidence = 'B', 'high'
                pair_type = 'nonfav_vs_fav'
            elif fav_a and fav_b:
                # Both favorites -> harder to decide
                preferred = random.choice(['A', 'B'])
                confidence = 'medium'
                pair_type = 'fav_vs_fav'
            else:
                # Neither favorite -> random preference
                preferred = random.choice(['A', 'B'])
                confidence = 'low'
                pair_type = 'nonfav_vs_nonfav'
            
            pairs.append({
                'title_a': title_a,
                'title_b': title_b,
                'preferred_title': preferred,
                'confidence': confidence,
                'pair_type': pair_type
            })
            
            # Add flipped version if augmenting
            if augment_flip:
                flipped_preferred = 'B' if preferred == 'A' else 'A'
                pairs.append({
                    'title_a': title_b,
                    'title_b': title_a,
                    'preferred_title': flipped_preferred,
                    'confidence': confidence,
                    'pair_type': pair_type + '_flipped'
                })
            
            if len(pairs) >= max_examples:
                break
        
        return pairs
    
    def _generate_mixed_pairs(self, max_examples: int, balanced: bool, augment_flip: bool) -> List[Dict]:
        """Generate only clear signal pairs: fav_vs_nonfav and nonfav_vs_fav"""
        pairs = []
        
        # Only generate fav_vs_nonfav pairs (clear signal only)
        print(f"Target distribution:")
        print(f"  Fav vs Non-fav: {max_examples // 2}")
        print(f"  Non-fav vs Fav: {max_examples - (max_examples // 2)}")
        print(f"  Eliminated: fav_vs_fav and nonfav_vs_nonfav (confusing signals)")
        
        # Generate favorite vs non-favorite pairs (A=fav, B=nonfav, winner=A)
        fav_vs_nonfav_target = max_examples // 2
        for i in range(fav_vs_nonfav_target):
            fav_title = random.choice(self.favorites)
            nonfav_title = random.choice(self.non_favorites)
            
            pairs.append({
                'title_a': fav_title,
                'title_b': nonfav_title,
                'preferred_title': 'A',  # Favorite wins
                'confidence': 'high',
                'pair_type': 'fav_vs_nonfav'
            })
        
        # Generate non-favorite vs favorite pairs (A=nonfav, B=fav, winner=B)
        nonfav_vs_fav_target = max_examples - fav_vs_nonfav_target
        for i in range(nonfav_vs_fav_target):
            nonfav_title = random.choice(self.non_favorites)
            fav_title = random.choice(self.favorites)
            
            pairs.append({
                'title_a': nonfav_title,
                'title_b': fav_title,
                'preferred_title': 'B',  # Favorite wins
                'confidence': 'high',
                'pair_type': 'nonfav_vs_fav'
            })
        
        return pairs
    
    def _generate_pair_type(self, list_a: List[str], list_b: List[str], 
                          preferred_rule: str, confidence: str, pair_type: str,
                          target_count: int, augment_flip: bool) -> List[Dict]:
        """Generate pairs between two lists with specified rules"""
        pairs = []
        
        for i in range(target_count):
            if augment_flip and i >= target_count // 2:
                # Second half: generate with augmentation
                break
            
            # Sample titles
            if list_a == list_b:
                # Same list, avoid duplicates
                if len(list_a) < 2:
                    continue
                title_a, title_b = random.sample(list_a, 2)
            else:
                # Different lists
                title_a = random.choice(list_a)
                title_b = random.choice(list_b)
            
            # Determine preferred title
            if preferred_rule == 'A':
                preferred = 'A'
            elif preferred_rule == 'B':
                preferred = 'B'
            else:  # 'random'
                preferred = random.choice(['A', 'B'])
            
            pairs.append({
                'title_a': title_a,
                'title_b': title_b,
                'preferred_title': preferred,
                'confidence': confidence,
                'pair_type': pair_type
            })
            
            # Add flipped version if augmenting
            if augment_flip:
                flipped_preferred = 'B' if preferred == 'A' else 'A'
                pairs.append({
                    'title_a': title_b,
                    'title_b': title_a,
                    'preferred_title': flipped_preferred,
                    'confidence': confidence,
                    'pair_type': pair_type + '_flipped'
                })
        
        return pairs
    
    def _print_pair_statistics(self, pairs: List[Dict]):
        """Print statistics about the generated pairs"""
        print(f"\n=== PAIR STATISTICS ===")
        
        if len(pairs) == 0:
            print("No pairs generated!")
            return
        
        # Count by pair type
        pair_types = {}
        confidences = {'high': 0, 'medium': 0, 'low': 0}
        preferences = {'A': 0, 'B': 0}
        
        for pair in pairs:
            pair_type = pair['pair_type']
            pair_types[pair_type] = pair_types.get(pair_type, 0) + 1
            confidences[pair['confidence']] += 1
            preferences[pair['preferred_title']] += 1
        
        print("By pair type:")
        for ptype, count in sorted(pair_types.items()):
            print(f"  {ptype}: {count} ({count/len(pairs)*100:.1f}%)")
        
        print("By confidence:")
        for conf, count in confidences.items():
            if count > 0:
                print(f"  {conf}: {count} ({count/len(pairs)*100:.1f}%)")
        
        print("By preference:")
        for pref, count in preferences.items():
            if count > 0:
                print(f"  {pref}: {count} ({count/len(pairs)*100:.1f}%)")
    
    def create_splits(self, 
                     pairs: List[Dict],
                     train_size: float = 0.7,
                     val_size: float = 0.15,
                     test_size: float = 0.15,
                     stratify_by: str = 'confidence') -> Dict[str, List[Dict]]:
        """
        Create train/val/test splits for pairwise data
        
        Args:
            pairs: List of pairwise examples
            train_size, val_size, test_size: split proportions
            stratify_by: 'confidence', 'pair_type', or None
        """
        print(f"Creating splits: {train_size:.1f}/{val_size:.1f}/{test_size:.1f}")
        print(f"Stratifying by: {stratify_by}")
        
        if stratify_by:
            # Create stratification labels
            if stratify_by == 'confidence':
                stratify_labels = [pair['confidence'] for pair in pairs]
            elif stratify_by == 'pair_type':
                stratify_labels = [pair['pair_type'] for pair in pairs]
            else:
                stratify_labels = None
        else:
            stratify_labels = None
        
        # First split: train vs temp
        train_pairs, temp_pairs = train_test_split_simple(
            pairs, test_size=(val_size + test_size),
            stratify=stratify_labels, random_state=42
        )
        
        # Second split: val vs test
        if len(temp_pairs) > 0:
            val_ratio = val_size / (val_size + test_size)
            
            # Stratify temp split if possible
            temp_stratify = None
            if stratify_labels:
                temp_indices = [i for i, pair in enumerate(pairs) if pair in temp_pairs]
                temp_stratify = [stratify_labels[i] for i in temp_indices]
            
            try:
                val_pairs, test_pairs = train_test_split_simple(
                    temp_pairs, test_size=(1-val_ratio),
                    stratify=temp_stratify, random_state=42
                )
            except (ValueError, IndexError):
                # If stratification fails, split without it
                val_pairs, test_pairs = train_test_split_simple(
                    temp_pairs, test_size=(1-val_ratio), random_state=42
                )
        else:
            val_pairs, test_pairs = [], []
        
        splits = {
            'train': train_pairs,
            'val': val_pairs,
            'test': test_pairs
        }
        
        self._print_split_statistics(splits)
        return splits
    
    def _print_split_statistics(self, splits: Dict[str, List[Dict]]):
        """Print statistics about the splits"""
        print(f"\n=== SPLIT STATISTICS ===")
        
        for split_name, split_pairs in splits.items():
            if len(split_pairs) == 0:
                print(f"{split_name:5s}: 0 pairs")
                continue
            
            confidences = {'high': 0, 'medium': 0, 'low': 0}
            pair_types = {}
            
            for pair in split_pairs:
                confidences[pair['confidence']] += 1
                ptype = pair['pair_type']
                pair_types[ptype] = pair_types.get(ptype, 0) + 1
            
            print(f"{split_name:5s}: {len(split_pairs):4d} pairs")
            
            # Show confidence distribution
            conf_str = f"High: {confidences['high']}, Med: {confidences['medium']}, Low: {confidences['low']}"
            print(f"       Confidence - {conf_str}")
            
            # Show top pair types
            top_types = sorted(pair_types.items(), key=lambda x: x[1], reverse=True)[:3]
            type_str = ", ".join([f"{ptype}: {count}" for ptype, count in top_types])
            print(f"       Pair types - {type_str}")

if __name__ == "__main__":
    # Test the pairwise data loader
    loader = PairwiseDataLoader("/Users/edmar/Code/taste/export.csv")
    data = loader.load_and_process()
    
    print("\n" + "="*50)
    print("TESTING PAIRWISE EXAMPLE GENERATION")
    
    # Test different strategies
    strategies = ['fav_vs_nonfav', 'mixed', 'all_combinations']
    
    for strategy in strategies:
        print(f"\n--- Testing {strategy} strategy ---")
        pairs = loader.generate_pairwise_examples(
            max_examples=100,
            pair_strategy=strategy,
            balanced=True,
            augment_flip=True
        )
        
        # Show some examples
        print("\nSample pairs:")
        for i, pair in enumerate(pairs[:3]):
            winner_title = pair['title_a'] if pair['preferred_title'] == 'A' else pair['title_b']
            print(f"{i+1}. [{pair['pair_type']}] {pair['confidence']} confidence")
            print(f"   A: {pair['title_a'][:60]}...")
            print(f"   B: {pair['title_b'][:60]}...")
            print(f"   Winner: {pair['preferred_title']} -> {winner_title[:60]}...")
        
        # Create splits
        print(f"\n--- Creating splits for {strategy} ---")
        splits = loader.create_splits(pairs, stratify_by='confidence')
        
        print(f"Train: {len(splits['train'])}, Val: {len(splits['val'])}, Test: {len(splits['test'])}")