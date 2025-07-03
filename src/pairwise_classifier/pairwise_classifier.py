#!/usr/bin/env python3
"""
Main API interface for the pairwise comparison classifier

This module provides a high-level interface for using the pairwise comparison
system to compare article titles and rank them by preference.
"""

import os
import json
from typing import List, Dict, Tuple, Optional
from pairwise_predictor import PairwiseClassifier
from pairwise_data_loader import PairwiseDataLoader

class PairwiseAPI:
    """
    High-level API for pairwise title comparison and ranking
    
    This class provides an easy-to-use interface for:
    - Comparing two titles
    - Ranking multiple titles
    - Loading and managing trained models
    - Integration with taste data
    """
    
    def __init__(self, model_path: str = None, model_name: str = "openai/gpt-4o-mini"):
        """
        Initialize the PairwiseAPI
        
        Args:
            model_path: Path to a trained model (optional)
            model_name: LLM model to use if no trained model provided
        """
        self.classifier = PairwiseClassifier(model_name=model_name)
        self.model_path = model_path
        self.metadata = None
        
        if model_path and os.path.exists(model_path):
            self.load_model(model_path)
    
    def load_model(self, model_path: str):
        """Load a trained pairwise model"""
        self.model_path = model_path
        self.classifier.load_model(model_path)
        
        # Load metadata
        metadata_path = model_path.replace('.json', '_metadata.json')
        if os.path.exists(metadata_path):
            with open(metadata_path, 'r') as f:
                self.metadata = json.load(f)
        
        print(f"Loaded pairwise model: {self.metadata.get('model_name', 'unknown')} ")
        print(f"Trained on {self.metadata.get('training_examples', 0)} examples")
    
    def compare_titles(self, title_a: str, title_b: str) -> Dict:
        """
        Compare two titles and determine which is more likely to be favorite
        
        Args:
            title_a: First title to compare
            title_b: Second title to compare
            
        Returns:
            Dict with keys:
            - winner: 'A' or 'B'
            - preferred_title: The actual title text that won
            - confidence: 'high', 'medium', or 'low'
            - reasoning: Explanation of the decision
        """
        result = self.classifier.compare(title_a, title_b)
        return {
            'winner': result['winner'],
            'preferred_title': result['preferred_title'],
            'confidence': result['confidence'],
            'reasoning': result.get('reasoning', ''),
            'title_a': title_a,
            'title_b': title_b
        }
    
    def rank_titles(self, titles: List[str], method: str = 'tournament') -> List[Dict]:
        """
        Rank a list of titles by preference using pairwise comparisons
        
        Args:
            titles: List of title strings to rank
            method: Ranking method ('tournament', 'round_robin')
            
        Returns:
            List of dicts with keys:
            - title: The title text
            - rank: Position in ranking (1 = best)
            - score: Normalized score (0-1)
            - wins: Number of wins in comparisons
        """
        if len(titles) <= 1:
            return [{'title': title, 'rank': 1, 'score': 1.0, 'wins': 0} for title in titles]
        
        if method == 'tournament':
            return self._tournament_ranking(titles)
        elif method == 'round_robin':
            return self._round_robin_ranking(titles)
        else:
            raise ValueError(f"Unknown ranking method: {method}")
    
    def _tournament_ranking(self, titles: List[str]) -> List[Dict]:
        """Rank using tournament-style comparisons (efficient for large lists)"""
        return self.classifier.rank_titles(titles)
    
    def _round_robin_ranking(self, titles: List[str]) -> List[Dict]:
        """Rank using round-robin comparisons (more accurate but slower)"""
        # Score each title by counting wins against all others
        scores = {title: 0 for title in titles}
        total_comparisons = {title: 0 for title in titles}
        comparison_details = {}
        
        # Compare each pair once
        for i, title_a in enumerate(titles):
            for j, title_b in enumerate(titles[i+1:], i+1):
                result = self.classifier.compare(title_a, title_b)
                winner = result['winner']
                confidence_weight = {'high': 1.0, 'medium': 0.7, 'low': 0.5}[result['confidence']]
                
                # Record detailed comparison
                comparison_details[(title_a, title_b)] = result
                
                if winner == 'A':
                    scores[title_a] += confidence_weight
                else:
                    scores[title_b] += confidence_weight
                
                total_comparisons[title_a] += 1
                total_comparisons[title_b] += 1
        
        # Normalize scores and rank
        normalized_scores = {}
        for title in titles:
            if total_comparisons[title] > 0:
                normalized_scores[title] = scores[title] / total_comparisons[title]
            else:
                normalized_scores[title] = 0.0
        
        # Sort by score
        ranked_titles = sorted(titles, key=lambda t: normalized_scores[t], reverse=True)
        
        # Create result with ranks
        result = []
        for rank, title in enumerate(ranked_titles, 1):
            result.append({
                'title': title,
                'rank': rank,
                'score': normalized_scores[title],
                'wins': scores[title],
                'total_comparisons': total_comparisons[title]
            })
        
        return result
    
    def find_best_title(self, titles: List[str]) -> Dict:
        """
        Find the single best title from a list
        
        Args:
            titles: List of titles to compare
            
        Returns:
            Dict with best title and comparison details
        """
        if len(titles) == 0:
            return None
        elif len(titles) == 1:
            return {'title': titles[0], 'rank': 1, 'score': 1.0}
        
        ranking = self.rank_titles(titles)
        best = ranking[0]
        
        return {
            'best_title': best['title'],
            'rank': best['rank'],
            'score': best['score'],
            'total_titles': len(titles),
            'ranking': ranking[:5]  # Top 5 for context
        }
    
    def compare_against_favorites(self, new_titles: List[str], csv_path: str) -> List[Dict]:
        """
        Compare new titles against known favorites from the taste data
        
        Args:
            new_titles: List of new titles to evaluate
            csv_path: Path to the taste CSV data
            
        Returns:
            List of dicts with comparison results
        """
        # Load favorites from data
        loader = PairwiseDataLoader(csv_path)
        loader.load_and_process()
        
        # Sample some favorites for comparison
        favorite_sample = loader.favorites[:10]  # Use top 10 favorites
        
        results = []
        for new_title in new_titles:
            wins = 0
            total_comparisons = len(favorite_sample)
            comparison_details = []
            
            for favorite_title in favorite_sample:
                result = self.compare_titles(new_title, favorite_title)
                if result['winner'] == 'A':  # new_title wins
                    wins += 1
                
                comparison_details.append({
                    'favorite': favorite_title,
                    'winner': result['winner'],
                    'confidence': result['confidence']
                })
            
            win_rate = wins / total_comparisons if total_comparisons > 0 else 0.0
            
            results.append({
                'title': new_title,
                'win_rate': win_rate,
                'wins': wins,
                'total_comparisons': total_comparisons,
                'likely_favorite': win_rate > 0.5,
                'comparison_details': comparison_details[:3]  # Top 3 for brevity
            })
        
        # Sort by win rate
        results.sort(key=lambda x: x['win_rate'], reverse=True)
        
        return results
    
    def batch_compare(self, title_pairs: List[Tuple[str, str]]) -> List[Dict]:
        """
        Compare multiple pairs of titles in batch
        
        Args:
            title_pairs: List of (title_a, title_b) tuples
            
        Returns:
            List of comparison results
        """
        results = []
        for title_a, title_b in title_pairs:
            result = self.compare_titles(title_a, title_b)
            results.append(result)
        
        return results
    
    def get_model_info(self) -> Dict:
        """Get information about the loaded model"""
        if self.metadata:
            return {
                'model_path': self.model_path,
                'model_name': self.metadata.get('model_name'),
                'training_examples': self.metadata.get('training_examples'),
                'timestamp': self.metadata.get('timestamp'),
                'pair_strategy': self.metadata.get('pair_strategy'),
                'accuracy': self.metadata.get('quick_evaluation', {}).get('accuracy')
            }
        else:
            return {
                'model_path': self.model_path,
                'model_name': self.classifier.model_name,
                'trained': self.classifier.trained_module is not None
            }
    
    def suggest_better_titles(self, title: str, alternatives: List[str]) -> List[Dict]:
        """
        Compare a title against alternatives and suggest improvements
        
        Args:
            title: Original title
            alternatives: List of alternative titles
            
        Returns:
            List of alternatives ranked by preference over original
        """
        improvements = []
        
        for alt_title in alternatives:
            result = self.compare_titles(title, alt_title)
            
            if result['winner'] == 'B':  # Alternative wins
                improvements.append({
                    'alternative': alt_title,
                    'confidence': result['confidence'],
                    'reasoning': result.get('reasoning', ''),
                    'improvement': True
                })
            else:
                improvements.append({
                    'alternative': alt_title,
                    'confidence': result['confidence'],
                    'reasoning': result.get('reasoning', ''),
                    'improvement': False
                })
        
        # Sort by improvements first, then by confidence
        improvements.sort(key=lambda x: (x['improvement'], x['confidence'] == 'high'), reverse=True)
        
        return improvements

def main():
    """Example usage of the PairwiseAPI"""
    import sys
    
    # Default model path
    model_path = "src/pairwise_classifier/models/pairwise_model.json"
    
    # Check if model path provided
    if len(sys.argv) > 1:
        model_path = sys.argv[1]
    
    try:
        # Initialize API
        print("Initializing Pairwise Comparison API...")
        api = PairwiseAPI(model_path=model_path)
        
        # Example 1: Compare two titles
        print("\n" + "="*50)
        print("EXAMPLE 1: COMPARING TWO TITLES")
        print("="*50)
        
        title_a = "Why You Should Start a Blog Right Now"
        title_b = "10 Common Mistakes in Web Development"
        
        result = api.compare_titles(title_a, title_b)
        print(f"Title A: {title_a}")
        print(f"Title B: {title_b}")
        print(f"Winner: {result['winner']} -> {result['preferred_title']}")
        print(f"Confidence: {result['confidence']}")
        print(f"Reasoning: {result['reasoning']}")
        
        # Example 2: Rank multiple titles
        print("\n" + "="*50)
        print("EXAMPLE 2: RANKING MULTIPLE TITLES")
        print("="*50)
        
        titles = [
            "Why You Should Start a Blog Right Now",
            "10 Common Mistakes in Web Development",
            "The Future of Artificial Intelligence",
            "How to Build Better Software",
            "Machine Learning for Beginners"
        ]
        
        ranking = api.rank_titles(titles)
        print("Rankings:")
        for item in ranking:
            print(f"{item['rank']}. {item['title']} (score: {item['score']:.3f})")
        
        # Example 3: Find best title
        print("\n" + "="*50)
        print("EXAMPLE 3: FINDING BEST TITLE")
        print("="*50)
        
        best = api.find_best_title(titles)
        print(f"Best title: {best['best_title']}")
        print(f"Score: {best['score']:.3f}")
        
        # Example 4: Model info
        print("\n" + "="*50)
        print("EXAMPLE 4: MODEL INFORMATION")
        print("="*50)
        
        model_info = api.get_model_info()
        for key, value in model_info.items():
            print(f"{key}: {value}")
        
        print("\n🎉 PairwiseAPI demo completed successfully!")
        
    except Exception as e:
        print(f"❌ Demo failed: {e}")
        print("Make sure you have a trained model at the specified path.")
        print("Train a model first using: poetry run python src/pairwise_classifier/pairwise_train.py")

if __name__ == "__main__":
    main()