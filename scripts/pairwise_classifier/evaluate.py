#!/usr/bin/env python3
"""
Evaluation script for the pairwise comparison classifier
"""

import os
import json
import random
from datetime import datetime
from typing import Dict, List
from pairwise_data_loader import PairwiseDataLoader
from pairwise_predictor import PairwiseClassifier

class PairwiseEvaluator:
    def __init__(self, model_path: str = None):
        self.model_path = model_path
        self.classifier = None
        self.metadata = None
        
    def load_model(self, model_path: str = None):
        """Load a trained pairwise model"""
        if model_path:
            self.model_path = model_path
        
        if self.model_path is None:
            raise ValueError("No model path provided")
        
        print(f"Loading pairwise model from: {self.model_path}")
        
        # Load metadata first
        metadata_path = self.model_path.replace('.json', '_metadata.json')
        try:
            with open(metadata_path, 'r') as f:
                self.metadata = json.load(f)
            print(f"Loaded metadata: {self.metadata.get('model_name', 'unknown')} model")
            print(f"Trained on: {self.metadata.get('training_examples', 0)} examples")
        except Exception as e:
            print(f"Warning: Could not load metadata: {e}")
            self.metadata = {}
        
        # Initialize classifier and load model
        model_name = self.metadata.get('model_name', 'openai/gpt-4o-mini')
        self.classifier = PairwiseClassifier(model_name=model_name)
        self.classifier.load_model(self.model_path)
    
    def evaluate_comprehensive(self, 
                             csv_path: str,
                             max_test_examples: int = 500,
                             include_baselines: bool = True,
                             pair_strategy: str = 'mixed',
                             show_examples: bool = True) -> Dict:
        """Comprehensive evaluation of the pairwise model"""
        
        print("="*60)
        print("COMPREHENSIVE PAIRWISE EVALUATION")
        print("="*60)
        
        # Load test data
        print("Loading and generating test data...")
        loader = PairwiseDataLoader(csv_path)
        data = loader.load_and_process()
        
        # Generate test pairs
        test_pairs = loader.generate_pairwise_examples(
            max_examples=max_test_examples * 2,  # Generate more for better test coverage
            balanced=True,
            pair_strategy=pair_strategy,
            augment_flip=False  # Don't augment test data
        )
        
        # Limit to requested size
        if len(test_pairs) > max_test_examples:
            test_pairs = test_pairs[:max_test_examples]
        
        print(f"Evaluating on {len(test_pairs)} test pairs")
        
        # Main model evaluation
        print(f"\n" + "="*40)
        print("MODEL EVALUATION")
        print("="*40)
        
        model_results = self.classifier.evaluate(test_pairs)
        
        # Baseline evaluations
        baseline_results = {}
        if include_baselines:
            print(f"\n" + "="*40)
            print("BASELINE COMPARISONS")
            print("="*40)
            
            baseline_results = self._evaluate_baselines(test_pairs)
        
        # Detailed analysis
        print(f"\n" + "="*40)
        print("DETAILED ANALYSIS")
        print("="*40)
        
        detailed_analysis = self._detailed_analysis(test_pairs, show_examples)
        
        # Compile comprehensive results
        comprehensive_results = {
            'timestamp': datetime.now().isoformat(),
            'model_path': self.model_path,
            'test_examples': len(test_pairs),
            'model_metadata': self.metadata,
            'model_results': model_results,
            'baseline_results': baseline_results,
            'detailed_analysis': detailed_analysis,
            'test_strategy': pair_strategy
        }
        
        # Summary
        print(f"\n" + "="*40)
        print("EVALUATION SUMMARY")
        print("="*40)
        
        self._print_summary(comprehensive_results)
        
        return comprehensive_results
    
    def _evaluate_baselines(self, test_pairs: List[Dict]) -> Dict:
        """Evaluate baseline strategies"""
        baselines = {
            'random': self._random_baseline,
            'always_a': self._always_a_baseline,
            'always_b': self._always_b_baseline,
            'confidence_based': self._confidence_based_baseline
        }
        
        baseline_results = {}
        
        for baseline_name, baseline_func in baselines.items():
            print(f"Evaluating {baseline_name} baseline...")
            
            correct = 0
            predictions = []
            
            for pair in test_pairs:
                pred = baseline_func(pair)
                is_correct = pred == pair['preferred_title']
                if is_correct:
                    correct += 1
                predictions.append(pred)
            
            accuracy = correct / len(test_pairs)
            
            baseline_results[baseline_name] = {
                'accuracy': accuracy,
                'correct': correct,
                'total': len(test_pairs),
                'predictions': predictions[:10]  # Save first 10 for inspection
            }
            
            print(f"  {baseline_name}: {accuracy:.3f} accuracy")
        
        return baseline_results
    
    def _random_baseline(self, pair: Dict) -> str:
        """Random choice baseline"""
        return random.choice(['A', 'B'])
    
    def _always_a_baseline(self, pair: Dict) -> str:
        """Always choose A baseline"""
        return 'A'
    
    def _always_b_baseline(self, pair: Dict) -> str:
        """Always choose B baseline"""
        return 'B'
    
    def _confidence_based_baseline(self, pair: Dict) -> str:
        """Choose based on ground truth confidence (oracle baseline)"""
        if pair['confidence'] == 'high':
            return pair['preferred_title']  # Use ground truth when confident
        else:
            return random.choice(['A', 'B'])  # Random when uncertain
    
    def _detailed_analysis(self, test_pairs: List[Dict], show_examples: bool = True) -> Dict:
        """Detailed analysis of model performance"""
        
        # Group by pair type and confidence
        by_pair_type = {}
        by_confidence = {'high': [], 'medium': [], 'low': []}
        
        correct_by_type = {}
        correct_by_confidence = {'high': 0, 'medium': 0, 'low': 0}
        
        for pair in test_pairs:
            # Get model prediction
            pred = self.classifier.compare(pair['title_a'], pair['title_b'])
            is_correct = pred['winner'] == pair['preferred_title']
            
            # Group by pair type
            pair_type = pair['pair_type']
            if pair_type not in by_pair_type:
                by_pair_type[pair_type] = []
                correct_by_type[pair_type] = 0
            
            by_pair_type[pair_type].append(pair)
            if is_correct:
                correct_by_type[pair_type] += 1
            
            # Group by ground truth confidence
            gt_confidence = pair['confidence']
            by_confidence[gt_confidence].append(pair)
            if is_correct:
                correct_by_confidence[gt_confidence] += 1
        
        # Calculate accuracies
        accuracy_by_type = {}
        for pair_type, pairs in by_pair_type.items():
            if len(pairs) > 0:
                accuracy_by_type[pair_type] = correct_by_type[pair_type] / len(pairs)
            else:
                accuracy_by_type[pair_type] = 0.0
        
        accuracy_by_confidence = {}
        for conf, pairs in by_confidence.items():
            if len(pairs) > 0:
                accuracy_by_confidence[conf] = correct_by_confidence[conf] / len(pairs)
            else:
                accuracy_by_confidence[conf] = 0.0
        
        print("Performance by pair type:")
        for pair_type, accuracy in sorted(accuracy_by_confidence.items()):
            count = len(by_pair_type.get(pair_type, []))
            print(f"  {pair_type}: {accuracy:.3f} ({count} examples)")
        
        print("\nPerformance by ground truth confidence:")
        for conf, accuracy in accuracy_by_confidence.items():
            count = len(by_confidence[conf])
            print(f"  {conf}: {accuracy:.3f} ({count} examples)")
        
        # Show some examples if requested
        examples = {}
        if show_examples:
            print("\nExample predictions:")
            
            # Show some correct and incorrect predictions
            correct_examples = []
            incorrect_examples = []
            
            for pair in test_pairs[:20]:  # Check first 20
                pred = self.classifier.compare(pair['title_a'], pair['title_b'])
                is_correct = pred['winner'] == pair['preferred_title']
                
                example = {
                    'title_a': pair['title_a'][:50] + "...",
                    'title_b': pair['title_b'][:50] + "...",
                    'ground_truth': pair['preferred_title'],
                    'predicted': pred['winner'],
                    'confidence': pred['confidence'],
                    'pair_type': pair['pair_type'],
                    'gt_confidence': pair['confidence']
                }
                
                if is_correct and len(correct_examples) < 3:
                    correct_examples.append(example)
                elif not is_correct and len(incorrect_examples) < 3:
                    incorrect_examples.append(example)
            
            print("\nCorrect predictions:")
            for i, ex in enumerate(correct_examples):
                print(f"  {i+1}. [{ex['pair_type']}] GT: {ex['ground_truth']}, Pred: {ex['predicted']} ({ex['confidence']})")
                print(f"     A: {ex['title_a']}")
                print(f"     B: {ex['title_b']}")
            
            print("\nIncorrect predictions:")
            for i, ex in enumerate(incorrect_examples):
                print(f"  {i+1}. [{ex['pair_type']}] GT: {ex['ground_truth']}, Pred: {ex['predicted']} ({ex['confidence']})")
                print(f"     A: {ex['title_a']}")
                print(f"     B: {ex['title_b']}")
            
            examples = {
                'correct': correct_examples,
                'incorrect': incorrect_examples
            }
        
        return {
            'accuracy_by_pair_type': accuracy_by_type,
            'accuracy_by_confidence': accuracy_by_confidence,
            'distribution_by_pair_type': {k: len(v) for k, v in by_pair_type.items()},
            'distribution_by_confidence': {k: len(v) for k, v in by_confidence.items()},
            'examples': examples
        }
    
    def _print_summary(self, results: Dict):
        """Print evaluation summary"""
        model_acc = results['model_results']['accuracy']
        
        print(f"Model Performance:")
        print(f"  Overall Accuracy: {model_acc:.3f}")
        print(f"  Test Examples: {results['test_examples']}")
        
        # Confidence breakdown
        conf_acc = results['model_results']['confidence_accuracy']
        print(f"  High Confidence: {conf_acc['high']:.3f}")
        print(f"  Medium Confidence: {conf_acc['medium']:.3f}")
        print(f"  Low Confidence: {conf_acc['low']:.3f}")
        
        # Baseline comparison
        if results['baseline_results']:
            print(f"\nBaseline Comparison:")
            for baseline, result in results['baseline_results'].items():
                improvement = model_acc - result['accuracy']
                print(f"  vs {baseline}: {result['accuracy']:.3f} (improvement: +{improvement:.3f})")
        
        # Best performing categories
        detailed = results['detailed_analysis']
        best_pair_type = max(detailed['accuracy_by_pair_type'].items(), key=lambda x: x[1])
        worst_pair_type = min(detailed['accuracy_by_pair_type'].items(), key=lambda x: x[1])
        
        print(f"\nBest pair type: {best_pair_type[0]} ({best_pair_type[1]:.3f})")
        print(f"Worst pair type: {worst_pair_type[0]} ({worst_pair_type[1]:.3f})")

def main():
    import sys
    
    # Default parameters
    model_path = "src/pairwise_classifier/models/pairwise_model.json"
    csv_path = "/Users/edmar/Code/taste/export.csv"
    max_test_examples = 500
    include_baselines = True
    pair_strategy = 'mixed'
    show_examples = True
    
    # Parse command line arguments
    for i, arg in enumerate(sys.argv[1:], 1):
        if arg.endswith('.json'):
            model_path = arg
        elif arg in ['mixed', 'fav_vs_nonfav', 'all_combinations']:
            pair_strategy = arg
        elif arg.lower() in ['no_baselines', 'no_baseline']:
            include_baselines = False
        elif arg.lower() in ['no_examples']:
            show_examples = False
        else:
            try:
                max_test_examples = int(arg)
            except ValueError:
                print(f"Warning: Ignoring unrecognized argument: {arg}")
    
    print(f"Parameters:")
    print(f"  Model path: {model_path}")
    print(f"  Max test examples: {max_test_examples}")
    print(f"  Include baselines: {include_baselines}")
    print(f"  Pair strategy: {pair_strategy}")
    print(f"  Show examples: {show_examples}")
    
    # Run evaluation
    evaluator = PairwiseEvaluator()
    
    try:
        evaluator.load_model(model_path)
        results = evaluator.evaluate_comprehensive(
            csv_path=csv_path,
            max_test_examples=max_test_examples,
            include_baselines=include_baselines,
            pair_strategy=pair_strategy,
            show_examples=show_examples
        )
        
        # Save results
        results_path = model_path.replace('.json', '_evaluation_results.json')
        try:
            with open(results_path, 'w') as f:
                json.dump(results, f, indent=2, default=str)
            print(f"\n✅ Evaluation results saved to: {results_path}")
        except Exception as e:
            print(f"⚠️ Could not save evaluation results: {e}")
        
        print(f"\n🎉 Evaluation completed successfully!")
        
    except Exception as e:
        print(f"❌ Evaluation failed: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()