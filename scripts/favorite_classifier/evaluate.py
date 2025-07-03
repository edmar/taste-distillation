#!/usr/bin/env python3
"""
Evaluation script for the taste prediction model - focused only on evaluation
"""

import json
import os
from datetime import datetime
import pandas as pd
import dspy
from dspy.evaluate import Evaluate
from data_loader import TasteDataLoader
from favorite_predictor import TasteClassifier

# DSPy metric functions
def accuracy_metric(example, pred, trace=None):
    """Simple accuracy metric for DSPy evaluation"""
    return example.is_favorite == pred.is_favorite

def f1_metric(examples, predictions, trace=None):
    """F1 score metric for DSPy evaluation"""
    if not examples or not predictions:
        return 0.0
    
    true_positives = sum(1 for ex, pred in zip(examples, predictions) 
                        if ex.is_favorite and pred.is_favorite)
    false_positives = sum(1 for ex, pred in zip(examples, predictions) 
                         if not ex.is_favorite and pred.is_favorite)
    false_negatives = sum(1 for ex, pred in zip(examples, predictions) 
                         if ex.is_favorite and not pred.is_favorite)
    
    precision = true_positives / (true_positives + false_positives) if (true_positives + false_positives) > 0 else 0
    recall = true_positives / (true_positives + false_negatives) if (true_positives + false_negatives) > 0 else 0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
    
    return f1

def evaluate_model(
    csv_path: str,
    model_path: str = 'src/favorite_classifier/models/trained_model.json',
    test_size: int = 200,
    split_method: str = 'random',
    include_baseline: bool = False,
    balanced: bool = True,
    results_save_path: str = 'results/evaluation_results.json'
):
    """Evaluate a trained taste prediction model using DSPy evaluation framework"""
    
    print("="*60)
    print("FAVORITE CLASSIFIER EVALUATION (DSPy)")
    print("="*60)
    
    # Check if model exists
    if not os.path.exists(model_path):
        print(f"❌ Model not found at: {model_path}")
        print(f"Please train a model first: poetry run python src/favorite_classifier/train.py")
        return None
    
    # Load data
    print("Loading data...")
    loader = TasteDataLoader(csv_path)
    data = loader.load_and_process()
    splits = loader.create_splits(split_method=split_method)
    
    # Prepare test dataset in DSPy format with optional balancing
    test_df = splits['test']
    
    if balanced:
        # Create balanced test set
        favorites = test_df[test_df['has_favorite'] == True]
        non_favorites = test_df[test_df['has_favorite'] == False]
        
        # Calculate how many of each type we need
        half_size = test_size // 2
        
        # Sample equal numbers of each class
        if len(favorites) >= half_size and len(non_favorites) >= half_size:
            sampled_favorites = favorites.sample(n=half_size, random_state=42)
            sampled_non_favorites = non_favorites.sample(n=half_size, random_state=42)
            balanced_df = pd.concat([sampled_favorites, sampled_non_favorites]).sample(frac=1, random_state=42)
            print(f"Using balanced test set: {half_size} favorites, {half_size} non-favorites")
        else:
            # If we don't have enough of one class, use all available and balance with the other
            min_class_size = min(len(favorites), len(non_favorites))
            balanced_size = min(min_class_size * 2, test_size)
            half_balanced = balanced_size // 2
            
            sampled_favorites = favorites.sample(n=half_balanced, random_state=42)
            sampled_non_favorites = non_favorites.sample(n=half_balanced, random_state=42)
            balanced_df = pd.concat([sampled_favorites, sampled_non_favorites]).sample(frac=1, random_state=42)
            print(f"Using balanced test set: {half_balanced} favorites, {half_balanced} non-favorites")
        
        test_df = balanced_df
    else:
        # Use original unbalanced approach
        if len(test_df) > test_size:
            print(f"Using first {test_size} test examples (from {len(test_df)} available) - UNBALANCED")
            test_df = test_df.head(test_size)
    
    # Convert to DSPy format
    devset = []
    for _, row in test_df.iterrows():
        example = dspy.Example(
            title=row['title_clean'],
            is_favorite=row['has_favorite']
        ).with_inputs('title')
        devset.append(example)
    
    test_favorites = sum(1 for ex in devset if ex.is_favorite)
    print(f"Test set: {len(devset)} examples ({test_favorites} favorites, {len(devset)-test_favorites} non-favorites)")
    
    # Initialize classifier and load trained model
    print(f"Loading trained model from: {model_path}")
    classifier = TasteClassifier()
    
    try:
        classifier.load_model(model_path)
        print("✅ Model loaded successfully")
    except Exception as e:
        print(f"❌ Failed to load model: {e}")
        return None
    
    results = {}
    
    # Baseline evaluation (untrained model) if requested
    if include_baseline:
        print("\n" + "="*40)
        print("BASELINE EVALUATION (DSPy)")
        print("="*40)
        
        baseline_classifier = TasteClassifier(classifier.model_name)
        baseline_evaluator = Evaluate(
            devset=devset,
            num_threads=1,
            display_progress=True,
            display_table=3
        )
        
        # Create wrapper for baseline classifier
        def baseline_predict_wrapper(title):
            result = baseline_classifier.predict(title)
            return dspy.Prediction(is_favorite=result['is_favorite'], reasoning=result.get('reasoning', ''))
        
        baseline_score = baseline_evaluator(baseline_predict_wrapper, metric=accuracy_metric)
        print(f"Baseline Accuracy: {baseline_score:.3f}")
        results['baseline'] = {'accuracy': baseline_score}
    
    # Evaluate trained model using DSPy
    print("\n" + "="*40)
    print("TRAINED MODEL EVALUATION (DSPy)")
    print("="*40)
    
    # Create DSPy evaluator
    evaluator = Evaluate(
        devset=devset,
        num_threads=1,
        display_progress=True,
        display_table=5,
        return_outputs=True
    )
    
    # Create a wrapper function for DSPy evaluation
    def predict_wrapper(title):
        """Wrapper to make TasteClassifier compatible with DSPy evaluation"""
        result = classifier.predict(title)
        # Return a DSPy-compatible result object
        return dspy.Prediction(is_favorite=result['is_favorite'], reasoning=result.get('reasoning', ''))
    
    # Run evaluation
    score, outputs = evaluator(predict_wrapper, metric=accuracy_metric)
    print(f"Trained Model Accuracy: {score:.3f}")
    
    # Calculate additional metrics manually from outputs
    predictions = [output.is_favorite for _, output, _ in outputs]
    actuals = [example.is_favorite for example in devset]
    
    # Calculate detailed metrics
    detailed_metrics = calculate_detailed_metrics(actuals, predictions)
    results['trained'] = detailed_metrics
    results['trained']['accuracy'] = score
    
    # Comparison
    if include_baseline:
        print("\n" + "="*40)
        print("PERFORMANCE COMPARISON")
        print("="*40)
        
        improvement = score - results['baseline']['accuracy']
        print(f"Accuracy improvement: {improvement:+.3f}")
        results['improvements'] = {'accuracy': improvement}
    
    # Detailed analysis with DSPy outputs
    print("\n" + "="*40)
    print("PREDICTION ANALYSIS (DSPy)")
    print("="*40)
    
    analysis = analyze_dspy_predictions(devset, outputs)
    results['analysis'] = analysis
    
    # Save results
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    evaluation_results = {
        'timestamp': timestamp,
        'model_path': model_path,
        'test_size': len(devset),
        'test_favorites': test_favorites,
        'split_method': split_method,
        'include_baseline': include_baseline,
        'balanced': balanced,
        'evaluation_framework': 'DSPy',
        'results': results
    }
    
    # Create results directory
    os.makedirs(os.path.dirname(results_save_path), exist_ok=True)
    
    # Add timestamp to filename
    base_path, ext = os.path.splitext(results_save_path)
    timestamped_path = f"{base_path}_dspy_{timestamp}{ext}"
    
    with open(timestamped_path, 'w') as f:
        json.dump(evaluation_results, f, indent=2)
    
    print(f"\n💾 Results saved to: {timestamped_path}")
    
    # Summary
    print("\n" + "="*40)
    print("EVALUATION SUMMARY")
    print("="*40)
    print(f"✅ DSPy evaluation completed")
    print(f"🎯 Test accuracy: {score:.3f}")
    print(f"🎯 Test F1-score: {detailed_metrics['f1_score']:.3f}")
    
    if include_baseline:
        print(f"📈 Accuracy improvement: {results['improvements']['accuracy']:+.3f}")
    
    return evaluation_results

def calculate_detailed_metrics(actuals, predictions):
    """Calculate precision, recall, F1 score from predictions"""
    true_positives = sum(1 for a, p in zip(actuals, predictions) if a and p)
    false_positives = sum(1 for a, p in zip(actuals, predictions) if not a and p)
    false_negatives = sum(1 for a, p in zip(actuals, predictions) if a and not p)
    true_negatives = sum(1 for a, p in zip(actuals, predictions) if not a and not p)
    
    precision = true_positives / (true_positives + false_positives) if (true_positives + false_positives) > 0 else 0
    recall = true_positives / (true_positives + false_negatives) if (true_positives + false_negatives) > 0 else 0
    f1_score = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
    
    return {
        'precision': precision,
        'recall': recall,
        'f1_score': f1_score,
        'true_positives': true_positives,
        'false_positives': false_positives,
        'false_negatives': false_negatives,
        'true_negatives': true_negatives
    }

def analyze_dspy_predictions(devset, outputs):
    """Analyze DSPy prediction patterns"""
    analysis = {
        'True Positives': {'count': 0, 'examples': []},
        'False Positives': {'count': 0, 'examples': []},
        'False Negatives': {'count': 0, 'examples': []},
        'True Negatives': {'count': 0, 'examples': []}
    }
    
    for example, output, _ in outputs:
        actual = example.is_favorite
        predicted = output.is_favorite
        
        if predicted and actual:
            category = 'True Positives'
        elif predicted and not actual:
            category = 'False Positives'
        elif not predicted and actual:
            category = 'False Negatives'
        else:
            category = 'True Negatives'
        
        analysis[category]['count'] += 1
        
        # Add example if we have less than 3
        if len(analysis[category]['examples']) < 3:
            reasoning = getattr(output, 'reasoning', '') if hasattr(output, 'reasoning') else ''
            analysis[category]['examples'].append({
                'title': example.title,
                'reasoning': reasoning[:100] if reasoning else ''
            })
    
    # Print analysis
    for category, data in analysis.items():
        if data['count'] > 0:
            print(f"\n{category}: {data['count']} examples")
            for i, example in enumerate(data['examples']):
                print(f"  {i+1}. {example['title'][:70]}...")
                if example['reasoning']:
                    print(f"     Reasoning: {example['reasoning'][:80]}...")
            
            if data['count'] > len(data['examples']):
                print(f"     ... and {data['count'] - len(data['examples'])} more")
    
    return analysis

def evaluate_classifier(classifier, test_examples, name):
    """Evaluate classifier performance"""
    print(f"Evaluating {name} model on {len(test_examples)} examples...")
    
    predictions = []
    actuals = []
    correct = 0
    
    for ex in test_examples:
        pred = classifier.predict(ex['title'])
        predictions.append(pred['is_favorite'])
        actuals.append(ex['is_favorite'])
        if pred['is_favorite'] == ex['is_favorite']:
            correct += 1
    
    # Calculate metrics
    accuracy = correct / len(test_examples)
    
    # Precision, recall, F1 for favorites
    true_positives = sum(1 for p, a in zip(predictions, actuals) if p and a)
    false_positives = sum(1 for p, a in zip(predictions, actuals) if p and not a)
    false_negatives = sum(1 for p, a in zip(predictions, actuals) if not p and a)
    true_negatives = sum(1 for p, a in zip(predictions, actuals) if not p and not a)
    
    precision = true_positives / (true_positives + false_positives) if (true_positives + false_positives) > 0 else 0
    recall = true_positives / (true_positives + false_negatives) if (true_positives + false_negatives) > 0 else 0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
    
    results = {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1_score': f1,
        'true_positives': true_positives,
        'false_positives': false_positives,
        'false_negatives': false_negatives,
        'true_negatives': true_negatives,
        'total_examples': len(test_examples),
        'correct_predictions': correct
    }
    
    print(f"Results:")
    print(f"  Accuracy:  {accuracy:.3f} ({correct}/{len(test_examples)})")
    print(f"  Precision: {precision:.3f}")
    print(f"  Recall:    {recall:.3f}")
    print(f"  F1-Score:  {f1:.3f}")
    print(f"  Confusion Matrix: TP={true_positives}, FP={false_positives}, FN={false_negatives}, TN={true_negatives}")
    
    return results

def analyze_predictions(classifier, test_examples):
    """Analyze prediction patterns in detail"""
    
    # Get predictions with reasoning
    results = []
    for ex in test_examples:
        pred = classifier.predict(ex['title'])
        results.append({
            'title': ex['title'],
            'actual': ex['is_favorite'],
            'predicted': pred['is_favorite'],
            'reasoning': pred['reasoning'],
            'correct': pred['is_favorite'] == ex['is_favorite']
        })
    
    # Categorize predictions
    categories = {
        'True Positives': [r for r in results if r['predicted'] and r['actual']],
        'False Positives': [r for r in results if r['predicted'] and not r['actual']],
        'False Negatives': [r for r in results if not r['predicted'] and r['actual']],
        'True Negatives': [r for r in results if not r['predicted'] and not r['actual']]
    }
    
    analysis = {}
    
    for category, items in categories.items():
        analysis[category] = {
            'count': len(items),
            'examples': []
        }
        
        if items:
            print(f"\n{category}: {len(items)} examples")
            sample_size = min(3, len(items))
            for i, item in enumerate(items[:sample_size]):
                print(f"  {i+1}. {item['title'][:70]}...")
                if item['reasoning']:
                    print(f"     Reasoning: {item['reasoning'][:80]}...")
                
                # Store for JSON
                analysis[category]['examples'].append({
                    'title': item['title'],
                    'reasoning': item['reasoning'][:100] if item['reasoning'] else ''
                })
            
            if len(items) > sample_size:
                print(f"     ... and {len(items) - sample_size} more")
            print()
    
    return analysis

if __name__ == "__main__":
    import sys
    
    # Default parameters
    csv_path = "/Users/edmar/Code/taste/export.csv"
    model_path = 'src/favorite_classifier/models/trained_model.json'
    test_size = 200
    include_baseline = False
    balanced = True
    
    # Parse command line arguments flexibly
    for i, arg in enumerate(sys.argv[1:], 1):
        if arg.lower() in ['true', 'yes', '1'] and i == 2:
            include_baseline = True
        elif arg.lower() in ['false', 'no', '0'] and i == 2:
            include_baseline = False
        elif arg.lower() in ['unbalanced', 'imbalanced', 'false'] and 'balanc' in arg.lower():
            balanced = False
        elif arg.lower() in ['balanced', 'balance', 'true'] and 'balanc' in arg.lower():
            balanced = True
        else:
            try:
                test_size = int(arg)
            except ValueError:
                print(f"Warning: Ignoring unrecognized argument: {arg}")
    
    print(f"Parameters:")
    print(f"  Test size: {test_size}")
    print(f"  Include baseline: {include_baseline}")
    print(f"  Balanced evaluation: {balanced}")
    print(f"  Model path: {model_path}")
    
    # Run evaluation
    result = evaluate_model(
        csv_path=csv_path,
        model_path=model_path,
        test_size=test_size,
        include_baseline=include_baseline,
        balanced=balanced
    )
    
    if result:
        print(f"\n🎉 Evaluation completed successfully!")
    else:
        print(f"\n💥 Evaluation failed!")
        sys.exit(1)