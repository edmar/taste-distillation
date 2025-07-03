#!/usr/bin/env python3
"""
Simple evaluation script for DSPy pairwise classifier
"""

import os
import sys
import json
import pandas as pd
import dspy
from dspy.evaluate import Evaluate

# Disable DSPy caching to avoid database conflicts in parallel evaluation
os.environ['DSPY_CACHEDIR'] = '/tmp/dspy_cache_disable'

# Add parent directory to path to import dspy_pairwise
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from dspy_pairwise import PairwiseComparisonModule

def accuracy_metric(example, pred, trace=None):
    """Simple accuracy metric for DSPy pairwise evaluation"""
    return example.preferred_title == pred.preferred_title

def evaluate_model(test_size: int = None, model_path: str = 'saved/models/dspy_pairwise', num_threads: int = 16):
    """Simple evaluation of DSPy pairwise classifier"""
    
    print("="*50)
    print("DSPy PAIRWISE CLASSIFIER EVALUATION")
    print("="*50)
    
    # Load test data
    csv_path = "data/processed/dspy_pairwise/test/pairwise_classifier.json"
    if not os.path.exists(csv_path):
        print(f"❌ Test data not found at: {csv_path}")
        print("Please run prepare.py first to create test data")
        return None
    
    # Load JSON test data
    with open(csv_path, 'r') as f:
        test_data = json.load(f)
    
    # Sample test examples (use all if test_size is None or large enough)
    if test_size is None or test_size >= len(test_data):
        print(f"Evaluating on full test set: {len(test_data)} examples")
    else:
        test_data = test_data[:test_size]  # Take first N examples
        print(f"Evaluating on {len(test_data)} test examples (sampled from full set)")
    
    # Convert to DSPy format
    test_examples = []
    for item in test_data:
        example = dspy.Example(
            title_a=item['title_a'],
            title_b=item['title_b'],
            preferred_title=item['preference'],
            confidence=item['confidence']
        ).with_inputs('title_a', 'title_b')
        test_examples.append(example)
    
    # Load rubric
    rubric_path = "saved/rubrics/personal_taste_rubric.txt"
    rubric = None
    if os.path.exists(rubric_path):
        with open(rubric_path, 'r') as f:
            rubric = f.read()
        print("✅ Loaded taste rubric")
    
    # Configure DSPy with cache disabled  
    dspy.configure(lm=dspy.LM("openai/gpt-4o-mini", cache=False), cache=False)
    
    # Initialize module
    module = PairwiseComparisonModule(use_reasoning=True, rubric=rubric)
    
    # Load trained model if available
    # Handle both .json and non-.json paths
    if model_path.endswith('.json'):
        json_path = model_path
        base_path = model_path[:-5]  # Remove .json extension
    else:
        json_path = f"{model_path}.json"
        base_path = model_path
    
    if os.path.exists(json_path):
        module.load(json_path)
        print(f"✅ Loaded trained model from: {json_path}")
    else:
        print(f"⚠️ No trained model found at: {json_path}")
        print("Using untrained model")
    
    # Run evaluation with configurable threading
    try:
        evaluator = Evaluate(
            devset=test_examples,
            num_threads=num_threads,
            display_progress=True,
            display_table=5,
            max_errors=10  # Allow some errors without failing
        )
        
        accuracy = evaluator(module, metric=accuracy_metric)
    except Exception as e:
        print(f"❌ Parallel evaluation failed: {e}")
        print("🔄 Falling back to single-threaded evaluation...")
        
        # Fallback to single-threaded evaluation
        evaluator = Evaluate(
            devset=test_examples,
            num_threads=1,
            display_progress=True,
            display_table=5
        )
        
        accuracy = evaluator(module, metric=accuracy_metric)
    
    print(f"\n🎯 Accuracy: {accuracy:.3f}")
    return accuracy


def detailed_evaluation(test_size: int = None, model_path: str = 'saved/models/dspy_pairwise'):
    """More detailed evaluation with confidence breakdown"""
    
    print("\n" + "="*50)
    print("DETAILED PAIRWISE EVALUATION")
    print("="*50)
    
    # Load test data
    csv_path = "data/processed/dspy_pairwise/test/pairwise_classifier.json"
    if not os.path.exists(csv_path):
        print(f"❌ Test data not found at: {csv_path}")
        return None
    
    with open(csv_path, 'r') as f:
        test_data = json.load(f)
    
    if test_size is not None and test_size < len(test_data):
        test_data = test_data[:test_size]
    
    # Load rubric
    rubric_path = "saved/rubrics/personal_taste_rubric.txt"
    rubric = None
    if os.path.exists(rubric_path):
        with open(rubric_path, 'r') as f:
            rubric = f.read()
    
    # Configure DSPy
    dspy.configure(lm=dspy.LM("openai/gpt-4o-mini", cache=False), cache=False)
    
    # Initialize and load model
    module = PairwiseComparisonModule(use_reasoning=True, rubric=rubric)
    
    json_path = f"{model_path}.json" if not model_path.endswith('.json') else model_path
    if os.path.exists(json_path):
        module.load(json_path)
        print(f"✅ Loaded trained model from: {json_path}")
    else:
        print(f"⚠️ Using untrained model")
    
    # Manual evaluation for detailed analysis
    correct = 0
    total = len(test_data)
    
    confidence_breakdown = {'high': {'correct': 0, 'total': 0}, 
                           'medium': {'correct': 0, 'total': 0}, 
                           'low': {'correct': 0, 'total': 0}}
    
    examples_shown = 0
    max_examples = 3
    
    print(f"\nEvaluating {total} test pairs...")
    
    for i, item in enumerate(test_data):
        # Get prediction
        try:
            result = module(title_a=item['title_a'], title_b=item['title_b'])
            predicted = result.preferred_title
        except Exception as e:
            print(f"Error on example {i}: {e}")
            predicted = 'A'  # Default fallback
        
        # Check correctness
        is_correct = predicted == item['preference']
        if is_correct:
            correct += 1
        
        # Track by confidence
        gt_confidence = item['confidence']
        confidence_breakdown[gt_confidence]['total'] += 1
        if is_correct:
            confidence_breakdown[gt_confidence]['correct'] += 1
        
        # Show some examples
        if examples_shown < max_examples:
            status = "✅" if is_correct else "❌"
            print(f"\n{status} Example {i+1}:")
            print(f"   A: {item['title_a'][:60]}...")
            print(f"   B: {item['title_b'][:60]}...")
            print(f"   Ground truth: {item['preference']} | Predicted: {predicted}")
            print(f"   Confidence: {gt_confidence}")
            examples_shown += 1
    
    # Calculate metrics
    overall_accuracy = correct / total
    
    print(f"\n" + "="*40)
    print("DETAILED RESULTS")
    print("="*40)
    print(f"Overall Accuracy: {overall_accuracy:.3f} ({correct}/{total})")
    
    print(f"\nBy Ground Truth Confidence:")
    for conf in ['high', 'medium', 'low']:
        data = confidence_breakdown[conf]
        if data['total'] > 0:
            acc = data['correct'] / data['total']
            print(f"  {conf.capitalize()}: {acc:.3f} ({data['correct']}/{data['total']})")
        else:
            print(f"  {conf.capitalize()}: No examples")
    
    return overall_accuracy


if __name__ == "__main__":
    import sys
    
    # Check for help flag
    if '--help' in sys.argv or '-h' in sys.argv:
        print("""
DSPy Pairwise Classifier Evaluation Script

Usage: python evaluate.py [OPTIONS]

Options:
  <number>           Number of test examples to evaluate (default: all)
  --model=PATH       Path to trained model (default: saved/models/dspy_pairwise)
  --model PATH       Path to trained model (alternative format)
  --threads=N        Number of parallel threads (default: 16)
  --detailed         Run detailed evaluation with confidence breakdown
  -h, --help         Show this help message

Examples:
  python evaluate.py                                      # Evaluate all test examples
  python evaluate.py 200                                 # Evaluate 200 examples  
  python evaluate.py --model=my_model                    # Use custom model path
  python evaluate.py --model saved/models/dspy_pairwise_001.json  # Use specific model
  python evaluate.py 50 --threads=8                      # 50 examples with 8 threads
  python evaluate.py --detailed                          # Run detailed analysis
        """)
        sys.exit(0)
    
    # Parse command line arguments
    test_size = None  # Default: use all test examples
    model_path = 'saved/models/dspy_pairwise'
    num_threads = 16
    run_detailed = False
    
    i = 1
    while i < len(sys.argv):
        arg = sys.argv[i]
        if arg.lower() == 'all':
            test_size = 1000  # Large number to ensure we use full test set
        elif arg == '--detailed':
            run_detailed = True
        elif arg.startswith('--model='):
            model_path = arg.split('=')[1]
        elif arg == '--model' and i + 1 < len(sys.argv):
            # Support --model PATH format
            model_path = sys.argv[i + 1]
            i += 1  # Skip the next argument
        elif arg.startswith('--threads='):
            num_threads = int(arg.split('=')[1])
        else:
            try:
                test_size = int(arg)
            except ValueError:
                print(f"Warning: Ignoring unrecognized argument: {arg}")
        i += 1
    
    # Run evaluation
    if run_detailed:
        accuracy = detailed_evaluation(test_size=test_size, model_path=model_path)
    else:
        accuracy = evaluate_model(test_size=test_size, model_path=model_path, num_threads=num_threads)
    
    if accuracy is not None:
        print(f"\n✅ Evaluation completed! Accuracy: {accuracy:.3f}")
    else:
        print(f"\n❌ Evaluation failed!")
        sys.exit(1)