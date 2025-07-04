#!/usr/bin/env python3
"""
Simple evaluation script for DSPy favorite classifier
"""

import os
import sys
import pandas as pd
import dspy
from dspy.evaluate import Evaluate

# Disable DSPy caching to avoid database conflicts in parallel evaluation
os.environ['DSPY_CACHEDIR'] = '/tmp/dspy_cache_disable'

# Add parent directory to path to import dspy_favorite
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from dspy_favorite import TastePredictionModule

def accuracy_metric(example, pred, trace=None):
    """Simple accuracy metric for DSPy evaluation"""
    return example.is_favorite == pred.is_favorite

def evaluate_model(test_size: int = None, model_path: str = 'saved/models/dspy_favorite', num_threads: int = 16):
    """Simple evaluation of DSPy favorite classifier"""
    
    print("="*50)
    print("DSPy FAVORITE CLASSIFIER EVALUATION")
    print("="*50)
    
    # Load test data
    csv_path = "data/processed/reader_favorite/test/favorite_classifier.csv"
    if not os.path.exists(csv_path):
        print(f"❌ Test data not found at: {csv_path}")
        print("Please run prepare.py first to create test data")
        return None
    
    df = pd.read_csv(csv_path)
    
    # Sample test examples (use all if test_size is None or large enough)
    if test_size is None or test_size >= len(df):
        print(f"Evaluating on full test set: {len(df)} examples")
    else:
        df = df.sample(n=test_size, random_state=42)
        print(f"Evaluating on {len(df)} test examples (sampled from {len(pd.read_csv(csv_path))})")
    
    # Convert to DSPy format
    test_examples = []
    for _, row in df.iterrows():
        example = dspy.Example(
            title=row['Title'],
            is_favorite=row['has_favorite']
        ).with_inputs('title')
        test_examples.append(example)
    
    # Load rubric
    rubric_path = "saved/rubrics/personal_taste_rubric.txt"
    rubric = None
    if os.path.exists(rubric_path):
        with open(rubric_path, 'r') as f:
            rubric = f.read()
        print("✅ Loaded taste rubric")
    
    # Configure DSPy with cache disabled  
    dspy.configure(lm=dspy.LM("openai/gpt-4o", cache=False), cache=False)
    
    # Initialize module
    module = TastePredictionModule(use_reasoning=True, rubric=rubric)
    
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
        print(f"📊 Running baseline evaluation with untrained model")
    
    # Run evaluation with configurable threading
    try:
        evaluator = Evaluate(
            devset=test_examples,
            num_threads=num_threads,
            display_progress=True,
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
            display_progress=True
        )
        
        accuracy = evaluator(module, metric=accuracy_metric)
    
    print(f"\n🎯 Accuracy: {accuracy:.3f}")
    return accuracy


if __name__ == "__main__":
    import sys
    
    # Check for help flag
    if '--help' in sys.argv or '-h' in sys.argv:
        print("""
DSPy Favorite Classifier Evaluation Script

Usage: python evaluate.py [OPTIONS]

Options:
  <number>           Number of test examples to evaluate (default: all)
  --model=PATH       Path to trained model (default: saved/models/dspy_favorite)
  --model PATH       Path to trained model (alternative format)
  --threads=N        Number of parallel threads (default: 16)
  -h, --help         Show this help message

Examples:
  python evaluate.py                                      # Evaluate all test examples
  python evaluate.py 200                                 # Evaluate 200 examples  
  python evaluate.py --model=my_model                    # Use custom model path
  python evaluate.py --model saved/models/dspy_favorite_001.json  # Use specific model
  python evaluate.py 50 --threads=8                      # 50 examples with 8 threads
        """)
        sys.exit(0)
    
    # Parse command line arguments
    test_size = None  # Default: use all test examples
    model_path = 'saved/models/dspy_favorite'
    num_threads = 16
    
    i = 1
    while i < len(sys.argv):
        arg = sys.argv[i]
        if arg.lower() == 'all':
            test_size = None  # Use all examples
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
    accuracy = evaluate_model(test_size=test_size, model_path=model_path, num_threads=num_threads)
    
    if accuracy is not None:
        print(f"\n✅ Evaluation completed! Accuracy: {accuracy:.3f}")
    else:
        print(f"\n❌ Evaluation failed!")
        sys.exit(1)