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

# Add shared utilities
sys.path.append(os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), 'shared'))
from logging_utils import tee_output
from config import get_config

def accuracy_metric(example, pred, trace=None):
    """Simple accuracy metric for DSPy evaluation"""
    return example.is_favorite == pred.is_favorite

def evaluate_model(test_size: int = None, model_path: str = None, num_threads: int = None, show_progress: bool = True, dataset_path: str = None, llm_model: str = None):
    """Simple evaluation of DSPy favorite classifier"""
    
    # Extract model name for logging
    if model_path:
        log_name = os.path.basename(model_path).replace('.json', '')
    else:
        log_name = 'dspy_favorite'
    
    with tee_output(log_name, 'eval', show_progress):
        print("="*50)
        print("DSPy FAVORITE CLASSIFIER EVALUATION")
        print("="*50)
        
        return _evaluate_model_impl(test_size, model_path, num_threads, dataset_path, llm_model)

def _evaluate_model_impl(test_size: int = None, model_path: str = None, num_threads: int = None, dataset_path: str = None, llm_model: str = None):
    """Implementation of evaluation logic"""
    
    # Load evaluation configuration
    config = get_config("evaluate")
    eval_config = config.get_evaluation_config()
    paths_config = config.get_paths_config()
    datasets_config = config.get_datasets_config()
    
    # Set defaults from config
    if model_path is None:
        model_path = paths_config.get('default_model', 'saved/models/dspy_favorite')
    if num_threads is None:
        num_threads = eval_config.get('num_threads', 16)
    if test_size is None:
        test_size = eval_config.get('default_test_size', None)
    llm_name = config.get_llm_name(llm_model)
    
    # Load test data
    if dataset_path is None:
        # Use default dataset from config
        json_path = paths_config.get('default_dataset', 'data/reader_favorite/test/dspy_examples.json')
    else:
        json_path = dataset_path
    
    if not os.path.exists(json_path):
        print(f"❌ Test data not found at: {json_path}")
        print("Please run prepare.py first to create test data")
        return None
    
    # Load JSON data and convert to DataFrame
    import json
    with open(json_path, 'r') as f:
        data = json.load(f)
    
    # Convert to DataFrame format
    df = pd.DataFrame(data)
    
    # Sample test examples (use all if test_size is None or large enough)
    if test_size is None or test_size >= len(df):
        print(f"Evaluating on full test set: {len(df)} examples")
    else:
        df = df.sample(n=test_size, random_state=42)
        print(f"Evaluating on {len(df)} test examples (sampled from {len(data)})")
    
    # Convert to DSPy format - detect the label field
    label_field = 'is_favorite' if 'is_favorite' in df.columns else 'is_shortlist'
    test_examples = []
    for _, row in df.iterrows():
        example = dspy.Example(
            title=row['title'],
            is_favorite=row[label_field]  # Use same field name for consistency
        ).with_inputs('title')
        test_examples.append(example)
    
    # Load rubric
    rubric_path = paths_config.get('rubric', "saved/rubrics/personal_taste_rubric.txt")
    rubric = None
    if os.path.exists(rubric_path):
        with open(rubric_path, 'r') as f:
            rubric = f.read()
        print("✅ Loaded taste rubric")
    
    # Configure DSPy with LLM from config
    llm_config = config.get_llm_config()
    cache_enabled = llm_config.get('cache', False)
    print(f"🤖 Using LLM: {llm_name}")
    dspy.configure(lm=dspy.LM(llm_name, cache=cache_enabled), cache=cache_enabled)
    
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
  --model PATH       Path to trained model
  --dataset PATH     Path to test dataset JSON file
  --threads N        Number of parallel threads (default: from config)
  --llm MODEL        LLM model to use (default: from config)
  --no-progress      Hide progress bars on console
  -h, --help         Show this help message

Examples:
  python evaluate.py                                      # Evaluate all test examples on favorite data
  python evaluate.py 200                                 # Evaluate 200 examples on favorite data
  python evaluate.py --model saved/models/dspy_favorite_001.json  # Use specific model
  python evaluate.py --dataset data/processed/reader_shortlist/test/dspy_examples.json  # Use shortlist data
  python evaluate.py 50 --threads 8                      # 50 examples with 8 threads
  python evaluate.py --no-progress                       # Hide progress bars
        """)
        sys.exit(0)
    
    # Parse command line arguments
    test_size = None  # Default: use all test examples
    model_path = 'saved/models/dspy_favorite'  # Default model path
    num_threads = None  # Will be set from config
    show_progress = True
    dataset_path = None
    llm_model = None  # Will be set from config
    
    i = 1
    while i < len(sys.argv):
        arg = sys.argv[i]
        if arg.lower() == 'all':
            test_size = None  # Use all examples
        elif arg == '--model' and i + 1 < len(sys.argv):
            model_path = sys.argv[i + 1]
            i += 1  # Skip the next argument
        elif arg == '--threads' and i + 1 < len(sys.argv):
            num_threads = int(sys.argv[i + 1])
            i += 1  # Skip the next argument
        elif arg == '--no-progress':
            show_progress = False
        elif arg == '--dataset' and i + 1 < len(sys.argv):
            dataset_path = sys.argv[i + 1]
            i += 1  # Skip the next argument
        elif arg == '--llm' and i + 1 < len(sys.argv):
            llm_model = sys.argv[i + 1]
            i += 1  # Skip the next argument
        else:
            try:
                test_size = int(arg)
            except ValueError:
                print(f"Warning: Ignoring unrecognized argument: {arg}")
        i += 1
    
    # Run evaluation
    accuracy = evaluate_model(test_size=test_size, model_path=model_path, num_threads=num_threads, show_progress=show_progress, dataset_path=dataset_path, llm_model=llm_model)
    
    if accuracy is not None:
        print(f"\n✅ Evaluation completed! Accuracy: {accuracy:.3f}")
    else:
        print(f"\n❌ Evaluation failed!")
        sys.exit(1)