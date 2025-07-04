#!/usr/bin/env python3
"""
Simple evaluation script for DSPy favorite classifier
"""

import os
import sys
import json
import dspy
import argparse
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


def evaluate_model(args):
    """Evaluate DSPy favorite classifier"""
    
    # Load configuration
    config = get_config("evaluate")
    eval_config = config.get_evaluation_config()
    paths_config = config.get_paths_config()
    
    # Set defaults from config
    dataset_path = args.dataset or paths_config.get('default_dataset')
    model_path = args.model or paths_config.get('default_model')
    num_threads = args.threads or eval_config.get('num_threads', 16)
    llm_name = config.get_llm_name(args.llm)
    
    # Extract model name for logging
    log_name = os.path.basename(model_path).replace('.json', '') if model_path else 'dspy_favorite'
    
    with tee_output(log_name, 'eval', not args.no_progress):
        print("="*50)
        print("DSPy FAVORITE CLASSIFIER EVALUATION")
        print("="*50)
        print(f"📂 Dataset: {dataset_path}")
        
        # Load test data
        if not os.path.exists(dataset_path):
            print(f"❌ Test data not found at: {dataset_path}")
            return None
        
        with open(dataset_path, 'r') as f:
            data = json.load(f)
        
        # Sample if requested
        if args.test_size and args.test_size < len(data):
            import random
            random.seed(42)
            data = random.sample(data, args.test_size)
            print(f"Evaluating on {len(data)} test examples (sampled)")
        else:
            print(f"Evaluating on full test set: {len(data)} examples")
        
        # Convert to DSPy format
        label_field = 'is_favorite' if 'is_favorite' in data[0] else 'is_shortlist'
        test_examples = [
            dspy.Example(title=item['title'], is_favorite=item[label_field]).with_inputs('title')
            for item in data
        ]
        
        # Load rubric
        rubric_path = paths_config.get('rubric')
        rubric = None
        if rubric_path and os.path.exists(rubric_path):
            with open(rubric_path, 'r') as f:
                rubric = f.read()
            print("✅ Loaded taste rubric")
        
        # Configure DSPy
        llm_config = config.get_llm_config(args.llm)
        cache_enabled = llm_config.get('cache', False)
        print(f"🤖 Using LLM: {llm_name}")
        
        # Build LM parameters with any special requirements
        lm_params = {'cache': cache_enabled}
        if 'temperature' in llm_config:
            lm_params['temperature'] = llm_config['temperature']
        if 'max_tokens' in llm_config:
            lm_params['max_tokens'] = llm_config['max_tokens']
            
        dspy.configure(lm=dspy.LM(llm_name, **lm_params), cache=cache_enabled)
        
        # Initialize module
        module = TastePredictionModule(use_reasoning=True, rubric=rubric)
        
        # Load trained model if available
        model_json_path = model_path if model_path.endswith('.json') else f"{model_path}.json"
        if os.path.exists(model_json_path):
            module.load(model_json_path)
            print(f"✅ Loaded trained model from: {model_json_path}")
        else:
            print(f"📊 Running baseline evaluation with untrained model")
        
        # Run evaluation
        evaluator = Evaluate(devset=test_examples, num_threads=num_threads, display_progress=True)
        accuracy = evaluator(module, metric=accuracy_metric)
        
        print(f"\n🎯 Accuracy: {accuracy:.3f}")
        return accuracy


def main():
    parser = argparse.ArgumentParser(description='DSPy Favorite Classifier Evaluation')
    parser.add_argument('test_size', nargs='?', type=int, help='Number of test examples (default: all)')
    parser.add_argument('--dataset', help='Path to test dataset JSON file')
    parser.add_argument('--model', help='Path to trained model')
    parser.add_argument('--llm', help='LLM model to use')
    parser.add_argument('--threads', type=int, help='Number of parallel threads')
    parser.add_argument('--no-progress', action='store_true', help='Hide progress bars')
    
    args = parser.parse_args()
    
    accuracy = evaluate_model(args)
    
    if accuracy is not None:
        print(f"\n✅ Evaluation completed! Accuracy: {accuracy:.3f}")
    else:
        print(f"\n❌ Evaluation failed!")
        sys.exit(1)


if __name__ == "__main__":
    main()