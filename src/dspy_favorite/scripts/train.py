#!/usr/bin/env python3
"""
Training script for DSPy favorite classifier
"""

import os
import sys
import json
import pandas as pd
import dspy
import argparse
from datetime import datetime

# Disable DSPy caching to avoid database conflicts in parallel training
os.environ['DSPY_CACHEDIR'] = '/tmp/dspy_cache_disable'

# Add parent directory to path to import dspy_favorite
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from dspy_favorite import TastePredictionModule

def accuracy_metric(example, pred, trace=None):
    """Simple accuracy metric for DSPy training"""
    return example.is_favorite == pred.is_favorite

def train_model(
    data_path: str = "data/processed/reader_favorite",
    model_name: str = "openai/gpt-4.1",
    optimizer_type: str = "mipro",
    max_train_examples: int = None,
    max_bootstrapped_demos: int = 4,
    max_labeled_demos: int = 4,
    num_threads: int = 16,
    auto_level: str = "light",
    save_path: str = "saved/models/dspy_favorite"
):
    """Train the DSPy favorite classifier"""
    
    print("="*50)
    print("DSPy FAVORITE CLASSIFIER TRAINING")
    print("="*50)
    print(f"Model: {model_name} | Optimizer: {optimizer_type} | Examples: {max_train_examples or 'all'}")
    
    # Load training data
    train_path = f"{data_path}/train/dspy_examples.json"
    
    if not os.path.exists(train_path):
        print(f"❌ Training data not found at: {train_path}")
        print("Please run prepare.py first to create training data")
        return False
    
    # Load training examples
    with open(train_path, 'r') as f:
        train_data = json.load(f)
    
    
    # Convert to DSPy Examples
    trainset = []
    subset = train_data if max_train_examples is None else train_data[:max_train_examples]
    for item in subset:
        example = dspy.Example(
            title=item['title'],
            is_favorite=item['is_favorite']
        ).with_inputs('title')
        trainset.append(example)
    
    
    # Show training statistics
    favorites = [ex for ex in trainset if ex.is_favorite]
    non_favorites = [ex for ex in trainset if not ex.is_favorite]
    
    print(f"Training examples: {len(trainset)} ({len(favorites)} favorites, {len(non_favorites)} non-favorites)")
    
    # Load rubric if available
    rubric_path = "saved/rubrics/personal_taste_rubric.txt"
    rubric = None
    if os.path.exists(rubric_path):
        with open(rubric_path, 'r') as f:
            rubric = f.read()
        print("✅ Loaded taste rubric")
    
    # Configure DSPy with cache disabled
    dspy.configure(lm=dspy.LM(model_name, cache=False), cache=False)
    
    # Initialize module
    print(f"\n🔧 Initializing module...")
    module = TastePredictionModule(use_reasoning=True, rubric=rubric)
    
    # Set up optimizer
    print(f"\n🎯 Setting up {optimizer_type} optimizer...")
    
    if optimizer_type.lower() == "mipro":
        teleprompter = dspy.MIPROv2(
            metric=accuracy_metric,
            auto=auto_level,
            num_threads=num_threads
        )
        compile_kwargs = {
            "max_bootstrapped_demos": max_bootstrapped_demos,
            "max_labeled_demos": max_labeled_demos,
            "requires_permission_to_run": False
        }
    elif optimizer_type.lower() == "bootstrap":
        teleprompter = dspy.BootstrapFewShot(
            metric=accuracy_metric,
            max_bootstrapped_demos=max_bootstrapped_demos,
            max_labeled_demos=max_labeled_demos,
            max_rounds=1,
            max_errors=10
        )
        compile_kwargs = {
            "requires_permission_to_run": False
        }
    else:
        print(f"❌ Unknown optimizer type: {optimizer_type}")
        return False
    
    # Optimize the model
    print(f"\n🚀 Starting optimization...")
    optimized_module = teleprompter.compile(
        module,
        trainset=trainset,
        **compile_kwargs
    )
    
    print("✅ Optimization completed successfully!")
    
    # Save the optimized model
    print(f"\n💾 Saving model...")
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    
    # Find next available model number
    base_path = save_path
    counter = 1
    while os.path.exists(f"{base_path}_{counter:03d}.json"):
        counter += 1
    numbered_path = f"{base_path}_{counter:03d}"
    
    # Save DSPy model
    optimized_module.save(f"{numbered_path}.json")
    print(f"✅ Model saved to: {numbered_path}.json")
    
    # Save training metadata
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    metadata = {
        'timestamp': timestamp,
        'model_name': model_name,
        'optimizer_type': optimizer_type,
        'training_examples': len(trainset),
        'favorite_examples': len(favorites),
        'non_favorite_examples': len(non_favorites),
        'max_train_examples': max_train_examples,
        'max_bootstrapped_demos': max_bootstrapped_demos,
        'max_labeled_demos': max_labeled_demos,
        'num_threads': num_threads,
        'auto_level': auto_level,
        'has_rubric': rubric is not None,
        'rubric': rubric,
        'save_path': numbered_path
    }
    
    metadata_path = f"{numbered_path}_metadata.json"
    with open(metadata_path, 'w') as f:
        json.dump(metadata, f, indent=2)
    print(f"✅ Metadata saved to: {metadata_path}")
    
    print(f"\n✅ Training completed! Model saved to: {numbered_path}.json")
    
    return True

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='DSPy Favorite Classifier Training Script')
    parser.add_argument('max_examples', nargs='?', type=int, help='Maximum training examples to use')
    parser.add_argument('optimizer', nargs='?', choices=['mipro', 'bootstrap'], help='Optimizer type')
    parser.add_argument('auto_level', nargs='?', choices=['light', 'medium', 'heavy'], help='Auto optimization level for MIPROv2')
    parser.add_argument('--threads', type=int, help='Number of parallel threads')
    parser.add_argument('--model', help='Language model to use')
    
    args = parser.parse_args()
    
    # Build kwargs dict, only including non-None values
    kwargs = {k: v for k, v in {
        'max_train_examples': args.max_examples,
        'optimizer_type': args.optimizer,
        'auto_level': args.auto_level,
        'num_threads': args.threads,
        'model_name': args.model
    }.items() if v is not None}
    
    success = train_model(**kwargs)
    
    if not success:
        sys.exit(1)