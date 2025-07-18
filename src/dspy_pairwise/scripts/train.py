#!/usr/bin/env python3
"""
Training script for DSPy pairwise classifier
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

# Add parent directory to path to import dspy_pairwise
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from dspy_pairwise import PairwiseComparisonModule

# Add shared utilities
sys.path.append(os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), 'shared'))
from logging_utils import tee_output

def accuracy_metric(example, pred, trace=None):
    """Simple accuracy metric for DSPy training"""
    return example.preferred_title == pred.preferred_title

def train_model(
    data_path: str = "data/processed/reader_favorite_pairwise",
    model_name: str = "openai/gpt-4.1",
    optimizer_type: str = "mipro",
    max_train_examples: int = None,
    max_bootstrapped_demos: int = 4,
    max_labeled_demos: int = 4,
    num_threads: int = 16,
    auto_level: str = "light",
    save_path: str = "saved/models/dspy_pairwise",
    train_dataset_path: str = None,
    show_progress: bool = True
):
    """Train the DSPy pairwise classifier"""
    
    # Extract model name for logging
    model_name_for_log = os.path.basename(save_path)
    
    with tee_output(model_name_for_log, 'train', show_progress):
        print("="*50)
        print("DSPy PAIRWISE CLASSIFIER TRAINING")
        print("="*50)
        print(f"Model: {model_name} | Optimizer: {optimizer_type} | Examples: {max_train_examples or 'all'}")
        
        return _train_model_impl(
            data_path, model_name, optimizer_type, max_train_examples,
            max_bootstrapped_demos, max_labeled_demos, num_threads, auto_level, save_path, train_dataset_path
        )

def _train_model_impl(
    data_path: str = "data/processed/reader_favorite_pairwise",
    model_name: str = "openai/gpt-4.1",
    optimizer_type: str = "mipro",
    max_train_examples: int = None,
    max_bootstrapped_demos: int = 4,
    max_labeled_demos: int = 4,
    num_threads: int = 16,
    auto_level: str = "light",
    save_path: str = "saved/models/dspy_pairwise",
    train_dataset_path: str = None,
):
    """Implementation of training logic"""
    
    # Load training data
    if train_dataset_path is not None:
        train_path = train_dataset_path
    else:
        train_path = f"{data_path}/train/dspy_examples.json"
    
    
    if not os.path.exists(train_path):
        print(f"❌ Training data not found at: {train_path}")
        if train_dataset_path is None:
            print("Please run prepare.py first to create training data")
        return False
    
    print(f"📂 Loading training data from: {train_path}")

    # Load training examples
    with open(train_path, 'r') as f:
        train_data = json.load(f)
    
    
    # Convert to DSPy Examples
    trainset = []
    subset = train_data if max_train_examples is None else train_data[:max_train_examples]
    for item in subset:
        example = dspy.Example(
            title_a=item['title_a'],
            title_b=item['title_b'],
            preferred_title=item['preferred_title'],
            confidence=item['confidence']
        ).with_inputs('title_a', 'title_b')
        trainset.append(example)
    
    # Show training statistics
    preferences_a = [ex for ex in trainset if ex.preferred_title == 'A']
    preferences_b = [ex for ex in trainset if ex.preferred_title == 'B']
    high_confidence = [ex for ex in trainset if ex.confidence == 'high']
    
    print(f"Training examples: {len(trainset)} ({len(preferences_a)} prefer A, {len(preferences_b)} prefer B)")
    print(f"High confidence examples: {len(high_confidence)}")
    
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
    module = PairwiseComparisonModule(use_reasoning=True, rubric=rubric)
    
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
        'preference_a_examples': len(preferences_a),
        'preference_b_examples': len(preferences_b),
        'high_confidence_examples': len(high_confidence),
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
    parser = argparse.ArgumentParser(description='DSPy Pairwise Classifier Training Script')
    parser.add_argument('max_examples', nargs='?', type=int, help='Maximum training examples to use')
    parser.add_argument('optimizer', nargs='?', choices=['mipro', 'bootstrap'], help='Optimizer type')
    parser.add_argument('auto_level', nargs='?', choices=['light', 'medium', 'heavy'], help='Auto optimization level for MIPROv2')
    parser.add_argument('--threads', type=int, help='Number of parallel threads')
    parser.add_argument('--model', help='Language model to use')
    parser.add_argument('--train-data', '--dataset', dest='train_data', help='Path to training dataset file')
    parser.add_argument('--no-progress', action='store_true', help='Hide progress bars on console')
    
    args = parser.parse_args()
    
    # Build kwargs dict, only including non-None values
    kwargs = {k: v for k, v in {
        'max_train_examples': args.max_examples,
        'optimizer_type': args.optimizer,
        'auto_level': args.auto_level,
        'num_threads': args.threads,
        'model_name': args.model,
        'train_dataset_path': args.train_data,
        'show_progress': not args.no_progress
    }.items() if v is not None}
    
    success = train_model(**kwargs)
    
    if not success:
        sys.exit(1)