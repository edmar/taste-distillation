#!/usr/bin/env python3
"""
Training script for DSPy pairwise classifier
"""

import os
import sys
import json
import pandas as pd
import dspy
from datetime import datetime

# Disable DSPy caching to avoid database conflicts in parallel training
os.environ['DSPY_CACHEDIR'] = '/tmp/dspy_cache_disable'

# Add parent directory to path to import dspy_pairwise
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from dspy_pairwise import PairwiseComparisonModule

def accuracy_metric(example, pred, trace=None):
    """Simple accuracy metric for DSPy training"""
    return example.preferred_title == pred.preferred_title

def train_model(
    data_path: str = "data/processed/dspy_pairwise",
    model_name: str = "openai/gpt-4o-mini",
    optimizer_type: str = "mipro",
    max_train_examples: int = None,
    max_bootstrapped_demos: int = 4,
    max_labeled_demos: int = 4,
    num_threads: int = 16,
    auto_level: str = "light",
    save_path: str = "saved/models/dspy_pairwise"
):
    """Train the DSPy pairwise classifier"""
    
    print("="*50)
    print("DSPy PAIRWISE CLASSIFIER TRAINING")
    print("="*50)
    print(f"Model: {model_name} | Optimizer: {optimizer_type} | Examples: {max_train_examples or 'all'}")
    
    # Load training data
    train_path = f"{data_path}/train/dspy_examples.json"
    val_path = f"{data_path}/val/dspy_examples.json"
    
    if not os.path.exists(train_path):
        print(f"❌ Training data not found at: {train_path}")
        print("Please run prepare.py first to create training data")
        return False
    
    # Load training examples
    with open(train_path, 'r') as f:
        train_data = json.load(f)
    
    # Load validation examples if available
    val_data = []
    if os.path.exists(val_path):
        with open(val_path, 'r') as f:
            val_data = json.load(f)
    
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
    
    valset = []
    for item in val_data:
        example = dspy.Example(
            title_a=item['title_a'],
            title_b=item['title_b'],
            preferred_title=item['preferred_title'],
            confidence=item['confidence']
        ).with_inputs('title_a', 'title_b')
        valset.append(example)
    
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
    # Check for help flag
    if '--help' in sys.argv or '-h' in sys.argv:
        print("""
DSPy Pairwise Classifier Training Script

Usage: python train.py [OPTIONS]

Options:
  <number>           Maximum training examples to use (default: all)
  mipro, bootstrap   Optimizer type (default: mipro)
  light, medium, heavy  Auto optimization level for MIPROv2 (default: light)
  --threads=N        Number of parallel threads (default: 16)
  --model=MODEL      Language model to use (default: openai/gpt-4o-mini)
  -h, --help         Show this help message

Examples:
  python train.py                    # Train with all examples using MIPROv2
  python train.py 500               # Train with 500 examples
  python train.py bootstrap         # Use bootstrap optimizer
  python train.py 1000 mipro heavy  # 1000 examples, MIPROv2, heavy optimization
  python train.py --threads=8       # Use 8 parallel threads
  python train.py --model=openai/gpt-4o  # Use different model
        """)
        sys.exit(0)
    
    # Default parameters
    data_path = "data/processed/dspy_pairwise"
    model_name = "openai/gpt-4o-mini"
    optimizer_type = "mipro"
    max_train_examples = None  # Default to all training examples
    max_bootstrapped_demos = 4
    max_labeled_demos = 4
    num_threads = 16
    auto_level = "light"
    
    # Parse command line arguments
    for arg in sys.argv[1:]:
        if arg.lower() in ['mipro', 'bootstrap']:
            optimizer_type = arg.lower()
        elif arg.lower() in ['light', 'medium', 'heavy']:
            auto_level = arg.lower()
        elif arg.startswith('--threads='):
            num_threads = int(arg.split('=')[1])
        elif arg.startswith('--model='):
            model_name = arg.split('=')[1]
        else:
            try:
                max_train_examples = int(arg)
            except ValueError:
                print(f"Warning: Ignoring unrecognized argument: {arg}")
    
    # Run training
    success = train_model(
        data_path=data_path,
        model_name=model_name,
        optimizer_type=optimizer_type,
        max_train_examples=max_train_examples,
        max_bootstrapped_demos=max_bootstrapped_demos,
        max_labeled_demos=max_labeled_demos,
        num_threads=num_threads,
        auto_level=auto_level
    )
    
    if not success:
        sys.exit(1)