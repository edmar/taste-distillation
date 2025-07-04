#!/usr/bin/env python3
"""
Training script for DSPy favorite classifier
"""

import os
import sys
import json
import yaml
import dspy
import argparse
from datetime import datetime

# Disable DSPy caching to avoid database conflicts in parallel training
os.environ['DSPY_CACHEDIR'] = '/tmp/dspy_cache_disable'

# Add parent directory to path to import dspy_favorite
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from dspy_favorite import TastePredictionModule

# Add shared utilities
sys.path.append(os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), 'shared'))
from logging_utils import tee_output
from config import get_config

def accuracy_metric(example, pred, trace=None):
    """Simple accuracy metric for DSPy training"""
    return example.is_favorite == pred.is_favorite

def train_model(
    data_path: str = None,
    llm_model: str = None,
    optimizer_type: str = None,
    max_train_examples: int = None,
    max_bootstrapped_demos: int = None,
    max_labeled_demos: int = None,
    num_threads: int = None,
    auto_level: str = None,
    save_path: str = None,
    show_progress: bool = True
):
    """Train the DSPy favorite classifier"""
    
    # Load config
    config = get_config("train")
    paths_config = config.get_paths_config()
    training_config = config.get_training_config()
    
    # Set defaults from config
    if data_path is None:
        data_path = paths_config.get('default_dataset', 'data/reader_favorite')
    if save_path is None:
        save_path = paths_config.get('models_dir', 'saved/models') + '/dspy_favorite'
    
    # Extract model name for logging
    model_name_for_log = os.path.basename(save_path)
    
    with tee_output(model_name_for_log, 'train', show_progress):
        print("="*50)
        print("DSPy FAVORITE CLASSIFIER TRAINING")
        print("="*50)
        print(f"📂 Dataset: {data_path}")
        print(f"🤖 LLM: {llm_model or config.get_llm_name()} | ⚙️  Optimizer: {optimizer_type or training_config.get('optimizer', 'mipro')} | 📊 Examples: {max_train_examples or 'all'}")
        
        return _train_model_impl(
            data_path, llm_model, optimizer_type, max_train_examples,
            max_bootstrapped_demos, max_labeled_demos, num_threads, auto_level, save_path
        )

def _train_model_impl(
    data_path: str,
    llm_model: str = None,
    optimizer_type: str = None,
    max_train_examples: int = None,
    max_bootstrapped_demos: int = None,
    max_labeled_demos: int = None,
    num_threads: int = None,
    auto_level: str = None,
    save_path: str = None
):
    """Implementation of training logic"""
    
    # Load training configuration
    config = get_config("train")
    training_config = config.get_training_config()
    paths_config = config.get_paths_config()
    
    # Load full config to access optimizers section
    with open(config.config_path, 'r') as f:
        full_config = yaml.safe_load(f)
    optimizers_config = full_config.get('optimizers', {})
    
    # Set defaults from config
    llm_name = config.get_llm_name(llm_model)
    if optimizer_type is None:
        optimizer_type = training_config.get('optimizer', 'mipro')
    
    # Get optimizer-specific config
    optimizer_config = optimizers_config.get(optimizer_type, {})
    
    # Set defaults with priority: args > optimizer_config > training_config > hardcoded
    if max_train_examples is None:
        max_train_examples = training_config.get('max_train_examples', None)
    if max_bootstrapped_demos is None:
        max_bootstrapped_demos = optimizer_config.get('max_bootstrapped_demos', training_config.get('max_bootstrapped_demos', 4))
    if max_labeled_demos is None:
        max_labeled_demos = optimizer_config.get('max_labeled_demos', training_config.get('max_labeled_demos', 4))
    if num_threads is None:
        num_threads = training_config.get('num_threads', 16)
    if auto_level is None:
        auto_level = optimizer_config.get('auto_level', training_config.get('auto_level', 'light'))
    
    print(f"🤖 Using LLM: {llm_name}")
    print(f"⚙️  Training config: {optimizer_type} optimizer, {auto_level} auto level, {num_threads} threads")
    print(f"📋 Demos: {max_bootstrapped_demos} bootstrapped, {max_labeled_demos} labeled")
    
    # Load training data - handle both directory and file paths
    if data_path.endswith('.json'):
        # Full file path provided
        train_path = data_path
    else:
        # Directory path provided, append train file
        train_path = f"{data_path}/train/dspy_examples.json"
    
    if not os.path.exists(train_path):
        print(f"❌ Training data not found at: {train_path}")
        print("Please run prepare.py first to create training data")
        return False
    
    # Load training examples
    with open(train_path, 'r') as f:
        train_data = json.load(f)
    
    
    # Convert to DSPy Examples - detect the label field
    label_field = 'is_favorite' if 'is_favorite' in train_data[0] else 'is_shortlist'
    trainset = []
    subset = train_data if max_train_examples is None else train_data[:max_train_examples]
    for item in subset:
        example = dspy.Example(
            title=item['title'],
            is_favorite=item[label_field]  # Use same field name for consistency
        ).with_inputs('title')
        trainset.append(example)
    
    
    # Show training statistics
    favorites = [ex for ex in trainset if ex.is_favorite]
    non_favorites = [ex for ex in trainset if not ex.is_favorite]
    
    print(f"Training examples: {len(trainset)} ({len(favorites)} favorites, {len(non_favorites)} non-favorites)")
    
    # Load rubric if available
    rubric_path = paths_config.get('rubric', "saved/rubrics/personal_taste_rubric.txt")
    rubric = None
    if os.path.exists(rubric_path):
        with open(rubric_path, 'r') as f:
            rubric = f.read()
        print("✅ Loaded taste rubric")
    
    # Configure DSPy with LLM from config
    llm_config = config.get_llm_config(llm_model)
    cache_enabled = llm_config.get('cache', False)
    
    # Build LM parameters with any special requirements
    lm_params = {'cache': cache_enabled}
    if 'temperature' in llm_config:
        lm_params['temperature'] = llm_config['temperature']
    if 'max_tokens' in llm_config:
        lm_params['max_tokens'] = llm_config['max_tokens']
        
    dspy.configure(lm=dspy.LM(llm_name, **lm_params), cache=cache_enabled)
    
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
            "max_labeled_demos": max_labeled_demos
        }
    elif optimizer_type.lower() == "bootstrap":
        # Get bootstrap-specific config with fallbacks
        max_rounds = optimizer_config.get('max_rounds', 1)
        max_errors = optimizer_config.get('max_errors', 10)
        
        teleprompter = dspy.BootstrapFewShot(
            metric=accuracy_metric,
            max_bootstrapped_demos=max_bootstrapped_demos,
            max_labeled_demos=max_labeled_demos,
            max_rounds=max_rounds,
            max_errors=max_errors
        )
        compile_kwargs = {}
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
        'llm_model': llm_name,
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
    parser.add_argument('--llm', help='Language model to use')
    parser.add_argument('--dataset', help='Dataset path to use')
    parser.add_argument('--no-progress', action='store_true', help='Hide progress bars on console')
    
    args = parser.parse_args()
    
    # Build kwargs dict, only including non-None values
    kwargs = {k: v for k, v in {
        'max_train_examples': args.max_examples,
        'optimizer_type': args.optimizer,
        'auto_level': args.auto_level,
        'num_threads': args.threads,
        'llm_model': args.llm,
        'data_path': args.dataset,
        'show_progress': not args.no_progress
    }.items() if v is not None}
    
    success = train_model(**kwargs)
    
    if not success:
        sys.exit(1)