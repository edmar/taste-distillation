#!/usr/bin/env python3
"""
Training script for the favorite classifier - focused only on training
"""

import os
import json
from datetime import datetime
from data_loader import TasteDataLoader
from favorite_predictor import TasteClassifier

def train_model(
    csv_path: str,
    split_method: str = 'random',
    model_name: str = 'openai/gpt-4o-mini',
    max_train_examples: int = 500,
    balanced: bool = False,
    optimizer_type: str = 'auto',
    use_validation: bool = True,
    model_save_path: str = 'src/favorite_classifier/models/trained_model.json',
    metadata_save_path: str = 'src/favorite_classifier/models/training_metadata.json'
):
    """Train the favorite classifier and save it"""
    
    print("="*60)
    print("FAVORITE CLASSIFIER TRAINING")
    print("="*60)
    
    # Load and split data
    print("Loading data...")
    loader = TasteDataLoader(csv_path)
    data = loader.load_and_process()
    splits = loader.create_splits(split_method=split_method)
    
    # Get training examples with optional balancing
    train_examples = loader.get_training_examples(splits, balanced=balanced, max_examples=max_train_examples)
    
    # Get validation examples if requested
    val_examples = None
    if use_validation:
        val_examples = loader.get_training_examples(
            {'train': splits['val']}, 
            balanced=balanced, 
            max_examples=min(200, max_train_examples // 4)  # Smaller validation set
        )
    
    # Show sample training examples
    favorites = [ex for ex in train_examples if ex['is_favorite']]
    non_favorites = [ex for ex in train_examples if not ex['is_favorite']]
    
    print(f"\nTraining set composition:")
    print(f"  Total examples: {len(train_examples)}")
    print(f"  Favorites: {len(favorites)} ({len(favorites)/len(train_examples)*100:.1f}%)")
    print(f"  Non-favorites: {len(non_favorites)} ({len(non_favorites)/len(train_examples)*100:.1f}%)")
    
    if val_examples:
        val_favorites = [ex for ex in val_examples if ex['is_favorite']]
        val_non_favorites = [ex for ex in val_examples if not ex['is_favorite']]
        print(f"\nValidation set composition:")
        print(f"  Total examples: {len(val_examples)}")
        print(f"  Favorites: {len(val_favorites)} ({len(val_favorites)/len(val_examples)*100:.1f}%)")
        print(f"  Non-favorites: {len(val_non_favorites)} ({len(val_non_favorites)/len(val_examples)*100:.1f}%)")
    
    print(f"\nOptimization settings:")
    print(f"  Optimizer type: {optimizer_type}")
    print(f"  Use validation: {use_validation}")
    
    print(f"\nSample favorite titles:")
    for ex in favorites[:3]:
        print(f"  + {ex['title']}")
    
    print(f"\nSample non-favorite titles:")
    for ex in non_favorites[:3]:
        print(f"  - {ex['title']}")
    
    # Initialize classifier
    print(f"\nInitializing classifier with model: {model_name}")
    classifier = TasteClassifier(model_name=model_name)
    
    # Train the model
    print("\n" + "="*40)
    print("TRAINING")
    print("="*40)
    
    try:
        print("Starting training...")
        trained_module = classifier.train(
            train_examples, 
            val_examples=val_examples, 
            optimizer_type=optimizer_type
        )
        training_successful = True
        print("✅ Training completed successfully!")
    except Exception as e:
        print(f"❌ Training failed: {e}")
        training_successful = False
        trained_module = None
    
    # Save results
    if training_successful:
        print("\n" + "="*40)
        print("SAVING MODEL")
        print("="*40)
        
        # Save the trained model
        classifier.save_model(model_save_path)
        
        # Save training metadata
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        metadata = {
            'timestamp': timestamp,
            'split_method': split_method,
            'model_name': model_name,
            'training_examples': len(train_examples),
            'favorite_examples': len(favorites),
            'non_favorite_examples': len(non_favorites),
            'max_train_examples': max_train_examples,
            'balanced': balanced,
            'optimizer_type': optimizer_type,
            'use_validation': use_validation,
            'validation_examples': len(val_examples) if val_examples else 0,
            'training_successful': training_successful,
            'model_save_path': model_save_path,
            'data_split_stats': {
                'train_size': len(splits['train']),
                'val_size': len(splits['val']),
                'test_size': len(splits['test'])
            }
        }
        
        # Create metadata directory
        os.makedirs(os.path.dirname(metadata_save_path), exist_ok=True)
        
        try:
            with open(metadata_save_path, 'w') as f:
                json.dump(metadata, f, indent=2)
            print(f"✅ Training metadata saved to: {metadata_save_path}")
        except Exception as e:
            print(f"⚠️ Failed to save training metadata: {e}")
            print("Training was successful, but metadata could not be saved.")
        
        # Summary
        print("\n" + "="*40)
        print("TRAINING SUMMARY")
        print("="*40)
        print(f"✅ Model successfully trained and saved")
        print(f"📊 Training examples: {len(train_examples)}")
        print(f"💾 Model saved to: {model_save_path}")
        print(f"📝 Metadata saved to: {metadata_save_path}")
        print(f"\nNext steps:")
        print(f"  1. Run evaluation: poetry run python src/favorite_classifier/evaluate.py")
        print(f"  2. Test predictions on new titles")
        
        return {
            'success': True,
            'model_path': model_save_path,
            'metadata_path': metadata_save_path,
            'training_examples': len(train_examples)
        }
    else:
        print("\n❌ Training failed - no model saved")
        return {
            'success': False,
            'error': 'Training failed'
        }

if __name__ == "__main__":
    import sys
    
    # Default parameters
    csv_path = "/Users/edmar/Code/taste/export.csv"
    split_method = 'random'
    max_examples = 500
    balanced = False
    optimizer_type = 'auto'
    
    # Parse command line arguments flexibly
    for i, arg in enumerate(sys.argv[1:], 1):
        if arg in ['random', 'temporal']:
            split_method = arg
        elif arg.lower() in ['balanced', 'balance', 'true']:
            balanced = True
        elif arg.lower() in ['bootstrap', 'bootstrap_random', 'mipro', 'auto']:
            optimizer_type = arg.lower()
        else:
            try:
                max_examples = int(arg)
            except ValueError:
                print(f"Warning: Ignoring unrecognized argument: {arg}")
    
    print(f"Parameters:")
    print(f"  Split method: {split_method}")
    print(f"  Max training examples: {max_examples}")
    print(f"  Balanced sampling: {balanced}")
    print(f"  Optimizer type: {optimizer_type}")
    
    # Run training
    result = train_model(
        csv_path=csv_path,
        split_method=split_method,
        max_train_examples=max_examples,
        balanced=balanced,
        optimizer_type=optimizer_type
    )
    
    if result['success']:
        print(f"\n🎉 Training completed successfully!")
    else:
        print(f"\n💥 Training failed!")
        sys.exit(1)