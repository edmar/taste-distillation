#!/usr/bin/env python3
"""
Training script for the pairwise comparison classifier
"""

import os
import json
from datetime import datetime
from pairwise_data_loader import PairwiseDataLoader
from pairwise_predictor import PairwiseClassifier

def train_pairwise_model(
    csv_path: str,
    model_name: str = 'openai/gpt-4o-mini',
    max_train_examples: int = 1000,
    pair_strategy: str = 'mixed',
    balanced: bool = True,
    augment_flip: bool = True,
    optimizer_type: str = 'auto',
    use_validation: bool = True,
    stratify_by: str = 'confidence',
    model_save_path: str = 'src/pairwise_classifier/models/pairwise_model.json',
    metadata_save_path: str = 'src/pairwise_classifier/models/pairwise_training_metadata.json'
):
    """Train the pairwise comparison classifier and save it"""
    
    print("="*60)
    print("PAIRWISE COMPARISON CLASSIFIER TRAINING")
    print("="*60)
    
    # Load and process data
    print("Loading data...")
    loader = PairwiseDataLoader(csv_path)
    data = loader.load_and_process()
    
    # Generate pairwise training examples
    print("Generating pairwise examples...")
    all_pairs = loader.generate_pairwise_examples(
        max_examples=max_train_examples * 2,  # Generate more, then split
        balanced=balanced,
        pair_strategy=pair_strategy,
        augment_flip=augment_flip
    )
    
    # Create splits
    print("Creating train/val/test splits...")
    splits = loader.create_splits(
        all_pairs,
        train_size=0.7,
        val_size=0.15,
        test_size=0.15,
        stratify_by=stratify_by
    )
    
    train_examples = splits['train']
    val_examples = splits['val'] if use_validation else None
    test_examples = splits['test']
    
    # Limit training examples if needed
    if len(train_examples) > max_train_examples:
        print(f"Limiting training examples from {len(train_examples)} to {max_train_examples}")
        train_examples = train_examples[:max_train_examples]
    
    # Show sample training examples
    print(f"\nTraining set composition:")
    print(f"  Total examples: {len(train_examples)}")
    
    # Count by confidence and pair type
    train_confidences = {'high': 0, 'medium': 0, 'low': 0}
    train_pair_types = {}
    for ex in train_examples:
        train_confidences[ex['confidence']] += 1
        ptype = ex['pair_type']
        train_pair_types[ptype] = train_pair_types.get(ptype, 0) + 1
    
    print(f"  Confidence distribution: High: {train_confidences['high']}, Medium: {train_confidences['medium']}, Low: {train_confidences['low']}")
    
    top_pair_types = sorted(train_pair_types.items(), key=lambda x: x[1], reverse=True)[:3]
    print(f"  Top pair types: {', '.join([f'{ptype}: {count}' for ptype, count in top_pair_types])}")
    
    if val_examples:
        val_confidences = {'high': 0, 'medium': 0, 'low': 0}
        for ex in val_examples:
            val_confidences[ex['confidence']] += 1
        print(f"\nValidation set composition:")
        print(f"  Total examples: {len(val_examples)}")
        print(f"  Confidence distribution: High: {val_confidences['high']}, Medium: {val_confidences['medium']}, Low: {val_confidences['low']}")
    
    print(f"\nTest set: {len(test_examples)} examples")
    
    print(f"\nOptimization settings:")
    print(f"  Optimizer type: {optimizer_type}")
    print(f"  Use validation: {use_validation}")
    print(f"  Pair strategy: {pair_strategy}")
    print(f"  Balanced: {balanced}")
    print(f"  Augment flip: {augment_flip}")
    
    print(f"\nSample training pairs:")
    for i, ex in enumerate(train_examples[:3]):
        winner_title = ex['title_a'] if ex['preferred_title'] == 'A' else ex['title_b']
        print(f"  {i+1}. [{ex['pair_type']}] {ex['confidence']} confidence")
        print(f"     A: {ex['title_a'][:50]}...")
        print(f"     B: {ex['title_b'][:50]}...")
        print(f"     Winner: {ex['preferred_title']} -> {winner_title[:50]}...")
    
    # Initialize classifier
    print(f"\nInitializing pairwise classifier with model: {model_name}")
    classifier = PairwiseClassifier(model_name=model_name)
    
    # Train the model
    print("\n" + "="*40)
    print("TRAINING")
    print("="*40)
    
    try:
        print("Starting pairwise training...")
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
            'model_name': model_name,
            'model_type': 'pairwise_comparison',
            'training_examples': len(train_examples),
            'validation_examples': len(val_examples) if val_examples else 0,
            'test_examples': len(test_examples),
            'max_train_examples': max_train_examples,
            'pair_strategy': pair_strategy,
            'balanced': balanced,
            'augment_flip': augment_flip,
            'optimizer_type': optimizer_type,
            'use_validation': use_validation,
            'stratify_by': stratify_by,
            'training_successful': training_successful,
            'model_save_path': model_save_path,
            'data_stats': {
                'total_favorites': len(loader.favorites),
                'total_non_favorites': len(loader.non_favorites),
                'total_generated_pairs': len(all_pairs)
            },
            'split_stats': {
                'train_pairs': len(splits['train']),
                'val_pairs': len(splits['val']),
                'test_pairs': len(splits['test'])
            },
            'training_confidence_distribution': train_confidences,
            'training_pair_types': train_pair_types
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
        
        # Quick evaluation on test set
        if len(test_examples) > 0:
            print("\n" + "="*40)
            print("QUICK EVALUATION")
            print("="*40)
            
            try:
                # Take a small sample for quick evaluation
                test_sample = test_examples[:min(50, len(test_examples))]
                print(f"Evaluating on {len(test_sample)} test examples...")
                
                eval_results = classifier.evaluate(test_sample)
                
                print(f"Quick evaluation results:")
                print(f"  Overall accuracy: {eval_results['accuracy']:.3f}")
                print(f"  High confidence accuracy: {eval_results['confidence_accuracy']['high']:.3f}")
                print(f"  Medium confidence accuracy: {eval_results['confidence_accuracy']['medium']:.3f}")
                print(f"  Low confidence accuracy: {eval_results['confidence_accuracy']['low']:.3f}")
                
                # Add to metadata
                metadata['quick_evaluation'] = eval_results
                
                # Re-save metadata with evaluation results
                with open(metadata_save_path, 'w') as f:
                    json.dump(metadata, f, indent=2)
                
            except Exception as e:
                print(f"⚠️ Quick evaluation failed: {e}")
        
        # Summary
        print("\n" + "="*40)
        print("TRAINING SUMMARY")
        print("="*40)
        print(f"✅ Pairwise model successfully trained and saved")
        print(f"📊 Training examples: {len(train_examples)} pairs")
        print(f"💾 Model saved to: {model_save_path}")
        print(f"📝 Metadata saved to: {metadata_save_path}")
        print(f"\nNext steps:")
        print(f"  1. Run full evaluation: poetry run python src/pairwise_classifier/pairwise_evaluate.py")
        print(f"  2. Test pairwise comparisons on new title pairs")
        print(f"  3. Use for ranking multiple titles")
        
        return {
            'success': True,
            'model_path': model_save_path,
            'metadata_path': metadata_save_path,
            'training_examples': len(train_examples),
            'test_examples': len(test_examples)
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
    max_examples = 1000
    pair_strategy = 'mixed'
    balanced = True
    augment_flip = True
    optimizer_type = 'auto'
    
    # Parse command line arguments flexibly
    for i, arg in enumerate(sys.argv[1:], 1):
        if arg in ['mixed', 'fav_vs_nonfav', 'all_combinations']:
            pair_strategy = arg
        elif arg.lower() in ['balanced', 'balance', 'true']:
            balanced = True
        elif arg.lower() in ['unbalanced', 'unbalance', 'false']:
            balanced = False
        elif arg.lower() in ['flip', 'augment', 'augment_flip']:
            augment_flip = True
        elif arg.lower() in ['no_flip', 'no_augment']:
            augment_flip = False
        elif arg.lower() in ['bootstrap', 'bootstrap_random', 'mipro', 'auto']:
            optimizer_type = arg.lower()
        else:
            try:
                max_examples = int(arg)
            except ValueError:
                print(f"Warning: Ignoring unrecognized argument: {arg}")
    
    print(f"Parameters:")
    print(f"  Max training examples: {max_examples}")
    print(f"  Pair strategy: {pair_strategy}")
    print(f"  Balanced sampling: {balanced}")
    print(f"  Augment flip: {augment_flip}")
    print(f"  Optimizer type: {optimizer_type}")
    
    # Run training
    result = train_pairwise_model(
        csv_path=csv_path,
        max_train_examples=max_examples,
        pair_strategy=pair_strategy,
        balanced=balanced,
        augment_flip=augment_flip,
        optimizer_type=optimizer_type
    )
    
    if result['success']:
        print(f"\n🎉 Pairwise training completed successfully!")
    else:
        print(f"\n💥 Pairwise training failed!")
        sys.exit(1)