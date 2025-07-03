import dspy
from typing import List, Dict

class FavoritePredictor(dspy.Signature):
    """Predict whether an article will be marked as favorite based on its title."""
    
    title = dspy.InputField(desc="Article title")
    is_favorite = dspy.OutputField(desc="Whether this article should be marked as favorite (True/False)")

class TastePredictionModule(dspy.Module):
    """DSPy module for predicting favorite articles from titles"""
    
    def __init__(self):
        super().__init__()
        self.predict_favorite = dspy.ChainOfThought(FavoritePredictor)
    
    def forward(self, title: str):
        """Predict if an article title indicates a favorite"""
        result = self.predict_favorite(title=title)
        return result

class TasteClassifier:
    """High-level interface for the taste prediction system"""
    
    def __init__(self, model_name: str = "openai/gpt-4o-mini"):
        # Configure DSPy
        self.lm = dspy.LM(model_name)
        dspy.configure(lm=self.lm)
        
        # Initialize module
        self.module = TastePredictionModule()
        self.trained_module = None
        self.model_name = model_name
    
    def train(self, train_examples: List[Dict], val_examples: List[Dict] = None, optimizer_type: str = "auto"):
        """Train the taste prediction module with improved optimization"""
        print(f"Training on {len(train_examples)} examples...")
        if val_examples:
            print(f"Validation set: {len(val_examples)} examples")
        
        # Convert to DSPy examples
        dspy_examples = []
        for ex in train_examples:
            dspy_ex = dspy.Example(
                title=ex['title'],
                is_favorite=ex['is_favorite']
            ).with_inputs('title')
            dspy_examples.append(dspy_ex)
        
        # Convert validation examples if provided
        val_dspy_examples = None
        if val_examples:
            val_dspy_examples = []
            for ex in val_examples:
                val_ex = dspy.Example(
                    title=ex['title'],
                    is_favorite=ex['is_favorite']
                ).with_inputs('title')
                val_dspy_examples.append(val_ex)
        
        # Choose optimizer based on data size and type
        num_examples = len(train_examples)
        
        if optimizer_type == "auto":
            if num_examples < 20:
                optimizer_type = "bootstrap"
            elif num_examples < 100:
                optimizer_type = "bootstrap_random"
            else:
                optimizer_type = "mipro"
        
        print(f"Selected optimizer: {optimizer_type}")
        
        # Set up optimizer with better configuration
        if optimizer_type == "bootstrap":
            optimizer = dspy.BootstrapFewShot(
                metric=self._accuracy_metric,
                metric_threshold=0.6,  # Only accept good bootstrap examples
                max_bootstrapped_demos=min(8, num_examples // 4),  # Scale with data
                max_labeled_demos=min(16, num_examples // 2),     # Scale with data
                max_rounds=2  # Multiple rounds for better examples
            )
        elif optimizer_type == "bootstrap_random":
            optimizer = dspy.BootstrapFewShotWithRandomSearch(
                metric=self._accuracy_metric,
                metric_threshold=0.6,
                max_bootstrapped_demos=min(6, num_examples // 4),
                max_labeled_demos=min(12, num_examples // 3),
                num_candidate_programs=8,  # Try multiple configurations
                num_threads=2  # Parallel optimization
            )
        elif optimizer_type == "mipro":
            # Use MIPROv2 for larger datasets with conservative settings
            optimizer = dspy.MIPROv2(
                metric=self._accuracy_metric,
                auto="light",  # Light optimization with preset configurations
                num_threads=1,  # Single thread to avoid file handle issues
                max_errors=10  # Allow some errors during optimization
            )
        else:
            raise ValueError(f"Unknown optimizer type: {optimizer_type}")
        
        # Train with validation set if available (only some optimizers support valset)
        print(f"Running {optimizer_type} optimization...")
        try:
            if val_dspy_examples and optimizer_type in ['mipro']:
                # Only MIPROv2 supports validation set
                self.trained_module = optimizer.compile(
                    student=self.module,
                    trainset=dspy_examples,
                    valset=val_dspy_examples[:min(len(val_dspy_examples), 20)],  # Smaller val size
                    requires_permission_to_run=False  # Bypass confirmation
                )
            else:
                # Fall back to training set only for bootstrap optimizers
                if optimizer_type == 'mipro':
                    self.trained_module = optimizer.compile(
                        student=self.module,
                        trainset=dspy_examples,
                        requires_permission_to_run=False
                    )
                else:
                    self.trained_module = optimizer.compile(
                        student=self.module,
                        trainset=dspy_examples
                    )
        except Exception as e:
            print(f"Optimizer compile failed: {e}")
            # Fallback: try basic compilation without special parameters
            try:
                self.trained_module = optimizer.compile(
                    student=self.module,
                    trainset=dspy_examples
                )
            except Exception as e2:
                print(f"Basic compilation also failed: {e2}")
                raise e2
        
        print("Training completed!")
        return self.trained_module
    
    def save_model(self, filepath: str):
        """Save the trained model using DSPy's state-only saving with robust error handling"""
        import json
        import os
        import time
        
        if self.trained_module is None:
            raise ValueError("No trained model to save. Train the model first.")
        
        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        
        # Determine file paths
        if filepath.endswith('.json'):
            json_path = filepath
            metadata_path = filepath.replace('.json', '_metadata.json')
            pkl_path = filepath.replace('.json', '.pkl')
        else:
            json_path = filepath.replace('.pkl', '.json')
            metadata_path = filepath.replace('.pkl', '_metadata.json')
            pkl_path = filepath
        
        # Try multiple saving strategies
        saved_successfully = False
        
        # Strategy 1: Try JSON state-only saving
        try:
            self.trained_module.save(json_path, save_program=False)
            print(f"✅ Model state saved to: {json_path}")
            saved_successfully = True
        except Exception as e:
            print(f"⚠️ JSON save failed: {e}")
            
            # Strategy 2: Try pickle saving as fallback
            try:
                print("Trying pickle format as fallback...")
                import pickle
                with open(pkl_path, 'wb') as f:
                    pickle.dump(self.trained_module, f)
                print(f"✅ Model saved as pickle to: {pkl_path}")
                saved_successfully = True
            except Exception as e2:
                print(f"⚠️ Pickle save also failed: {e2}")
                
                # Strategy 3: Try saving just the demos and signature
                try:
                    print("Trying manual state extraction...")
                    manual_state = {
                        'demos': getattr(self.trained_module.predict_favorite, 'demos', []),
                        'signature': getattr(self.trained_module.predict_favorite, 'signature', None),
                    }
                    
                    # Add small delay to avoid file handle issues
                    time.sleep(0.1)
                    
                    with open(json_path, 'w') as f:
                        json.dump(manual_state, f, indent=2, default=str)
                    print(f"✅ Manual state saved to: {json_path}")
                    saved_successfully = True
                except Exception as e3:
                    print(f"❌ All save strategies failed: {e3}")
        
        # Save metadata if any strategy worked
        if saved_successfully:
            try:
                # Add small delay to avoid file handle conflicts
                time.sleep(0.1)
                
                metadata = {
                    'model_name': self.model_name,
                    'trained': True,
                    'json_path': json_path,
                    'pkl_path': pkl_path if os.path.exists(pkl_path) else None
                }
                
                with open(metadata_path, 'w') as f:
                    json.dump(metadata, f, indent=2)
                
                print(f"✅ Metadata saved to: {metadata_path}")
                
            except Exception as e:
                print(f"⚠️ Metadata save failed: {e}")
        else:
            print("❌ Warning: Could not save model using any method")
            print("The model is still trained and can be used in this session.")
            print("Model persistence across sessions may not be available.")
    
    def load_model(self, filepath: str):
        """Load a trained model using DSPy's state loading"""
        import json
        
        try:
            # Load metadata
            if filepath.endswith('.json'):
                metadata_path = filepath.replace('.json', '_metadata.json')
                json_path = filepath
            else:
                metadata_path = filepath.replace('.pkl', '_metadata.json')
                json_path = filepath.replace('.pkl', '.json')
                
            with open(metadata_path, 'r') as f:
                metadata = json.load(f)
            
            self.model_name = metadata['model_name']
            json_path = metadata.get('json_path', json_path)
            
            # Reconfigure DSPy with the same model
            self.lm = dspy.LM(self.model_name)
            dspy.configure(lm=self.lm)
            
            # Recreate the same program architecture, then load state
            self.trained_module = TastePredictionModule()
            self.trained_module.load(json_path)
            
            print(f"Model loaded from: {json_path}")
            
        except Exception as e:
            print(f"Warning: Could not load model: {e}")
            print("Please retrain the model in this session.")
            raise ValueError(f"Failed to load model: {e}")
    
    def predict(self, title: str) -> Dict:
        """Predict if a title indicates a favorite article"""
        if self.trained_module is None:
            module = self.module
        else:
            module = self.trained_module
        
        result = module(title=title)
        
        # Parse the boolean result
        is_fav_str = str(result.is_favorite).lower()
        is_favorite = is_fav_str in ['true', '1', 'yes']
        
        return {
            'title': title,
            'is_favorite': is_favorite,
            'raw_prediction': result.is_favorite,
            'reasoning': getattr(result, 'reasoning', '')
        }
    
    def predict_batch(self, titles: List[str]) -> List[Dict]:
        """Predict favorites for multiple titles"""
        return [self.predict(title) for title in titles]
    
    def evaluate(self, test_examples: List[Dict]) -> Dict:
        """Evaluate the model on test examples"""
        print(f"Evaluating on {len(test_examples)} examples...")
        
        predictions = []
        actuals = []
        
        for ex in test_examples:
            pred = self.predict(ex['title'])
            predictions.append(pred['is_favorite'])
            actuals.append(ex['is_favorite'])
        
        # Calculate metrics
        correct = sum(1 for p, a in zip(predictions, actuals) if p == a)
        accuracy = correct / len(test_examples)
        
        # Calculate precision, recall for favorites
        true_positives = sum(1 for p, a in zip(predictions, actuals) if p and a)
        false_positives = sum(1 for p, a in zip(predictions, actuals) if p and not a)
        false_negatives = sum(1 for p, a in zip(predictions, actuals) if not p and a)
        
        precision = true_positives / (true_positives + false_positives) if (true_positives + false_positives) > 0 else 0
        recall = true_positives / (true_positives + false_negatives) if (true_positives + false_negatives) > 0 else 0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
        
        results = {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1_score': f1,
            'total_examples': len(test_examples),
            'correct_predictions': correct,
            'true_positives': true_positives,
            'false_positives': false_positives,
            'false_negatives': false_negatives
        }
        
        print(f"Accuracy: {accuracy:.3f}")
        print(f"Precision: {precision:.3f}")
        print(f"Recall: {recall:.3f}")
        print(f"F1-Score: {f1:.3f}")
        
        return results
    
    def _accuracy_metric(self, example, pred, trace=None):
        """Metric for DSPy optimization"""
        # Parse prediction
        pred_str = str(pred.is_favorite).lower()
        pred_bool = pred_str in ['true', '1', 'yes']
        
        # Compare to actual
        return pred_bool == example.is_favorite

if __name__ == "__main__":
    # Test the predictor with some examples
    from data_loader import TasteDataLoader
    
    print("Loading data...")
    loader = TasteDataLoader("/Users/edmar/Code/taste/export.csv")
    data = loader.load_and_process()
    splits = loader.create_splits(split_method='random')
    
    # Get small sample for testing
    train_examples = loader.get_training_examples(splits)[:50]  # Small sample for testing
    
    print("Creating classifier...")
    classifier = TasteClassifier()
    
    print("Testing prediction (before training)...")
    test_title = "Why You Should Start a Blog Right Now"
    result = classifier.predict(test_title)
    print(f"Title: {result['title']}")
    print(f"Predicted favorite: {result['is_favorite']}")
    print(f"Reasoning: {result['reasoning']}")
    
    print("\nTesting with a few training examples...")
    for i, ex in enumerate(train_examples[:3]):
        result = classifier.predict(ex['title'])
        actual = ex['is_favorite']
        print(f"{i+1}. {ex['title'][:60]}...")
        print(f"   Predicted: {result['is_favorite']}, Actual: {actual}")
        print(f"   Match: {'✓' if result['is_favorite'] == actual else '✗'}")
        print()