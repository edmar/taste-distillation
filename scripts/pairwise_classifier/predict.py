import dspy
from typing import List, Dict

class PairwiseComparison(dspy.Signature):
    """Compare two article titles and predict which one is more likely to be marked as favorite."""
    
    title_a = dspy.InputField(desc="First article title to compare")
    title_b = dspy.InputField(desc="Second article title to compare")
    preferred_title = dspy.OutputField(desc="Which title is more likely to be favorite: 'A' or 'B'")
    confidence = dspy.OutputField(desc="Confidence level in the prediction: 'high', 'medium', or 'low'")

class PairwiseComparisonModule(dspy.Module):
    """DSPy module for pairwise comparison of article titles"""
    
    def __init__(self):
        super().__init__()
        self.compare_titles = dspy.ChainOfThought(PairwiseComparison)
    
    def forward(self, title_a: str, title_b: str):
        """Compare two titles and predict which is more likely to be favorite"""
        result = self.compare_titles(title_a=title_a, title_b=title_b)
        return result

class PairwiseClassifier:
    """High-level interface for the pairwise comparison system"""
    
    def __init__(self, model_name: str = "openai/gpt-4o-mini"):
        # Configure DSPy
        self.lm = dspy.LM(model_name)
        dspy.configure(lm=self.lm)
        
        # Initialize module
        self.module = PairwiseComparisonModule()
        self.trained_module = None
        self.model_name = model_name
    
    def train(self, train_examples: List[Dict], val_examples: List[Dict] = None, optimizer_type: str = "auto"):
        """Train the pairwise comparison module"""
        print(f"Training on {len(train_examples)} pairwise examples...")
        if val_examples:
            print(f"Validation set: {len(val_examples)} examples")
        
        # Convert to DSPy examples
        dspy_examples = []
        for ex in train_examples:
            dspy_ex = dspy.Example(
                title_a=ex['title_a'],
                title_b=ex['title_b'],
                preferred_title=ex['preferred_title'],
                confidence=ex['confidence']
            ).with_inputs('title_a', 'title_b')
            dspy_examples.append(dspy_ex)
        
        # Convert validation examples if provided
        val_dspy_examples = None
        if val_examples:
            val_dspy_examples = []
            for ex in val_examples:
                val_ex = dspy.Example(
                    title_a=ex['title_a'],
                    title_b=ex['title_b'],
                    preferred_title=ex['preferred_title'],
                    confidence=ex['confidence']
                ).with_inputs('title_a', 'title_b')
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
                metric=self._pairwise_accuracy_metric,
                metric_threshold=0.6,
                max_bootstrapped_demos=min(8, num_examples // 4),
                max_labeled_demos=min(16, num_examples // 2),
                max_rounds=2
            )
        elif optimizer_type == "bootstrap_random":
            optimizer = dspy.BootstrapFewShotWithRandomSearch(
                metric=self._pairwise_accuracy_metric,
                metric_threshold=0.6,
                max_bootstrapped_demos=min(6, num_examples // 4),
                max_labeled_demos=min(12, num_examples // 3),
                num_candidate_programs=8,
                num_threads=2
            )
        elif optimizer_type == "mipro":
            optimizer = dspy.MIPROv2(
                metric=self._pairwise_accuracy_metric,
                auto="light",
                num_threads=1,
                max_errors=10
            )
        else:
            raise ValueError(f"Unknown optimizer type: {optimizer_type}")
        
        # Train with validation set if available (only some optimizers support valset)
        print(f"Running {optimizer_type} optimization...")
        try:
            if val_dspy_examples and optimizer_type in ['mipro']:
                self.trained_module = optimizer.compile(
                    student=self.module,
                    trainset=dspy_examples,
                    valset=val_dspy_examples[:min(len(val_dspy_examples), 20)],
                    requires_permission_to_run=False
                )
            else:
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
        """Save the trained model using DSPy's state-only saving"""
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
                        'demos': getattr(self.trained_module.compare_titles, 'demos', []),
                        'signature': getattr(self.trained_module.compare_titles, 'signature', None),
                    }
                    
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
                time.sleep(0.1)
                
                metadata = {
                    'model_name': self.model_name,
                    'trained': True,
                    'model_type': 'pairwise_comparison',
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
            self.trained_module = PairwiseComparisonModule()
            self.trained_module.load(json_path)
            
            print(f"Pairwise model loaded from: {json_path}")
            
        except Exception as e:
            print(f"Warning: Could not load model: {e}")
            print("Please retrain the model in this session.")
            raise ValueError(f"Failed to load model: {e}")
    
    def compare(self, title_a: str, title_b: str) -> Dict:
        """Compare two titles and predict which is more likely to be favorite"""
        if self.trained_module is None:
            module = self.module
        else:
            module = self.trained_module
        
        result = module(title_a=title_a, title_b=title_b)
        
        # Parse the preferred title result
        preferred = str(result.preferred_title).upper()
        if preferred not in ['A', 'B']:
            # Fallback parsing
            if 'A' in preferred and 'B' not in preferred:
                preferred = 'A'
            elif 'B' in preferred and 'A' not in preferred:
                preferred = 'B'
            else:
                preferred = 'A'  # Default fallback
        
        # Parse confidence level
        confidence = str(result.confidence).lower()
        if confidence not in ['high', 'medium', 'low']:
            confidence = 'medium'  # Default fallback
        
        return {
            'title_a': title_a,
            'title_b': title_b,
            'winner': preferred,
            'preferred_title': title_a if preferred == 'A' else title_b,
            'confidence': confidence,
            'raw_prediction': result.preferred_title,
            'raw_confidence': result.confidence,
            'reasoning': getattr(result, 'reasoning', '')
        }
    
    def compare_batch(self, title_pairs: List[tuple]) -> List[Dict]:
        """Compare multiple pairs of titles"""
        return [self.compare(pair[0], pair[1]) for pair in title_pairs]
    
    def rank_titles(self, titles: List[str]) -> List[Dict]:
        """Rank a list of titles using pairwise comparisons (simple tournament)"""
        if len(titles) <= 1:
            return [{'title': title, 'rank': 1, 'score': 1.0} for title in titles]
        
        # Score each title by counting wins against all others
        scores = {title: 0 for title in titles}
        total_comparisons = {title: 0 for title in titles}
        
        # Compare each pair
        for i, title_a in enumerate(titles):
            for j, title_b in enumerate(titles):
                if i != j:
                    result = self.compare(title_a, title_b)
                    winner = result['winner']
                    confidence_weight = {'high': 1.0, 'medium': 0.7, 'low': 0.5}[result['confidence']]
                    
                    if winner == 'A':
                        scores[title_a] += confidence_weight
                    else:
                        scores[title_b] += confidence_weight
                    
                    total_comparisons[title_a] += 1
                    total_comparisons[title_b] += 1
        
        # Normalize scores and rank
        normalized_scores = {}
        for title in titles:
            if total_comparisons[title] > 0:
                normalized_scores[title] = scores[title] / total_comparisons[title]
            else:
                normalized_scores[title] = 0.0
        
        # Sort by score
        ranked_titles = sorted(titles, key=lambda t: normalized_scores[t], reverse=True)
        
        # Create result with ranks
        result = []
        for rank, title in enumerate(ranked_titles, 1):
            result.append({
                'title': title,
                'rank': rank,
                'score': normalized_scores[title],
                'wins': scores[title],
                'total_comparisons': total_comparisons[title]
            })
        
        return result
    
    def evaluate(self, test_examples: List[Dict]) -> Dict:
        """Evaluate the model on test examples"""
        print(f"Evaluating on {len(test_examples)} pairwise examples...")
        
        correct_predictions = 0
        confidence_correct = {'high': 0, 'medium': 0, 'low': 0}
        confidence_total = {'high': 0, 'medium': 0, 'low': 0}
        
        for ex in test_examples:
            pred = self.compare(ex['title_a'], ex['title_b'])
            
            # Check if prediction is correct
            is_correct = pred['winner'] == ex['preferred_title']
            if is_correct:
                correct_predictions += 1
                confidence_correct[pred['confidence']] += 1
            
            confidence_total[pred['confidence']] += 1
        
        # Calculate metrics
        accuracy = correct_predictions / len(test_examples)
        
        # Calculate confidence calibration
        confidence_accuracy = {}
        for conf_level in ['high', 'medium', 'low']:
            if confidence_total[conf_level] > 0:
                confidence_accuracy[conf_level] = confidence_correct[conf_level] / confidence_total[conf_level]
            else:
                confidence_accuracy[conf_level] = 0.0
        
        results = {
            'accuracy': accuracy,
            'total_examples': len(test_examples),
            'correct_predictions': correct_predictions,
            'confidence_distribution': confidence_total,
            'confidence_accuracy': confidence_accuracy
        }
        
        print(f"Overall Accuracy: {accuracy:.3f}")
        print(f"High Confidence Accuracy: {confidence_accuracy['high']:.3f} ({confidence_total['high']} examples)")
        print(f"Medium Confidence Accuracy: {confidence_accuracy['medium']:.3f} ({confidence_total['medium']} examples)")
        print(f"Low Confidence Accuracy: {confidence_accuracy['low']:.3f} ({confidence_total['low']} examples)")
        
        return results
    
    def _pairwise_accuracy_metric(self, example, pred, trace=None):
        """Metric for DSPy optimization - checks both preference and confidence"""
        # Parse prediction
        pred_title = str(pred.preferred_title).upper()
        if pred_title not in ['A', 'B']:
            # Fallback parsing
            if 'A' in pred_title and 'B' not in pred_title:
                pred_title = 'A'
            elif 'B' in pred_title and 'A' not in pred_title:
                pred_title = 'B'
            else:
                return False  # Invalid prediction
        
        # Check if preference is correct
        preference_correct = pred_title == example.preferred_title
        
        # Parse confidence
        pred_confidence = str(pred.confidence).lower()
        confidence_valid = pred_confidence in ['high', 'medium', 'low']
        
        # Return True only if both preference and confidence are reasonable
        return preference_correct and confidence_valid

if __name__ == "__main__":
    # Test the predictor with some examples
    print("Creating pairwise classifier...")
    classifier = PairwiseClassifier()
    
    print("Testing comparison (before training)...")
    title_a = "Why You Should Start a Blog Right Now"
    title_b = "10 Common Mistakes in Web Development"
    
    result = classifier.compare(title_a, title_b)
    print(f"Title A: {result['title_a']}")
    print(f"Title B: {result['title_b']}")
    print(f"Winner: {result['winner']} ({result['preferred_title']})")
    print(f"Confidence: {result['confidence']}")
    print(f"Reasoning: {result['reasoning']}")
    
    print("\nTesting ranking with multiple titles...")
    titles = [
        "Why You Should Start a Blog Right Now",
        "10 Common Mistakes in Web Development", 
        "The Future of Artificial Intelligence",
        "How to Build Better Software"
    ]
    
    ranked = classifier.rank_titles(titles)
    print("Rankings:")
    for item in ranked:
        print(f"{item['rank']}. {item['title']} (score: {item['score']:.3f})")