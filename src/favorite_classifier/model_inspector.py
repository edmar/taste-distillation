#!/usr/bin/env python3
"""
DSPy Model Inspector - Comprehensive utility for inspecting trained models

This script provides various inspection capabilities for DSPy models:
- Model state analysis
- Prompt structure visualization  
- Demonstration analysis
- Performance comparison
"""

import json
import os
import sys
from typing import Dict, List, Any
import dspy
from favorite_predictor import TasteClassifier


class ModelInspector:
    """Utility class for inspecting DSPy models"""
    
    def __init__(self, model_path: str = "models/trained_model.json"):
        self.model_path = model_path
        self.model_data = None
        self.classifier = None
        
    def load_model_data(self) -> bool:
        """Load and parse the model JSON file"""
        if not os.path.exists(self.model_path):
            print(f"❌ Model not found at: {self.model_path}")
            return False
        
        try:
            with open(self.model_path, 'r') as f:
                self.model_data = json.load(f)
            return True
        except Exception as e:
            print(f"❌ Error reading model file: {e}")
            return False
    
    def show_model_overview(self):
        """Show basic model information"""
        if not self.model_data:
            return
        
        print("="*60)
        print("DSPy MODEL OVERVIEW")
        print("="*60)
        
        # Top-level structure
        print("📁 Model Structure:")
        for key in self.model_data.keys():
            print(f"  • {key}")
        
        # Predictor info
        if 'predict_favorite.predict' in self.model_data:
            predictor = self.model_data['predict_favorite.predict']
            demos = predictor.get('demos', [])
            traces = predictor.get('traces', [])
            
            print(f"\n📊 Training Results:")
            print(f"  • Demonstrations: {len(demos)}")
            print(f"  • Traces: {len(traces)}")
            
            if demos:
                favorites = [d for d in demos if str(d.get('is_favorite')).lower() == 'true']
                print(f"  • Favorite demos: {len(favorites)} ({len(favorites)/len(demos)*100:.1f}%)")
        
        # Metadata
        if 'metadata' in self.model_data:
            meta = self.model_data['metadata']
            print(f"\n⚙️ Environment:")
            if 'dependency_versions' in meta:
                for dep, ver in meta['dependency_versions'].items():
                    print(f"  • {dep}: {ver}")
    
    def show_prompt_structure(self):
        """Show the DSPy prompt structure and format"""
        if not self.model_data:
            return
        
        predictor = self.model_data.get('predict_favorite.predict', {})
        signature = predictor.get('signature', {})
        demos = predictor.get('demos', [])
        
        print("\n" + "="*60)
        print("PROMPT STRUCTURE ANALYSIS")
        print("="*60)
        
        # Instructions
        instructions = signature.get('instructions', 'None')
        print(f"📋 Instructions: \"{instructions}\"")
        
        # Field structure
        print(f"\n🏗️ Field Structure:")
        fields = signature.get('fields', [])
        for field in fields:
            prefix = field.get('prefix', '')
            desc = field.get('description', '')
            print(f"  • {prefix} → {desc}")
        
        # Show reconstructed prompt with examples
        if demos:
            print(f"\n📜 RECONSTRUCTED PROMPT (with {len(demos)} demonstrations):")
            print("-" * 60)
            
            prompt = self._reconstruct_prompt(demos[:3])  # Show first 3 demos
            print(prompt)
    
    def _reconstruct_prompt(self, demos: List[Dict], test_title: str = "Example Article Title") -> str:
        """Reconstruct what the DSPy prompt would look like"""
        prompt = "Given the fields `title`, produce the fields `reasoning`, `is_favorite`.\n\n"
        prompt += "---\n\n"
        prompt += "Follow the following format.\n\n"
        prompt += "Title: {title}\n"
        prompt += "Reasoning: Let's think step by step in order to {reasoning}\n"
        prompt += "Is Favorite: {is_favorite}\n\n"
        prompt += "---\n\n"
        
        # Add demonstrations
        for demo in demos:
            prompt += f"Title: {demo.get('title', 'N/A')}\n"
            reasoning = demo.get('reasoning', '')
            if reasoning:
                prompt += f"Reasoning: Let's think step by step in order to {reasoning}\n"
            prompt += f"Is Favorite: {demo.get('is_favorite', 'N/A')}\n\n"
            prompt += "---\n\n"
        
        # Current input
        prompt += f"Title: {test_title}\n"
        prompt += "Reasoning: Let's think step by step in order to"
        
        return prompt
    
    def analyze_demonstrations(self):
        """Analyze the quality and content of demonstrations"""
        if not self.model_data:
            return
        
        predictor = self.model_data.get('predict_favorite.predict', {})
        demos = predictor.get('demos', [])
        
        if not demos:
            print("\n❌ No demonstrations found in model")
            return
        
        print("\n" + "="*60)
        print("DEMONSTRATION ANALYSIS")
        print("="*60)
        
        # Basic stats
        favorites = [d for d in demos if str(d.get('is_favorite')).lower() == 'true']
        non_favorites = [d for d in demos if str(d.get('is_favorite')).lower() == 'false']
        
        print(f"📊 Statistics:")
        print(f"  • Total: {len(demos)}")
        print(f"  • Favorites: {len(favorites)} ({len(favorites)/len(demos)*100:.1f}%)")
        print(f"  • Non-favorites: {len(non_favorites)} ({len(non_favorites)/len(demos)*100:.1f}%)")
        
        # Show examples
        print(f"\n✅ FAVORITE Examples:")
        for i, demo in enumerate(favorites[:3]):
            print(f"  {i+1}. \"{demo.get('title', 'N/A')}\"")
            reasoning = demo.get('reasoning', '')
            if reasoning:
                short_reasoning = reasoning[:80] + "..." if len(reasoning) > 80 else reasoning
                print(f"     → {short_reasoning}")
        
        print(f"\n❌ NON-FAVORITE Examples:")
        for i, demo in enumerate(non_favorites[:3]):
            print(f"  {i+1}. \"{demo.get('title', 'N/A')}\"")
            reasoning = demo.get('reasoning', '')
            if reasoning:
                short_reasoning = reasoning[:80] + "..." if len(reasoning) > 80 else reasoning
                print(f"     → {short_reasoning}")
    
    def test_model_prediction(self, test_title: str = None):
        """Load model and test a prediction"""
        print("\n" + "="*60)
        print("MODEL PREDICTION TEST")
        print("="*60)
        
        try:
            self.classifier = TasteClassifier()
            self.classifier.load_model(self.model_path)
            print("✅ Model loaded successfully")
            
            if test_title is None:
                test_title = "How to Build Better AI Applications with Modern Tools"
            
            print(f"\n🧪 Testing: \"{test_title}\"")
            result = self.classifier.predict(test_title)
            
            print(f"📊 Result:")
            print(f"  • Prediction: {result['is_favorite']}")
            reasoning = result.get('reasoning', 'No reasoning provided')
            print(f"  • Reasoning: {reasoning[:150]}{'...' if len(reasoning) > 150 else ''}")
            
        except Exception as e:
            print(f"❌ Error testing model: {e}")
    
    def compare_models(self, other_model_path: str):
        """Compare this model with another model"""
        print("\n" + "="*60)
        print("MODEL COMPARISON")
        print("="*60)
        
        try:
            with open(other_model_path, 'r') as f:
                other_data = json.load(f)
            
            # Compare demonstrations
            current_demos = self.model_data.get('predict_favorite.predict', {}).get('demos', [])
            other_demos = other_data.get('predict_favorite.predict', {}).get('demos', [])
            
            print(f"📊 Demonstration Comparison:")
            print(f"  • Current model: {len(current_demos)} demos")
            print(f"  • Other model: {len(other_demos)} demos")
            
            # Compare balance
            if current_demos:
                current_fav = len([d for d in current_demos if str(d.get('is_favorite')).lower() == 'true'])
                current_balance = current_fav / len(current_demos) * 100
                print(f"  • Current balance: {current_balance:.1f}% favorites")
            
            if other_demos:
                other_fav = len([d for d in other_demos if str(d.get('is_favorite')).lower() == 'true'])
                other_balance = other_fav / len(other_demos) * 100
                print(f"  • Other balance: {other_balance:.1f}% favorites")
                
        except Exception as e:
            print(f"❌ Error comparing models: {e}")
    
    def export_prompt_template(self, output_path: str = "prompt_template.txt"):
        """Export the reconstructed prompt as a template"""
        if not self.model_data:
            return
        
        predictor = self.model_data.get('predict_favorite.predict', {})
        demos = predictor.get('demos', [])
        
        prompt = self._reconstruct_prompt(demos, "{NEW_ARTICLE_TITLE}")
        
        try:
            with open(output_path, 'w') as f:
                f.write(prompt)
            print(f"\n💾 Prompt template exported to: {output_path}")
        except Exception as e:
            print(f"❌ Error exporting template: {e}")


def main():
    """Main function with command line interface"""
    if len(sys.argv) < 2:
        print("Usage: python model_inspector.py <command> [model_path] [options]")
        print("\nCommands:")
        print("  overview     - Show model overview")
        print("  prompt       - Show prompt structure") 
        print("  demos        - Analyze demonstrations")
        print("  test         - Test model prediction")
        print("  all          - Show all information")
        print("  compare      - Compare two models (requires second model path)")
        print("  export       - Export prompt template")
        print("\nExample:")
        print("  python model_inspector.py all")
        print("  python model_inspector.py test models/trained_model.json")
        print("  python model_inspector.py compare models/model1.json models/model2.json")
        return
    
    command = sys.argv[1]
    model_path = sys.argv[2] if len(sys.argv) > 2 else "models/trained_model.json"
    
    inspector = ModelInspector(model_path)
    
    if not inspector.load_model_data():
        return
    
    if command == "overview":
        inspector.show_model_overview()
    elif command == "prompt":
        inspector.show_prompt_structure()
    elif command == "demos":
        inspector.analyze_demonstrations()
    elif command == "test":
        test_title = sys.argv[3] if len(sys.argv) > 3 else None
        inspector.test_model_prediction(test_title)
    elif command == "all":
        inspector.show_model_overview()
        inspector.show_prompt_structure()
        inspector.analyze_demonstrations()
        inspector.test_model_prediction()
    elif command == "compare":
        if len(sys.argv) < 4:
            print("❌ Compare requires two model paths")
            return
        other_path = sys.argv[3]
        inspector.compare_models(other_path)
    elif command == "export":
        output_path = sys.argv[3] if len(sys.argv) > 3 else "prompt_template.txt"
        inspector.export_prompt_template(output_path)
    else:
        print(f"❌ Unknown command: {command}")


if __name__ == "__main__":
    main()