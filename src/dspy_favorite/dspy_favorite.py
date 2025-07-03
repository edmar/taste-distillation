"""
DSPy Favorite Classifier

DSPy signatures and modules for binary taste classification.
"""

import dspy
from typing import Optional


def create_favorite_predictor_signature(rubric: Optional[str] = None):
    """Create a FavoritePredictor signature with rubric embedded in instructions."""
    
    instruction = "Predict whether an article will be marked as favorite based on its title."
    
    if rubric:
        instruction += f"\n\nUse the following taste rubric as guidance:\n{rubric}"
    
    return dspy.Signature(
        "title -> is_favorite: bool",
        instruction
    )


def create_favorite_predictor_with_reasoning_signature(rubric: Optional[str] = None):
    """Create a FavoritePredictorWithReasoning signature with rubric embedded in instructions."""
    
    instruction = "Predict whether an article will be marked as favorite based on its title with detailed reasoning."
    
    if rubric:
        instruction += f"\n\nUse the following taste rubric as guidance:\n{rubric}"
    
    return dspy.Signature(
        "title -> is_favorite: bool, reasoning: str",
        instruction
    )


class TastePredictionModule(dspy.Module):
    """DSPy module for predicting favorite articles from titles."""
    
    def __init__(self, use_reasoning: bool = True, rubric: Optional[str] = None):
        super().__init__()
        self.use_reasoning = use_reasoning
        
        if use_reasoning:
            signature = create_favorite_predictor_with_reasoning_signature(rubric)
            self.predict_favorite = dspy.ChainOfThought(signature)
        else:
            signature = create_favorite_predictor_signature(rubric)
            self.predict_favorite = dspy.Predict(signature)
    
    def forward(self, title: str):
        """Predict if an article title indicates a favorite."""
        result = self.predict_favorite(title=title)
        return result