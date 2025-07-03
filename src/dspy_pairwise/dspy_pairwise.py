"""
DSPy Pairwise Classifier

DSPy signatures and modules for pairwise taste comparison.
"""

import dspy
from typing import Optional


def create_pairwise_comparison_signature(rubric: Optional[str] = None):
    """Create a PairwiseComparison signature with rubric embedded in instructions."""
    
    instruction = "Compare two article titles and predict which one is more likely to be marked as favorite. Return 'A' or 'B' as the preferred title."
    
    if rubric:
        instruction += f"\n\nUse the following taste rubric as guidance:\n{rubric}"
    
    return dspy.Signature(
        "title_a, title_b -> preferred_title: str, confidence: str",
        instruction
    )


def create_pairwise_comparison_with_reasoning_signature(rubric: Optional[str] = None):
    """Create a PairwiseComparisonWithReasoning signature with rubric embedded in instructions."""
    
    instruction = "Compare two article titles and predict which one is more likely to be marked as favorite with detailed reasoning. Return 'A' or 'B' as the preferred title."
    
    if rubric:
        instruction += f"\n\nUse the following taste rubric as guidance:\n{rubric}"
    
    return dspy.Signature(
        "title_a, title_b -> preferred_title: str, confidence: str, reasoning: str",
        instruction
    )


class PairwiseComparisonModule(dspy.Module):
    """DSPy module for pairwise comparison of article titles."""
    
    def __init__(self, use_reasoning: bool = True, rubric: Optional[str] = None):
        super().__init__()
        self.use_reasoning = use_reasoning
        
        if use_reasoning:
            signature = create_pairwise_comparison_with_reasoning_signature(rubric)
            self.compare_titles = dspy.ChainOfThought(signature)
        else:
            signature = create_pairwise_comparison_signature(rubric)
            self.compare_titles = dspy.Predict(signature)
    
    def forward(self, title_a: str, title_b: str):
        """Compare two titles and predict which is more likely to be favorite."""
        result = self.compare_titles(title_a=title_a, title_b=title_b)
        return result