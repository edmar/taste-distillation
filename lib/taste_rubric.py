"""
Taste Rubric Generator

This module generates a detailed taste rubric by analyzing a user's favorite articles.
The rubric captures common themes, patterns, and decision criteria that can be used
for both binary classification and pairwise comparison tasks.
"""

import json
import os
from typing import List, Dict, Any, Optional
import pandas as pd
from openai import OpenAI
from pathlib import Path

class TasteRubricGenerator:
    """Generates a taste rubric from favorite articles."""
    
    def __init__(self, model_name: str = "gpt-4o-mini"):
        """Initialize the taste rubric generator."""
        self.client = OpenAI()
        self.model_name = model_name
    
    def generate_rubric(self, favorite_titles: List[str]) -> str:
        """
        Generate a taste rubric from favorite article titles.
        
        Args:
            favorite_titles: List of favorite article titles
            
        Returns:
            Generated taste rubric as a string
        """
        return self._generate_rubric_from_titles(favorite_titles)
    
    def _generate_rubric_from_titles(self, favorite_titles: List[str]) -> str:
        """Generate the actual rubric using OpenAI API."""
        
        # Create a sample of titles to analyze (limit to avoid token limits)
        sample_titles = favorite_titles[:50] if len(favorite_titles) > 50 else favorite_titles
        
        prompt = f"""
You are an expert at analyzing reading preferences and taste patterns. I will provide you with a list of article titles that a user has marked as "favorites" or found particularly interesting.

Your task is to analyze these titles and create a comprehensive "taste rubric" that captures:

1. **Common Themes**: What topics, domains, or subjects appear frequently?
2. **Content Patterns**: What types of content does this person seem to prefer (technical, philosophical, practical, etc.)?
3. **Style Preferences**: What writing styles or approaches seem to resonate?
4. **Decision Criteria**: What makes an article likely to be interesting vs not interesting to this person?

Here are the favorite article titles to analyze:

{"\n".join(f"- {title}" for title in sample_titles)}

Please create a detailed taste rubric that could be used by an AI system to

The rubric should be specific enough to be actionable but general enough to apply to new content.

Format your response as a clear, structured rubric that begins with "TASTE RUBRIC:" and includes specific criteria and examples.
"""

        try:
            response = self.client.chat.completions.create(
                model=self.model_name,
                messages=[
                    {"role": "system", "content": "You are an expert at analyzing reading preferences and creating taste rubrics."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.7,
                max_tokens=2000
            )
            
            rubric = response.choices[0].message.content
            return rubric
            
        except Exception as e:
            print(f"Error generating rubric: {e}")
            return self._get_fallback_rubric()
    
    def _get_fallback_rubric(self) -> str:
        """Fallback rubric if API call fails."""
        return """
TASTE RUBRIC:

This user appears to prefer articles that:
1. Contain technical or analytical content
2. Discuss emerging technologies or trends
3. Offer practical insights or actionable advice
4. Explore philosophical or conceptual topics
5. Come from reputable sources or thought leaders

For binary classification:
- Articles matching 3+ criteria are likely interesting
- Articles with none of these elements are likely not interesting

For pairwise comparison:
- Prioritize articles with more technical depth
- Prefer practical over purely theoretical content
- Choose articles from more authoritative sources
"""
    
    def load_favorites_from_csv(self, csv_path: str, title_column: str = "title") -> List[str]:
        """Load favorite titles from a CSV file."""
        try:
            df = pd.read_csv(csv_path)
            # Assuming favorites are marked with a boolean column or specific value
            if "favorite" in df.columns:
                favorites_df = df[df["favorite"] == True]
            elif "is_favorite" in df.columns:
                favorites_df = df[df["is_favorite"] == True]
            else:
                # If no favorite column, assume all titles are favorites
                favorites_df = df
            
            return favorites_df[title_column].tolist()
            
        except Exception as e:
            print(f"Error loading favorites from CSV: {e}")
            return []
    
    def analyze_rubric_quality(self, rubric: str, favorite_titles: List[str]) -> Dict[str, Any]:
        """Analyze the quality of the generated rubric."""
        
        analysis_prompt = f"""
Analyze this taste rubric and provide a quality assessment:

RUBRIC:
{rubric}

SAMPLE TITLES IT WAS BASED ON:
{"\n".join(f"- {title}" for title in favorite_titles[:10])}

Please evaluate:
1. Specificity: How specific vs generic are the criteria?
2. Actionability: How easy would it be to apply these criteria?
3. Coverage: How well does it capture the variety in the sample titles?
4. Consistency: Are the criteria internally consistent?

Provide a JSON response with scores (1-10) for each dimension and brief explanations.
"""
        
        try:
            response = self.client.chat.completions.create(
                model=self.model_name,
                messages=[
                    {"role": "system", "content": "You are evaluating the quality of a taste rubric. Respond with valid JSON."},
                    {"role": "user", "content": analysis_prompt}
                ],
                temperature=0.3,
                max_tokens=500
            )
            
            analysis = json.loads(response.choices[0].message.content)
            return analysis
            
        except Exception as e:
            print(f"Error analyzing rubric quality: {e}")
            return {"error": str(e)}


def main():
    """Demo of the taste rubric generator."""
    
    # Initialize generator
    generator = TasteRubricGenerator()
    
    # Example favorite titles (you'd normally load these from your data)
    sample_favorites = [
        "The Future of AI and Machine Learning",
        "Understanding Quantum Computing",
        "Building Better Software Architecture",
        "The Philosophy of Technology",
        "Data Science Best Practices",
        "Modern Web Development Trends",
        "The Ethics of Artificial Intelligence",
        "Functional Programming Concepts",
        "Database Design Patterns",
        "The Psychology of User Experience"
    ]
    
    # Generate rubric
    print("Generating taste rubric...")
    rubric = generator.generate_rubric(sample_favorites)
    
    print("\n" + "="*50)
    print("GENERATED TASTE RUBRIC:")
    print("="*50)
    print(rubric)
    
    # Analyze rubric quality
    print("\n" + "="*50)
    print("RUBRIC QUALITY ANALYSIS:")
    print("="*50)
    analysis = generator.analyze_rubric_quality(rubric, sample_favorites)
    print(json.dumps(analysis, indent=2))


if __name__ == "__main__":
    main()