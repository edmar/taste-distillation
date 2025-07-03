# Pairwise Comparison Classifier

A DSPy-based system for comparing article titles and determining which one is more likely to be marked as favorite. Unlike binary classification, this system performs pairwise comparisons to rank titles by preference.

## Core Concept

Given two titles A and B, the system predicts:
- Which title (A or B) is more likely to be favorite
- Confidence level: "high", "medium", or "low"
- Reasoning for the decision

## Files

### Pairwise Comparison System (NEW)
- `pairwise_predictor.py` - Core pairwise DSPy model implementation
- `pairwise_data_loader.py` - Generates pairwise training examples from taste data
- `pairwise_train.py` - Training script for pairwise model
- `pairwise_evaluate.py` - Comprehensive evaluation with baselines
- `pairwise_classifier.py` - High-level API for pairwise comparisons

### Original Binary Classification System
- `train.py` - Training script with multiple optimizer options
- `evaluate.py` - Evaluation script with balanced testing
- `favorite_predictor.py` - Core DSPy model implementation
- `data_loader.py` - Data loading and preprocessing utilities
- `model_inspector.py` - Model inspection and analysis tool

## Quick Start

### Pairwise Comparison System

#### Training
```bash
# Train pairwise model with mixed pair strategy
poetry run python pairwise_train.py 1000 mixed balanced

# Train with specific parameters
poetry run python pairwise_train.py 500 fav_vs_nonfav bootstrap_random
```

#### Evaluation
```bash
# Comprehensive evaluation with baselines
poetry run python pairwise_evaluate.py 500

# Evaluate specific model
poetry run python pairwise_evaluate.py models/pairwise_model.json 200
```

#### Using the API
```python
from pairwise_classifier import PairwiseAPI

# Initialize with trained model
api = PairwiseAPI("models/pairwise_model.json")

# Compare two titles
result = api.compare_titles("Title A", "Title B")
print(f"Winner: {result['winner']} with {result['confidence']} confidence")

# Rank multiple titles
titles = ["Title 1", "Title 2", "Title 3"]
ranking = api.rank_titles(titles)
for item in ranking:
    print(f"{item['rank']}. {item['title']} (score: {item['score']:.3f})")
```

### Original Binary Classification System

#### Training
```bash
# Train with balanced data and auto optimizer selection
poetry run python train.py 50 balanced

# Train with specific optimizer
poetry run python train.py 100 balanced bootstrap_random
```

#### Evaluation
```bash
# Evaluate with balanced test set
poetry run python evaluate.py 50

# Include baseline comparison
poetry run python evaluate.py 50 true
```

### Model Inspection
```bash
# Show complete model analysis
poetry run python model_inspector.py all

# Just show prompt structure
poetry run python model_inspector.py prompt models/trained_model.json

# Test a specific prediction
poetry run python model_inspector.py test models/trained_model.json "Your Article Title Here"

# Compare two models
poetry run python model_inspector.py compare models/model1.json models/model2.json
```

## Model Optimization

The system supports multiple DSPy optimizers:

- **BootstrapFewShot**: Best for small datasets (< 20 examples)
- **BootstrapFewShotWithRandomSearch**: Good for medium datasets (20-100 examples)  
- **MIPROv2**: For large datasets (> 100 examples) - experimental

## Pairwise Comparison Features

### Training Data Generation
- **Favorite vs Non-favorite pairs**: Clear signal (high confidence)
- **Favorite vs Favorite pairs**: Harder decisions (medium confidence)  
- **Non-favorite vs Non-favorite pairs**: Random preference (low confidence)
- **Data augmentation**: Includes both (A,B) and (B,A) versions
- **Balanced sampling**: Equal distribution across pair types

### Pair Strategies
- **mixed**: Balanced mix of all pair types (recommended)
- **fav_vs_nonfav**: Only clear favorite vs non-favorite pairs
- **all_combinations**: All possible combinations with heuristic labeling

### Ranking Methods
- **Tournament**: Efficient pairwise elimination for large lists
- **Round-robin**: Comprehensive comparison of every pair

### Evaluation
- **Baseline comparisons**: Random, always-A, always-B, confidence-based
- **Confidence calibration**: High confidence predictions should be more accurate
- **Performance by pair type**: Analyze model strength on different comparison types

## Original Binary Classification Features

- **Balanced Training**: Automatically balances favorites vs non-favorites
- **Multiple Optimizers**: Auto-selects best optimizer based on data size
- **Comprehensive Evaluation**: Uses DSPy's evaluation framework
- **Model Inspection**: Detailed analysis of prompts and demonstrations
- **Robust Error Handling**: Graceful fallbacks for optimization issues