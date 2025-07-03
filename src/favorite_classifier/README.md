# Favorite Classifier

A DSPy-based system for predicting whether articles will be marked as favorites based on their titles.

## Files

- `train.py` - Training script with multiple optimizer options
- `evaluate.py` - Evaluation script with balanced testing
- `favorite_predictor.py` - Core DSPy model implementation
- `data_loader.py` - Data loading and preprocessing utilities
- `model_inspector.py` - Model inspection and analysis tool

## Quick Start

### Training
```bash
# Train with balanced data and auto optimizer selection
poetry run python train.py 50 balanced

# Train with specific optimizer
poetry run python train.py 100 balanced bootstrap_random
```

### Evaluation
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

## Key Features

- **Balanced Training**: Automatically balances favorites vs non-favorites
- **Multiple Optimizers**: Auto-selects best optimizer based on data size
- **Comprehensive Evaluation**: Uses DSPy's evaluation framework
- **Model Inspection**: Detailed analysis of prompts and demonstrations
- **Robust Error Handling**: Graceful fallbacks for optimization issues