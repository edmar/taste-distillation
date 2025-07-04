# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is a machine learning research project focused on "taste distillation" - training models to predict personal preferences (favorites) from article titles using Hacker News data. The project uses DSPy (Declarative Self-improving Python) framework for building language model programs.

## Technology Stack

- **Language**: Python (3.10-3.12)
- **ML Framework**: DSPy (2.6.24) for building language model programs
- **LLM Provider**: OpenAI (GPT-4o-mini as default, configurable)
- **Package Manager**: uv
- **Dependencies**: scikit-learn, pandas, jupyter, beautifulsoup4, pyyaml
- **Configuration**: YAML-based centralized config system

## Common Development Commands

### Environment Setup
```bash
# Install dependencies
uv sync
```

### Configuration Management
```bash
# Configuration is split into separate files:
# - src/dspy_favorite/config/evaluate.yaml: Evaluation settings, default datasets, LLM models
# - src/dspy_favorite/config/train.yaml: Training settings, optimizers, LLM models

# Set environment to control which config to use:
export TASTE_ENV=development  # or production, testing (default: default)

# Override LLM model for any script:
--llm openai/gpt-4o          # Use GPT-4o instead of default
--llm openai/gpt-4o-mini     # Use GPT-4o-mini (default)

# Default datasets from src/dspy_favorite/config/evaluate.yaml:
# - favorite: data/reader_favorite/test/dspy_examples.json
# - shortlist: data/reader_shortlist/test/dspy_examples.json
```

### Complete Development Workflow
```bash
# 1. Generate taste rubric from data
uv run python scripts/generate_rubric.py

# 2. Prepare data for training using unified preparation script
uv run python lib/reader/prepare.py --data data/raw/reader_export.csv  # Prepares all datasets
uv run python lib/reader/prepare.py --data data/raw/reader_export.csv --datasets favorite  # Only favorites
uv run python lib/reader/prepare.py --data data/raw/reader_export.csv --datasets shortlist  # Only shortlist

# 3. Train models (with various options)
# Binary classification - can use either dataset
uv run python src/dspy_favorite/scripts/train.py --dataset data/reader_favorite  # For favorites
uv run python src/dspy_favorite/scripts/train.py --dataset data/reader_shortlist  # For shortlist
uv run python src/dspy_favorite/scripts/train.py --model openai/gpt-4o 100 mipro  # Custom LLM, 100 examples, MIPRO optimizer

# Pairwise comparison - can use either dataset
uv run python src/dspy_pairwise/scripts/train.py --dataset data/reader_favorite_pairwise  # For favorites
uv run python src/dspy_pairwise/scripts/train.py --dataset data/reader_shortlist_pairwise  # For shortlist

# 4. Evaluate trained models
uv run python src/dspy_favorite/scripts/evaluate.py  # Uses default favorite dataset from config
uv run python src/dspy_favorite/scripts/evaluate.py --dataset data/reader_shortlist/test/dspy_examples.json
uv run python src/dspy_favorite/scripts/evaluate.py --llm openai/gpt-4o --dataset data/reader_shortlist/test/dspy_examples.json  # Custom LLM
uv run python src/dspy_pairwise/scripts/evaluate.py --dataset data/reader_favorite_pairwise
uv run python src/dspy_pairwise/scripts/evaluate.py --dataset data/reader_shortlist_pairwise

# 5. Run predictions on new data
uv run python src/dspy_favorite/scripts/predict.py --title "Your Article Title"
uv run python src/dspy_pairwise/scripts/predict.py --title-a "Title A" --title-b "Title B"
```

### Training Script Parameters
- **Split methods**: `random`, `temporal`
- **Balancing**: `balanced`, `balance`, `true`
- **Optimizers**: `bootstrap`, `bootstrap_random`, `mipro`, `auto`
- **Max examples**: Any integer (e.g., `500`, `1000`)

### Evaluation Script Parameters
- **Test size**: Any integer (e.g., `200`, `100`)
- **Baseline comparison**: `true`, `false`
- **Balanced evaluation**: `balanced`, `unbalanced`

## Architecture Overview

### Core Model Types
1. **Binary Classifier** (`src/dspy_favorite/`): Binary classification for both favorites and shortlist tasks (is_favorite/is_shortlist: True/False)
2. **Pairwise Classifier** (`src/dspy_pairwise/`): Preference ranking between pairs of articles for both favorite and shortlist data

Both models are task-agnostic and work with any dataset in the correct format. The task is determined by which dataset you point the model to:
- Use `reader_favorite`/`reader_favorite_pairwise` for predicting favorites
- Use `reader_shortlist`/`reader_shortlist_pairwise` for predicting shortlist items

### DSPy Implementation Pattern
Each model follows a standardized structure:
- `model.py` - DSPy signatures and modules
- `scripts/prepare.py` - Data preparation
- `scripts/train.py` - Model training with optimizers
- `scripts/evaluate.py` - Model evaluation
- `scripts/predict.py` - Inference on new data

### Key Directories
- `/data/` - Data storage with raw (`reader_export.csv`) and processed splits
- `/src/` - DSPy-based models (dspy_favorite, dspy_pairwise)
- `/lib/` - Shared utilities including unified data preparation (lib/reader/prepare.py)
- `/saved/` - Model checkpoints, rubrics, and training logs
- `/scripts/` - High-level utility scripts

### Data Flow
1. Raw data (`data/raw/reader_export.csv`) contains article titles, URLs, and user preferences
2. Unified data preparation (`lib/reader/prepare.py`) creates all datasets: reader_favorite, reader_shortlist, reader_*_pairwise
3. Rubric generation creates taste guidelines from user data
4. Models are trained using DSPy optimizers (BootstrapFinetune, MIPROv2) with dataset-specific parameters
5. Evaluation compares model performance against baselines for each dataset
6. Predictions generate preference scores for new articles using the appropriate trained model

## Key Features

- **Rubric Integration**: Models can use generated "taste rubrics" to guide predictions
- **Multiple Optimizers**: Support for BootstrapFinetune, MIPROv2, and other DSPy optimizers
- **Flexible Data Splits**: Random or temporal splitting strategies
- **Balanced Sampling**: Option to balance positive/negative examples during training
- **Comprehensive Logging**: All experiments logged in `saved/logs/`

## Testing

Currently minimal testing infrastructure. Test files should be added to `/tests/` directory. No automated test runner configured yet.

## Notes

- All scripts should be run from the project root directory using uv run
- Models and logs are automatically saved in `saved/` directory with timestamps
- OpenAI API key required for DSPy models (set in environment)
- Data preparation must be run before training new models
- The project follows DSPy best practices with clean separation of model definition, training, and evaluation