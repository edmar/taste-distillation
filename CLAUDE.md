# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is a machine learning research project focused on "taste distillation" - training models to predict personal preferences (favorites) from article titles using Hacker News data. The project uses DSPy (Declarative Self-improving Python) framework for building language model programs.

## Technology Stack

- **Language**: Python (3.10-3.12)
- **ML Framework**: DSPy (2.6.24) for building language model programs
- **LLM Provider**: OpenAI (GPT-4o-mini as default)
- **Package Manager**: uv
- **Dependencies**: scikit-learn, pandas, jupyter, beautifulsoup4

## Common Development Commands

### Environment Setup
```bash
# Install dependencies
uv sync
```

### Complete Development Workflow
```bash
# 1. Generate taste rubric from data
uv run python scripts/generate_rubric.py

# 2. Prepare data for training
uv run python src/dspy_favorite/scripts/prepare.py

# 3. Train model (with various options)
uv run python src/dspy_favorite/scripts/train.py
uv run python src/dspy_favorite/scripts/train.py balanced
uv run python src/dspy_favorite/scripts/train.py 500 bootstrap
uv run python src/dspy_favorite/scripts/train.py 1000 mipro

# 4. Evaluate trained model
uv run python src/dspy_favorite/scripts/evaluate.py
uv run python src/dspy_favorite/scripts/evaluate.py 200 true balanced

# 5. Run predictions on new data
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
1. **Favorite Classifier** (`src/dspy_favorite/`): Binary classification (is_favorite: True/False)
2. **Pairwise Classifier** (`src/dspy_pairwise/`): Preference ranking between pairs of articles

### DSPy Implementation Pattern
Each model follows a standardized structure:
- `model.py` - DSPy signatures and modules
- `scripts/prepare.py` - Data preparation
- `scripts/train.py` - Model training with optimizers
- `scripts/evaluate.py` - Model evaluation
- `scripts/predict.py` - Inference on new data

### Key Directories
- `/data/` - Data storage with raw (`reader_export.csv`) and processed splits
- `/src/` - DSPy-based models
- `/lib/` - Shared utilities including Hacker News scraping
- `/saved/` - Model checkpoints, rubrics, and training logs
- `/scripts/` - High-level utility scripts

### Data Flow
1. Raw data (`data/raw/reader_export.csv`) contains article titles, URLs, and user preferences
2. Data preparation creates train/validation/test splits
3. Rubric generation creates taste guidelines from user data
4. Models are trained using DSPy optimizers (BootstrapFinetune, MIPROv2)
5. Evaluation compares model performance against baselines
6. Predictions generate preference scores for new articles

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