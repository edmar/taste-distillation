# Taste Distillation Research

A machine learning research project for training models to predict personal preferences from article titles using Hacker News data and the DSPy framework.

## Overview

This project focuses on "taste distillation" - the process of training AI models to understand and predict personal preferences based on reading behavior. Using article titles from Hacker News, we build models that can:

- **Binary Classification**: Predict whether a specific article will be marked as "favorite" 
- **Pairwise Comparison**: Rank articles by preference when given multiple options
- **Taste Rubric Generation**: Create personalized preference guidelines from historical data

## Architecture

### Core Components

The project implements two main model architectures using DSPy:

1. **Favorite Classifier** (`src/dspy_favorite/`) - Binary classification for individual articles
2. **Pairwise Classifier** (`src/dspy_pairwise/`) - Comparative ranking between article pairs

### DSPy Implementation

Each model follows a standardized DSPy structure:
- `model.py` - DSPy signatures and modules with Chain-of-Thought reasoning
- `scripts/prepare.py` - Data preprocessing and train/validation/test splits
- `scripts/train.py` - Model training with DSPy optimizers (BootstrapFinetune, MIPROv2)
- `scripts/evaluate.py` - Performance evaluation and baseline comparison
- `scripts/predict.py` - Inference on new data

### Key Features

- **Rubric Integration**: Models incorporate generated taste rubrics to guide predictions
- **Multiple Optimizers**: Support for BootstrapFinetune, MIPROv2, and other DSPy optimizers
- **Flexible Data Handling**: Random or temporal data splits, balanced sampling options
- **Comprehensive Logging**: All experiments tracked in `saved/logs/`

## Quick Start

### Prerequisites

- Python 3.10-3.12
- OpenAI API key (for LLM inference)
- uv package manager

### Installation

```bash
# Clone the repository
git clone <repository-url>
cd taste

# Install dependencies
uv sync
```

### Basic Usage

```bash
# 1. Generate taste rubric from existing data
uv run python scripts/generate_rubric.py

# 2. Prepare training data
uv run python src/dspy_favorite/scripts/prepare.py

# 3. Train model
uv run python src/dspy_favorite/scripts/train.py

# 4. Evaluate model
uv run python src/dspy_favorite/scripts/evaluate.py

# 5. Make predictions
uv run python src/dspy_favorite/scripts/predict.py
```

### Training Options

The training script supports various configurations:

```bash
# Basic training
uv run python src/dspy_favorite/scripts/train.py

# Using bootstrap optimizer
uv run python src/dspy_favorite/scripts/train.py 1000 bootstrap

# Using MIPROv2 optimizer
uv run python src/dspy_favorite/scripts/train.py 1000 mipro
```

**Parameters:**
- **Size**: Number of training examples (e.g., 500, 1000)
- **Balancing**: `balanced` for equal positive/negative samples
- **Optimizers**: `bootstrap`, `bootstrap_random`, `mipro`, `auto`
- **Split methods**: `random`, `temporal`

### Evaluation Options

```bash
# Evaluate with 200 test examples, compare to baseline
uv run python src/dspy_favorite/scripts/evaluate.py 200

# Quick evaluation with defaults
uv run python src/dspy_favorite/scripts/evaluate.py
```

## Data Structure

```
data/
├── raw/
│   ├── reader_export.csv           # Raw article data with user preferences
│   └── hn_export_3dm4r_small_20250703_182120.csv  # Hacker News data
└── processed/
    ├── reader_favorite/            # Binary classification splits (main dataset)
    │   ├── train/
    │   ├── val/
    │   └── test/
    ├── reader_pairwise/            # Pairwise comparison splits (main dataset)
    │   ├── train/
    │   ├── val/
    │   └── test/
    ├── hn_favorite/                # HN binary classification splits
    │   ├── train/
    │   ├── val/
    │   └── test/
    └── hn_pairwise/                # HN pairwise comparison splits
        ├── train/
        ├── val/
        └── test/
```

## Model Performance

### Latest Evaluation Results

Based on comprehensive evaluation detailed in `EVALUATION_REPORT.md`:

#### Baseline Performance (Untrained Models)
- **Favorite Classifier**: 62-64% accuracy (above random due to taste rubric integration)
- **Pairwise Classifier**: 62% accuracy (above random)
- Both models benefit significantly from comprehensive taste rubrics

#### Trained Model Performance
- **Favorite Classifier (dspy_favorite_001)**: 62% accuracy (no improvement over baseline)
- **Pairwise Classifier (dspy_pairwise_002)**: 72% accuracy (+10% improvement)
- **Pairwise Classifier (dspy_pairwise_003)**: Latest model version

### Evaluation Metrics
- **Accuracy**: Binary classification correctness
- **Precision/Recall**: Positive class performance
- **F1-Score**: Harmonic mean of precision and recall
- **Baseline Comparison**: Performance vs. random/majority baselines

Results are saved in `results/` with detailed metrics and evaluation reports in `EVALUATION_REPORT.md`.

## Technology Stack

- **DSPy** (2.6.24) - Language model programming framework
- **OpenAI** - LLM provider (GPT-4o-mini default)
- **scikit-learn** - ML utilities and evaluation metrics
- **pandas** - Data manipulation
- **beautifulsoup4** - Web scraping utilities
- **uv** - Python package management

## Project Structure

```
├── src/
│   ├── dspy_favorite/          # Binary classification models
│   │   ├── dspy_favorite.py    # Model definition
│   │   └── scripts/            # Training, evaluation, prediction scripts
│   └── dspy_pairwise/          # Pairwise comparison models
│       ├── dspy_pairwise.py    # Model definition
│       └── scripts/            # Training, evaluation, prediction scripts
├── lib/
│   ├── hn_scraper/             # Hacker News data collection
│   │   ├── hn_scraper.py       # Web scraping utilities
│   │   └── prepare.py          # Data preparation
│   └── taste_rubric.py         # Taste rubric generation
├── scripts/
│   └── generate_rubric.py      # Top-level rubric generation
├── saved/
│   ├── models/                 # Trained model checkpoints (4 models)
│   │   ├── dspy_favorite_001.json
│   │   ├── dspy_pairwise_001.json
│   │   ├── dspy_pairwise_002.json
│   │   └── dspy_pairwise_003.json
│   ├── rubrics/                # Generated taste rubrics
│   │   └── personal_taste_rubric.txt
│   └── logs/                   # Training and evaluation logs
├── data/                       # Raw and processed datasets
├── results/                    # Evaluation results (JSON format)
├── notebooks/                  # Jupyter notebooks for analysis
│   └── report.ipynb           # Analysis notebook
├── EVALUATION_REPORT.md        # Comprehensive evaluation report
└── tests/                      # Test framework (minimal)
```

## Key Research Findings

### Model Insights
1. **Taste Rubric Integration**: Comprehensive taste rubrics provide significant baseline performance improvement (62-64% vs. 50% random)
2. **Pairwise vs. Binary**: Pairwise comparison models show better learning capability than binary classification
3. **Training Effects**: Binary models hit performance ceiling, while pairwise models demonstrate 10% improvement
4. **Task Difficulty**: Moderate accuracy (62-72%) reflects inherent challenges of taste prediction from limited context

### Taste Rubric Components
- **7 Common Themes**: Self-improvement, knowledge management, writing, technology/AI, psychology, networking, meta-reflection
- **4 Content Patterns**: Practical/actionable, analytical/conceptual, reflective/philosophical, personal/anecdotal
- **5 Style Preferences**: Clear/direct, structured/list-based, engaging/conversational, meta/self-referential, concise/deep
- **7 Decision Criteria**: Practical takeaways, thinking frameworks, contrarian ideas, personal experience, etc.

## Research Applications

This framework can be extended for:
- **Content Recommendation**: Personalized article suggestions
- **Preference Learning**: Understanding decision-making patterns
- **Taste Transfer**: Applying learned preferences to new domains
- **Interactive Systems**: Building taste-aware user interfaces

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Run tests: `uv run python -m pytest`
5. Submit a pull request

## License

MIT License - see LICENSE file for details
