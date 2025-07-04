# Taste Distillation Experiments - Updated Plan with DSPy Architecture

## Today's Goal
Demonstrate the two key approaches from the article:
1. **True In-Context Learning**: Pure zero/few-shot classification without any optimization
2. **Fine-Tuned Model**: Actual model training using OpenAI fine-tuning

## Part 1: True In-Context Taste Models (2-3 hours)

### Experiment 1.1: Taste Rubric Generation
- Feed the LLM samples of favorite articles
- Have it generate a detailed "taste rubric" that captures:
  - Common themes and patterns in favorites
  - Specific criteria for what makes an article appealing
  - Decision framework for pairwise comparisons
- Use this rubric as a prompt for classification
- Compare rubric-based approach vs other in-context methods
- Implementation: `scripts/generate_rubric.py`
- Output: `saved/rubrics/taste_rubric.json`

### Experiment 1.2: DSPy Models
- Implement DSPy modules for BOTH tasks:
  - Binary: "Is this article title interesting?"
  - Pairwise: "Which of these two titles is more interesting?"
- Test with AND without taste rubric to measure rubric impact
- Evaluation experiments:
  1. **Baseline evaluation**: Run evaluate.py on unoptimized model (with/without rubric)
  2. **Train optimization**: Run train.py to optimize with BootstrapFewShot/MIPRO
  3. **Optimized evaluation**: Run evaluate.py on optimized model (with/without rubric)
- Implementation following DSPy architecture:
  - `src/in_context_taste_models/dspy_favorite/`
    - `model.py` - DSPy ChainOfThought module with rubric support
    - `metrics.py` - Custom metrics for optimization
    - `scripts/prepare.py` - Prepare DSPy Example objects
    - `scripts/train.py` - Optimize with BootstrapFewShot/MIPRO
    - `scripts/evaluate.py` - Evaluate model performance
    - `scripts/predict.py` - Run inference
  - `src/in_context_taste_models/dspy_pairwise/`
    - Similar structure for pairwise task
- Saved models: `saved/models/dspy_favorite/`, `saved/models/dspy_pairwise/`

## Part 2: Learned Taste Models (3-4 hours)

### Experiment 2.1: Data Preparation for Fine-Tuning
- Convert dataset to OpenAI fine-tuning format for BOTH:
  - Binary classification (is favorite: yes/no)
  - Pairwise comparison (which title is preferred: A or B)
- Create train/validation splits for both tasks
- Implementation: 
  - `src/learned_taste_models/prepare_favorite_data.py`
  - `src/learned_taste_models/prepare_pairwise_data.py`
- Output: `data/processed/openai_format/`

### Experiment 2.2: Fine-Tune GPT-4o-mini
- Use OpenAI fine-tuning API for both models:
  - Model 1: Binary favorite classifier
  - Model 2: Pairwise preference classifier
- Document costs and training time for each
- Implementation: 
  - `src/learned_taste_models/train_favorite.py`
  - `src/learned_taste_models/train_pairwise.py`
- Saved models: Track fine-tuning job IDs in `saved/models/fine_tuned/`

### Experiment 2.3: Evaluation & Comparison
- Compare all approaches for BOTH tasks:
  - **Unoptimized DSPy**: Without rubric, With rubric
  - **Optimized DSPy**: Without rubric, With rubric
  - **Fine-tuned**: Binary classifier, Pairwise classifier
- Create comprehensive comparison matrix showing:
  - Accuracy, precision, recall
  - Cost per 1000 classifications
  - Impact of taste rubric on each method
- Implementation: `src/evaluation/compare_all.py`
- Results: `results/comparison_matrix.csv`

## Key Metrics to Track
- Accuracy, Precision, Recall
- Cost per 1000 classifications
- Latency
- Explanation quality (qualitative)

## Project Structure (Following DSPy Architecture)

```
taste/
├── config/
│   └── config.yaml                    # Central configuration
├── data/
│   ├── raw/                          # Original HN datasets
│   └── processed/                    # Processed data
│       ├── dspy_examples/            # DSPy Example objects
│       └── openai_format/            # OpenAI fine-tuning format
├── src/
│   ├── in_context_taste_models/      # In-context learning approaches
│   │   ├── dspy_favorite/            # DSPy favorite classifier (baseline & optimized)
│   │   └── dspy_pairwise/            # DSPy pairwise classifier (baseline & optimized)
│   ├── learned_taste_models/         # Fine-tuning approaches
│   └── evaluation/                   # Comparison scripts
├── lib/                              # Shared utilities
├── saved/
│   ├── models/                       # Trained models
│   ├── rubrics/                      # Generated taste rubrics
│   └── logs/                         # Training logs
├── scripts/                          # Top-level scripts
│   └── generate_rubric.py            # Generate taste rubric
└── results/                          # Evaluation outputs
```

## Implementation Order

1. **Setup Configuration** (`config/config.yaml`)
2. **Generate Taste Rubric** (`scripts/generate_rubric.py`)
3. **Implement DSPy Models** (following DSPy architecture)
   - Create model.py, metrics.py, scripts/ for each task
   - Run baseline experiment: evaluate.py on unoptimized models
4. **Optimize DSPy Models**
   - Run prepare.py → train.py → evaluate.py pipeline
   - Compare unoptimized vs optimized performance
5. **Prepare Fine-Tuning Data** (OpenAI format conversion)
6. **Fine-Tune Models** (OpenAI API)
7. **Run Comprehensive Evaluation** (`src/evaluation/compare_all.py`)
