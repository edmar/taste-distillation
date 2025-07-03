# Taste Distillation Experiments - Simplified Plan

## Current State Analysis
We have implemented classifiers using DSPy that do **prompt optimization**, not true in-context learning or fine-tuning:
- **What we have**: DSPy optimizers (BootstrapFewShot, MIPRO) that find optimal prompts and demonstrations
- **What this is**: Sophisticated prompt engineering, not model training
- **Performance**: 28% baseline → 48% accuracy with optimized prompts

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
- Implementation: `src/in_context_taste_models/taste_rubric.py`

### Experiment 1.2: Baseline (Zero-Shot)
- Create simple classifiers for BOTH tasks with NO training/optimization:
  - Binary: "Is this article title interesting?"
  - Pairwise: "Which of these two titles is more interesting?"
- Test with AND without taste rubric to measure rubric impact
- Implementation: 
  - `src/in_context_taste_models/baseline_favorite.py`
  - `src/in_context_taste_models/baseline_pairwise.py`
  - Both files support `--use_rubric` flag

### Experiment 1.3: DSPy Optimized
- Use DSPy optimization (BootstrapFewShot/MIPRO) for both tasks
- Test with AND without taste rubric integration
- Compare 4 variants:
  - DSPy optimized without rubric (current implementation)
  - DSPy optimized with rubric
  - Baseline without rubric
  - Baseline with rubric
- Implementation:
  - `src/in_context_taste_models/dspy_favorite.py`
  - `src/in_context_taste_models/dspy_pairwise.py`
  - Both files support `--use_rubric` flag

## Part 2: Learned Taste Models (3-4 hours)

### Experiment 2.1: Data Preparation for Fine-Tuning
- Convert dataset to OpenAI fine-tuning format for BOTH:
  - Binary classification (is favorite: yes/no)
  - Pairwise comparison (which title is preferred: A or B)
- Create train/validation splits for both tasks
- Implementation: 
  - `src/learned_taste_models/prepare_favorite_data.py`
  - `src/learned_taste_models/prepare_pairwise_data.py`

### Experiment 2.2: Fine-Tune GPT-4o-mini
- Use OpenAI fine-tuning API for both models:
  - Model 1: Binary favorite classifier
  - Model 2: Pairwise preference classifier
- Document costs and training time for each
- Implementation: 
  - `src/learned_taste_models/train_favorite.py`
  - `src/learned_taste_models/train_pairwise.py`

### Experiment 2.3: Evaluation & Comparison
- Compare all approaches for BOTH tasks:
  - **Baseline**: Without rubric, With rubric
  - **DSPy Optimized**: Without rubric, With rubric
  - **Fine-tuned**: Binary classifier, Pairwise classifier
- Create comprehensive comparison matrix showing:
  - Accuracy, precision, recall
  - Cost per 1000 classifications
  - Impact of taste rubric on each method
- Implementation: `src/evaluation/compare_all.py`

## Key Metrics to Track
- Accuracy, Precision, Recall
- Cost per 1000 classifications
- Latency
- Explanation quality (qualitative)
