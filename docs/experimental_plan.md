# Experimental Plan for Taste Distillation Article

## Overview
This document outlines the experimental approach to demonstrate taste distillation across three different data contexts: personal reading data (Readwise Reader), public data (Hacker News), and combined datasets. Additionally, we test across different LLMs to show robustness of the approach.
3
## Experimental Structure

### LLM Models to Test
- **GPT-4.1-mini**: `openai/gpt-4.1-mini` (default)
- **GPT-4.1**: `openai/gpt-4.1`
- **O3**: `openai/o3`

*Note: Each experiment should be run with all three LLMs to test robustness.*

## Favorite Binary Classification Experiments

### Dataset A: Readwise Reader (Shortlist)

**Baseline Evaluation (Rubric Only)**
```bash
# Test with each LLM
uv run python src/dspy_favorite/scripts/evaluate.py --dataset data/reader_shortlist/test/dspy_examples.json --llm openai/gpt-4.1-mini
uv run python src/dspy_favorite/scripts/evaluate.py --dataset data/reader_shortlist/test/dspy_examples.json --llm openai/gpt-4.1
uv run python src/dspy_favorite/scripts/evaluate.py --dataset data/reader_shortlist/test/dspy_examples.json --llm openai/o3
```

**Train Model**
```bash
# Train with each LLM
uv run python src/dspy_favorite/scripts/train.py --dataset data/reader_shortlist mipro --llm openai/gpt-4.1-mini
uv run python src/dspy_favorite/scripts/train.py --dataset data/reader_shortlist mipro --llm openai/gpt-4.1
uv run python src/dspy_favorite/scripts/train.py --dataset data/reader_shortlist mipro --llm openai/o3
```

**Evaluate Trained Model**
```bash
# Evaluate each trained model
uv run python src/dspy_favorite/scripts/evaluate.py --model saved/models/dspy_favorite_XXX --dataset data/reader_shortlist/test/dspy_examples.json --llm openai/gpt-4.1-mini
uv run python src/dspy_favorite/scripts/evaluate.py --model saved/models/dspy_favorite_XXX --dataset data/reader_shortlist/test/dspy_examples.json --llm openai/gpt-4.1
uv run python src/dspy_favorite/scripts/evaluate.py --model saved/models/dspy_favorite_XXX --dataset data/reader_shortlist/test/dspy_examples.json --llm openai/o3
```

### Dataset B: Hacker News

**Baseline Evaluation (Rubric Only)**
```bash
# Test with each LLM
uv run python src/dspy_favorite/scripts/evaluate.py --dataset data/hn_favorite/test/dspy_examples.json --llm openai/gpt-4.1-mini
uv run python src/dspy_favorite/scripts/evaluate.py --dataset data/hn_favorite/test/dspy_examples.json --llm openai/gpt-4.1
uv run python src/dspy_favorite/scripts/evaluate.py --dataset data/hn_favorite/test/dspy_examples.json --llm openai/o3
```

**Train Model**
```bash
# Train with each LLM
uv run python src/dspy_favorite/scripts/train.py --dataset data/hn_favorite mipro --llm openai/gpt-4.1-mini
uv run python src/dspy_favorite/scripts/train.py --dataset data/hn_favorite mipro --llm openai/gpt-4.1
uv run python src/dspy_favorite/scripts/train.py --dataset data/hn_favorite mipro --llm openai/o3
```

**Evaluate Trained Model**
```bash
# Evaluate each trained model
uv run python src/dspy_favorite/scripts/evaluate.py --model saved/models/dspy_favorite_XXX --dataset data/hn_favorite/test/dspy_examples.json --llm openai/gpt-4.1-mini
uv run python src/dspy_favorite/scripts/evaluate.py --model saved/models/dspy_favorite_XXX --dataset data/hn_favorite/test/dspy_examples.json --llm openai/gpt-4.1
uv run python src/dspy_favorite/scripts/evaluate.py --model saved/models/dspy_favorite_XXX --dataset data/hn_favorite/test/dspy_examples.json --llm openai/o3
```

### Dataset C: Combined Dataset

**Prepare Combined Dataset**
```bash
# Create script to merge reader_shortlist and hn_favorite datasets
uv run python scripts/combine_datasets.py --datasets reader_shortlist,hn_favorite --output data/combined_favorite
```

**Baseline Evaluation (Rubric Only)**
```bash
# Test with each LLM
uv run python src/dspy_favorite/scripts/evaluate.py --dataset data/combined_favorite/test/dspy_examples.json --llm openai/gpt-4.1-mini
uv run python src/dspy_favorite/scripts/evaluate.py --dataset data/combined_favorite/test/dspy_examples.json --llm openai/gpt-4.1
uv run python src/dspy_favorite/scripts/evaluate.py --dataset data/combined_favorite/test/dspy_examples.json --llm openai/o3
```

**Train Model**
```bash
# Train with each LLM
uv run python src/dspy_favorite/scripts/train.py --dataset data/combined_favorite mipro --llm openai/gpt-4.1-mini
uv run python src/dspy_favorite/scripts/train.py --dataset data/combined_favorite mipro --llm openai/gpt-4.1
uv run python src/dspy_favorite/scripts/train.py --dataset data/combined_favorite mipro --llm openai/o3
```

**Evaluate Trained Model**
```bash
# Evaluate each trained model
uv run python src/dspy_favorite/scripts/evaluate.py --model saved/models/dspy_favorite_XXX --dataset data/combined_favorite/test/dspy_examples.json --llm openai/gpt-4.1-mini
uv run python src/dspy_favorite/scripts/evaluate.py --model saved/models/dspy_favorite_XXX --dataset data/combined_favorite/test/dspy_examples.json --llm openai/gpt-4.1
uv run python src/dspy_favorite/scripts/evaluate.py --model saved/models/dspy_favorite_XXX --dataset data/combined_favorite/test/dspy_examples.json --llm openai/o3
```

---

## Pairwise Comparison Experiments

### Dataset A: Readwise Reader (Shortlist)

**Baseline Evaluation (Rubric Only)**
```bash
# Test with each LLM
uv run python src/dspy_pairwise/scripts/evaluate.py --dataset data/reader_shortlist_pairwise/test/dspy_examples.json --llm openai/gpt-4.1-mini
uv run python src/dspy_pairwise/scripts/evaluate.py --dataset data/reader_shortlist_pairwise/test/dspy_examples.json --llm openai/gpt-4.1
uv run python src/dspy_pairwise/scripts/evaluate.py --dataset data/reader_shortlist_pairwise/test/dspy_examples.json --llm openai/o3
```

**Train Model**
```bash
# Train with each LLM
uv run python src/dspy_pairwise/scripts/train.py --dataset data/reader_shortlist_pairwise mipro --llm openai/gpt-4.1-mini
uv run python src/dspy_pairwise/scripts/train.py --dataset data/reader_shortlist_pairwise mipro --llm openai/gpt-4.1
uv run python src/dspy_pairwise/scripts/train.py --dataset data/reader_shortlist_pairwise mipro --llm openai/o3
```

**Evaluate Trained Model**
```bash
# Evaluate each trained model
uv run python src/dspy_pairwise/scripts/evaluate.py --model saved/models/dspy_pairwise_XXX --dataset data/reader_shortlist_pairwise/test/dspy_examples.json --llm openai/gpt-4.1-mini
uv run python src/dspy_pairwise/scripts/evaluate.py --model saved/models/dspy_pairwise_XXX --dataset data/reader_shortlist_pairwise/test/dspy_examples.json --llm openai/gpt-4.1
uv run python src/dspy_pairwise/scripts/evaluate.py --model saved/models/dspy_pairwise_XXX --dataset data/reader_shortlist_pairwise/test/dspy_examples.json --llm openai/o3
```

### Dataset B: Hacker News

**Baseline Evaluation (Rubric Only)**
```bash
# Test with each LLM
uv run python src/dspy_pairwise/scripts/evaluate.py --dataset data/hn_pairwise/test/dspy_examples.json --llm openai/gpt-4.1-mini
uv run python src/dspy_pairwise/scripts/evaluate.py --dataset data/hn_pairwise/test/dspy_examples.json --llm openai/gpt-4.1
uv run python src/dspy_pairwise/scripts/evaluate.py --dataset data/hn_pairwise/test/dspy_examples.json --llm openai/o3
```

**Train Model**
```bash
# Train with each LLM
uv run python src/dspy_pairwise/scripts/train.py --dataset data/hn_pairwise mipro --llm openai/gpt-4.1-mini
uv run python src/dspy_pairwise/scripts/train.py --dataset data/hn_pairwise mipro --llm openai/gpt-4.1
uv run python src/dspy_pairwise/scripts/train.py --dataset data/hn_pairwise mipro --llm openai/o3
```

**Evaluate Trained Model**
```bash
# Evaluate each trained model
uv run python src/dspy_pairwise/scripts/evaluate.py --model saved/models/dspy_pairwise_XXX --dataset data/hn_pairwise/test/dspy_examples.json --llm openai/gpt-4.1-mini
uv run python src/dspy_pairwise/scripts/evaluate.py --model saved/models/dspy_pairwise_XXX --dataset data/hn_pairwise/test/dspy_examples.json --llm openai/gpt-4.1
uv run python src/dspy_pairwise/scripts/evaluate.py --model saved/models/dspy_pairwise_XXX --dataset data/hn_pairwise/test/dspy_examples.json --llm openai/o3
```

### Dataset C: Combined Dataset

**Prepare Combined Dataset**
```bash
# Create script to merge reader_shortlist_pairwise and hn_pairwise datasets
uv run python scripts/combine_datasets.py --datasets reader_shortlist_pairwise,hn_pairwise --output data/combined_pairwise
```

**Baseline Evaluation (Rubric Only)**
```bash
# Test with each LLM
uv run python src/dspy_pairwise/scripts/evaluate.py --dataset data/combined_pairwise/test/dspy_examples.json --llm openai/gpt-4.1-mini
uv run python src/dspy_pairwise/scripts/evaluate.py --dataset data/combined_pairwise/test/dspy_examples.json --llm openai/gpt-4.1
uv run python src/dspy_pairwise/scripts/evaluate.py --dataset data/combined_pairwise/test/dspy_examples.json --llm openai/o3
```

**Train Model**
```bash
# Train with each LLM
uv run python src/dspy_pairwise/scripts/train.py --dataset data/combined_pairwise mipro --llm openai/gpt-4.1-mini
uv run python src/dspy_pairwise/scripts/train.py --dataset data/combined_pairwise mipro --llm openai/gpt-4.1
uv run python src/dspy_pairwise/scripts/train.py --dataset data/combined_pairwise mipro --llm openai/o3
```

**Evaluate Trained Model**
```bash
# Evaluate each trained model
uv run python src/dspy_pairwise/scripts/evaluate.py --model saved/models/dspy_pairwise_XXX --dataset data/combined_pairwise/test/dspy_examples.json --llm openai/gpt-4.1-mini
uv run python src/dspy_pairwise/scripts/evaluate.py --model saved/models/dspy_pairwise_XXX --dataset data/combined_pairwise/test/dspy_examples.json --llm openai/gpt-4.1
uv run python src/dspy_pairwise/scripts/evaluate.py --model saved/models/dspy_pairwise_XXX --dataset data/combined_pairwise/test/dspy_examples.json --llm openai/o3
```

---

## Results Summary Tables

### 1. Favorite Binary Classification Results
| Dataset | LLM | Baseline | Trained | Improvement |
|---------|-----|----------|---------|-------------|
| Readwise | GPT-4.1-mini | [Result] | [Result] | [Delta] |
| Readwise | GPT-4.1 | [Result] | [Result] | [Delta] |
| Readwise | O3 | [Result] | [Result] | [Delta] |
| Hacker News | GPT-4.1-mini | [Result] | [Result] | [Delta] |
| Hacker News | GPT-4.1 | [Result] | [Result] | [Delta] |
| Hacker News | O3 | [Result] | [Result] | [Delta] |
| Combined | GPT-4.1-mini | [Result] | [Result] | [Delta] |
| Combined | GPT-4.1 | [Result] | [Result] | [Delta] |
| Combined | O3 | [Result] | [Result] | [Delta] |

### 2. Pairwise Comparison Results
| Dataset | LLM | Baseline | Trained | Improvement |
|---------|-----|----------|---------|-------------|
| Readwise | GPT-4.1-mini | [Result] | [Result] | [Delta] |
| Readwise | GPT-4.1 | [Result] | [Result] | [Delta] |
| Readwise | O3 | [Result] | [Result] | [Delta] |
| Hacker News | GPT-4.1-mini | [Result] | [Result] | [Delta] |
| Hacker News | GPT-4.1 | [Result] | [Result] | [Delta] |
| Hacker News | O3 | [Result] | [Result] | [Delta] |
| Combined | GPT-4.1-mini | [Result] | [Result] | [Delta] |
| Combined | GPT-4.1 | [Result] | [Result] | [Delta] |
| Combined | O3 | [Result] | [Result] | [Delta] |

---

## Analysis Points

### For Each Task Type:
1. **Baseline Performance**: Does in-context learning with rubric alone show meaningful performance?
2. **Optimization Impact**: How much does DSPy optimization improve performance?
3. **Dataset Comparison**: How does performance vary across datasets?
4. **Reasoning Quality**: What patterns do the models identify?

### Binary vs Pairwise Comparison:
1. **Approach Effectiveness**: Which method (binary classification vs pairwise) works better for taste modeling?
2. **Dataset Sensitivity**: Does one approach work better on certain datasets?
3. **Model Consistency**: Are preference predictions consistent between approaches?
4. **Computational Efficiency**: What are the trade-offs in performance vs cost?

### Cross-Dataset Analysis:
1. **Personal vs Public**: How does performance differ between personal reading data (Readwise) vs public data (HN)?
2. **Rubric Effectiveness**: Does the taste rubric work better on one dataset than the other?
3. **Combined Performance**: Does combining datasets improve or hurt performance?
4. **Generalization**: Does the combined model capture a more robust taste profile?

### LLM Comparisons:
1. **Model Capability**: How do different LLMs (GPT-4.1-mini, GPT-4.1, O3) perform on taste distillation?
2. **Cost vs Performance**: Is the performance difference between models worth the cost difference?
3. **Reasoning Quality**: Which model provides better explanations for taste decisions?
4. **Consistency**: Are taste predictions consistent across different LLMs?
5. **Optimization Response**: Do all models benefit equally from DSPy optimization?

---

## Key Deliverables for Article

1. **Quantitative Evidence**: Performance metrics showing taste can be captured across different LLMs
2. **Qualitative Examples**: Compelling reasoning examples from the models
3. **Learning Curves**: Visual demonstration of improvement with optimization
4. **Comparative Analysis**: Clear differences between approaches, datasets, and LLMs
5. **Cost-Performance Analysis**: Trade-offs between model capability and computational cost
6. **Robustness Evidence**: Consistent taste distillation across different model architectures
7. **Practical Implications**: What this means for building personalized AI systems

---

## Implementation Notes

- All experiments should use consistent evaluation metrics across LLMs
- Save all model outputs for qualitative analysis
- Document any unexpected findings or errors
- Keep detailed logs of all experiment runs with LLM specifications
- Note the model version (XXX) after training for proper evaluation
- Track computational costs and time for each LLM
- Test order should be randomized to avoid bias
- Consider rate limiting for API calls across different models
- **Total experiments**: 18 baseline + 18 trained = 36 evaluations per dataset group
- **Grand total**: 108 experiments across all groups and LLMs