# Fine-Tuning OpenAI Models for Taste Distillation

This document outlines how to leverage OpenAI's fine-tuning capabilities to create models that can predict user preferences (favorites and shortlist items) using two approaches: binary classification and pairwise comparison.

## Overview

We'll create three fine-tuned models:
1. **Binary Classifier (SFT)**: Predicts if an article is a favorite/shortlist item
2. **Pairwise Classifier (SFT)**: Determines preference between two articles using standard fine-tuning
3. **Pairwise Classifier (DPO)**: Uses Direct Preference Optimization for improved preference learning

## 1. Binary Classification Model (Supervised Fine-Tuning)

### Data Preparation

Convert your existing DSPy examples to OpenAI's chat format:

```python
import json
from openai import OpenAI

client = OpenAI()

def convert_to_openai_format(dspy_examples, rubric_content):
    """Convert DSPy binary examples to OpenAI chat format"""
    training_data = []
    
    system_message = f"""You are a personal preference predictor. Based on the user's taste rubric below, 
    determine if an article title would be a favorite.
    
    Taste Rubric:
    {rubric_content}
    
    Respond with only 'true' or 'false'."""
    
    for example in dspy_examples:
        messages = [
            {"role": "system", "content": system_message},
            {"role": "user", "content": f"Title: {example['title']}"},
            {"role": "assistant", "content": str(example['is_favorite']).lower()}
        ]
        training_data.append({"messages": messages})
    
    return training_data

# Load your data
with open('data/reader_favorite/train/dspy_examples.json', 'r') as f:
    train_examples = json.load(f)

with open('saved/rubrics/taste_rubric.txt', 'r') as f:
    rubric = f.read()

# Convert and save
training_data = convert_to_openai_format(train_examples, rubric)
with open('data/openai_training_binary.jsonl', 'w') as f:
    for item in training_data:
        f.write(json.dumps(item) + '\n')
```

### Training the Binary Model

```python
# Upload training file
training_file = client.files.create(
    file=open("data/openai_training_binary.jsonl", "rb"),
    purpose="fine-tune"
)

# Create fine-tuning job
binary_job = client.fine_tuning.jobs.create(
    training_file=training_file.id,
    model="gpt-4o-mini-2024-07-18",  # or gpt-3.5-turbo for cost efficiency
    suffix="taste-binary",
    hyperparameters={
        "n_epochs": 3,
        "batch_size": 8
    }
)

print(f"Binary classification job created: {binary_job.id}")
```

## 2. Pairwise Comparison Model (Standard Fine-Tuning)

### Data Preparation

```python
def convert_pairwise_to_openai_format(pairwise_examples, rubric_content):
    """Convert DSPy pairwise examples to OpenAI chat format"""
    training_data = []
    
    system_message = f"""You are a personal preference predictor. Based on the user's taste rubric below, 
    determine which of two article titles the user would prefer.
    
    Taste Rubric:
    {rubric_content}
    
    Respond with only 'A' or 'B'."""
    
    for example in pairwise_examples:
        messages = [
            {"role": "system", "content": system_message},
            {"role": "user", "content": f"A: {example['title_a']}\nB: {example['title_b']}"},
            {"role": "assistant", "content": example['preferred']}
        ]
        training_data.append({"messages": messages})
    
    return training_data

# Load pairwise data
with open('data/reader_favorite_pairwise/train/dspy_examples.json', 'r') as f:
    pairwise_train = json.load(f)

# Convert and save
pairwise_data = convert_pairwise_to_openai_format(pairwise_train, rubric)
with open('data/openai_training_pairwise_sft.jsonl', 'w') as f:
    for item in pairwise_data:
        f.write(json.dumps(item) + '\n')
```

### Training the Pairwise SFT Model

```python
# Upload and train
pairwise_file = client.files.create(
    file=open("data/openai_training_pairwise_sft.jsonl", "rb"),
    purpose="fine-tune"
)

pairwise_job = client.fine_tuning.jobs.create(
    training_file=pairwise_file.id,
    model="gpt-4o-mini-2024-07-18",
    suffix="taste-pairwise-sft",
    hyperparameters={
        "n_epochs": 3,
        "batch_size": 8
    }
)
```

## 3. Pairwise Comparison Model (Direct Preference Optimization)

### Data Preparation for DPO

DPO requires a different format with explicit preferred/rejected pairs:

```python
def convert_to_dpo_format(pairwise_examples, rubric_content):
    """Convert pairwise examples to DPO format"""
    dpo_data = []
    
    system_message = f"""You are a personal preference predictor. Based on the user's taste rubric, 
    explain which article title would be preferred and why.
    
    Taste Rubric:
    {rubric_content}"""
    
    for example in pairwise_examples:
        # Create the user prompt
        user_content = f"Which article would be preferred?\nA: {example['title_a']}\nB: {example['title_b']}"
        
        # Determine preferred and rejected responses
        if example['preferred'] == 'A':
            preferred_response = f"I would choose A: {example['title_a']}. This aligns better with the taste preferences."
            rejected_response = f"I would choose B: {example['title_b']}. This seems more interesting."
        else:
            preferred_response = f"I would choose B: {example['title_b']}. This aligns better with the taste preferences."
            rejected_response = f"I would choose A: {example['title_a']}. This seems more interesting."
        
        dpo_example = {
            "messages": [
                {"role": "system", "content": system_message},
                {"role": "user", "content": user_content}
            ],
            "preferred": [
                {"role": "assistant", "content": preferred_response}
            ],
            "rejected": [
                {"role": "assistant", "content": rejected_response}
            ]
        }
        dpo_data.append(dpo_example)
    
    return dpo_data

# Convert and save DPO training data
dpo_training_data = convert_to_dpo_format(pairwise_train, rubric)
with open('data/openai_training_dpo.jsonl', 'w') as f:
    for item in dpo_training_data:
        f.write(json.dumps(item) + '\n')
```

### Training the DPO Model

```python
# Upload DPO training file
dpo_file = client.files.create(
    file=open("data/openai_training_dpo.jsonl", "rb"),
    purpose="fine-tune"
)

# Create DPO fine-tuning job
dpo_job = client.fine_tuning.jobs.create(
    training_file=dpo_file.id,
    model="gpt-4o-mini-2024-07-18",
    suffix="taste-pairwise-dpo",
    method={
        "type": "dpo",
        "dpo": {
            "hyperparameters": {
                "n_epochs": 2,
                "beta": 0.1,  # Controls preference strength
                "batch_size": 8
            }
        }
    }
)

print(f"DPO job created: {dpo_job.id}")
```

## 4. Evaluation and Comparison

### Monitoring Training Progress

```python
def monitor_job(job_id):
    """Monitor fine-tuning job progress"""
    while True:
        status = client.fine_tuning.jobs.retrieve(job_id)
        print(f"Status: {status.status}")
        
        if status.status == "succeeded":
            print(f"Model ready: {status.fine_tuned_model}")
            return status.fine_tuned_model
        elif status.status == "failed":
            print(f"Job failed: {status.error}")
            return None
        
        time.sleep(60)  # Check every minute

# Wait for models to complete
binary_model_id = monitor_job(binary_job.id)
pairwise_sft_model_id = monitor_job(pairwise_job.id)
pairwise_dpo_model_id = monitor_job(dpo_job.id)
```

### Evaluation Script

```python
def evaluate_binary_model(model_id, test_examples):
    """Evaluate binary classification model"""
    correct = 0
    
    for example in test_examples:
        response = client.chat.completions.create(
            model=model_id,
            messages=[
                {"role": "user", "content": f"Title: {example['title']}"}
            ],
            temperature=0,
            max_tokens=10
        )
        
        prediction = response.choices[0].message.content.strip().lower()
        actual = str(example['is_favorite']).lower()
        
        if prediction == actual:
            correct += 1
    
    accuracy = correct / len(test_examples)
    print(f"Binary model accuracy: {accuracy:.2%}")
    return accuracy

def evaluate_pairwise_model(model_id, test_examples, model_type="SFT"):
    """Evaluate pairwise comparison model"""
    correct = 0
    
    for example in test_examples:
        response = client.chat.completions.create(
            model=model_id,
            messages=[
                {"role": "user", "content": f"A: {example['title_a']}\nB: {example['title_b']}"}
            ],
            temperature=0,
            max_tokens=50 if model_type == "DPO" else 10
        )
        
        content = response.choices[0].message.content.strip()
        
        # Extract choice from response
        if model_type == "DPO":
            # DPO model gives explanations, extract A or B
            prediction = 'A' if 'A:' in content or 'choose A' in content else 'B'
        else:
            prediction = content.upper()
        
        if prediction == example['preferred']:
            correct += 1
    
    accuracy = correct / len(test_examples)
    print(f"Pairwise {model_type} model accuracy: {accuracy:.2%}")
    return accuracy

# Run evaluations
with open('data/reader_favorite/test/dspy_examples.json', 'r') as f:
    binary_test = json.load(f)

with open('data/reader_favorite_pairwise/test/dspy_examples.json', 'r') as f:
    pairwise_test = json.load(f)

# Evaluate all models
binary_acc = evaluate_binary_model(binary_model_id, binary_test)
pairwise_sft_acc = evaluate_pairwise_model(pairwise_sft_model_id, pairwise_test, "SFT")
pairwise_dpo_acc = evaluate_pairwise_model(pairwise_dpo_model_id, pairwise_test, "DPO")
```

## 5. Using the Fine-Tuned Models

### Binary Prediction

```python
def predict_favorite(title, model_id):
    """Predict if a title would be a favorite"""
    response = client.chat.completions.create(
        model=model_id,
        messages=[
            {"role": "user", "content": f"Title: {title}"}
        ],
        temperature=0,
        max_tokens=10
    )
    return response.choices[0].message.content.strip().lower() == 'true'

# Example usage
is_favorite = predict_favorite(
    "New breakthrough in quantum computing achieves 99.9% error correction",
    binary_model_id
)
```

### Pairwise Comparison

```python
def compare_titles(title_a, title_b, model_id, model_type="SFT"):
    """Compare two titles and return preference"""
    response = client.chat.completions.create(
        model=model_id,
        messages=[
            {"role": "user", "content": f"A: {title_a}\nB: {title_b}"}
        ],
        temperature=0,
        max_tokens=50 if model_type == "DPO" else 10
    )
    
    content = response.choices[0].message.content.strip()
    
    if model_type == "DPO":
        # Parse DPO explanation
        return {
            "preferred": 'A' if 'A:' in content or 'choose A' in content else 'B',
            "explanation": content
        }
    else:
        return {"preferred": content.upper()}
```

## Expected Benefits

### Binary Classification (SFT)
- **Pros**: Simple, fast inference, cost-effective
- **Cons**: May not capture nuanced preferences
- **Best for**: Quick filtering of content

### Pairwise SFT
- **Pros**: Direct comparison, simple output format
- **Cons**: No explanation, may be inconsistent
- **Best for**: A/B testing scenarios

### Pairwise DPO
- **Pros**: Better preference modeling, provides explanations, more consistent
- **Cons**: Higher inference cost, requires more training data
- **Best for**: Understanding preference reasoning, higher accuracy needs

## Cost Considerations

- **Training**: ~$0.008 per 1K tokens
- **Inference**: ~$0.0006 per 1K tokens (gpt-4o-mini)
- **DPO Training**: Typically requires 2x training time due to preference pairs

## Integration with Existing DSPy System

You can use these fine-tuned models as drop-in replacements or as ensemble components:

```python
class OpenAIFavoriteBinarySignature(dspy.Signature):
    """Use fine-tuned OpenAI model for prediction"""
    title = dspy.InputField()
    is_favorite = dspy.OutputField()

class OpenAIPredictor(dspy.Module):
    def __init__(self, model_id):
        self.model_id = model_id
    
    def forward(self, title):
        # Call OpenAI fine-tuned model
        prediction = predict_favorite(title, self.model_id)
        return dspy.Prediction(is_favorite=prediction)
```

This approach allows you to compare DSPy-optimized models with OpenAI fine-tuned models directly.