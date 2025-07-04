# DSPy Implementation Tutorial

This tutorial provides step-by-step instructions for implementing DSPy models using a clean, standardized project architecture.

## Project Architecture Overview

```
project_root/
├── config/
│   └── config.yaml              # Central config for paths, hyperparams, DSPy flags
│
├── src/                         # Source code for models and scripts
│   └── modelA/                  # Model-specific implementation and scripts
│       ├── modelA.py            # DSPy model implementation
│       ├── metrics.py           # Custom metrics for evaluation
│       └── scripts/             # Model-specific scripts
│           ├── train.py         # Train DSPy module using defined signatures and optimizer
│           ├── evaluate.py      # Evaluate model on validation or test set using defined metrics
│           ├── predict.py       # Run inference on new data
│           └── prepare.py       # Load and split raw data into train/valid/test sets
│
├── data/                        # Data storage
│   ├── raw/                     # Untouched original input data
│   └── processed/               # Preprocessed & structured data
│       ├── train/               # Training split
│       ├── valid/               # Validation split
│       └── test/                # Test split
│
├── lib/                         # Shared utilities and metrics
│   └── utilityA.py              # Model-specific utilities
│
├── tests/                       # Unit and integration tests
│
├── saved/                       # Persisted outputs (e.g. checkpoints, logs)
│   ├── models/                  # Trained DSPy modules / checkpoints
│   │   └── modelA/              # Model-specific checkpoints
│   │       ├── 01-checkpoint/   # First saved checkpoint
│   │       └── 02-checkpoint/   # Second saved checkpoint
│   └── logs/                    # Training/eval logs
│       ├── 01-logs/             # Logs from run #1
│       └── 02-logs/             # Logs from run #2
│
├── notebooks/                   # Jupyter notebooks for EDA, quick prototyping
│
└── results/                     # Evaluation outputs, visualizations, or analysis reports
```

## Development Order

1. **Configure Project** (setup `config/config.yaml`)
2. **Define DSPy Models** (implement `src/modelA/model.py`)
3. **Define Custom Metrics** (implement `src/modelA/metrics.py`)
4. **Create Data Preparation** (implement `src/modelA/scripts/prepare.py`)
5. **Create Evaluation Script** (implement `src/modelA/scripts/evaluate.py`)
6. **Create Training Script** (implement `src/modelA/scripts/train.py`)
7. **Create Inference Script** (implement `src/modelA/scripts/predict.py`)

**Note**: This structure separates concerns cleanly - configuration, core logic, execution scripts, and data are all in their proper places with versioned checkpoints for reproducibility.

---

## Step 0: Project Configuration

### 0.1 Create Central Configuration

Before implementing any DSPy components, set up centralized configuration management.

**File Location**: `config/config.yaml`

**Key Configuration Sections:**
- **Data Paths**: Raw data locations, processed data directories
- **Model Settings**: DSPy model types, language model configurations
- **Training Parameters**: Optimization settings, hyperparameters
- **Evaluation Settings**: Metrics, validation procedures
- **Logging**: Output directories, checkpoint versioning

**Configuration includes sections for:**
- Data paths and splitting ratios
- Model selection and language model settings
- Training parameters and optimization settings
- Evaluation metrics and rubric configuration
- Logging and checkpoint management

---

## Step 1: Define DSPy Models

### 1.1 Create DSPy Signatures and Modules

Define all your DSPy signatures and modules for the specific model.

**File Location**: `src/modelA/model.py`

**Key Components:**
- **Signature Definitions**: Clear input/output interfaces with type hints
- **Module Classes**: DSPy modules that orchestrate signatures
- **Factory Functions**: Create models based on configuration
- **Model Registry**: Easy access to different model variants

**Design Patterns:**
- Use `dspy.ChainOfThought` for reasoning tasks
- Use `dspy.Predict` for direct classification
- Make model choice configurable through constructor parameters
- Implement factory pattern for easy model instantiation

**Implementation includes:**
- Signature definitions with clear input/output interfaces
- Module classes using ChainOfThought or Predict
- Factory functions for configurable model creation
- Type hints and documentation for clarity

---

## Step 2: Define Custom Metrics

### 2.1 Create Evaluation Metrics

Centralize all evaluation metrics and scoring functions based on DSPy's metric system.

**File Location**: `src/modelA/metrics.py`

**Understanding DSPy Metrics:**
- **Function Signature**: `metric(example, prediction, trace=None) -> float/int/bool`
- **Parameters**: 
  - `example`: From your training/dev set (contains expected outputs)
  - `prediction`: Output from your DSPy program
  - `trace`: Optional parameter for optimization (enables advanced tricks)
- **Return Values**:
  - For evaluation/optimization (`trace=None`): Return float/int score
  - For bootstrapping (`trace` is not None): Return bool (pass/fail)

**Key Metric Types:**
- **Simple Metrics**: Direct comparisons (accuracy, exact match, F1)
- **AI-Feedback Metrics**: Use LMs to assess complex outputs
- **Composite Metrics**: Check multiple properties and aggregate scores
- **DSPy Program Metrics**: Full DSPy programs as metrics (can be optimized)

**Implementation includes:**
- Simple accuracy metrics for basic evaluation
- Complex multi-property validation functions
- AI-feedback metrics using DSPy signatures
- Comprehensive evaluation utilities
- Built-in DSPy metric utilities

**Metric Design Best Practices:**
- **Start Simple**: Begin with basic accuracy/exact match, iterate to complex metrics
- **Use AI Feedback**: For long-form outputs, use smaller DSPy programs to check quality
- **Handle the `trace` Parameter**: Different behavior for evaluation vs bootstrapping
- **Make Metrics Learnable**: Consider making your metrics DSPy programs themselves
- **Aggregate Thoughtfully**: Combine multiple dimensions for comprehensive assessment

---

## Step 3: Create Data Preparation Script

### 3.1 Data Preparation Goals

Transform raw data into DSPy-compatible format with proper train/valid/test splits.

**File Location**: `src/modelA/scripts/prepare.py`

**Key Functions:**
- **Configuration Loading**: Read settings from `config/config.yaml`
- **Data Loading**: Read various raw data formats
- **DSPy Example Creation**: Convert to `dspy.Example` objects
- **Data Splitting**: Create train/valid/test splits
- **Data Saving**: Export to organized directory structure

**Implementation includes:**
- Configuration loading from YAML files
- Raw data loading and validation
- DSPy Example object creation
- Train/validation/test splitting
- Organized data saving with proper directory structure

---

## Step 4: Create Evaluation Script

### 4.1 Baseline Evaluation

Establish baseline performance before optimization.

**File Location**: `src/modelA/scripts/evaluate.py`

**Key Functions:**
- **Configuration Management**: Load all settings from config
- **Data Loading**: Load processed datasets
- **Model Initialization**: Create models from config
- **Rubric Integration**: Optionally load taste rubrics
- **Comprehensive Evaluation**: Use custom metrics from `src/modelA/metrics.py`

**Implementation includes:**
- Model loading from checkpoints or fresh creation
- Validation data loading and processing
- Rubric integration for evaluation context
- DSPy Evaluate utility for standardized evaluation
- Comprehensive metrics collection and reporting
- Results saving and logging

---

## Step 5: Create Training Script

### 5.1 Model Training/Optimization

Optimize DSPy models using configured optimizers.

**File Location**: `src/modelA/scripts/train.py`

**Key Functions:**
- **Optimizer Configuration**: Set up MIPROv2, BootstrapFinetune, etc.
- **Training Pipeline**: Baseline → Optimization → Evaluation
- **Checkpoint Management**: Save models with versioning
- **Training Logs**: Comprehensive logging of training process

**Implementation includes:**
- Training and validation data loading
- Model creation using factory functions
- Optimizer configuration (MIPROv2, BootstrapFinetune, etc.)
- Model compilation and optimization
- Checkpoint saving with metadata and versioning
- Training progress logging and monitoring

---

## Step 6: Create Inference Script

### 6.1 Model Inference

Run inference on new data using trained models.

**File Location**: `src/modelA/scripts/predict.py`

**Key Functions:**
- **Model Loading**: Load trained checkpoints
- **Batch Inference**: Process multiple examples efficiently
- **Result Formatting**: Structure outputs for downstream use
- **Error Handling**: Graceful handling of prediction failures


## Implementation Summary

### Complete Development Workflow

1. **Project Setup** → Configure `config/config.yaml` with all settings
2. **Core Implementation** → Build `src/modelA/model.py` and `src/modelA/metrics.py`
3. **Data Pipeline** → Implement `src/modelA/scripts/prepare.py` for data preparation
4. **Evaluation Framework** → Create `src/modelA/scripts/evaluate.py` for baseline metrics
5. **Training Pipeline** → Build `src/modelA/scripts/train.py` for model optimization
6. **Inference System** → Implement `src/modelA/scripts/predict.py` for inference use


### Next Steps for Implementation

1. Set up the directory structure with proper `config/config.yaml`
2. Implement your specific DSPy signatures and modules in `src/modelA/model.py`
3. Create custom metrics relevant to your task in `src/modelA/metrics.py`
4. Build the data preparation pipeline in `src/modelA/scripts/prepare.py`
5. Implement evaluation and training scripts with proper checkpoint management in `src/modelA/scripts/`
6. Create inference capabilitiesin `src/modelA/scripts/predict.py`

This standardized structure provides a solid foundation for any DSPy project while maintaining flexibility for task-specific requirements.
