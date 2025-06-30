# MathCoRL - Usage Guide

## 🔭 Overview

MathCoRL implements **In-Context Reinforcement Learning** for mathematical reasoning, comparing three approaches:

1. **🎯 Zero-shot FPP**: Function Prototype Prompting without examples
2. **🎲 FPP + Random Examples**: FPP with randomly selected examples  
3. **🤖 FPP + Policy Network**: FPP with policy-selected examples

## 📋 Pipeline Architecture

The pipeline consists of 3 main scripts with consistent ArgumentParser interface:

```
📦 generate_candidates.py  →  🎓 train_policy.py  →  🏆 run_comparison.py
      Step 1                       Step 2                    Step 3
```

### Pipeline Components

| Script | Function | Description |
|--------|----------|-------------|
| `generate_candidates.py` | Candidate Generation | Extract mathematical problems and generate code solutions |
| `train_policy.py` | Policy Training | Train neural networks to select relevant examples |
| `run_comparison.py` | Method Comparison | Compare zero-shot, random, and policy-based approaches |
| `test_pipeline.py` | Pipeline Testing | End-to-end validation of complete pipeline |

## 🔧 Environment Setup

### 1. Configure Environment
```bash
# Copy and configure environment
cp env.example .env
# Edit .env with your OpenAI API key

# Model configuration:
DEFAULT_MODEL=gpt-4.1-mini           # Chat model for code generation
EMBEDDING_MODEL=text-embedding-3-small  # Embedding model for similarity
TEMPERATURE=0.1                      # Temperature for generation
```

### 2. Supported Datasets
| Dataset | Domain | Description |
|---------|--------|-------------|
| **GSM8K** | Elementary Math | Grade School Math word problems |
| **SVAMP** | Arithmetic | Simple arithmetic word problems |
| **TabMWP** | Tabular Math | Math problems involving tables and charts |
| **TAT-QA** | Financial Reasoning | Table-and-text QA for financial data |
| **FinQA** | Financial Analysis | Complex financial reasoning and calculations |

## 📦 Step 1: Generate Candidates

Generate training examples from mathematical reasoning datasets.

### Basic Usage
```bash
# Generate 100 candidates for FinQA
python generate_candidates.py --dataset FinQA --n-candidates 100

# Generate candidates with verbose output
python generate_candidates.py --dataset GSM8K --n-candidates 50 --verbose
```

### Options
- `--dataset, -d`: Choose dataset (GSM8K, SVAMP, TabMWP, TAT-QA, FinQA)
- `--n-candidates, -n`: Number of candidates to generate (default: 100)
- `--output-dir, -o`: Output directory (default: candidates)
- `--model`: OpenAI chat model (default: from config)
- `--embedding-model`: OpenAI embedding model (default: from config)
- `--overwrite`: Overwrite existing files
- `--verbose, -v`: Enable verbose logging

### What It Does
1. **Load Dataset**: Parse mathematical problems from JSON/JSONL files
2. **Generate Code**: Use Function Prototype Prompting to create Python solutions
3. **Create Embeddings**: Generate semantic embeddings for problem context + question
4. **Validate Solutions**: Execute code and verify correctness
5. **Save Candidates**: Store validated examples for training

## 🎓 Step 2: Train Policy Network

Train neural networks to select the most relevant examples for each problem.

### Basic Usage
```bash
# Train with default settings
python train_policy.py --dataset FinQA --epochs 5

# Train with custom hyperparameters
python train_policy.py --dataset GSM8K --epochs 3 --lr 3e-4
```

### Dataset-Specific Configurations
| Dataset | Examples (k) | Pool Size | Learning Rate | Recommended Epochs |
|---------|--------------|-----------|---------------|-------------------|
| GSM8K   | 2           | 20        | 3e-4          | 3                |
| SVAMP   | 2           | 15        | 3e-4          | 4                |
| TabMWP  | 2           | 25        | 2e-4          | 4                |
| TAT-QA  | 3           | 25        | 2e-4          | 5                |
| FinQA   | 2           | 30        | 1e-4          | 5                |

### Options
- `--dataset, -d`: Dataset to train on (required)
- `--epochs, -e`: Number of training epochs (default: 5)
- `--lr`: Learning rate (default: dataset-specific)
- `--pool-size`: Candidate pool size for training (default: dataset-specific)
- `--k`: Number of examples to select (default: dataset-specific)
- `--samples-per-epoch`: Samples per epoch (default: all candidates)
- `--models-dir`: Model save directory (default: models)
- `--overwrite`: Overwrite existing models
- `--verbose, -v`: Enable verbose logging

### Training Process
1. **Load Candidates**: Read generated candidates with embeddings
2. **Policy Network**: Initialize neural network with attention mechanism
3. **PPO Training**: Use Proximal Policy Optimization with multi-objective rewards
4. **Model Saving**: Save best and final models during training
5. **History Logging**: Record training metrics and loss curves

## 🏆 Step 3: Run Comparison

Compare the three approaches on test samples to evaluate Policy Network effectiveness.

### Basic Usage
```bash
# Compare methods on 10 samples
python run_comparison.py --dataset FinQA --samples 10

# Save detailed results to JSON
python run_comparison.py --dataset GSM8K --samples 20 --save-results
```

### Options
- `--dataset, -d`: Dataset to test on (required)
- `--samples, -s`: Number of test samples (default: 10)
- `--seed`: Random seed for reproducibility (default: 42)
- `--save-results`: Save detailed JSON results
- `--skip-policy`: Skip Policy Network method if model unavailable
- `--quiet, -q`: Suppress detailed output during testing
- `--verbose, -v`: Enable verbose logging

### Comparison Methods

#### 1. 🎯 Zero-shot FPP
- **Description**: Function Prototype Prompting without examples
- **Usage**: Baseline method using only problem description
- **Implementation**: Direct GPT call with function prototypes

#### 2. 🎲 FPP + Random Examples
- **Description**: FPP with randomly selected training examples
- **Usage**: Control method to test value of any examples vs. smart selection
- **Implementation**: Random sampling from candidate pool

#### 3. 🤖 FPP + Policy Network
- **Description**: FPP with policy-selected relevant examples
- **Usage**: Main method using trained neural network for example selection
- **Implementation**: Policy network selects most relevant examples based on embeddings

### Output Format
```
🏆 FINAL COMPARISON RESULTS
============================================================
Dataset: [DATASET_NAME]
Samples tested: [N]
Best method: [BEST_METHOD] (X.X%)
------------------------------------------------------------
🎯 Zero-shot FPP      : X.X%
🎲 FPP + Random       : X.X%
🤖 FPP + Policy Net   : X.X%
============================================================
📈 Policy Network improvement: ±X.X% over zero-shot
```

## 🧪 Pipeline Testing

Test the complete pipeline end-to-end or individual components.

### Complete Pipeline Test
```bash
# Quick test (recommended for first time)
python test_pipeline.py --dataset GSM8K --quick-test

# Full test with larger parameters
python test_pipeline.py --dataset FinQA --full-test
```

### Component Testing
```bash
# Test only candidate generation
python test_pipeline.py --dataset FinQA --candidates-only

# Test only policy training (requires existing candidates)
python test_pipeline.py --dataset FinQA --training-only

# Test only comparison (requires existing candidates & models)
python test_pipeline.py --dataset FinQA --comparison-only
```

### Test Configurations
| Test Type | Candidates | Epochs | Samples | Use Case |
|-----------|------------|--------|---------|----------|
| Quick Test | 20 | 2 | 5 | Initial validation |
| Full Test | 50 | 3 | 10 | Standard testing |
| Custom | Configurable | Configurable | Configurable | Development |

## 📁 Directory Structure

```
MathCoRL/
├── 🔨 generate_candidates.py    # Step 1: Candidate generation
├── 🎓 train_policy.py           # Step 2: Policy network training
├── 🏆 run_comparison.py         # Step 3: Method comparison
├── 🧪 test_pipeline.py          # Complete pipeline testing
├── comparison_study_generic.py  # Backend comparison module
├── mint/                        # Core library
│   ├── config.py               # Configuration management
│   ├── icrl/                   # In-Context RL components
│   │   ├── candidate_generator.py  # Candidate generation logic
│   │   ├── trainer.py          # Policy network training
│   │   ├── policy_network.py   # Neural network architecture
│   │   └── evaluator.py        # Policy network evaluation
│   ├── prompts.py              # Prompt templates
│   ├── utils.py                # Utility functions
│   └── ...
├── candidates/                  # Generated training candidates
│   ├── GSM8K.json
│   ├── FinQA.json
│   └── ...
├── models/                      # Trained policy network models
│   ├── GSM8K_policy_best.pt
│   ├── FinQA_policy_best.pt
│   └── ...
├── results/                     # Comparison results
│   ├── GSM8K_comparison_*.json
│   └── ...
├── logs/                        # Training logs
│   └── *_training_history.json
├── datasets/                    # Dataset files
├── templates/                   # Prompt templates
└── env.example                  # Environment configuration template
```

## 🎯 Example Workflows

### New Dataset Setup
```bash
# Complete workflow for new dataset
python generate_candidates.py --dataset FinQA --n-candidates 100
python train_policy.py --dataset FinQA --epochs 5
python run_comparison.py --dataset FinQA --samples 10 --save-results
```

### Hyperparameter Exploration
```bash
# Try different learning rates
python train_policy.py --dataset GSM8K --epochs 3 --lr 1e-4 --overwrite
python train_policy.py --dataset GSM8K --epochs 3 --lr 3e-4 --overwrite
python train_policy.py --dataset GSM8K --epochs 3 --lr 5e-4 --overwrite

# Compare results
python run_comparison.py --dataset GSM8K --samples 20
```

### Batch Testing
```bash
# Test all datasets
for dataset in GSM8K SVAMP TabMWP TAT-QA FinQA; do
    python test_pipeline.py --dataset $dataset --quick-test
done
```

### Development Workflow
```bash
# Iterative development cycle
python test_pipeline.py --dataset GSM8K --candidates-only  # Test candidate generation
python test_pipeline.py --dataset GSM8K --training-only    # Test policy training
python test_pipeline.py --dataset GSM8K --comparison-only  # Test comparison
```

## 🔍 Troubleshooting

### Common Issues

1. **Missing candidates**: Run `generate_candidates.py` first
2. **Missing models**: Run `train_policy.py` before comparison
3. **OpenAI API errors**: Check your API key in `.env` file
4. **Out of memory**: Reduce `--pool-size` or `--samples-per-epoch`
5. **File exists errors**: Use `--overwrite` flag to replace existing files

### Validation Commands
```bash
# Check if all components are working
python test_pipeline.py --dataset GSM8K --quick-test

# Verify file structure
ls candidates/ models/ results/ logs/

# Check model compatibility
python -c "import torch; print('PyTorch version:', torch.__version__)"
```

### Debug Mode
```bash
# Enable verbose output for debugging
python generate_candidates.py --dataset GSM8K --n-candidates 5 --verbose
python train_policy.py --dataset GSM8K --epochs 1 --verbose
python run_comparison.py --dataset GSM8K --samples 3 --verbose
```

## 🏗️ Architecture Details

### Policy Network Architecture
- **Input**: Problem embedding (1536-D) + Candidate embeddings (N×1536-D)
- **Architecture**: Multi-head attention with feed-forward layers
- **Output**: Probability distribution over candidates
- **Training**: PPO with multi-objective reward (accuracy + similarity + diversity)

### Embedding Strategy
- **Problem Representation**: `f"{context}\n\n{question}"` for datasets with context, `question` only for GSM8K
- **Model**: OpenAI `text-embedding-3-small` (1536 dimensions)
- **Consistency**: Standardized across candidate generation, training, and inference

### Code Generation
- **Method**: Function Prototype Prompting (FPP)
- **Model**: GPT-4.1-mini for reliable code generation
- **Templates**: LangChain templates for consistent prompting
- **Execution**: Safe Python execution with result validation

## 📋 Configuration Reference

### Environment Variables
```bash
# Required
OPENAI_API_KEY=your_openai_api_key_here

# Model Configuration
DEFAULT_MODEL=gpt-4.1-mini
EMBEDDING_MODEL=text-embedding-3-small
TEMPERATURE=0.1
MAX_TOKENS=1000

# Optional
LOG_LEVEL=INFO
DEBUG_MODE=false
```

### Script Arguments
All scripts support:
- `--help`: Show detailed help information
- `--verbose, -v`: Enable verbose logging
- `--dataset, -d`: Specify dataset (required for main scripts)

### File Formats
- **Candidates**: JSON with embedding vectors and code solutions
- **Models**: PyTorch checkpoint files (.pt)
- **Results**: JSON with detailed accuracy metrics and metadata
- **Logs**: JSON with training history and metrics

---

This guide provides complete instructions for using MathCoRL's In-Context Reinforcement Learning pipeline. For additional technical details, refer to the source code documentation in the `mint/` directory. 