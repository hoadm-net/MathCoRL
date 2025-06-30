# MathCoRL - Mathematical Intelligence Library

A comprehensive library for mathematical problem solving using advanced prompting methods. This project implements and compares five state-of-the-art techniques for mathematical reasoning with Large Language Models.

## 🎯 Overview

MathCoRL provides a unified framework for mathematical problem solving using five different prompting strategies:

1. **FPP (Function Prototype Prompting)** - *Our novel method*
2. **CoT (Chain-of-Thought)** - Step-by-step reasoning baseline
3. **PoT (Program of Thoughts)** - Code generation baseline  
4. **Zero-Shot** - Direct solving baseline
5. **PAL (Program-aided Language Models)** - Reasoning + code hybrid

## 📊 Methods Comparison

| Method | Type | Reasoning | Code Generation | Accuracy | Interpretability |
|--------|------|-----------|----------------|----------|-----------------|
| **FPP** | Novel | Function-guided | ✅ With prototypes | High | High |
| **CoT** | Baseline | Natural language | ❌ | Good | High |
| **PoT** | Baseline | Minimal | ✅ Pure code | High | Low |
| **Zero-Shot** | Baseline | None | ❌ | Basic | Low |
| **PAL** | Baseline | Natural + Code | ✅ Hybrid | High | Medium |

## 🚀 Quick Start

### Installation

```bash
# Clone the repository
git clone https://github.com/mathcorl/MathCoRL.git
cd MathCoRL

# Install dependencies
pip install -r requirements.txt

# Install in development mode
pip install -e .

# Set up environment
cp env.example .env
# Edit .env and add your OpenAI API key
```

### Basic Usage

#### Single Problem Solving

```bash
# Using the unified interface
python mathcorl.py solve --method fpp --question "What is 15 + 27?"
python mathcorl.py solve --method cot --question "John has 20 apples. He gives 8 to his friend. How many are left?"
python mathcorl.py solve --method pot --question "Calculate the average of 10, 20, 30"
python mathcorl.py solve --method zero_shot --question "What is 5 * 6?"
python mathcorl.py solve --method pal --question "A pizza is cut into 8 slices. If 3 are eaten, how many remain?"
```

#### Interactive Mode

```bash
python mathcorl.py interactive
```

Choose from 5 methods:
- 1: FPP (Function Prototype Prompting)
- 2: CoT (Chain-of-Thought) 
- 3: PoT (Program of Thoughts)
- 4: Zero-Shot
- 5: PAL (Program-aided Language Models)

#### Dataset Testing

```bash
# Test specific method on dataset
python mathcorl.py test --method fpp --dataset SVAMP --limit 100
python mathcorl.py test --method cot --dataset GSM8K --limit 50
python mathcorl.py test --method pot --dataset TabMWP --limit 30

# Compare methods
python mathcorl.py compare --dataset SVAMP --limit 20
```

## 🧠 Methods Explained

### 1. FPP (Function Prototype Prompting) - *Our Method*

**Novel approach** that guides LLMs to generate code using predefined function prototypes.

**Key Features:**
- Pre-defined mathematical function library
- Structured code generation
- High accuracy and interpretability
- Robust error handling

**Example:**
```python
from mint import FunctionPrototypePrompting

fpp = FunctionPrototypePrompting()
result = fpp.solve("John has 25 marbles. He gives 7 to his friend.")
# Generates code like: sub(25, 7)
```

### 2. CoT (Chain-of-Thought)

**Baseline method** that generates step-by-step reasoning in natural language.

**Key Features:**
- Human-like reasoning steps
- High interpretability
- Good for understanding problem logic
- No code execution

**Example:**
```python
from mint import ChainOfThoughtPrompting

cot = ChainOfThoughtPrompting()
result = cot.solve("What is 15 + 27?")
# Generates: "Step 1: Add 15 and 27..."
```

### 3. PoT (Program of Thoughts)

**Baseline method** that generates Python code to solve numerical problems.

**Key Features:**
- Pure code generation
- High computational accuracy
- Minimal reasoning text
- Direct execution

**Example:**
```python
from mint import ProgramOfThoughtsPrompting

pot = ProgramOfThoughtsPrompting()
result = pot.solve("Calculate average of 10, 20, 30")
# Generates: "answer = (10 + 20 + 30) / 3"
```

### 4. Zero-Shot

**Simple baseline** that asks the model to solve problems directly without examples.

**Key Features:**
- No reasoning guidance
- Fastest method
- Basic accuracy
- Good baseline for comparison

**Example:**
```python
from mint import ZeroShotPrompting

zs = ZeroShotPrompting()
result = zs.solve("What is 5 * 6?")
# Direct answer: "30"
```

### 5. PAL (Program-aided Language Models)

**Hybrid method** that combines natural language reasoning with code generation.

**Key Features:**
- Both reasoning and code
- Interpretable + accurate
- Best of both worlds
- Two-stage process

**Example:**
```python
from mint import ProgramAidedLanguageModel

pal = ProgramAidedLanguageModel()
result = pal.solve("A train travels 120 miles in 2 hours. What is its speed?")
# Generates reasoning + code: speed = distance / time
```

## 📚 Supported Datasets

The library supports comprehensive testing on multiple mathematical reasoning datasets:

- **SVAMP** - Simple math word problems
- **GSM8K** - Grade school math problems  
- **TabMWP** - Tabular math word problems
- **TAT-QA** - Table-based Q&A
- **FinQA** - Financial reasoning

Each dataset uses appropriate tolerance functions for accurate evaluation.

See [README_DATASETS.md](README_DATASETS.md) for detailed dataset information.

## 🏗️ Architecture

### Unified Framework

```
MathCoRL/
├── mint/                    # Core library
│   ├── core.py             # FPP implementation
│   ├── cot.py              # Chain-of-Thought
│   ├── pot.py              # Program of Thoughts  
│   ├── zero_shot.py        # Zero-Shot prompting
│   ├── pal.py              # Program-aided LM
│   ├── cli.py              # Unified CLI interface
│   ├── testing.py          # Testing framework
│   ├── evaluation.py       # Evaluation metrics
│   └── utils.py            # Utility functions
├── mathcorl.py             # Main unified interface
└── datasets/               # Test datasets
```

### Configuration Management

All methods use consistent configuration from `.env`:

```bash
# Model Configuration
DEFAULT_MODEL=gpt-3.5-turbo
TEMPERATURE=0.1
MAX_TOKENS=1000

# OpenAI API
OPENAI_API_KEY=your_key_here
```

## 🔧 API Reference

### Python API

```python
# Import all methods
from mint import (
    FunctionPrototypePrompting,        # FPP
    ChainOfThoughtPrompting,           # CoT  
    ProgramOfThoughtsPrompting,        # PoT
    ZeroShotPrompting,                 # Zero-Shot
    ProgramAidedLanguageModel,         # PAL
    solve_math_problem,                # Convenience function
)

# Solve problems
fpp = FunctionPrototypePrompting()
result = fpp.solve_detailed("Your question here")

# Testing framework
from mint import TestRunner, create_fpp_solver

solver = create_fpp_solver()
runner = TestRunner('FPP', solver)
results = runner.test_dataset('SVAMP', limit=10)
```

### CLI Reference

```bash
# Unified interface
python mathcorl.py [command] [options]

Commands:
  solve        Solve a single problem
  test         Test method on dataset  
  compare      Compare methods
  interactive  Interactive mode
  datasets     List available datasets

# Examples
python mathcorl.py solve --method fpp --question "What is 15 + 27?"
python mathcorl.py test --method cot --dataset SVAMP --limit 50
python mathcorl.py compare --dataset GSM8K --limit 20
python mathcorl.py interactive
```

## 🧪 Testing and Evaluation

### Running Tests

```bash
# Test single method
python mathcorl.py test --method fpp --dataset SVAMP --limit 100

# Test all methods
for method in fpp cot pot zero_shot pal; do
    python mathcorl.py test --method $method --dataset SVAMP --limit 50
done

# Compare methods
python mathcorl.py compare --dataset SVAMP --limit 20
```

### Custom Evaluation

```python
from mint import TestRunner, create_fpp_solver

# Create custom test runner
solver = create_fpp_solver()
runner = TestRunner('FPP', solver)

# Test on custom dataset
results = runner.test_dataset('SVAMP', limit=100, verbose=True)

# Access detailed results
accuracy = results['accuracy']
correct_count = results['correct_predictions']
detailed_results = results['results']
```

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🤝 Contributing

We welcome contributions! Please see our contributing guidelines for details.