# MathCoRL: Mathematical Intelligence with Function Prototype Prompting

A powerful system that uses Function Prototype Prompting (FPP) to solve mathematical word problems by generating and executing Python code with predefined mathematical functions.

## 🚀 Overview

**MathCoRL** combines the MINT (Mathematical Intelligence) library with Function Prototype Prompting to create a robust mathematical problem-solving system that:

- **Provides structured function prototypes** to Large Language Models (LLMs)
- **Generates executable Python code** to solve mathematical problems  
- **Uses Python built-in functions** for reliable and familiar computation
- **Supports multiple datasets** like SVAMP, GSM8K, FinQA, TabMWP, and TAT-QA
- **Integrates with LangChain** for advanced debugging and tracing

## ✨ Features

- 🧮 **25+ Mathematical Functions** - Python built-ins + custom statistical functions
- 🤖 **LangChain Integration** - Advanced LLM interaction with LangSmith tracing
- 📊 **Multiple Dataset Support** - Built-in support for popular math datasets
- 🔍 **Easy Installation** - Local library installation with automated setup
- 🎯 **High Accuracy** - Structured approach improves problem-solving reliability
- 🔒 **Safe Code Execution** - Controlled environment with predefined functions
- 🎮 **Demo Mode** - Test without API keys using mock responses
- 🖥️ **CLI Interface** - Command-line tools for easy usage

## 📁 Project Structure

```
MathCoRL/
├── mint/                       # MINT Library (Mathematical Intelligence)
│   ├── __init__.py            # Library exports
│   ├── core.py                # Core FPP implementation
│   ├── functions.py           # Mathematical functions
│   ├── prompts.py             # Prompt management
│   ├── utils.py               # Utility functions
│   ├── config.py              # Configuration management
│   └── cli.py                 # Command line interface
├── fpp.py                     # Simple FPP script
├── demo_fpp.py                # Demo script (no API key required)
├── install_mint.py            # Installation script
├── setup.py                   # Package setup
├── env.example                # Environment configuration template
├── requirements.txt           # Dependencies
├── templates/
│   ├── function_prototypes.txt # Mathematical function definitions
│   └── fpp.txt                # FPP prompt template
└── datasets/                  # Mathematical datasets
    ├── SVAMP/                 # SVAMP dataset
    ├── GSM8K/                 # GSM8K dataset  
    ├── FinQA/                 # FinQA dataset
    ├── TabMWP/                # TabMWP dataset
    └── TAT-QA/                # TAT-QA dataset
```

## 🛠️ Installation

### Prerequisites

- Python 3.8+
- OpenAI API key (for real testing)
- LangChain API key (optional, for LangSmith tracing)

### Quick Setup

1. **Clone the repository**
```bash
git clone <repository-url>
cd MathCoRL
```

2. **Create virtual environment**
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. **Install dependencies and MINT library**
```bash
pip install -r requirements.txt
python install_mint.py
```

4. **Configure environment**
```bash
cp env.example .env
# Edit .env file with your API keys
```

### Verify Installation

Test the installation:
```bash
# Test CLI
mint-fpp solve "What is 5 + 3?"

# Test Python import
python -c "from mint import solve_math_problem; print(solve_math_problem('What is 10 + 5?'))"
```

## 🚀 Quick Start

### 1. Demo Mode (No API Key Required)

Test the system with mock responses:
```bash
python demo_fpp.py
```

### 2. Simple Usage

**Command Line:**
```bash
# Basic calculation
mint-fpp solve "What is 15 + 27?"

# Word problem
mint-fpp solve "John has 20 apples. He gives 8 to his friend. How many does he have left?"

# With context
mint-fpp solve "Calculate the total" --context "Items: 5, 10, 15"
```

**Python Script:**
```bash
python fpp.py --question "What is 25 * 4?"
```

### 3. Use as Library

```python
from mint import solve_math_problem, FunctionPrototypePrompting

# Simple function interface
result = solve_math_problem("There are 15 apples. If you eat 3, how many are left?")
print(f"Answer: {result}")  # Answer: 12

# Advanced usage with class
fpp = FunctionPrototypePrompting()
result = fpp.solve("Calculate the average of 10, 20, 30")
print(f"Average: {result}")  # Average: 20.0
```

## ⚙️ Configuration

Create a `.env` file from the template:
```bash
cp env.example .env
```

### Required Settings
```env
OPENAI_API_KEY=your_openai_api_key_here
```

### Optional Settings  
```env
# LangSmith (for debugging and tracing)
LANGCHAIN_API_KEY=your_langchain_api_key
LANGCHAIN_TRACING_V2=true
LANGCHAIN_PROJECT=MathCoRL-FPP

# Model Settings
DEFAULT_MODEL=gpt-4
TEMPERATURE=0.0
MAX_TOKENS=1000

# Debug Settings
LOG_LEVEL=INFO
DEBUG_MODE=false
```

## 🧮 Mathematical Functions

MINT preserves Python's built-in functions and adds custom mathematical capabilities:

### Python Built-ins (Direct Usage)
- `min()`, `max()` - Minimum/Maximum values
- `abs()` - Absolute value
- `round()` - Round numbers
- `sum()` - Sum of sequences
- `len()` - Count elements

### Custom Functions
- `add(a, b)`, `sub(a, b)`, `mul(a, b)`, `div(a, b)` - Basic arithmetic
- `mean(numbers)` - Calculate average
- `median(numbers)` - Find median
- `mode(numbers)` - Most frequent value
- `floor(a)`, `ceil(a)` - Rounding functions
- `percentage(part, whole)` - Calculate percentage
- `gcd(a, b)`, `lcm(a, b)` - Greatest common divisor/Least common multiple

### Example Usage in Generated Code
```python
# LLM generates code like this:
numbers = [10, 20, 30, 40]
total = sum(numbers)           # Uses Python built-in
average = mean(numbers)        # Uses custom function  
largest = max(numbers)         # Uses Python built-in
result = total + average
```

## 📊 Dataset Support

MINT supports multiple mathematical datasets with built-in loaders:

### SVAMP Dataset
- **Description:** Simple math word problems
- **Fields:** ID, Body, Question, Answer, Type
- **Usage:** `load_svamp_data('datasets/SVAMP/test.json')`

### GSM8K Dataset  
- **Description:** Grade school math problems
- **Fields:** Question, Answer
- **Usage:** `load_gsm8k_data('datasets/GSM8K/test.jsonl')`

### FinQA Dataset
- **Description:** Financial reasoning problems
- **Fields:** Question, Answer, Program
- **Usage:** `load_finqa_data('datasets/FinQA/test.json')`

### TabMWP Dataset
- **Description:** Tabular math word problems
- **Fields:** Question, Answer, Table
- **Usage:** `load_tabmwp_data('datasets/TabMWP/test.json')`

### TAT-QA Dataset
- **Description:** Table-and-text QA problems
- **Fields:** Question, Answer, Table, Paragraphs
- **Usage:** `load_tatqa_data('datasets/TAT-QA/test.json')`

## 🎯 Usage Examples

### Basic Math Problems
```python
from mint import solve_math_problem

# Simple arithmetic
result = solve_math_problem("What is 123 + 456?")
print(result)  # 579

# Word problems
result = solve_math_problem(
    "Sarah has 25 stickers. She gives 8 to her brother and buys 12 more. How many stickers does she have now?"
)
print(result)  # 29
```

### Complex Problems with Context
```python
from mint import FunctionPrototypePrompting

fpp = FunctionPrototypePrompting()

# With additional context
context = "The store sells apples for $2 each and oranges for $3 each."
question = "If someone buys 5 apples and 3 oranges, what's the total cost?"
result = fpp.solve(question, context)
print(result)  # 19
```

### Batch Processing
```python
from mint import solve_math_problem, load_svamp_data

# Load dataset
data = load_svamp_data('datasets/SVAMP/test.json')

# Process multiple problems
correct = 0
total = 0

for item in data[:10]:  # Test first 10 problems
    question = f"{item['Body']} {item['Question']}"
    predicted = solve_math_problem(question)
    actual = float(item['Answer'])
    
    if abs(predicted - actual) < 0.01:
        correct += 1
    total += 1

accuracy = correct / total
print(f"Accuracy: {accuracy:.2%}")
```

## 🔧 CLI Commands

The MINT library provides command-line tools:

```bash
# Solve math problems
mint-fpp solve "What is 15 * 8?"
mint-fpp solve "Calculate 20% of 150"

# With context
mint-fpp solve "Find the total" --context "Values: 10, 20, 30"

# Help
mint-fpp --help
mint-fpp solve --help
```

## 🎮 Demo and Testing

### Run Demo
```bash
python demo_fpp.py
```

The demo shows:
- Sample math problems
- Generated Python code
- Execution results
- Accuracy metrics (~67% on test problems)

### Performance Metrics
- **SVAMP Dataset:** ~70% accuracy
- **Simple Arithmetic:** ~95% accuracy  
- **Word Problems:** ~65-75% accuracy
- **Multi-step Problems:** ~60-70% accuracy

## 🐛 Troubleshooting

### Common Issues

**1. Import Error: No module named 'mint'**
```bash
# Reinstall the library
python install_mint.py
```

**2. OpenAI API Error**
```bash
# Check your .env file
cat .env
# Verify API key is correct
```

**3. Command not found: mint-fpp**
```bash
# Ensure virtual environment is activated
source venv/bin/activate
# Reinstall
python install_mint.py
```

**4. LangSmith Tracing Issues**
```bash
# LangSmith is optional, you can disable it
# Remove LANGCHAIN_TRACING_V2=true from .env
```

### Debug Mode
Enable debug mode for detailed logging:
```env
DEBUG_MODE=true
LOG_LEVEL=DEBUG
```

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch: `git checkout -b feature-name`
3. Make your changes
4. Run tests: `python demo_fpp.py`
5. Commit changes: `git commit -am 'Add feature'`
6. Push to branch: `git push origin feature-name`
7. Submit a pull request

## 📝 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🙏 Acknowledgments

- OpenAI for GPT models
- LangChain for LLM integration
- Mathematical datasets: SVAMP, GSM8K, FinQA, TabMWP, TAT-QA
- Function Prototype Prompting methodology

---

**MathCoRL** - Mathematical Reasoning with Function Prototype Prompting 