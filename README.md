# MathCoRL: Mathematical Intelligence with Function Prototype Prompting

A powerful system that uses Function Prototype Prompting (FPP) to solve mathematical word problems by generating and executing Python code with predefined mathematical functions.

## 🚀 Overview

**MathCoRL** combines the MINT (Mathematical Intelligence) library with Function Prototype Prompting to create a robust mathematical problem-solving system that:

- **Provides structured function prototypes** to Large Language Models (LLMs)
- **Generates executable Python code** to solve mathematical problems  
- **Uses Python built-in functions** for reliable and familiar computation
- **Supports multiple datasets** like SVAMP, GSM8K, FinQA, TabMWP, and TAT-QA
- **Integrates with LangChain** for advanced debugging and tracing
- **Includes dataset testing tools** with progress tracking and accuracy calculation
- **Features advanced tolerance functions** for semantic equivalence evaluation

## ✨ Features

- 🧮 **25+ Mathematical Functions** - Python built-ins + custom statistical functions
- 🤖 **LangChain Integration** - Advanced LLM interaction with LangSmith tracing
- 📊 **Multiple Dataset Support** - Built-in support for 5 popular math datasets
- 🔍 **Easy Installation** - Local library installation with automated setup
- 🎯 **High Accuracy** - Structured approach improves problem-solving reliability
- 🔒 **Safe Code Execution** - Controlled environment with predefined functions
- 🎮 **Demo Mode** - Test without API keys using mock responses
- 🖥️ **CLI Interface** - Command-line tools for easy usage
- 📈 **Dataset Testing** - Comprehensive testing tools with progress bars and accuracy metrics
- 🎯 **Smart Evaluation** - Advanced tolerance functions for semantic equivalence

## 📁 Project Structure

```
MathCoRL/
├── mint/                       # MINT Library (Mathematical Intelligence)
│   ├── __init__.py            # Library exports
│   ├── core.py                # Core FPP implementation
│   ├── functions.py           # Mathematical functions
│   ├── prompts.py             # Prompt management
│   ├── utils.py               # Utility functions & dataset loaders
│   ├── config.py              # Configuration management
│   └── cli.py                 # Command line interface
├── fpp.py                     # Simple FPP script
├── fpp_prompting.py           # Advanced dataset testing tool
├── install_mint.py            # Installation script
├── setup.py                   # Package setup
├── env.example                # Environment configuration template
├── requirements.txt           # Dependencies
├── templates/
│   ├── function_prototypes.txt # Mathematical function definitions
│   └── fpp.txt                # FPP prompt template
├── results/                   # Test results directory
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

### 3. Dataset Testing

**Test individual datasets:**
```bash
# Test SVAMP dataset (first 50 samples)
python fpp_prompting.py SVAMP --limit 50

# Test GSM8K with verbose output
python fpp_prompting.py GSM8K --limit 100 -v

# Test FinQA and save to custom directory
python fpp_prompting.py FinQA --limit 100 --output my_results

# Test without saving results
python fpp_prompting.py TabMWP --limit 20 --no-save
```

### 4. Use as Library

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

## 📊 Dataset Support & Performance

MINT supports multiple mathematical datasets with specialized evaluation functions:

| Dataset | Description | Accuracy | Special Features |
|---------|-------------|----------|------------------|
| **SVAMP** | Simple math word problems | 96-100% | Basic arithmetic problems |
| **GSM8K** | Grade school math problems | 94% | Multi-step reasoning |
| **TabMWP** | Tabular math word problems | 90-100% | Table processing + fraction handling |
| **TAT-QA** | Table-and-text QA problems | 80%+ | Special rounding tolerance (±0.05) |
| **FinQA** | Financial reasoning problems | 89% | Semantic equivalence (0.92 ≈ 92%) |

### Dataset-Specific Tolerance Functions

- **Standard**: `±1e-3` tolerance for floating point precision
- **TAT-QA**: Enhanced tolerance with rounding checks (0-3 decimal places)
- **FinQA**: Candidate generation system for semantic equivalence:
  - Handles percentage conversion (val/100, val*100)
  - Multiple rounding levels (0-3 decimal places)
  - Prioritizes semantic meaning over exact numerical match

### Advanced Evaluation Features

- **Progress Tracking**: Real-time progress bars with tqdm
- **Automatic Accuracy Calculation**: Built-in evaluation metrics
- **Result Storage**: JSON format with generated code included
- **Flexible Testing**: Configurable sample limits and output options

## ⚙️ Configuration

Create a `.env` file from the template:
```bash
cp env.example .env
```

### Required Settings
```env
OPENAI_API_KEY=your_openai_api_key
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

## 🎯 Advanced Usage Examples

### Dataset Testing with Different Configurations

```bash
# Quick test on small sample
python fpp_prompting.py SVAMP --limit 10 -v

# Comprehensive evaluation
python fpp_prompting.py FinQA --limit 100

# Test all datasets
for dataset in SVAMP GSM8K TabMWP TAT-QA FinQA; do
    python fpp_prompting.py $dataset --limit 50
done
```

### Using Dataset Loaders in Python

```python
from mint.utils import load_svamp_test_data, load_finqa_test_data

# Load SVAMP data
svamp_data = load_svamp_test_data('datasets/SVAMP/test.json')
print(f"Loaded {len(svamp_data)} SVAMP problems")

# Load FinQA data  
finqa_data = load_finqa_test_data('datasets/FinQA/test.json')
print(f"Loaded {len(finqa_data)} FinQA problems")

# Process individual problems
for problem in svamp_data[:5]:
    print(f"Q: {problem['question']}")
    print(f"A: {problem['ground_truth']}")
```

### Custom Tolerance Functions

```python
from mint.utils import FinQA_generate_candidates

# Generate semantic equivalents for FinQA evaluation
value = 0.92
candidates = FinQA_generate_candidates(value)
print(candidates)  # [0.92, 1.0, 0.9, 0.92, 0.920, 0.01, 0.0, 0.01, 0.010, 92.0, 92.0, 92.0, 92.000]

# This allows 0.92 to match with 92% semantically
```

## 📈 Testing and Evaluation

The system includes comprehensive testing capabilities:

### Test Results Format
```json
{
    "dataset": "FinQA",
    "total_samples": 100,
    "correct_predictions": 89,
    "accuracy": 89.0,
    "results": [
        {
            "question": "what is the average payment volume per transaction?",
            "context": "...",
            "ground_truth": 127.4,
            "result": 127.4,
            "code": "american_express_payment_volume = 637\n...",
            "correct": true
        }
    ]
}
```

### Key Metrics
- **Accuracy**: Percentage of correct predictions
- **Code Generation**: Python code is saved for analysis
- **Error Analysis**: Failed cases are tracked for debugging
- **Semantic Matching**: Advanced tolerance functions for realistic evaluation

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
from mint.utils import load_svamp_test_data
from mint import FunctionPrototypePrompting

# Load dataset
data = load_svamp_test_data('datasets/SVAMP/test.json')
fpp = FunctionPrototypePrompting()

# Process multiple problems
correct = 0
total = 0

for item in data[:10]:  # Test first 10 problems
    question = item['question']
    context = item.get('context', '')
    predicted = fpp.solve(question, context, show_code=False)
    actual = item['ground_truth']
    
    if abs(predicted - actual) < 0.01:
        correct += 1
    total += 1

accuracy = correct / total
print(f"Accuracy: {accuracy:.2%}")
```

## 🔧 CLI Commands

The system provides multiple command-line interfaces:

### Basic FPP Script
```bash
# Simple problem solving
python fpp.py --question "What is 15 * 8?"
python fpp.py --question "Calculate 20% of 150"
```

### Advanced Dataset Testing
```bash
# Test specific datasets
python fpp_prompting.py SVAMP --limit 50
python fpp_prompting.py GSM8K --limit 100 -v
python fpp_prompting.py FinQA --limit 100

# Comprehensive testing
python fpp_prompting.py TAT-QA --limit 50 --output custom_results
python fpp_prompting.py TabMWP --limit 20 --no-save
```

### MINT Library CLI
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
- Performance metrics

### Performance Metrics

Our comprehensive testing across multiple datasets shows:

| Dataset | Accuracy | Problem Type | Notes |
|---------|----------|--------------|-------|
| **SVAMP** | 96-100% | Simple arithmetic word problems | Excellent performance on basic math |
| **GSM8K** | 94% | Grade school math problems | Strong multi-step reasoning |
| **TabMWP** | 90-100% | Table-based math problems | Good table processing abilities |
| **TAT-QA** | 80%+ | Complex table & text reasoning | Advanced rounding tolerance |
| **FinQA** | 89% | Financial calculations | Semantic equivalence matching |

### Key Improvements
- **Smart Tolerance Functions**: Dataset-specific evaluation criteria
- **Semantic Equivalence**: FinQA handles percentage format variations (0.92 ≈ 92%)
- **Progress Tracking**: Real-time progress bars for large dataset testing
- **Comprehensive Logging**: Detailed results with generated code storage

## 🐛 Troubleshooting

### Common Issues

**1. Import Error: No module named 'mint'**
```bash
# Reinstall the library
python install_mint.py

# Verify installation
python -c "import mint; print('MINT installed successfully')"
```

**2. OpenAI API Error**
```bash
# Check your .env file
cat .env
# Verify API key is correct and has sufficient credits
```

**3. Dataset Loading Issues**
```bash
# Ensure datasets are in correct location
ls datasets/SVAMP/test.json
ls datasets/GSM8K/test.jsonl
ls datasets/FinQA/test.json

# Check file permissions
chmod 644 datasets/*/*.json*
```

**4. Low Accuracy on Custom Problems**
- Ensure questions are clearly formulated
- Provide sufficient context for complex problems
- Use appropriate dataset-specific tolerance functions

**5. Progress Bar Issues**
```bash
# Install tqdm if missing
pip install tqdm
```

### Debug Mode
Enable detailed logging:
```env
DEBUG_MODE=true
LOG_LEVEL=DEBUG
```

### Testing Your Setup
```bash
# Quick verification
python fpp_prompting.py SVAMP --limit 5 -v

# Full system test
python fpp_prompting.py GSM8K --limit 10
```

## 🤝 Contributing

We welcome contributions! Here's how to get started:

1. **Fork the repository**
2. **Create a feature branch**: `git checkout -b feature-name`
3. **Make your changes**:
   - Add new dataset loaders in `mint/utils.py`
   - Implement custom tolerance functions
   - Improve mathematical function coverage
4. **Test your changes**:
   ```bash
   python fpp_prompting.py SVAMP --limit 10
   python demo_fpp.py
   ```
5. **Commit changes**: `git commit -am 'Add feature: description'`
6. **Push to branch**: `git push origin feature-name`
7. **Submit a pull request**

### Development Areas
- Additional mathematical datasets
- Enhanced tolerance functions
- New mathematical function prototypes
- Performance optimizations
- Documentation improvements

## 📝 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🙏 Acknowledgments

- **OpenAI** for GPT models and API
- **LangChain** for LLM integration framework
- **Mathematical Datasets**:
  - SVAMP: Simple math word problems
  - GSM8K: Grade school mathematics
  - FinQA: Financial reasoning questions
  - TabMWP: Tabular math word problems  
  - TAT-QA: Table-and-text question answering
- **Function Prototype Prompting** methodology
- **tqdm** for progress tracking
- **Research Community** for mathematical reasoning benchmarks

## 🚀 Future Roadmap

- [ ] Additional dataset support (MATH, MathQA, etc.)
- [ ] Enhanced mathematical function library
- [ ] Multi-language support
- [ ] Web interface for easy testing
- [ ] Batch processing optimizations
- [ ] Advanced error analysis tools

---

**MathCoRL** - Mathematical Intelligence with Function Prototype Prompting

*Empowering Large Language Models with structured mathematical reasoning capabilities* 