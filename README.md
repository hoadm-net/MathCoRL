# MathCoRL: Mathematical Intelligence with Advanced Prompting Methods

A powerful system that implements **Function Prototype Prompting (FPP)** and **Chain-of-Thought (CoT)** prompting to solve mathematical word problems through different reasoning approaches.

## 🚀 Overview

**MathCoRL** combines the MINT (Mathematical Intelligence) library with two complementary prompting methodologies:

### 🔧 Function Prototype Prompting (FPP)
- **Provides structured function prototypes** to Large Language Models (LLMs)
- **Generates executable Python code** to solve mathematical problems  
- **Uses Python built-in functions** for reliable and familiar computation
- **High accuracy** through structured code generation

### 🧠 Chain-of-Thought (CoT) Prompting  
- **Step-by-step reasoning** with explicit thinking process
- **Few-shot examples** to guide mathematical problem solving
- **Natural language explanations** for transparent reasoning
- **Strong performance** on word problems through logical decomposition

Both methods support **multiple datasets** like SVAMP, GSM8K, FinQA, TabMWP, and TAT-QA with **dataset-specific tolerance functions** for accurate evaluation.

## 📦 Installation

```bash
# Clone the repository
git clone https://github.com/your-username/MathCoRL.git
cd MathCoRL

# Create and activate virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Set up environment variables
cp env.example .env
# Edit .env with your OpenAI API key
```

## 🔧 CLI Commands

The system provides multiple command-line interfaces for different use cases:

### Function Prototype Prompting (FPP)

#### Basic FPP Script
```bash
# Simple problem solving
python fpp.py --question "What is 15 + 27?"
python fpp.py --question "John has 20 apples..." --context "Additional info"
```

#### FPP Dataset Testing
```bash
# Test on different datasets
python fpp_prompting.py SVAMP --limit 50
python fpp_prompting.py GSM8K --limit 100 -v
python fpp_prompting.py FinQA --limit 100 --output results
python fpp_prompting.py TAT-QA --limit 50
python fpp_prompting.py TabMWP --limit 100
```

### Chain-of-Thought (CoT) Prompting

#### Basic CoT Script
```bash
# Simple problem solving with reasoning
python cot.py --question "What is 15 + 27?"
python cot.py --question "John has 20 apples. He gives 8 to his friend. How many does he have left?"
python cot.py --question "Calculate the average" --context "Numbers: 10, 20, 30"

# Hide reasoning steps
python cot.py --question "What is 15 + 27?" --no-reasoning
```

#### CoT Dataset Testing
```bash
# Test on different datasets with CoT
python cot_prompting.py SVAMP --limit 50
python cot_prompting.py GSM8K --limit 100 -v  
python cot_prompting.py FinQA --limit 100 --output cot_results
python cot_prompting.py TAT-QA --limit 50
python cot_prompting.py TabMWP --limit 100
```

## 📊 Performance Comparison

| Dataset | FPP Accuracy | CoT Accuracy | Best Method |
|---------|-------------|-------------|------------|
| SVAMP   | 96-100%     | 74-90%      | **FPP** ✨ |
| GSM8K   | 94%         | 64%         | **FPP** ✨ |
| TabMWP  | 90-100%     | ~70%*       | **FPP** ✨ |
| TAT-QA  | 80%+        | ~65%*       | **FPP** ✨ |
| FinQA   | 89%         | 64%         | **FPP** ✨ |

*CoT results on some datasets are preliminary

### Key Insights:
- **FPP excels** at structured mathematical computation through code generation
- **CoT provides** transparent reasoning but may struggle with complex calculations  
- **FPP is faster** for computational problems
- **CoT is better** for understanding reasoning process

## 🎯 Advanced Features

### Dataset-Specific Tolerance Functions

The system implements specialized evaluation methods for different datasets:

- **Standard tolerance** (±1e-3): SVAMP, GSM8K
- **TAT-QA tolerance**: Handles rounding with ±0.05 and percentage calculations
- **FinQA semantic equivalence**: Uses candidate generation (0.92 ≈ 92%)

### Supported Datasets

1. **SVAMP** - Simple word problems with arithmetic operations
2. **GSM8K** - Grade school math word problems  
3. **TabMWP** - Table-based math word problems with markdown formatting
4. **TAT-QA** - Table and text question answering with financial calculations
5. **FinQA** - Financial question answering with complex percentage formats

## 💻 Programming Interface

### Function Prototype Prompting
```python
from mint import FunctionPrototypePrompting

# Initialize FPP
fpp = FunctionPrototypePrompting()

# Solve single problem
result = fpp.solve("John has 10 apples. He buys 5 more. How many does he have?")
print(f"Answer: {result}")

# Batch processing
from mint.utils import load_svamp_test_data
data = load_svamp_test_data('datasets/SVAMP/test.json')

for item in data[:10]:
    question = item['question']
    context = item.get('context', '')
    predicted = fpp.solve_single(question, context, show_code=False)
    actual = item['ground_truth']
    print(f"Predicted: {predicted}, Actual: {actual}")
```

### Chain-of-Thought Prompting
```python
from mint.cot import ChainOfThoughtPrompting

# Initialize CoT
cot = ChainOfThoughtPrompting()

# Solve with full reasoning
result = cot.solve("John has 10 apples. He buys 5 more. How many does he have?")
print(f"Answer: {result['result']}")
print(f"Reasoning: {result['reasoning']}")

# Solve silently
result = cot.solve_silent("What is 15 + 27?")
print(f"Answer: {result['result']}")
```

## 🔧 Troubleshooting

### Common Issues

1. **OpenAI API Key**: Ensure your `.env` file contains `OPENAI_API_KEY=your_key_here`
2. **Virtual Environment**: Always activate venv before running scripts
3. **Dependencies**: Install all requirements with `pip install -r requirements.txt`
4. **Large Files**: Some datasets are large; use `--limit` for testing

### Performance Tips

- Use `--limit` parameter for quick testing
- Set appropriate timeout for API calls
- Monitor API usage and costs
- Use `--verbose` for debugging failed cases

## 📈 Development

### Project Structure
```
MathCoRL/
├── mint/                    # Core library
│   ├── core.py             # FPP implementation  
│   ├── cot.py              # CoT implementation
│   ├── utils.py            # Dataset loaders and utilities
│   ├── functions.py        # Mathematical function prototypes
│   └── prompts.py          # Prompt templates
├── datasets/               # Mathematical datasets
│   ├── SVAMP/
│   ├── GSM8K/
│   ├── FinQA/
│   ├── TabMWP/
│   └── TAT-QA/
├── results/                # Test results and outputs
├── fpp_prompting.py       # FPP dataset testing tool
├── cot_prompting.py       # CoT dataset testing tool  
├── fpp.py                 # Simple FPP script
├── cot.py                 # Simple CoT script
└── requirements.txt       # Dependencies
```

### Contributing

1. Fork the repository
2. Create feature branch (`git checkout -b feature/amazing-feature`)
3. Commit changes (`git commit -m 'Add amazing feature'`)
4. Push to branch (`git push origin feature/amazing-feature`)
5. Open Pull Request

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- OpenAI for GPT models
- LangChain for LLM integration
- The mathematical reasoning research community
- Contributors to the datasets used in evaluation 