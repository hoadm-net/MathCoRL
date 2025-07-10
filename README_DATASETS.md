# Datasets in MathCoRL

MathCoRL hỗ trợ 5 datasets cho mathematical reasoning research, mỗi dataset có đặc thù riêng về độ phức tạp và domain:

## 📊 **Dataset Overview**

| Dataset | Domain | Train | Test | Description | ICL k | Policy Status |
|---------|--------|-------|------|-------------|-------|---------------|
| **GSM8K** | Elementary Math | 7.5K | 1.3K | Grade School Math word problems | 2 | Available |
| **SVAMP** | Arithmetic Variations | 800 | 300 | Simple arithmetic with linguistic variations | 2 | Available |
| **TabMWP** | Tabular Math | 28.9K | 4.5K | Math problems involving tables and charts | 2 | Available |
| **TAT-QA** | Financial QA | 13.8K | 2.2K | Table-and-text QA for financial documents | 3 | Available |
| **FinQA** | Financial Analysis | 6.3K | 1.1K | Complex financial reasoning and calculations | 2 | Available |

## 🎯 **Dataset Details**

### 1. **GSM8K** - Grade School Math
**Domain**: Elementary Mathematics
**Complexity**: Basic to Intermediate

#### **Characteristics**
- **Problem Types**: Word problems involving basic arithmetic operations
- **Reasoning**: Multi-step calculations with real-world contexts
- **Language**: Natural, conversational problem statements
- **Solution Format**: Step-by-step numerical solutions

#### **Sample Problem Structure**
```json
{
  "question": "John has 20 apples. He gives 8 to his friend and buys 15 more. How many apples does he have now?",
  "answer": "20 - 8 + 15 = 27",
  "numerical_answer": "27"
}
```

#### **Research Applications**
- **Prompting Method Comparison**: Evaluate different reasoning approaches
- **ICL Example Selection**: Test different strategies for selecting helpful examples
- **Error Analysis**: Study failure modes in basic mathematical reasoning

### 2. **SVAMP** - Simple Variations on Arithmetic Math word Problems
**Domain**: Arithmetic with Linguistic Variations
**Complexity**: Basic

#### **Characteristics**
- **Problem Types**: Simple arithmetic with varied linguistic expressions
- **Focus**: Same mathematical operations presented with different wordings
- **Challenge**: Natural language understanding rather than complex computation
- **Solution Format**: Single numerical answers

#### **Sample Problem Structure**
```json
{
  "ID": "1",
  "Body": "John has some apples.",
  "Question": "If he has 20 apples and gives away 8, how many are left?", 
  "Answer": "12",
  "Type": "Subtraction"
}
```

#### **Research Applications**
- **Language Robustness**: Test how methods handle linguistic variations
- **Baseline Performance**: Establish lower bounds for method capabilities
- **Prompt Engineering**: Study how different prompting affects simple problems

### 3. **TabMWP** - Tabular Math Word Problems
**Domain**: Tabular Data Reasoning
**Complexity**: Intermediate to Advanced

#### **Characteristics**
- **Problem Types**: Math problems requiring table interpretation
- **Reasoning**: Combination of table lookup and mathematical operations
- **Modality**: Text + structured tabular data
- **Solution Format**: Multi-step reasoning with table references

#### **Sample Problem Structure**
```json
{
  "question": "Based on the table, what is the total revenue for Q1 and Q2?",
  "table": [
    ["Quarter", "Revenue"],
    ["Q1", "100000"],
    ["Q2", "150000"],
    ["Q3", "120000"]
  ],
  "answer": "250000",
  "solution": "Q1: 100000 + Q2: 150000 = 250000"
}
```

#### **Research Applications**
- **Multi-modal Reasoning**: Evaluate text+table understanding capabilities
- **Structured Data Processing**: Test ability to work with tabular information
- **Complex Problem Solving**: Study multi-step reasoning with structured data

### 4. **TAT-QA** - Table-And-Text Question Answering
**Domain**: Financial Table-Text QA
**Complexity**: Advanced

#### **Characteristics**
- **Problem Types**: Financial analysis requiring both table and text interpretation
- **Reasoning**: Complex multi-step financial calculations
- **Context**: Real financial documents and reports
- **Solution Format**: Detailed reasoning with financial domain knowledge

#### **Sample Problem Structure**
```json
{
  "question": "What was the percentage change in net income from 2019 to 2020?",
  "table": "Financial table with yearly data...",
  "context": "Company annual report context...",
  "answer": "15.2%",
  "derivation": "((2020_income - 2019_income) / 2019_income) * 100"
}
```

#### **Research Applications**
- **Financial Domain**: Test reasoning in specialized domains
- **Complex Document Understanding**: Evaluate multi-source information processing
- **Professional Problem Solving**: Study reasoning in real-world business contexts

### 5. **FinQA** - Financial Question Answering
**Domain**: Financial Analysis
**Complexity**: Advanced

#### **Characteristics**
- **Problem Types**: Complex financial reasoning and calculations
- **Reasoning**: Multi-step numerical computations with financial concepts
- **Documents**: Real financial reports and earnings calls
- **Solution Format**: Step-by-step financial analysis

#### **Sample Problem Structure**
```json
{
  "question": "What is the debt-to-equity ratio if total debt is $500M and equity is $300M?",
  "context": "Financial report context with detailed numbers...",
  "answer": "1.67",
  "program": ["debt = 500", "equity = 300", "ratio = debt / equity", "return ratio"]
}
```

#### **Research Applications**
- **Financial Expertise**: Test domain-specific reasoning capabilities
- **Multi-step Calculations**: Evaluate complex computational reasoning
- **Real-world Applications**: Study performance on actual business problems

## 🔧 **Implementation Details**

### Dataset-Specific Configurations

#### **ICL Examples (k)**
- **GSM8K, SVAMP, TabMWP, FinQA**: k=2 (2 examples for in-context learning)
- **TAT-QA**: k=3 (3 examples due to higher complexity)

#### **Candidate Pool Sizes**
- **GSM8K**: 20 candidates (sufficient for basic problems)
- **SVAMP**: 15 candidates (smaller dataset, fewer candidates needed)
- **TabMWP**: 25 candidates (more candidates for table reasoning)
- **TAT-QA**: 25 candidates (balanced for complex financial problems)
- **FinQA**: 30 candidates (largest pool for most complex domain)

#### **Training Configurations**
| Dataset | Learning Rate | Epochs | Expected Training Time | Memory Usage |
|---------|---------------|--------|------------------------|--------------|
| GSM8K   | 3e-4         | 3      | ~10 minutes            | Low          |
| SVAMP   | 3e-4         | 4      | ~5 minutes             | Low          |
| TabMWP  | 2e-4         | 4      | ~20 minutes            | Medium       |
| TAT-QA  | 2e-4         | 3      | ~15 minutes            | Medium       |
| FinQA   | 1e-4         | 5      | ~30 minutes            | High         |

### File Formats

#### **Dataset JSON Structure**
```json
{
  "train": [
    {
      "question": "Problem statement...",
      "answer": "Expected answer...",
      "context": "Additional context if applicable...",
      "table": "Table data if applicable...",
      "solution": "Step-by-step solution..."
    }
  ],
  "test": [...],
  "dev": [...]  // If applicable
}
```

#### **Generated Candidates Structure**
```json
[
  {
    "question": "Problem statement...",
    "context": "Problem context...",
    "code": "def solve():\n    # FPP solution\n    return answer",
    "expected_answer": "Ground truth answer",
    "embedding": [1536-dimensional vector],
    "is_correct": true,
    "execution_time": 0.05
  }
]
```

## 📁 **Data Preprocessing**

### Loading and Processing
```python
import json
from pathlib import Path

def load_dataset(dataset_name):
    """Load dataset with proper preprocessing"""
    dataset_path = Path(f"datasets/{dataset_name}")
    
    # Load train/test splits
    with open(dataset_path / "train.json") as f:
        train_data = json.load(f)
    with open(dataset_path / "test.json") as f:
        test_data = json.load(f)
    
    return train_data, test_data

# Usage for different datasets
gsm8k_train, gsm8k_test = load_dataset("GSM8K")
tatqa_train, tatqa_test = load_dataset("TAT-QA")
```

### Data Validation
```python
def validate_dataset_format(data, dataset_name):
    """Validate dataset follows expected format"""
    required_fields = ["question", "answer"]
    
    if dataset_name in ["TabMWP", "TAT-QA"]:
        required_fields.extend(["table", "context"])
    
    for item in data:
        for field in required_fields:
            assert field in item, f"Missing {field} in {dataset_name}"
    
    print(f"✅ {dataset_name} format validated")
```

## 🚀 **Usage Examples**

### Candidate Generation
```bash
# Generate candidates for different datasets
python generate_candidates.py --dataset GSM8K --n-candidates 50
python generate_candidates.py --dataset TAT-QA --n-candidates 100
python generate_candidates.py --dataset FinQA --n-candidates 150
```

### Policy Training
```bash
# Train policies with dataset-specific configurations
python train_policy.py --dataset GSM8K --epochs 3 --lr 3e-4
python train_policy.py --dataset TAT-QA --epochs 3 --lr 2e-4
python train_policy.py --dataset FinQA --epochs 5 --lr 1e-4
```

### Method Comparison
```bash
# Compare ICL methods on different datasets
python run_comparison.py --dataset GSM8K --samples 100
python run_comparison.py --dataset TAT-QA --samples 150
python run_comparison.py --dataset FinQA --samples 50
```

### Prompting Method Evaluation
```bash
# Test different prompting methods
python mathcorl.py test --method fpp --dataset SVAMP --limit 100
python mathcorl.py test --method cot --dataset GSM8K --limit 50
python mathcorl.py compare --dataset TabMWP --limit 30
```

## 🔬 **Research Considerations**

### Dataset Selection Guidelines

#### **For Algorithm Development**
- **Start with GSM8K**: Well-established, clear baselines, easier debugging
- **Progress to SVAMP**: Test language robustness and variation handling
- **Scale to TabMWP**: Evaluate multi-modal reasoning capabilities

#### **For Advanced Research**
- **TAT-QA**: Complex financial reasoning, real-world applications
- **FinQA**: Most challenging domain, sophisticated financial analysis

#### **For Comprehensive Evaluation**
- **Cross-dataset testing**: Train on one dataset, evaluate on others
- **Domain transfer**: Study how methods generalize across domains
- **Complexity scaling**: Compare performance as problem complexity increases

### Experimental Design Recommendations

#### **Sample Size Guidelines**
- **Initial testing**: 10-25 samples for quick validation
- **Method development**: 50-100 samples for reliable comparison
- **Final evaluation**: 100-200 samples for statistical significance

#### **Method Comparison Strategy**
1. **Baseline establishment**: Start with Random and Zero-shot
2. **Traditional methods**: Add KATE and CDS for comparison
3. **Novel approaches**: Include Policy Network for innovation

#### **Budget Planning**
- **Development phase**: Use smaller datasets (GSM8K, SVAMP) for cost efficiency
- **Validation phase**: Scale to larger, more complex datasets
- **Final evaluation**: Comprehensive testing across all relevant datasets

## 🛠️ **Troubleshooting**

### Common Issues

#### **Dataset Loading Errors**
```bash
# Verify dataset files exist
ls -la datasets/TAT-QA/
# Expected: train.json, test.json, dev.json (if applicable)

# Check JSON format
python -m json.tool datasets/TAT-QA/train.json > /dev/null
```

#### **Memory Issues with Large Datasets**
```python
# For large datasets, process in batches
def process_dataset_batch(dataset, batch_size=100):
    for i in range(0, len(dataset), batch_size):
        batch = dataset[i:i+batch_size]
        yield batch

# Usage
for batch in process_dataset_batch(large_dataset):
    process_batch(batch)
```

#### **Embedding Generation Issues**
```bash
# Check OpenAI API key and quota
python -c "import openai; print(openai.api_key)"

# Test embedding generation
python -c "from mint.utils import get_embedding; print(len(get_embedding('test')))"
```

### Performance Optimization

#### **Faster Candidate Generation**
```python
# Use parallel processing for candidate generation
from concurrent.futures import ThreadPoolExecutor

def generate_candidates_parallel(problems, n_workers=4):
    with ThreadPoolExecutor(max_workers=n_workers) as executor:
        results = list(executor.map(generate_single_candidate, problems))
    return results
```

#### **Memory-Efficient Processing**
```python
# Stream processing for large datasets
def stream_dataset(file_path):
    with open(file_path) as f:
        for line in f:
            yield json.loads(line)

# Generator-based processing
def process_dataset_efficiently(dataset_path):
    for item in stream_dataset(dataset_path):
        result = process_item(item)
        yield result
```

---

**📚 Dataset Diversity**: MathCoRL's dataset collection spans from elementary arithmetic to complex financial analysis, providing comprehensive coverage for mathematical reasoning research across different domains and complexity levels. 