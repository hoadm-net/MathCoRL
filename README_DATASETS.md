# MathCoRL - Dataset Guide

## 🎯 **Datasets for Dual Research Framework**

This guide describes the mathematical reasoning datasets used in MathCoRL's dual research framework:

- **📚 Task 1**: Prompting method comparison (FPP vs CoT vs PAL vs PoT vs Zero-shot)
- **🧠 Task 2**: ICL example selection comparison (Policy vs KATE vs CDS vs Random vs Zero-shot)

All datasets support both research directions with domain-specific configurations optimized for mathematical reasoning evaluation.

## 📊 **Dataset Overview**

| Dataset | Domain | Training | Test | Total | Complexity | Primary Challenge |
|---------|--------|----------|------|-------|------------|-------------------|
| **GSM8K** | Elementary Math | 7,473 | 1,319 | 8,792 | Basic | Multi-step arithmetic reasoning |
| **SVAMP** | Arithmetic Variations | 800 | 300 | 1,100 | Simple | Arithmetic with linguistic variations |
| **TabMWP** | Tabular Math | 28,876 | 4,153 | 33,029 | Medium | Table-based mathematical reasoning |
| **TAT-QA** | Financial Tables | 13,826 | 2,199 | 16,025 | Hard | Financial document comprehension |
| **FinQA** | Financial Analysis | 6,251 | 1,147 | 7,398 | Hard | Complex financial reasoning and calculations |

## 🔬 **Research Applications by Dataset**

### **Task 1: Prompting Method Research**
**Question**: Which prompting technique works best for each mathematical domain?

### **Task 2: ICL Example Selection Research**
**Question**: Which example selection strategy works best for each mathematical domain?

## 📋 **Dataset Details**

### **🧮 GSM8K (Grade School Math 8K)**

**Domain**: Elementary mathematics word problems  
**Source**: [OpenAI GSM8K](https://github.com/openai/grade-school-math)

#### **Problem Characteristics**
- **Type**: Word problems requiring 2-8 reasoning steps
- **Operations**: Basic arithmetic (addition, subtraction, multiplication, division)
- **Complexity**: Simple to moderate multi-step problems
- **Answer Format**: Numerical values with tolerance

#### **Example Problem**
```json
{
  "question": "Janet's ducks lay 16 eggs per day. She eats 3 for breakfast every morning and bakes 4 into muffins for her friends every day. She sells the remainder at the farmers' market for $2 per egg. How much does she make every day?",
  "answer": "18"
}
```

#### **Research Configuration**
| Setting | Task 1 (Prompting) | Task 2 (ICL) | Rationale |
|---------|-------------------|--------------|-----------|
| **Examples (k)** | N/A | 2 | Sufficient for pattern recognition |
| **Candidate Pool** | N/A | Small-Medium | Balanced diversity vs. efficiency |
| **Policy Training** | N/A | 3 epochs, 3e-4 LR | Fast convergence on clear patterns |

#### **Evaluation Method**
- **Tolerance**: ±1% for numerical answers
- **Extraction**: Extract final numerical value from generated solution
- **Validation**: Execute generated code and compare results

### **🔄 SVAMP (Simple Variations on Arithmetic Math word Problems)**

**Domain**: Arithmetic word problems with linguistic variations  
**Source**: [SVAMP Dataset](https://github.com/arkilpatel/SVAMP)

#### **Problem Characteristics**
- **Type**: Single-step arithmetic with question variations
- **Operations**: Basic operations with different phrasings
- **Complexity**: Simple arithmetic, complex language
- **Answer Format**: Numerical values

#### **Example Problem**
```json
{
  "question": "Each pack of dvds costs 76 dollars. If Melissa bought 50 packs, how much did she spend?",
  "answer": "3800"
}
```

#### **Research Configuration**
| Setting | Task 1 (Prompting) | Task 2 (ICL) | Rationale |
|---------|-------------------|--------------|-----------|
| **Examples (k)** | N/A | 2 | Simple problems need few examples |
| **Candidate Pool** | N/A | Small | Small dataset, focused candidates |
| **Policy Training** | N/A | 4 epochs, 3e-4 LR | Multiple epochs for language variation |

#### **Evaluation Method**
- **Tolerance**: Exact match for integer answers
- **Extraction**: Direct numerical extraction
- **Validation**: Simple arithmetic verification

### **📊 TabMWP (Tabular Math Word Problems)**

**Domain**: Mathematical reasoning with tables and charts  
**Source**: [TabMWP Dataset](https://github.com/lupantech/PromptPG)

#### **Problem Characteristics**
- **Type**: Problems requiring table/chart interpretation
- **Operations**: Arithmetic operations on tabular data
- **Complexity**: Medium complexity with visual reasoning
- **Answer Format**: Numerical values or text

#### **Example Problem**
```json
{
  "question": "Look at the table. How much more does a large pizza cost than a small pizza?",
  "table": "Size | Price\nSmall | $8\nMedium | $12\nLarge | $16",
  "answer": "8"
}
```

#### **Research Configuration**
| Setting | Task 1 (Prompting) | Task 2 (ICL) | Rationale |
|---------|-------------------|--------------|-----------|
| **Examples (k)** | N/A | 2 | Table structure provides context |
| **Candidate Pool** | N/A | Medium-Large | Large dataset supports bigger pools |
| **Policy Training** | N/A | 4 epochs, 2e-4 LR | Medium complexity needs more training |

#### **Evaluation Method**
- **Tolerance**: Moderate tolerance for numerical, exact for categorical
- **Context**: Include table/chart data in prompts
- **Validation**: Parse tabular data and verify calculations

### **📈 TAT-QA (Table-and-Text Question Answering)**

**Domain**: Financial question answering on tables and text  
**Source**: [TAT-QA Dataset](https://github.com/NExTplusplus/TAT-QA)

#### **Problem Characteristics**
- **Type**: Complex financial reasoning with mixed data sources
- **Operations**: Percentage calculations, comparisons, aggregations
- **Complexity**: High complexity with financial context
- **Answer Format**: Numerical values, percentages, spans

#### **Example Problem**
```json
{
  "question": "What is the percentage change in total revenue from 2019 to 2020?",
  "context": "Financial statements showing revenue data...",
  "table": "Year | Revenue\n2019 | $1,200M\n2020 | $1,320M",
  "answer": "10"
}
```

#### **Research Configuration**
| Setting | Task 1 (Prompting) | Task 2 (ICL) | Rationale |
|---------|-------------------|--------------|-----------|
| **Examples (k)** | N/A | 3 | Complex problems need more examples |
| **Candidate Pool** | N/A | Medium | Focused on financial reasoning |
| **Policy Training** | N/A | 5 epochs, 2e-4 LR | Complex domain needs more training |

#### **Evaluation Method**
- **Tolerance**: Higher tolerance for percentages and large values
- **Context**: Include both table and text context
- **Validation**: Financial calculation verification

### **💰 FinQA (Financial Question Answering)**

**Domain**: Complex financial analysis and calculations  
**Source**: [FinQA Dataset](https://github.com/czyssrs/FinQA)

#### **Problem Characteristics**
- **Type**: Multi-step financial calculations and analysis
- **Operations**: Complex financial formulas, multi-step reasoning
- **Complexity**: Highest complexity with domain expertise required
- **Answer Format**: Numerical values with financial context

#### **Example Problem**
```json
{
  "question": "What is the net change in net revenue during 2015 for entergy corporation?",
  "context": "Entergy Corporation reported revenues of $11,961.5 million in 2015 compared to $11,867.5 million in 2014...",
  "answer": "94.0"
}
```

#### **Research Configuration**
| Setting | Task 1 (Prompting) | Task 2 (ICL) | Rationale |
|---------|-------------------|--------------|-----------|
| **Examples (k)** | N/A | 2 | Quality over quantity for complex problems |
| **Candidate Pool** | N/A | Medium-Large | Large pool for diverse financial scenarios |
| **Policy Training** | N/A | 5 epochs, 1e-4 LR | Low LR for stable complex training |

#### **Evaluation Method**
- **Tolerance**: Higher tolerance for large financial figures
- **Context**: Rich financial document context
- **Validation**: Multi-step financial calculation verification

## 🔧 **Dataset-Specific Implementation Details**

### **Prompt Engineering by Dataset**

#### **Task 1: Prompting Method Adaptation**
```python
# GSM8K - Simple arithmetic focus
fpp_prompt = "Use basic math functions: add(), sub(), mul(), div()"

# TabMWP - Table processing emphasis  
fpp_prompt = "Use table functions: sum_column(), filter_table(), get_value()"

# FinQA - Financial calculation focus
fpp_prompt = "Use financial functions: percentage(), change_rate(), sum()"
```

#### **Task 2: ICL Context Formatting**
```python
# SVAMP - Question only
context = question

# TAT-QA - Table + text context
context = f"{table_data}\n\n{text_context}\n\n{question}"

# FinQA - Rich financial context
context = f"{financial_context}\n\nQuestion: {question}"
```

### **Embedding Strategy by Dataset**

| Dataset | Context Strategy | Embedding Focus | Rationale |
|---------|------------------|-----------------|-----------|
| **GSM8K** | Question only | Arithmetic patterns | Simple structure |
| **SVAMP** | Question + variation | Linguistic patterns | Language variation focus |
| **TabMWP** | Table + question | Structural patterns | Table relationship focus |
| **TAT-QA** | Full context | Domain patterns | Financial context crucial |
| **FinQA** | Rich context | Calculation patterns | Complex financial reasoning |

### **Tolerance Functions by Dataset**

```python
def evaluate_gsm8k(predicted, ground_truth):
    """GSM8K: ±1% tolerance for arithmetic"""
    return abs(predicted - ground_truth) <= abs(ground_truth * 0.01)

def evaluate_tabmwp(predicted, ground_truth):
    """TabMWP: ±2% tolerance for table calculations"""
    return abs(predicted - ground_truth) <= abs(ground_truth * 0.02)

def evaluate_finqa(predicted, ground_truth):
    """FinQA: ±5% tolerance for complex financial calculations"""
    return abs(predicted - ground_truth) <= abs(ground_truth * 0.05)
```

## 📈 **Research Methodology by Dataset**

### **Task 1: Prompting Method Research Applications**
| Dataset | Best Method Type | Characteristics | Research Value |
|---------|------------------|-----------------|----------------|
| **GSM8K** | Structured | Clear step-by-step reasoning | Tests basic reasoning |
| **SVAMP** | Precise | Consistent function usage | Tests precision vs flexibility |
| **TabMWP** | Programming | Code handles table operations | Tests programming paradigms |
| **TAT-QA** | Structured | Function structure clarifies operations | Tests complexity management |
| **FinQA** | Domain-specific | Mathematical functions match domain | Tests domain adaptation |

### **Task 2: ICL Method Research Applications**
| Dataset | Optimal Strategy | Key Success Factor | Research Insight |
|---------|------------------|-------------------|------------------|
| **GSM8K** | Adaptive Learning | Pattern recognition in arithmetic | AI adaptation effectiveness |
| **SVAMP** | Curriculum | Systematic difficulty progression | Complexity progression value |
| **TabMWP** | Similarity | Structural pattern matching | Semantic similarity effectiveness |
| **TAT-QA** | Learning | Complex pattern adaptation | Learning vs heuristics |
| **FinQA** | Advanced | Multi-step complexity handling | Advanced strategy necessity |

## 🎓 **Research Applications**

### **Cross-Dataset Analysis**
```bash
# Compare method effectiveness across domains
for dataset in GSM8K SVAMP TabMWP TAT-QA FinQA; do
    python mathcorl.py compare --dataset $dataset --limit 50
done

# Compare ICL strategies across domains  
for dataset in GSM8K SVAMP TabMWP TAT-QA FinQA; do
    python run_comparison.py --dataset $dataset --samples 30 --save-results
done
```

### **Domain Specialization Research**
- **Elementary Math** (GSM8K, SVAMP): Test basic reasoning capabilities
- **Structured Data** (TabMWP, TAT-QA): Evaluate table/chart interpretation
- **Financial Domain** (TAT-QA, FinQA): Study domain-specific reasoning

### **Complexity Analysis Research**
- **Simple** → **Complex**: SVAMP → GSM8K → TabMWP → TAT-QA → FinQA
- Study how method effectiveness changes with problem complexity
- Investigate whether ICL strategies adapt to complexity levels

## 🔍 **Dataset Preprocessing**

### **Loading and Validation**
```python
from mint.utils import load_dataset

# Load with automatic validation
dataset = load_dataset('GSM8K', split='test')

# Verify dataset integrity
print(f"Loaded {len(dataset)} samples")
print(f"Average question length: {avg_question_length}")
print(f"Answer type distribution: {answer_types}")
```

### **Quality Assurance**
- **Answer Validation**: All answers verified through multiple solution paths
- **Format Consistency**: Standardized JSON format across all datasets
- **Context Completeness**: All necessary context included for problem solving
- **Ground Truth Accuracy**: Human-verified ground truth labels

---

🎯 **Dataset selection guide**: 
- **Start with GSM8K** for basic evaluation
- **Use SVAMP** for linguistic variation testing  
- **Try TabMWP** for structured data reasoning
- **Test TAT-QA/FinQA** for domain expertise evaluation

This comprehensive dataset guide enables targeted research across mathematical reasoning domains, supporting both prompting method comparison and in-context learning strategy evaluation. 