# MathCoRL - Usage Guide

## 🔭 **Dual Research Framework Overview**

MathCoRL implements two complementary research directions for mathematical reasoning:

### **📚 Task 1: Prompting Method Comparison**
**Research Question**: Which prompting technique works best for mathematical reasoning?

- **Tool**: `mathcorl.py` (unified CLI interface)
- **Methods Compared**: FPP, CoT, PAL, PoT, Zero-shot
- **Use Case**: Compare different ways to prompt LLMs for mathematical problems
- **Output**: Method accuracy comparison across datasets

### **🧠 Task 2: In-Context Learning (ICL) Example Selection Comparison**
**Research Question**: Which example selection strategy works best for in-context learning?

- **Tool**: 3-script pipeline for comprehensive ICL research
- **Methods Compared**: FPP + Policy Network, FPP + KATE, FPP + CDS, FPP + Random, FPP + Zero-shot
- **Use Case**: Compare different ways to select demonstration examples for ICL
- **Output**: Example selection strategy accuracy comparison

## 📋 **Research Pipeline Architecture**

### **Task 1 Workflow** (Prompting Method Research)
```
mathcorl.py → mint.cli → {fpp, cot, pal, pot, zero_shot}.py → Results
```
**Single unified interface for comparing different prompting approaches**

### **Task 2 Workflow** (ICL Example Selection Research)
```
📦 generate_candidates.py  →  🎓 train_policy.py  →  🏆 run_comparison.py
      Step 1                       Step 2                    Step 3
  (Candidate Pool)            (Policy Training)      (Method Comparison)
```
**Three-step pipeline for end-to-end ICL research**

## 🧠 **Deep Dive: ICL Research Process**

### **Candidate Generation (Step 1)**
Transform raw datasets into structured candidate pools for example selection research.

#### **What Happens in Candidate Generation:**
1. **Problem Parsing**: Extract mathematical problems from dataset JSON/JSONL files
2. **FPP Code Generation**: Use Function Prototype Prompting to create Python solutions
3. **Semantic Embedding**: Generate embeddings using `text-embedding-3-small` model
4. **Solution Validation**: Execute generated code and verify correctness against ground truth
5. **Candidate Pool Creation**: Store validated problem-solution-embedding triplets

#### **Why This Matters:**
- Creates **standardized candidate pools** for fair ICL comparison
- Ensures **solution correctness** through automated validation
- Generates **semantic embeddings** for similarity-based methods (KATE, CDS)
- Provides **training data** for Policy Network learning

### **Policy Network Training (Step 2)**
Train neural networks to learn optimal example selection strategies.

#### **Policy Network Architecture:**
- **Input**: Problem embedding (1536D) + Candidate embeddings (1536D each)
- **Architecture**: Multi-head attention transformer (1536D → 768D, 8 heads)
- **Output**: Relevance scores for each candidate
- **Selection**: Top-k candidates based on learned scores

#### **Training Algorithm:**
- **Method**: Proximal Policy Optimization (PPO)
- **Reward Function**: Multi-objective combining:
  - **Correctness**: Whether selected examples lead to correct solutions
  - **Diversity**: Ensuring varied example types
  - **Efficiency**: Computational cost consideration
- **Training Data**: Generated candidates with ground truth validation

#### **Why Policy Network:**
- **Adaptive Selection**: Learns from data rather than using fixed rules
- **Context Awareness**: Considers problem characteristics for selection
- **Multi-Objective**: Balances correctness, diversity, and efficiency
- **Transferable**: Can generalize to unseen problem types

### **Method Comparison (Step 3)**
Comprehensive evaluation of different ICL example selection strategies.

#### **ICL Methods Explained:**

**🎯 FPP + Zero-shot**: 
- No examples provided, pure function prototype prompting
- **Purpose**: Baseline to measure raw model capability

**🎲 FPP + Random**:
- Random selection from candidate pool
- **Purpose**: Control method to measure value of any examples vs. smart selection

**🤖 FPP + Policy Network**:
- Neural network selects most relevant examples
- **Purpose**: AI-powered adaptive selection based on learned patterns

**🔍 FPP + KATE** (kNN-Augmented in-conText Example selection):
- Semantic similarity using cosine distance on embeddings
- **Purpose**: Simple but effective similarity-based approach

**📚 FPP + CDS** (Curriculum Demonstration Selection):
- Complexity-based partitioning with curriculum learning principles
- **Purpose**: Balanced difficulty progression following educational principles

## 🔧 **Environment Setup**

### **1. Configure Environment**
```bash
# Copy and configure environment
cp env.example .env
# Edit .env with your OpenAI API key

# Model configuration:
DEFAULT_MODEL=gpt-4.1-mini           # Chat model for reasoning and code generation
EMBEDDING_MODEL=text-embedding-3-small  # Embedding model for semantic similarity
TEMPERATURE=0.1                      # Temperature for generation (low for consistency)
```

### **2. Dataset Overview**
| Dataset | Domain | Training | Test | Complexity | ICL Examples (k) |
|---------|--------|----------|------|------------|------------------|
| **GSM8K** | Elementary Math | 7.5K | 1.3K | Basic arithmetic | 2 |
| **SVAMP** | Arithmetic Variations | 800 | 300 | Simple variations | 2 |
| **TabMWP** | Tabular Math | 28.9K | 4.5K | Table reasoning | 2 |
| **TAT-QA** | Financial Tables | 13.8K | 2.2K | Financial analysis | 3 |
| **FinQA** | Financial Text | 6.3K | 1.1K | Complex finance | 2 |

## 📚 **Task 1: Prompting Method Comparison**

### **Available Prompting Methods**

#### **🎯 Function Prototype Prompting (FPP)**
- **Approach**: Pre-define mathematical functions available to the model
- **Advantages**: Structured reasoning, reduced hallucination, explicit function usage
- **Best For**: Problems requiring precise mathematical operations

#### **💭 Chain-of-Thought (CoT)**
- **Approach**: Step-by-step reasoning with natural language explanations
- **Advantages**: Interpretable reasoning steps, human-like problem decomposition
- **Best For**: Multi-step word problems requiring logical reasoning

#### **🔧 Program-aided Language Models (PAL)**
- **Approach**: Generate executable Python code for mathematical computation
- **Advantages**: Programming flexibility, computational precision
- **Best For**: Problems requiring complex calculations or algorithmic thinking

#### **📝 Program-of-Thoughts (PoT)**
- **Approach**: Structured programming with explicit variable tracking
- **Advantages**: Systematic problem decomposition, algorithmic organization
- **Best For**: Problems requiring step-by-step computational procedures

#### **⚡ Zero-shot**
- **Approach**: Direct problem solving without examples or special prompting
- **Advantages**: Fast, minimal prompting, measures raw capability
- **Best For**: Baseline comparison and simple problems

### **Usage Examples**

```bash
# Single problem solving with different methods
python mathcorl.py solve --method fpp --question "What is the compound interest on $1000 at 5% for 3 years?"
python mathcorl.py solve --method cot --question "John has 20 apples. He gives 8 to his friend. How many are left?"
python mathcorl.py solve --method pal --question "Calculate the average of these numbers: 15, 23, 31, 42"

# Dataset evaluation with specific method
python mathcorl.py test --method fpp --dataset SVAMP --limit 100
python mathcorl.py test --method cot --dataset GSM8K --limit 50

# Compare all prompting methods
python mathcorl.py compare --dataset TabMWP --limit 30
```

## 🧠 **Task 2: ICL Example Selection Research**

### **Step 1: Generate Candidates**

Generate standardized candidate pools for ICL research.

```bash
# Basic candidate generation
python generate_candidates.py --dataset FinQA --n-candidates 100

# Generate with specific settings
python generate_candidates.py --dataset GSM8K --n-candidates 50 --verbose
```

#### **Dataset-Specific Candidate Pool Configurations**
| Dataset | Recommended Pool Size | Generation Time | Memory Usage |
|---------|----------------------|-----------------|--------------|
| GSM8K   | Small-Medium         | Fast            | Low          |
| SVAMP   | Small                | Fast            | Low          |
| TabMWP  | Medium-Large         | Medium          | Medium       |
| TAT-QA  | Medium               | Medium          | Medium       |
| FinQA   | Medium-Large         | Slow            | High         |

#### **Options**
- `--dataset, -d`: Choose dataset (GSM8K, SVAMP, TabMWP, TAT-QA, FinQA)
- `--n-candidates, -n`: Number of candidates to generate (default: 100)
- `--output-dir, -o`: Output directory (default: candidates)
- `--model`: OpenAI chat model (default: from config)
- `--embedding-model`: OpenAI embedding model (default: from config)
- `--overwrite`: Overwrite existing files
- `--verbose, -v`: Enable verbose logging

### **Step 2: Train Policy Network**

Train neural networks for intelligent example selection.

```bash
# Train with default settings
python train_policy.py --dataset FinQA --epochs 5

# Train with custom hyperparameters
python train_policy.py --dataset GSM8K --epochs 3 --lr 3e-4
```

#### **Dataset-Specific Training Configurations**
| Dataset | Learning Rate | Epochs | Pool Size | Training Time |
|---------|---------------|--------|-----------|---------------|
| GSM8K   | 3e-4         | 3      | 20        | ~15 min       |
| SVAMP   | 3e-4         | 4      | 15        | ~10 min       |
| TabMWP  | 2e-4         | 4      | 25        | ~20 min       |
| TAT-QA  | 2e-4         | 5      | 25        | ~25 min       |
| FinQA   | 1e-4         | 5      | 30        | ~30 min       |

#### **Training Process**
1. **Initialization**: Load candidates and create training pairs
2. **Policy Network**: Initialize transformer with attention mechanism
3. **PPO Training**: Use reinforcement learning with reward function
4. **Validation**: Evaluate on held-out candidates
5. **Model Saving**: Save best and final models

#### **Options**
- `--dataset, -d`: Dataset to train on (required)
- `--epochs, -e`: Number of training epochs (default: 5)
- `--lr`: Learning rate (default: dataset-specific)
- `--pool-size`: Candidate pool size (default: dataset-specific)
- `--k`: Number of examples to select (default: dataset-specific)
- `--models-dir`: Model save directory (default: models)
- `--overwrite`: Overwrite existing models
- `--verbose, -v`: Enable verbose logging

### **Step 3: Compare ICL Methods**

Comprehensive evaluation of example selection strategies.

```bash
# Compare all ICL methods
python run_comparison.py --dataset FinQA --samples 20 --save-results

# Compare specific methods
python run_comparison.py --dataset GSM8K --methods policy,kate,cds --samples 50

# Quick test with fewer samples
python run_comparison.py --dataset SVAMP --samples 5 --methods policy,random
```

#### **Method Selection Options**
```bash
# Compare similarity-based methods
python run_comparison.py --dataset FinQA --methods kate,cds --samples 10

# Compare AI vs. baseline methods
python run_comparison.py --dataset GSM8K --methods policy,random,zero-shot --samples 15

# Test only Policy Network
python run_comparison.py --dataset TabMWP --methods policy --samples 10

# Full comparison (all 5 methods)
python run_comparison.py --dataset TAT-QA --samples 30
```

#### **Options**
- `--dataset, -d`: Dataset to test on (required)
- `--samples, -s`: Number of test samples (default: 10)
- `--methods`: Comma-separated methods (default: zero-shot,random,policy,kate,cds)
- `--seed`: Random seed for reproducibility (default: 42)
- `--save-results`: Save detailed JSON results
- `--verbose, -v`: Enable verbose logging

## 📊 **Research Methodology Overview**

### **Task 1: Prompting Method Characteristics**
- **FPP (Function Prototype)**: Structured reasoning with explicit function definitions
- **PAL (Program-aided)**: Code generation combined with reasoning explanations
- **CoT (Chain-of-Thought)**: Step-by-step natural language reasoning
- **PoT (Program-of-Thoughts)**: Pure code generation approach
- **Zero-shot**: Direct problem solving baseline

### **Task 2: ICL Example Selection Approaches**
- **FPP + Policy Network**: Learned adaptive selection through neural networks
- **FPP + CDS**: Curriculum-based complexity partitioning
- **FPP + KATE**: Semantic similarity-based selection
- **FPP + Random**: Random selection control baseline
- **FPP + Zero-shot**: No examples baseline

## 🔬 **Research Applications & Insights**

### **For Prompting Research (Task 1)**
- **Method Effectiveness**: Compare structured vs. free-form reasoning
- **Domain Adaptation**: Which methods work best for different mathematical domains
- **Computational Efficiency**: Trade-offs between accuracy and computational cost
- **Error Analysis**: Understanding failure modes of different prompting approaches

### **For ICL Research (Task 2)**  
- **Example Selection Strategies**: Policy vs. similarity vs. curriculum vs. random
- **Training Data Requirements**: How much data needed for effective policy learning
- **Transfer Learning**: Whether policies trained on one dataset work on others
- **Complexity Analysis**: How mathematical complexity affects example selection

## 🛠️ **Advanced Usage**

### **Pipeline Testing**
```bash
# Test complete pipeline end-to-end
python test_pipeline.py --dataset GSM8K --quick-test

# Full pipeline test
python test_pipeline.py --dataset FinQA
```

### **Custom Evaluation**
```bash
# Save detailed results for analysis
python run_comparison.py --dataset FinQA --samples 50 --save-results

# Results saved to: results/finqa_comparison_50samples.json
```

### **Batch Experiments**
```bash
# Compare methods across multiple datasets
for dataset in GSM8K SVAMP TabMWP; do
    python run_comparison.py --dataset $dataset --samples 30 --save-results
done
```

## 🎓 **Best Practices**

### **For Prompting Research**
1. **Start Small**: Test with 10-20 samples first
2. **Use Consistent Settings**: Same temperature, model version
3. **Multiple Runs**: Use different random seeds for statistical validity
4. **Domain Analysis**: Different methods excel in different domains

### **For ICL Research**
1. **Adequate Candidates**: Generate 30+ candidates for reliable training
2. **Training Patience**: Allow sufficient epochs for policy convergence
3. **Validation Set**: Reserve test samples separate from training
4. **Baseline Comparison**: Always include random and zero-shot baselines

## 📈 **Resource Guidelines**

### **Computational Requirements Overview**
| Task | Time | Memory | API Usage | Resource Level |
|------|------|--------|-----------|----------------|
| Task 1 (50 samples) | Fast | Low | Moderate | Light |
| Task 2 Candidates | Medium | Medium | High | Medium |
| Task 2 Training | Medium | Medium | High | Medium |
| Task 2 Comparison | Fast | Low | Moderate | Light |

### **Dataset Complexity Levels**
| Dataset | Task 1 Complexity | Task 2 Complexity | Overall Difficulty |
|---------|------------------|------------------|-------------------|
| SVAMP   | Simple           | Simple           | Easy               |
| GSM8K   | Moderate         | Moderate         | Medium             |
| TabMWP  | Moderate         | Moderate         | Medium             |
| TAT-QA  | Complex          | Complex          | Hard               |
| FinQA   | Complex          | Complex          | Hard               |

---

🚀 **Ready to start your mathematical reasoning research!** Choose Task 1 for prompting method comparison or Task 2 for in-context learning research. 