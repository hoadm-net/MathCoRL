# MathCoRL - Complete Usage Guide

## 🔭 **Dual Research Framework Overview**

MathCoRL implements two complementary research directions for mathematical reasoning with comprehensive API tracking and cost monitoring:

### **📚 Task 1: Prompting Method Comparison**
**Research Question**: Which prompting technique works best for mathematical reasoning?

- **Tool**: `mathcorl.py` (unified CLI interface with API tracking)
- **Methods Compared**: FPP, CoT, PAL, PoT, Zero-shot
- **Use Case**: Compare different ways to prompt LLMs for mathematical problems
- **Output**: Method accuracy comparison across datasets with cost analysis
- **Features**: Real-time cost tracking, token monitoring, exportable results

### **🧠 Task 2: In-Context Learning (ICL) Example Selection Comparison**
**Research Question**: Which example selection strategy works best for in-context learning?

- **Tool**: 3-script pipeline for comprehensive ICL research
- **Methods Compared**: FPP + Policy Network, FPP + KATE, FPP + CDS, FPP + Random, FPP + Zero-shot
- **Use Case**: Compare different ways to select demonstration examples for ICL
- **Output**: Example selection strategy accuracy comparison
- **Features**: Neural policy training, multi-objective optimization, robust evaluation

## 📋 **Research Pipeline Architecture**

### **Task 1 Workflow** (Prompting Method Research)
```
mathcorl.py → mint.cli → {fpp, cot, pal, pot, zero_shot}.py → tracking.py → Results + Cost Analysis
```
**Single unified interface for comparing different prompting approaches with full API monitoring**

### **Task 2 Workflow** (ICL Example Selection Research)
```
📦 generate_candidates.py  →  🎓 train_policy.py  →  🏆 run_comparison.py
      Step 1                       Step 2                    Step 3
  (Candidate Pool)            (Policy Training)      (Method Comparison)
```
**Three-step pipeline for end-to-end ICL research with trained neural policies**

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
Train neural networks to learn optimal example selection strategies using reinforcement learning.

#### **Policy Network Architecture:**
- **Input**: Problem embedding (1536D) + Candidate embeddings (1536D each)
- **Architecture**: Multi-head attention transformer (1536D → 768D, 8 heads)
- **Output**: Relevance scores for each candidate
- **Selection**: Top-k candidates based on learned scores

#### **Training Algorithm:**
- **Method**: Proximal Policy Optimization (PPO)
- **Reward Function**: Multi-objective combining:
  - **Correctness**: Whether selected examples lead to correct solutions (60%)
  - **Semantic Similarity**: Cosine similarity between problem and examples (30%)
  - **Diversity**: Ensuring varied example types (10%)
- **Training Data**: Generated candidates with ground truth validation

#### **Training Process:**
1. **Problem Sampling**: Random selection of target problems from training set
2. **Candidate Pool Creation**: Sample diverse pool of potential examples
3. **Policy Selection**: Use current policy to select k examples
4. **GPT Evaluation**: Generate solution using selected examples, check correctness
5. **Reward Calculation**: Multi-objective reward based on correctness and quality
6. **Policy Update**: Backpropagate rewards to improve selection policy

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
- Neural network selects most relevant examples using learned policy
- **Purpose**: AI-powered adaptive selection based on learned patterns
- **Training**: PPO with multi-objective rewards

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
DEFAULT_MODEL=gpt-4o-mini           # Chat model for reasoning and code generation
EMBEDDING_MODEL=text-embedding-3-small  # Embedding model for semantic similarity
TEMPERATURE=0.1                      # Temperature for generation (low for consistency)
```

### **2. Dataset Overview**
| Dataset | Domain | Training | Test | Complexity | ICL Examples (k) | Policy Status |
|---------|--------|----------|------|------------|------------------|---------------|
| **GSM8K** | Elementary Math | 7.5K | 1.3K | Basic arithmetic | 2 | Available |
| **SVAMP** | Arithmetic Variations | 800 | 300 | Simple variations | 2 | Available |
| **TabMWP** | Tabular Math | 28.9K | 4.5K | Table reasoning | 2 | Available |
| **TAT-QA** | Financial Tables | 13.8K | 2.2K | Financial analysis | 3 | Available |
| **FinQA** | Financial Text | 6.3K | 1.1K | Complex finance | 2 | Available |

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

### **Usage Examples with API Tracking**

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

# Monitor API usage and costs
python mathcorl.py stats                    # Last 24 hours
python mathcorl.py stats --hours 12         # Last 12 hours
python mathcorl.py export --format csv      # Export to CSV
python mathcorl.py chart --type all --save  # Generate charts
```

## 🧠 **Task 2: ICL Example Selection Research**

### **Step 1: Generate Candidates**

Generate standardized candidate pools for ICL research.

```bash
# Basic candidate generation
python generate_candidates.py --dataset TAT-QA --n-candidates 100

# Generate with specific settings
python generate_candidates.py --dataset GSM8K --n-candidates 50 --verbose

# All available datasets
python generate_candidates.py --dataset FinQA --n-candidates 150
python generate_candidates.py --dataset SVAMP --n-candidates 75
python generate_candidates.py --dataset TabMWP --n-candidates 200
```

#### **Dataset-Specific Candidate Pool Configurations**
| Dataset | Recommended Pool Size | Generation Time | Memory Usage | Status |
|---------|----------------------|-----------------|--------------|--------|
| GSM8K   | 50-100               | Fast            | Low          | Available |
| SVAMP   | 25-75                | Fast            | Low          | Available |
| TabMWP  | 100-200              | Medium          | Medium       | Available |
| TAT-QA  | 100-150              | Medium          | Medium       | Available |
| FinQA   | 100-200              | Slow            | High         | Available |

#### **Options**
- `--dataset, -d`: Choose dataset (GSM8K, SVAMP, TabMWP, TAT-QA, FinQA)
- `--n-candidates, -n`: Number of candidates to generate (default: 100)
- `--output-dir, -o`: Output directory (default: candidates)
- `--model`: OpenAI chat model (default: from config)
- `--verbose, -v`: Enable detailed logging

### **Step 2: Train Policy Network**

Train neural policy networks for intelligent example selection.

```bash
# Train on different datasets
python train_policy.py --dataset TAT-QA --epochs 3
python train_policy.py --dataset GSM8K --epochs 3 --lr 3e-4
python train_policy.py --dataset FinQA --epochs 5 --lr 1e-4
python train_policy.py --dataset SVAMP --epochs 4 --lr 3e-4

# Advanced training options
python train_policy.py --dataset TabMWP --epochs 6 --verbose --save-best
python train_policy.py --dataset FinQA --epochs 5 --samples-per-epoch 50 --overwrite
```

#### **Training Process:**
1. **Candidate Loading**: Load generated candidates from Step 1
2. **Policy Initialization**: Create neural network with multi-head attention
3. **PPO Training**: Use reinforcement learning with reward function
4. **Validation**: Evaluate on held-out candidates
5. **Model Saving**: Save best and final models

#### **Training Configuration:**
The training process uses dataset-specific configurations optimized for each domain:
- **Learning rates**: Adjusted based on dataset complexity
- **Epochs**: More epochs for complex domains
- **Reward function**: Multi-objective balancing correctness, similarity, and diversity

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
# Full comparison on various datasets
python run_comparison.py --dataset TAT-QA --samples 150 --save-results
python run_comparison.py --dataset GSM8K --samples 100 --save-results

# Compare specific methods
python run_comparison.py --dataset GSM8K --methods policy,kate,cds --samples 50

# Quick test with fewer samples
python run_comparison.py --dataset SVAMP --samples 25 --methods policy,random

# Compare similarity-based methods only
python run_comparison.py --dataset FinQA --methods kate,cds --samples 30

# Test policy network only
python run_comparison.py --dataset TAT-QA --methods policy --samples 50
```

#### **Method Selection Options**
Available methods: `zero-shot`, `random`, `policy`, `kate`, `cds`

```bash
# AI vs. baseline methods
python run_comparison.py --dataset GSM8K --methods policy,random,zero-shot --samples 25

# Traditional similarity methods
python run_comparison.py --dataset FinQA --methods kate,cds,random --samples 20

# Full comparison (all 5 methods)
python run_comparison.py --dataset TAT-QA --samples 100  # Default: all methods
```

#### **Expected Output Format**
The comparison generates detailed results showing:
- Method-by-method accuracy scores
- Individual sample results with correctness flags
- Best performing method identification
- Statistical significance analysis

#### **Options**
- `--dataset, -d`: Dataset to test on (required)
- `--samples, -s`: Number of test samples (default: 10)
- `--methods`: Comma-separated methods (default: all 5 methods)
- `--seed`: Random seed for reproducibility (default: 42)
- `--save-results`: Save detailed JSON results
- `--verbose, -v`: Enable verbose logging

## 💰 **API Usage Tracking & Cost Monitoring**

### **Real-time Statistics**
```bash
# View usage statistics
python mathcorl.py stats                    # Last 24 hours
python mathcorl.py stats --hours 12         # Last 12 hours
python mathcorl.py stats --hours 1          # Last hour
```

**Sample Output Structure:**
- **Overview**: Total requests, success rate, tokens, cost, timing
- **By Method**: Breakdown per prompting/ICL method
- **By Model**: Usage statistics per AI model
- **Efficiency Metrics**: Cost-effectiveness analysis

### **Export and Analysis**
```bash
# Export usage data
python mathcorl.py export --format csv      # For Excel analysis
python mathcorl.py export --format json     # For programmatic analysis

# Generate visualization charts
python mathcorl.py chart --type all --save  # All chart types
python mathcorl.py chart --type cost --save # Cost analysis only
python mathcorl.py chart --type time --save # Time analysis only

# Clear tracking logs (with backup)
python mathcorl.py clear-logs
```

### **Cost Optimization Tips**
1. **Use gpt-4o-mini**: Default model for best cost/performance ratio
2. **Limit samples**: Start with small datasets for testing
3. **Monitor regularly**: Check `python mathcorl.py stats` frequently
4. **Export data**: Analyze patterns for optimization opportunities

## 🔬 **Research Applications & Best Practices**

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

### **Research Workflow Recommendations**

#### **Starting New Research**
1. **Begin with Task 1**: Compare prompting methods to understand baseline performance
2. **Monitor costs**: Use tracking to optimize API usage patterns
3. **Generate candidates**: Create diverse pools for ICL research
4. **Train policies**: Start with simpler datasets then expand to complex domains

#### **Experimental Design**
1. **Use consistent samples**: Set random seeds for reproducibility
2. **Save results**: Always use `--save-results` for later analysis
3. **Multiple runs**: Average results across different random seeds
4. **Track costs**: Monitor API usage to stay within budget

#### **Publication Pipeline**
1. **Export results**: Use JSON format for detailed analysis
2. **Generate charts**: Visual comparisons for papers
3. **Document configs**: Save all hyperparameters and settings
4. **Reproduce results**: Use saved models and seeds

## 🛠️ **Advanced Usage**

### **Custom Experiments**
```bash
# Batch experiments across datasets
for dataset in GSM8K SVAMP TabMWP TAT-QA FinQA; do
    python run_comparison.py --dataset $dataset --samples 50 --save-results
done

# Compare training configurations
python train_policy.py --dataset TAT-QA --epochs 5
python train_policy.py --dataset TAT-QA --epochs 10 --overwrite

# Cost analysis across methods
python mathcorl.py test --method fpp --dataset GSM8K --limit 100
python mathcorl.py test --method cot --dataset GSM8K --limit 100
python mathcorl.py stats
```

### **Troubleshooting**

#### **Policy Training Issues**
```bash
# Check candidates exist
ls -la candidates/TAT-QA.json

# Generate if missing
python generate_candidates.py --dataset TAT-QA --n-candidates 100

# Verify training
python train_policy.py --dataset TAT-QA --epochs 1 --verbose
```

#### **Evaluation Issues**
```bash
# Check model exists
ls -la models/TAT-QA_policy_best.pt

# Skip policy if model missing
python run_comparison.py --dataset TAT-QA --methods kate,cds,random --samples 10
```

#### **Cost Monitoring**
```bash
# Check current usage
python mathcorl.py stats --hours 1

# If quota exceeded, wait or upgrade plan
# Use smaller sample sizes for testing
python run_comparison.py --dataset TAT-QA --samples 10
```

---

**Happy Research!** 🚀 Whether you're comparing prompting methods, training neural policies, or optimizing API costs, MathCoRL provides comprehensive tools for mathematical reasoning research. 