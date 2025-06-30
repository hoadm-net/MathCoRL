# MathCoRL - Mathematical Intelligence with Reinforcement Learning

🚀 **Comprehensive framework for mathematical reasoning research with dual research capabilities**

## 🎯 **Dual Research Framework**

MathCoRL supports two complementary research directions:

### **📚 Task 1: Prompting Method Comparison**
Compare different prompting techniques for mathematical reasoning:
- **Tool**: `mathcorl.py` (unified CLI interface)
- **Methods**: FPP, CoT, PAL, PoT, Zero-shot
- **Purpose**: Evaluate which prompting strategy works best for mathematical problems

### **🧠 Task 2: In-Context Learning (ICL) Method Comparison**  
Compare different example selection strategies within Function Prototype Prompting:
- **Tool**: 3-script pipeline for end-to-end ICL research
- **Methods**: FPP + Policy Network, FPP + KATE, FPP + CDS, FPP + Random, FPP + Zero-shot
- **Purpose**: Evaluate which example selection strategy works best for in-context learning

## 📊 **Supported Research Datasets**

| Dataset | Domain | Size | Description |
|---------|--------|------|-------------|
| **GSM8K** | Elementary Math | 8.5K | Grade School Math word problems |
| **SVAMP** | Arithmetic | 1K | Simple arithmetic word problems with variations |
| **TabMWP** | Tabular Math | 38K | Math problems involving tables and charts |
| **TAT-QA** | Financial QA | 16K | Table-and-text QA for financial documents |
| **FinQA** | Financial Analysis | 8K | Complex financial reasoning and calculations |

Each dataset includes:
- **Training set**: For candidate generation and policy training
- **Test set**: For evaluation and comparison
- **Domain-specific complexity**: Different mathematical reasoning challenges

## 🚀 **Quick Start**

### **Task 1: Compare Prompting Methods**
```bash
# Single problem solving with different methods
python mathcorl.py solve --method fpp --question "What is 15 + 27?"
python mathcorl.py solve --method cot --question "John has 20 apples. He gives 8 to his friend. How many are left?"
python mathcorl.py solve --method pal --question "Calculate the average of 10, 20, 30"

# Dataset evaluation with different prompting methods  
python mathcorl.py test --method fpp --dataset SVAMP --limit 100
python mathcorl.py test --method cot --dataset GSM8K --limit 50
python mathcorl.py test --method pot --dataset TabMWP --limit 30

# Compare multiple prompting methods
python mathcorl.py compare --dataset SVAMP --limit 20
```

### **Task 2: Compare ICL Example Selection Methods**
```bash
# Step 1: Generate candidate examples with embeddings
python generate_candidates.py --dataset FinQA --n-candidates 100

# Step 2: Train Policy Network for example selection
python train_policy.py --dataset FinQA --epochs 5

# Step 3: Compare ICL example selection strategies
python run_comparison.py --dataset FinQA --samples 20 --save-results

# Compare specific ICL methods
python run_comparison.py --dataset GSM8K --methods policy,kate,cds --samples 50
```

## 📈 **Research Methodology Focus**

### **Task 1: Prompting Method Comparison**
- **FPP**: Strong structured reasoning with explicit function usage
- **CoT**: Excellent for multi-step reasoning with interpretable steps  
- **PAL**: High computational accuracy through programming flexibility
- **PoT**: Systematic problem decomposition with algorithmic thinking
- **Zero-shot**: Baseline capability measurement

### **Task 2: ICL Example Selection Strategies**
- **Policy Network**: Adaptive, learned selection strategies
- **KATE**: Semantic similarity-based, simple and effective approach
- **CDS**: Curriculum learning with balanced difficulty progression  
- **Random**: Control baseline for measuring example value
- **Zero-shot**: No-example baseline

## 🔧 **Configuration**

### **Environment Setup**
```bash
# Copy and configure API keys
cp env.example .env
# Edit .env with your OpenAI API key

# Model configuration
DEFAULT_MODEL=gpt-4.1-mini           # Primary reasoning model
EMBEDDING_MODEL=text-embedding-3-small  # Semantic embeddings  
TEMPERATURE=0.1                      # Generation temperature
```

### **Dataset-Specific Configurations**
| Dataset | ICL Examples (k) | Candidate Pool | Policy LR | Epochs |
|---------|------------------|----------------|-----------|--------|
| GSM8K   | 2               | 20             | 3e-4      | 3      |
| SVAMP   | 2               | 15             | 3e-4      | 4      |
| TabMWP  | 2               | 25             | 2e-4      | 4      |
| TAT-QA  | 3               | 25             | 2e-4      | 5      |
| FinQA   | 2               | 30             | 1e-4      | 5      |

## 📚 **Documentation**

- **[README_USAGE.md](README_USAGE.md)**: Detailed usage guide for both research tasks
- **[README_DATASETS.md](README_DATASETS.md)**: Dataset descriptions and preprocessing details
- **[README_POLICY_NETWORK.md](README_POLICY_NETWORK.md)**: Policy network architecture and training details

## 🎓 **Research Applications**

### **For Prompting Research**
- Compare prompting techniques across mathematical domains
- Evaluate structured vs. free-form reasoning approaches
- Study impact of function constraints on mathematical accuracy

### **For ICL Research**  
- Investigate optimal example selection strategies
- Study curriculum learning effects in mathematical reasoning
- Analyze policy network vs. similarity-based selection
- Explore reinforcement learning for in-context demonstration

## 🛠️ **Technical Architecture**

### **Core Components**

```
mint/                           # Core mathematical intelligence package
├── core.py                     # FPP solver and base functionality
├── cot.py, pal.py, pot.py     # Alternative prompting methods
├── zero_shot.py               # Zero-shot baseline
├── icrl/                      # In-Context Reinforcement Learning
│   ├── candidate_generator.py # Candidate extraction and validation
│   ├── policy_network.py     # Neural network for example selection
│   ├── trainer.py            # PPO training implementation
│   └── evaluator.py          # Policy evaluation and testing
├── utils.py                   # Mathematical evaluation utilities
└── config.py                 # Model and embedding configuration
```

### **Workflow Architecture**

**Task 1 Workflow** (Prompting Comparison):
```
mathcorl.py → mint.cli → {fpp, cot, pal, pot, zero_shot}.py → Results
```

**Task 2 Workflow** (ICL Comparison):
```
generate_candidates.py → train_policy.py → run_comparison.py → Results
       ↓                       ↓                    ↓
   Candidate Pool          Policy Model      Method Comparison
```

## 🤝 **Contributing**

MathCoRL welcomes research contributions in:
- **New prompting methods**: Additional structured reasoning approaches
- **ICL strategies**: Novel example selection algorithms  
- **Datasets**: Additional mathematical reasoning domains
- **Evaluation metrics**: Advanced mathematical correctness measures

## 📜 **License**

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

🚀 **Happy Mathematical Reasoning Research!** Whether you're studying prompting techniques or in-context learning strategies, MathCoRL provides the tools for comprehensive mathematical intelligence research.