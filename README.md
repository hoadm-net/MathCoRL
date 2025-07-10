# MathCoRL - Mathematical Intelligence with Reinforcement Learning

🚀 **Comprehensive framework for mathematical reasoning research with dual research capabilities and API tracking**

## 🎯 **Dual Research Framework**

MathCoRL supports two complementary research directions with comprehensive tracking and evaluation:

### **📚 Task 1: Prompting Method Comparison**
Compare different prompting techniques for mathematical reasoning:
- **Tool**: `mathcorl.py` (unified CLI interface)
- **Methods**: FPP, CoT, PAL, PoT, Zero-shot
- **Purpose**: Evaluate which prompting strategy works best for mathematical problems
- **Features**: Real-time API tracking, cost monitoring, comprehensive evaluation

### **🧠 Task 2: In-Context Learning (ICL) Method Comparison**  
Compare different example selection strategies within Function Prototype Prompting:
- **Tool**: 3-script pipeline for end-to-end ICL research
- **Methods**: FPP + Policy Network, FPP + KATE, FPP + CDS, FPP + Random, FPP + Zero-shot
- **Purpose**: Evaluate which example selection strategy works best for in-context learning
- **Features**: Neural policy networks, multi-objective training, comprehensive comparison

## 📊 **Supported Research Datasets**

| Dataset | Domain | Size | Description | ICL k | Policy Training |
|---------|--------|------|-------------|-------|-----------------|
| **GSM8K** | Elementary Math | 8.5K | Grade School Math word problems | 2 | Available |
| **SVAMP** | Arithmetic | 1K | Simple arithmetic word problems with variations | 2 | Available |
| **TabMWP** | Tabular Math | 38K | Math problems involving tables and charts | 2 | Available |
| **TAT-QA** | Financial QA | 16K | Table-and-text QA for financial documents | 3 | Available |
| **FinQA** | Financial Analysis | 8K | Complex financial reasoning and calculations | 2 | Available |

Each dataset includes:
- **Training set**: For candidate generation and policy training
- **Test set**: For evaluation and comparison
- **Domain-specific complexity**: Different mathematical reasoning challenges
- **API cost tracking**: Monitor usage and optimization

## 🚀 **Quick Start**

### **Task 1: Compare Prompting Methods (with API Tracking)**
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

# Monitor API usage and costs
python mathcorl.py stats
python mathcorl.py stats --hours 12
python mathcorl.py export --format csv
```

### **Task 2: Compare ICL Example Selection Methods**
```bash
# Step 1: Generate candidate examples with embeddings
python generate_candidates.py --dataset TAT-QA --n-candidates 100

# Step 2: Train Policy Network for example selection
python train_policy.py --dataset TAT-QA --epochs 3

# Step 3: Compare ICL example selection strategies
python run_comparison.py --dataset TAT-QA --samples 150 --save-results

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
- **Policy Network**: Adaptive, learned selection strategies using neural networks
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
DEFAULT_MODEL=gpt-4o-mini           # Primary reasoning model
EMBEDDING_MODEL=text-embedding-3-small  # Semantic embeddings  
TEMPERATURE=0.1                      # Generation temperature
```

### **Dataset-Specific Configurations**
| Dataset | ICL Examples (k) | Candidate Pool | Policy LR | Epochs | Training Status |
|---------|------------------|----------------|-----------|--------|------------------|
| GSM8K   | 2               | 20             | 3e-4      | 3      | Available        |
| SVAMP   | 2               | 15             | 3e-4      | 4      | Available        |
| TabMWP  | 2               | 25             | 2e-4      | 4      | Available        |
| TAT-QA  | 3               | 25             | 2e-4      | 3      | Available        |
| FinQA   | 2               | 30             | 1e-4      | 5      | Available        |

## 🛠️ **API Usage Tracking**

### **Real-time Monitoring**
- ✅ **100% Accurate Token Counting**: Direct from OpenAI API metadata
- ✅ **Precise Cost Calculation**: Based on actual token usage
- ✅ **Method Comparison**: Track efficiency across different approaches
- ✅ **Export Capabilities**: CSV/JSON export for analysis

### **Tracking Commands**
```bash
# View usage statistics
python mathcorl.py stats --hours 24

# Export usage data
python mathcorl.py export --format csv
python mathcorl.py export --format json

# Clear tracking logs
python mathcorl.py clear-logs

# Generate visual charts
python mathcorl.py chart --type all --save
```

## 📚 **Comprehensive Documentation**

- **[README_USAGE.md](README_USAGE.md)**: Complete usage guide for both research tasks
- **[README_DATASETS.md](README_DATASETS.md)**: Dataset descriptions and preprocessing details
- **[README_POLICY_NETWORK.md](README_POLICY_NETWORK.md)**: Policy network architecture and training details
- **[README_TRACKING.md](README_TRACKING.md)**: API usage tracking and cost monitoring
- **[README_CHARTS.md](README_CHARTS.md)**: Visualization and analysis tools
- **[USAGE_WITH_TRACKING.md](USAGE_WITH_TRACKING.md)**: Practical usage examples with tracking

## 🎓 **Research Applications**

### **For Prompting Research**
- Compare prompting techniques across mathematical domains
- Evaluate structured vs. free-form reasoning approaches
- Study impact of function constraints on mathematical accuracy
- Monitor computational costs and efficiency

### **For ICL Research**  
- Investigate optimal example selection strategies
- Study curriculum learning effects in mathematical reasoning
- Analyze policy network vs. similarity-based selection
- Explore reinforcement learning for in-context demonstration

### **For Cost Optimization Research**
- Compare method efficiency (accuracy per dollar)
- Study token usage patterns across different approaches
- Optimize API calls for budget-constrained environments

## 🛠️ **Technical Architecture**

### **Core Components**

```
mint/                           # Core mathematical intelligence package
├── core.py                     # FPP solver and base functionality
├── cot.py, pal.py, pot.py     # Alternative prompting methods
├── zero_shot.py               # Zero-shot baseline
├── tracking.py                # API usage tracking and monitoring
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
mathcorl.py → mint.cli → {fpp, cot, pal, pot, zero_shot}.py → tracking.py → Results
```

**Task 2 Workflow** (ICL Comparison):
```
generate_candidates.py → train_policy.py → run_comparison.py → Results
       ↓                       ↓                    ↓
   Candidate Pool          Policy Model      Method Comparison
```

## 🏆 **Key Features**

### **Implemented Capabilities**
- ✅ **Full Policy Network Training**: PPO-based training with multi-objective rewards
- ✅ **Comprehensive Evaluation**: Multi-method comparison framework
- ✅ **API Cost Tracking**: 100% accurate token counting and cost monitoring
- ✅ **Multi-Method Support**: 5 different ICL strategies fully implemented
- ✅ **Production Ready**: Complete CLI tools, logging, and result export

### **Research Contributions**
- 🎯 **Novel Policy Network Architecture**: Multi-head attention for example selection
- 📊 **Empirical Framework**: Systematic comparison of ICL methods
- 💰 **Cost Optimization**: Comprehensive tracking for budget-conscious research
- 🔄 **Reproducible Pipeline**: End-to-end research workflow with proper documentation

## 🤝 **Contributing**

MathCoRL welcomes research contributions in:
- **New prompting methods**: Additional structured reasoning approaches
- **ICL strategies**: Novel example selection algorithms  
- **Datasets**: Additional mathematical reasoning domains
- **Evaluation metrics**: Advanced mathematical correctness measures
- **Cost optimization**: More efficient API usage patterns

## 📜 **License**

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

🚀 **Happy Mathematical Reasoning Research!** Whether you're studying prompting techniques, in-context learning strategies, or optimizing API costs, MathCoRL provides comprehensive tools for mathematical intelligence research.