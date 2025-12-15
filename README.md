# MathCoRL - Mathematical Intelligence with Reinforcement Learning

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![arXiv](https://img.shields.io/badge/arXiv-2501.xxxxx-b31b1b.svg)](https://arxiv.org/)

Research framework for mathematical reasoning with dual LLM providers (OpenAI, Claude) and reinforcement learning-based example selection.

## 🎯 **Dual Research Framework**

MathCoRL supports two complementary research directions with comprehensive tracking, evaluation, and dual LLM provider support:

### **📚 Task 1: Prompting Method Comparison**
Compare different prompting techniques for mathematical reasoning:
- **Interface**: Unified CLI interface with dual provider support
- **Methods**: FPP, CoT, PAL, PoT, Zero-shot
- **Providers**: OpenAI (GPT-4o, GPT-4, GPT-3.5) & Claude (3.5 Sonnet, Opus, Haiku)
- **Purpose**: Evaluate which prompting strategy and provider works best for mathematical problems
- **Features**: Real-time API tracking, cost monitoring, comprehensive evaluation, interactive mode

### **🧠 Task 2: In-Context Learning (ICL) Method Comparison**  
Compare different example selection strategies within Function Prototype Prompting:
- **Pipeline**: 3-script workflow for end-to-end ICL research
- **Methods**: Policy Network, KATE, CDS, Random Selection, Zero-shot
- **Providers**: Full support for both OpenAI and Claude models
- **Purpose**: Evaluate which example selection strategy works best for in-context learning
- **Features**: Neural policy networks, multi-objective training, reinforcement learning

## 🤖 **Dual LLM Provider Support**

### **OpenAI Integration**
- **Models**: GPT-4o, GPT-4, GPT-3.5-turbo (all variants)
- **Features**: Complete API integration with accurate token counting
- **Pricing**: Real-time cost tracking with up-to-date pricing
- **Status**: ✅ Fully supported and tested

### **Claude Integration** 
- **Models**: Claude 3.5 Sonnet, Claude 3 Opus, Claude 3 Haiku (all variants)
- **Features**: Native Anthropic API integration via LangChain
- **Pricing**: Comprehensive cost tracking for all Claude models
- **Status**: ✅ Fully supported and tested

### **Provider Switching**
```bash
# Use OpenAI (default)
python -m mint.cli solve --method fpp --provider openai --question "What is 15 + 27?"

# Use Claude  
python -m mint.cli solve --method fpp --provider claude --question "What is 15 + 27?"

# Set default provider in environment
export LLM_PROVIDER=claude  # or openai
```

## 📊 **Supported Research Datasets**

| Dataset | Domain | Size | Description | ICL k | Both Providers |
|---------|--------|------|-------------|-------|----------------|
| **GSM8K** | Elementary Math | 8.5K | Grade School Math word problems | 2 | ✅ |
| **SVAMP** | Arithmetic | 1K | Simple arithmetic word problems with variations | 2 | ✅ |
| **TabMWP** | Tabular Math | 38K | Math problems involving tables and charts | 2 | ✅ |
| **TAT-QA** | Financial QA | 16K | Table-and-text QA for financial documents | 3 | ✅ |
| **FinQA** | Financial Analysis | 8K | Complex financial reasoning and calculations | 2 | ✅ |

Each dataset includes:
- **Training set**: For candidate generation and policy training
- **Test set**: For evaluation and comparison
- **Cross-provider evaluation**: Test with both OpenAI and Claude
- **API cost tracking**: Monitor usage across providers

## 🚀 **Quick Start**

### **Requirements**
- **Python**: 3.8+ (tested on 3.10, 3.11, 3.13)
- **Memory**: 4GB minimum, 8GB recommended for Policy Network training
- **Storage**: 2GB for datasets and embeddings
- **API Keys**: OpenAI or Anthropic account with API access

### **Installation**
```bash
# Clone repository
git clone https://github.com/your-username/MathCoRL.git
cd MathCoRL

# Install dependencies
pip install -r requirements.txt

# Configure API keys
cp env.example .env
# Edit .env with your API keys:
# OPENAI_API_KEY=your_openai_key
# ANTHROPIC_API_KEY=your_anthropic_key  
# LLM_PROVIDER=openai  # or claude
```

### **Task 1: Compare Prompting Methods**
```bash
# Single problem solving with different methods and providers
python -m mint.cli solve --method fpp --question "What is 15 + 27?" --provider openai
python -m mint.cli solve --method cot --question "John has 20 apples. He gives 8 to his friend. How many are left?" --provider claude
python -m mint.cli solve --method pal --question "Calculate the average of 10, 20, 30" --provider openai

# Dataset evaluation with cross-provider testing
python -m mint.cli test --method fpp --dataset SVAMP --limit 100 --provider openai
python -m mint.cli test --method cot --dataset GSM8K --limit 50 --provider claude
python -m mint.cli test --method pot --dataset TabMWP --limit 30 --provider openai

# Interactive problem-solving mode
python -m mint.cli interactive --provider claude
python -m mint.cli interactive --provider openai

# Monitor API usage across providers
python -m mint.cli stats
python -m mint.cli stats --hours 12 --provider claude
python -m mint.cli export --format csv
```

### **Task 2: ICL Example Selection Methods**
```bash
# Step 1: Generate candidate examples with embeddings
python generate_candidates.py --dataset TAT-QA --n-candidates 100 --seed 42

# Step 2: Train Policy Network for example selection  
python train_policy.py --dataset TAT-QA --epochs 3 --seed 42

# Step 3: Compare ICL example selection strategies
python run_comparison.py --dataset TAT-QA --samples 150 --save-results --seed 42

# Reproducibility: Use same seed for consistent results
python run_comparison.py --dataset GSM8K --methods policy,kate,cds --samples 50 --seed 123
```

## 🔧 **Advanced Features**

### **API Tracking & Cost Monitoring**
```bash
# Real-time usage statistics
python -m mint.cli stats                    # All providers, last 24h
python -m mint.cli stats --hours 12         # Last 12 hours
python -m mint.cli stats --provider claude  # Claude only

# Export detailed usage data
python -m mint.cli export --format csv      # CSV export
python -m mint.cli export --format json     # JSON export

# Generate cost analysis charts  
python -m mint.cli chart --type cost --save
python -m mint.cli chart --type comparison --save
python -m mint.cli chart --type usage --save
```

### **Method Comparison Tools**
```bash
# Compare all prompting methods on dataset
python -m mint.cli compare --dataset SVAMP --limit 50 --provider openai

# Cross-provider method comparison
python -m mint.cli compare --dataset GSM8K --limit 30 --provider claude

# Ablation studies
python run_ablation_study.py --dataset SVAMP --methods fpp,cot,pal
python run_ablation_triple.py --dataset TabMWP --samples 100
```

### **Visualization & Analysis**
```bash
# Generate performance charts
python -m mint.cli chart --type performance --save

# Export results for analysis
python -m mint.cli export --format csv --save-path results/

# View training progress
python -m mint.cli training-history --dataset GSM8K
```

## 📈 **Research Methodology**

### **Prompting Methods** 
- **FPP (Function Prototype Prompting)**: Structured reasoning with explicit function calls
- **CoT (Chain-of-Thought)**: Step-by-step reasoning with natural language explanations
- **PAL (Program-aided Language)**: Programming-based problem solving with code execution
- **PoT (Program of Thoughts)**: Algorithmic decomposition with systematic thinking
- **Zero-shot**: Direct problem solving without examples or special prompting

### **ICL Example Selection Strategies**
- **Policy Network**: Neural network trained with reinforcement learning for adaptive selection
- **KATE (k-Nearest Examples)**: Semantic similarity-based selection using embeddings
- **CDS (Curriculum-based Selection)**: Progressive difficulty-based example ordering
- **Random Selection**: Random sampling baseline for controlled comparison
- **Zero-shot**: No examples baseline for measuring ICL contribution

### **Cross-Provider Analysis**
- **Performance Comparison**: Accuracy and reasoning quality across OpenAI vs Claude
- **Cost Efficiency**: Token usage and cost per problem solved
- **Method Suitability**: Which methods work best with which providers
- **Scaling Behavior**: Performance changes with different model sizes

## 🛠️ **Technical Architecture**

### **Core Components**
```
mint/                              # Core package
├── cli.py                         # Unified command-line interface
├── config.py                      # Multi-provider configuration
├── tracking.py                    # Universal API tracking
├── reproducibility.py             # Seed fixing for reproducibility
├── core.py                        # FPP implementation
├── cot.py, pal.py, pot.py        # Alternative prompting methods
├── zero_shot.py                   # Zero-shot baseline
├── icrl/                          # In-Context RL components
│   ├── candidate_generator.py     # Training example extraction
│   ├── policy_network.py         # Neural selection model
│   ├── trainer.py                # PPO training implementation
│   └── evaluator.py              # Multi-method evaluation
├── utils.py                       # Evaluation utilities
└── testing.py                    # Testing framework
```

### **Multi-Provider Workflow**
```
CLI Interface → Provider Selection → Method Execution → Universal Tracking → Results
     ↓                 ↓                    ↓                   ↓
   User Input    [OpenAI|Claude]    [FPP|CoT|PAL|PoT]    Cost/Token Tracking
```

## 🏆 **Key Features**

### **Comprehensive Functionality**
- ✅ **Dual LLM Provider Support**: Full OpenAI and Claude integration
- ✅ **Universal API Tracking**: Accurate cost monitoring across providers
- ✅ **Reproducibility**: Comprehensive seed fixing for consistent results
- ✅ **Complete Method Suite**: 5 prompting methods + 5 ICL strategies
- ✅ **Interactive CLI**: Real-time problem solving and testing
- ✅ **Advanced Visualization**: Charts, exports, and analysis tools
- ✅ **Reinforcement Learning**: Policy network training for example selection
- ✅ **Production Ready**: Comprehensive logging, error handling, and documentation

### **Research Capabilities**
- 🔬 **Method Comparison**: Systematic evaluation of reasoning approaches
- 📊 **Cross-Provider Analysis**: Performance comparison between OpenAI and Claude
- 💰 **Cost Optimization**: Detailed tracking for budget-conscious research
- 🎯 **ICL Research**: Advanced in-context learning with neural selection
- 📈 **Scalability**: Support for large-scale dataset evaluation
- 🔄 **Reproducibility**: Comprehensive configuration and result tracking

## 📚 **Documentation**

Comprehensive guides available in [docs/](docs/) directory:

- **[Usage Guide](docs/usage.md)**: Complete usage guide for both research tasks
- **[API Tracking](docs/tracking.md)**: API usage tracking and cost monitoring
- **[Tracking Examples](docs/tracking-examples.md)**: Practical examples with tracking
- **[Claude Integration](docs/claude.md)**: Claude setup and configuration
- **[Datasets](docs/datasets.md)**: Dataset descriptions and preprocessing
- **[Policy Network](docs/policy-network.md)**: Neural network architecture and training
- **[Charts & Visualization](docs/charts.md)**: Analysis and visualization tools
- **[Technical Notes](docs/refactoring.md)**: Implementation details and refactoring history

## 🎓 **Research Applications**

### **Prompting Research**
- Compare structured vs. free-form reasoning approaches
- Evaluate mathematical reasoning capabilities across different LLMs
- Study cost-effectiveness of different prompting strategies
- Analyze reasoning quality and interpretability

### **In-Context Learning Research**
- Investigate optimal example selection strategies
- Study reinforcement learning for demonstration selection
- Compare neural vs. similarity-based selection methods
- Explore curriculum learning effects in mathematical reasoning

### **Cross-Provider Analysis**
- Evaluate reasoning capabilities: OpenAI vs Claude
- Compare cost efficiency across providers and methods
- Study model-specific optimal prompting strategies
- Analyze scaling laws for mathematical reasoning

### **Cost Optimization Research**
- Track accuracy per dollar across methods and providers
- Optimize API usage for budget-constrained environments
- Study token efficiency patterns in mathematical reasoning

## 🛠️ **Configuration Options**

### **Environment Variables**
```bash
# Provider configuration
LLM_PROVIDER=openai                    # Default: openai | claude
OPENAI_API_KEY=your_openai_key         # Required for OpenAI
ANTHROPIC_API_KEY=your_anthropic_key   # Required for Claude

# Model selection
OPENAI_MODEL=gpt-4o-mini              # OpenAI model choice
ANTHROPIC_MODEL=claude-3-5-sonnet-20241022  # Claude model choice

# Generation parameters  
TEMPERATURE=0.1                        # Response randomness
MAX_TOKENS=4000                       # Maximum response length
```

### **Advanced Configuration**
```python
# Programmatic configuration
from mint.config import create_llm_client, get_config

# Create provider-specific clients
openai_client = create_llm_client(provider="openai")
claude_client = create_llm_client(provider="claude")

# Access configuration
config = get_config()
print(f"Current provider: {config.provider}")
print(f"Current model: {config.get_current_model_name()}")
```

## 🤝 **Contributing**

See [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines on:
- Code style and testing requirements
- Pull request process
- Research contribution areas

## 🐛 **Troubleshooting**

### Common Issues

**Import Error**: `ModuleNotFoundError: No module named 'mint'`
```bash
pip install -e .  # Install package in development mode
```

**API Key Error**: `openai.error.AuthenticationError`
```bash
# Verify .env file exists and contains valid keys
cat .env | grep API_KEY
export OPENAI_API_KEY=your_key_here  # Set directly if needed
```

**CUDA/MPS Device Error**: `RuntimeError: MPS backend out of memory`
```bash
# Use CPU instead of GPU
export PYTORCH_ENABLE_MPS_FALLBACK=1
# Or reduce batch size in configs/hyperparameters.yaml
```

**Embedding Generation Slow**: Taking too long on large datasets
```bash
# Use smaller candidate pools
python generate_candidates.py --n-candidates 50  # Default is 100
```

**Policy Network Training Unstable**: Loss not decreasing
```bash
# Adjust learning rate and epochs in configs/hyperparameters.yaml
# Try: lr: 0.0001 (lower) or epochs: 5 (more training)
```

For additional support, see [documentation](docs/) or open an issue on GitHub.

## 🤝 **Contributing**

MathCoRL welcomes contributions in:
- **New Prompting Methods**: Additional structured reasoning approaches
- **LLM Provider Integration**: Support for new language models
- **ICL Strategies**: Novel example selection algorithms
- **Datasets**: Additional mathematical reasoning domains
- **Evaluation Metrics**: Advanced correctness and efficiency measures
- **Cost Optimization**: More efficient API usage patterns

## 📜 **License**

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
