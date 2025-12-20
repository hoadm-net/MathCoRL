# MathCoRL - Mathematical Intelligence with Reinforcement Learning

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

Research framework for mathematical reasoning with multiple LLM backends (OpenAI API, Claude API, Open-Source HuggingFace models) and reinforcement learning-based example selection.

## 🎯 **Multi-Backend Research Framework**

MathCoRL supports three LLM backends for comprehensive mathematical reasoning research:

### **🔌 LLM Provider Support**

#### **1. OpenAI API**
- **Models**: GPT-4o, GPT-4, GPT-3.5-turbo (all variants)
- **Features**: Complete API integration with accurate token counting
- **Status**: ✅ Fully supported and tested

#### **2. Claude API** 
- **Models**: Claude 3.5 Sonnet, Claude 3 Opus, Claude 3 Haiku
- **Features**: Native Anthropic API integration via LangChain
- **Status**: ✅ Fully supported and tested

#### **3. Open-Source Models (HuggingFace)**
- **Models**: 
  - DeepSeek-R1 (1.5B, 7B, 14B)
  - Qwen2.5-Math (7B, 72B)
- **Features**: Local GPU inference, zero API cost
- **Requirements**: CUDA GPU recommended (tested on RTX 3090 24GB)
- **Status**: ✅ Fully supported with unified interface

### **📚 Prompting Methods**

Compare different prompting techniques:
- **Zero-Shot**: Direct problem solving without examples
- **Few-Shot**: Random example selection from candidate pool
- **FPP (Function Prototype Prompting)**: With policy network example selection
- **CoT, PAL, PoT**: Additional baseline methods (API models only)

### **🧠 In-Context Learning (ICL) Research**
Compare example selection strategies:
- **Policy Network**: Reinforcement learning-based selection
- **KATE**: K-nearest neighbors with embeddings
- **CDS**: Clustering-based diverse selection
- **Random**: Baseline random sampling

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

# Configure API keys (optional for open-source models)
cp env.example .env
# Edit .env with your API keys:
# OPENAI_API_KEY=your_openai_key        # For API models
# ANTHROPIC_API_KEY=your_anthropic_key  # For Claude
# LLM_PROVIDER=openai                   # Default provider (openai/claude)
```

### **Quick Start Examples**

#### **Option 1: Open-Source Models (Zero Cost)**
```bash
# Test with DeepSeek-R1 7B on GSM8K
python mathcorl_os.py test --method zero_shot --model deepseek_r1_7b --dataset GSM8K --samples 10

# Compare all 3 methods (zero-shot, few-shot, fpp+policy)
python mathcorl_os.py compare --model deepseek_r1_7b --dataset GSM8K --samples 50

# Test with Qwen2.5-Math 7B
python mathcorl_os.py compare --model qwen_math_7b --dataset TAT-QA --samples 50

# Available models: deepseek_r1_7b, deepseek_r1_1.5b, qwen_math_7b, qwen_math_72b
```

#### **Option 2: API Models (OpenAI/Claude)**
```bash
# Single problem solving
python -m mint.cli solve --method fpp --question "What is 15 + 27?" --provider openai
python -m mint.cli solve --method cot --question "John has 20 apples..." --provider claude

# Dataset evaluation
python -m mint.cli test --method fpp --dataset SVAMP --limit 100 --provider openai
python -m mint.cli test --method cot --dataset GSM8K --limit 50 --provider claude

# Interactive mode
python -m mint.cli interactive --provider openai
```

### **Policy Network Training & ICL Research**
```bash
# Step 1: Generate candidate examples with embeddings
python generate_candidates.py --dataset TAT-QA --n-candidates 30 --seed 42

# Step 2: Train Policy Network for example selection  
python train_policy.py --dataset TAT-QA --epochs 20 --seed 42

# Step 3: Compare ICL methods (works with both API and open-source)
python run_comparison.py --dataset TAT-QA --samples 101 --seed 42

# Test with open-source models + policy network
python mathcorl_os.py test --method fpp_policy --model deepseek_r1_7b --dataset GSM8K --samples 50
```

## 🔧 **Advanced Features**

### **API Tracking & Cost Monitoring (API Models)**
```bash
# Real-time usage statistics
python -m mint.cli stats                    # All providers, last 24h
python -m mint.cli stats --hours 12         # Last 12 hours
python -m mint.cli stats --provider claude  # Claude only

# Export detailed usage data
python -m mint.cli export --format csv      # CSV export
python -m mint.cli export --format json     # JSON export
```

### **Ablation Studies**
```bash
# Pool size ablation (ICL research)
python run_pool_size_ablation.py --dataset GSM8K --samples 101

# Method comparison ablation
python run_ablation_study.py --dataset SVAMP --methods fpp,cot,pal
```

## 📈 **Research Methodology**

### **Prompting Methods** 
- **Zero-Shot**: Direct problem solving without examples
- **Few-Shot**: Random k examples from candidate pool
- **FPP (Function Prototype Prompting)**: Structured reasoning with math functions + policy network selection
- **CoT (Chain-of-Thought)**: Step-by-step natural language reasoning (API only)
- **PAL/PoT**: Program-based reasoning (API only)

### **ICL Example Selection Strategies**
- **Policy Network**: Reinforcement learning-based adaptive selection (1536D→768D transformer)
- **KATE**: k-Nearest neighbors with embedding similarity
- **CDS**: Clustering-based diverse selection
- **Random**: Baseline random sampling

### **Multi-Backend Architecture**
- **API Models**: OpenAI/Claude via REST APIs with token tracking
- **Open-Source**: HuggingFace models with local GPU inference
- **Unified Interface**: Same prompting methods across all backends
- **Cost Comparison**: $0 for open-source vs API pricing

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
