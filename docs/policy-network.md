# Policy Network for In-Context Reinforcement Learning (ICRL)

## 🎯 Overview

This document describes the **Policy Network approach** for mathematical reasoning using **In-Context Reinforcement Learning (ICRL)**. The core idea is to learn an intelligent example selection policy that chooses the most helpful few-shot examples for solving mathematical problems.

## 🧠 Core Concept: In-Context Reinforcement Learning

### The Problem
Traditional few-shot prompting relies on:
- **Random example selection** - No guarantee examples are relevant
- **Fixed example sets** - Same examples for all problems
- **Manual curation** - Expensive and doesn't scale

### The ICRL Solution
Instead of random selection, we train a **Policy Network** that:
1. **Analyzes the target problem** using semantic embeddings
2. **Evaluates candidate examples** from a large pool
3. **Selects optimal examples** that maximize solving success
4. **Learns from feedback** to improve selection over time

This transforms static few-shot prompting into **adaptive, learned example selection**.

## 🏗️ Architecture

### Policy Network Design
```
Input: [Problem Embedding (1536-D), Candidate Embeddings (N×1536)]
       ↓
   Input Projection (1536 → 768)
       ↓
   Multi-Head Attention (8 heads, 768-D)
       ↓
   Feed-Forward Network + Residual Connection
       ↓
   Scoring & Selection Head
       ↓
Output: [Probability Distribution over N candidates]
```

### Key Components

#### 1. **Multi-Head Attention Mechanism**
- **Purpose**: Captures complex relationships between problem and candidates
- **Architecture**: 8 attention heads with 768-dimensional representations
- **Benefit**: Allows model to focus on different semantic aspects simultaneously
- **Implementation**: PyTorch MultiheadAttention with batch_first=True

#### 2. **Multi-Objective Reward Function**
- **Correctness Reward (60%)**: Whether selected examples lead to correct solutions
- **Semantic Similarity (30%)**: Cosine similarity between problem and examples
- **Diversity Reward (10%)**: Ensuring varied example types
- **Total**: Weighted combination for balanced learning

#### 3. **Adaptive Temperature Scaling**
- **Problem**: Fixed temperature may be too sharp or too soft for all problems
- **Solution**: Learnable temperature parameter that adapts during training
- **Range**: Constrained between 0.1 and 2.0 for stability
- **Result**: Better probability distributions for example selection

## 📚 Training Methodology

### Phase 1: Candidate Generation
1. **Dataset Processing**: Extract mathematical problems from datasets (TAT-QA, GSM8K, etc.)
2. **Solution Generation**: Use Function Prototype Prompting (FPP) to generate code solutions
3. **Embedding Creation**: Convert problem+context to 1536-D embeddings using OpenAI text-embedding-3-small
4. **Quality Filtering**: Keep only candidates with valid, executable code
5. **Validation**: Execute code and verify correctness against ground truth

### Phase 2: Policy Training (PPO Implementation)
1. **Problem Sampling**: Random selection of target problems from training set
2. **Candidate Pool**: Create diverse pool of potential examples for each problem
3. **Policy Selection**: Use current policy to select k examples (typically k=2-3)
4. **GPT Evaluation**: Generate solution using selected examples, check correctness
5. **Reward Calculation**: Multi-objective reward combining correctness, similarity, diversity
6. **Policy Update**: PPO loss with KL divergence regularization and gradient clipping

### Training Objectives
- **Primary**: Maximize success rate of selected examples
- **Secondary**: Maintain selection consistency across similar problems
- **Regularization**: Contrastive loss for better representation learning

## 🔬 Technical Details

### Embedding Strategy
```python
# Problem representation
problem_text = f"{context} {question}"
problem_embedding = openai.embeddings(problem_text)

# Candidate representation  
candidate_text = f"{candidate_context} {candidate_question}"
candidate_embedding = openai.embeddings(candidate_text)
```

### Selection Algorithm
```python
def select_examples(policy_net, problem_emb, candidate_embs, k=3):
    # Forward pass through policy network
    probs = policy_net(problem_emb, candidate_embs)
    
    # Sample k examples based on learned probabilities
    selected_indices = torch.multinomial(probs, k, replacement=False)
    
    return [candidates[i] for i in selected_indices]
```

### Loss Functions

#### 1. **PPO Loss with Multi-Objective Rewards**
```python
# Calculate multi-objective reward
accuracy_reward = 1.0 if is_correct else 0.0
similarity_reward = F.cosine_similarity(problem_emb, example_embs.mean(dim=0)).item()
diversity_reward = 1.0 - F.cosine_similarity(example_embs[0], example_embs[1]).item()

total_reward = 0.6 * accuracy_reward + 0.3 * similarity_reward + 0.1 * diversity_reward

# PPO loss with advantages
ratio = new_probs / (old_probs.detach() + 1e-8)
clipped_ratio = torch.clamp(ratio, 1 - epsilon, 1 + epsilon)
policy_loss = -torch.min(ratio * advantages, clipped_ratio * advantages).mean()
```

#### 2. **Contrastive Loss**
```python
# Pull positive examples closer, push negative ones away
pos_sim = F.cosine_similarity(problem_emb, positive_embs, dim=-1).mean()
neg_sim = F.cosine_similarity(problem_emb, negative_embs, dim=-1).mean()
contrastive_loss = -torch.log(torch.exp(pos_sim) / (torch.exp(pos_sim) + torch.exp(neg_sim)))
```

## 🎓 Theoretical Foundation

### Why This Works

#### 1. **Semantic Similarity ≠ Utility**
The policy learns to distinguish between different types of similarity:
- **Surface similarity** (similar keywords, domain) - Traditional methods rely on this
- **Structural similarity** (similar reasoning patterns) - Policy captures this
- **Solution utility** (examples that actually help solve the problem) - Policy optimizes for this

#### 2. **Learned vs. Heuristic Selection**
Traditional methods use fixed heuristics (cosine similarity, complexity ranking), while our policy learns optimal selection from data:
- **Adaptive criteria**: Selection strategy adapts to problem characteristics
- **Context awareness**: Considers both problem and available candidates
- **Feedback learning**: Improves from success/failure experiences

#### 3. **Multi-Objective Optimization**
Unlike single-metric approaches, our reward function balances multiple objectives:
- **Correctness**: Primary goal of solving problems correctly
- **Diversity**: Prevents mode collapse to similar examples
- **Similarity**: Ensures relevance to target problem

### Comparison with Alternatives

| Approach | Selection Strategy | Adaptability | Learning | Training Required |
|----------|-------------------|--------------|----------|-------------------|
| **Random** | Uniform sampling | None | No | No |
| **KATE** | Semantic similarity | Static | No | No |
| **CDS** | Curriculum-based | Semi-static | No | No |
| **Policy (Ours)** | Learned optimization | Dynamic | Yes | Yes |

## 🚀 Implementation Guide

### Setup Requirements
```bash
pip install torch transformers openai pandas numpy tqdm
```

### Basic Usage
```python
from mint.icrl.policy_network import PolicyNetwork
from mint.icrl.evaluator import PolicyNetworkEvaluator
import torch

# Initialize components
policy_net = PolicyNetwork(emb_dim=1536, hidden_dim=768)
evaluator = PolicyNetworkEvaluator()

# Load trained model
checkpoint = torch.load('models/dataset_policy_best.pt', map_location='cpu')
policy_net.load_state_dict(checkpoint['model_state_dict'])

# Select examples for a new problem
selected_examples = evaluator.select_with_policy(
    policy_net, problem_dict, candidate_pool, k=3
)
```

### Training Your Own Policy
```python
from mint.icrl.trainer import PolicyNetworkTrainer

# Initialize trainer
trainer = PolicyNetworkTrainer(
    dataset_name='TAT-QA',
    candidates_dir='candidates',
    models_dir='models'
)

# Train for specified epochs
training_history = trainer.train(
    num_epochs=3,
    save_best=True
)

print(f"Training completed successfully")
```

### Evaluation Framework
```python
from mint.icrl.evaluator import PolicyNetworkEvaluator

# Load trained policy
evaluator = PolicyNetworkEvaluator()

# Compare with baselines
results = evaluator.evaluate_policy_vs_random(
    policy_net=policy_net,
    dataset_candidates=candidates,
    dataset_name='TAT-QA',
    n_trials=150
)

print(f"Policy vs Random comparison completed")
```

## 🔄 Dataset-Specific Configurations

### Recommended Settings
| Dataset | k (examples) | Pool Size | Learning Rate | Epochs | Expected Behavior |
|---------|--------------|-----------|---------------|--------|-------------------|
| **GSM8K** | 2 | 20 | 3e-4 | 3 | Fast convergence on arithmetic patterns |
| **SVAMP** | 2 | 15 | 3e-4 | 4 | Good handling of linguistic variations |
| **TabMWP** | 2 | 25 | 2e-4 | 4 | Effective table structure recognition |
| **TAT-QA** | 3 | 25 | 2e-4 | 3 | Complex financial reasoning support |
| **FinQA** | 2 | 30 | 1e-4 | 5 | Multi-step financial calculations |

### Training Commands
```bash
# Train on different datasets
python train_policy.py --dataset TAT-QA --epochs 3
python train_policy.py --dataset GSM8K --epochs 3 --lr 3e-4
python train_policy.py --dataset FinQA --epochs 5 --lr 1e-4
python train_policy.py --dataset SVAMP --epochs 4 --lr 3e-4
python train_policy.py --dataset TabMWP --epochs 4 --lr 2e-4
```

## 🔮 Future Directions

### Research Opportunities
1. **Cross-Dataset Transfer**: Train on one dataset, evaluate on others
2. **Dynamic k Selection**: Learn optimal number of examples per problem
3. **Hierarchical Policies**: Different policies for different problem types
4. **Meta-Learning**: Quick adaptation to new mathematical domains
5. **Multi-Modal**: Incorporate visual reasoning for TabMWP

### Engineering Improvements
1. **Efficient Inference**: Reduce selection time for real-time applications
2. **Distributed Training**: Scale to larger candidate pools and datasets
3. **Online Learning**: Continuously improve from user feedback
4. **Robustness**: Handle out-of-distribution problems gracefully

### Empirical Studies
1. **Ablation Studies**: Impact of different reward components
2. **Architecture Search**: Optimal network design for mathematical reasoning
3. **Data Efficiency**: Minimum training data for effective policies
4. **Failure Analysis**: When and why policy selection fails

## 📖 References & Related Work

This approach builds upon recent advances in:
- **In-Context Learning**: Understanding how LLMs use few-shot examples
- **Reinforcement Learning**: Policy gradient methods for discrete selection
- **Representation Learning**: Semantic embeddings for mathematical reasoning
- **Meta-Learning**: Learning to learn from examples

### Key Innovations
1. **Multi-Objective Reward**: Balancing correctness, similarity, and diversity
2. **Attention-Based Selection**: Using transformer architecture for example selection
3. **End-to-End Pipeline**: Complete system from candidate generation to evaluation
4. **Domain Adaptability**: Configurable for different mathematical reasoning domains

## 🤝 Contributing

Contributions are welcome! Key areas for improvement:
- Novel policy architectures for mathematical reasoning
- Better training objectives and reward functions
- Evaluation on additional mathematical domains
- Transfer learning between datasets

See the main README for development setup and contribution guidelines.

---

**🎯 Policy Network Vision**: This approach demonstrates how reinforcement learning can be applied to optimize in-context learning for mathematical reasoning, providing an adaptive alternative to traditional heuristic-based example selection methods. 