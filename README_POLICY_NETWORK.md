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
   Multi-Head Attention (8 heads)
       ↓
   Feed-Forward Network + Residual
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

#### 2. **Contrastive Learning Framework**
- **Positive Examples**: Successfully selected examples that led to correct solutions
- **Negative Examples**: Poorly selected examples that led to wrong solutions
- **Loss Function**: Contrastive loss that pulls positive examples closer, pushes negative ones away

#### 3. **Adaptive Temperature Scaling**
- **Problem**: Fixed temperature may be too sharp or too soft for all problems
- **Solution**: Learnable temperature parameter that adapts during training
- **Range**: Constrained between 0.1 and 2.0 for stability

## 📚 Training Methodology

### Phase 1: Candidate Generation
1. **Dataset Processing**: Extract mathematical problems from FinQA dataset
2. **Solution Generation**: Use Function Prototype Prompting (FPP) to generate code solutions
3. **Embedding Creation**: Convert problem+context to 1536-D embeddings using OpenAI text-embedding-3-small
4. **Quality Filtering**: Keep only candidates with valid, executable code

### Phase 2: Policy Training
1. **Problem Sampling**: Random selection of target problems from training set
2. **Candidate Pool**: Create diverse pool of potential examples for each problem
3. **Policy Selection**: Use current policy to select k examples (typically k=2)
4. **Execution & Evaluation**: Generate solution using selected examples, check correctness
5. **Loss Computation**: Calculate policy gradient loss based on success/failure
6. **Backpropagation**: Update policy parameters to improve selection

### Training Objectives
- **Primary**: Maximize success rate of selected examples
- **Secondary**: Minimize selection variance (consistency)
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
def select_examples(policy_net, problem_emb, candidate_embs, k=2):
    # Forward pass through policy network
    probs = policy_net(problem_emb, candidate_embs)
    
    # Sample k examples based on learned probabilities
    selected_indices = torch.multinomial(probs, k, replacement=False)
    
    return [candidates[i] for i in selected_indices]
```

### Loss Functions

#### 1. **Policy Gradient Loss**
```python
# REINFORCE-style policy gradient
log_probs = torch.log(action_probs)
policy_loss = -(log_probs * rewards).sum()
```

#### 2. **Contrastive Loss**
```python
# Pull positive examples closer, push negative ones away
pos_sim = cosine_similarity(problem_emb, positive_embs).mean()
neg_sim = cosine_similarity(problem_emb, negative_embs).mean()
contrastive_loss = -torch.log(exp(pos_sim) / (exp(pos_sim) + exp(neg_sim)))
```

## 🎓 Theoretical Foundation

### Why This Works

#### 1. **Semantic Similarity ≠ Utility**
Not all semantically similar examples are equally helpful for solving a problem. The policy learns to distinguish between:
- **Surface similarity** (similar keywords, domain)
- **Structural similarity** (similar reasoning patterns)
- **Solution similarity** (similar mathematical operations)

#### 2. **Contextual Adaptation**
The attention mechanism allows the policy to adapt selection criteria based on:
- **Problem complexity** (simple vs. multi-step reasoning)
- **Domain specifics** (financial vs. arithmetic problems)
- **Available examples** (quality and diversity of candidate pool)

#### 3. **Transfer Learning**
Pre-trained embeddings (OpenAI text-embedding-3-small) provide:
- **Rich semantic representations** from large-scale training
- **Domain knowledge** about mathematical and financial concepts
- **Compositional understanding** of complex problem structures

### Comparison with Alternatives

| Approach | Selection Strategy | Adaptability | Learning |
|----------|-------------------|--------------|----------|
| **Random** | Uniform sampling | None | No |
| **Retrieval** | Semantic similarity | Static | No |
| **Manual** | Human curation | Limited | No |
| **ICRL (Ours)** | Learned policy | Dynamic | Yes |

## 🚀 Implementation Guide

### Setup Requirements
```bash
pip install torch transformers openai pandas numpy
```

### Basic Usage
```python
from mint.icrl.policy_network import PolicyNetwork
from mint.icrl.candidate_generator import CandidateGenerator
from mint.icrl.evaluator import PolicyNetworkEvaluator

# Initialize components
policy_net = PolicyNetwork(emb_dim=1536, hidden_dim=768)
generator = CandidateGenerator()
evaluator = PolicyNetworkEvaluator()

# Load trained model
policy_net.load_state_dict(torch.load('models/policy_best.pt'))

# Select examples for a new problem
problem_embedding = generator.create_embedding(problem_text)
selected_examples = evaluator.select_with_policy(
    policy_net, problem_dict, candidate_pool, k=2
)
```

### Training Your Own Policy
```python
from mint.icrl.trainer import PolicyNetworkTrainer

# Initialize trainer
trainer = PolicyNetworkTrainer(
    policy_net=policy_net,
    candidates=candidates,
    learning_rate=1e-4,
    batch_size=16
)

# Train for specified epochs
trainer.train(
    n_epochs=10,
    save_path='models/my_policy.pt'
)
```

## 🔄 Evaluation Framework

### Metrics
1. **Selection Accuracy**: Success rate with policy-selected examples
2. **Improvement over Baseline**: Comparison with random selection
3. **Consistency**: Variance in selection quality across problems
4. **Efficiency**: Training time and convergence speed

### Evaluation Protocol
1. **Hold-out Test Set**: Separate problems not seen during training
2. **Multiple Runs**: Average results across different random seeds
3. **Ablation Studies**: Compare different architectural choices
4. **Human Evaluation**: Qualitative assessment of example relevance

## 🔮 Future Directions

### Research Opportunities
1. **Multi-Modal Learning**: Incorporate table/figure understanding
2. **Dynamic k Selection**: Learn optimal number of examples per problem
3. **Hierarchical Policies**: Different policies for different problem types
4. **Meta-Learning**: Quick adaptation to new mathematical domains

### Engineering Improvements
1. **Efficient Inference**: Reduce selection time for real-time applications
2. **Distributed Training**: Scale to larger candidate pools
3. **Online Learning**: Continuously improve from user feedback
4. **Robustness**: Handle out-of-distribution problems gracefully

## 📖 References

This approach builds upon recent advances in:
- **In-Context Learning**: Understanding how LLMs use few-shot examples
- **Reinforcement Learning**: Policy gradient methods for discrete selection
- **Representation Learning**: Semantic embeddings for mathematical reasoning
- **Meta-Learning**: Learning to learn from examples

## 🤝 Contributing

Contributions are welcome! Key areas for improvement:
- Novel policy architectures
- Better training objectives
- Evaluation metrics
- New application domains

See the main README for development setup and contribution guidelines.

---

*This Policy Network approach represents a significant advancement in making few-shot learning more intelligent and adaptive for mathematical reasoning tasks.* 