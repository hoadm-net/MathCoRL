# Policy Network Example Selection Rules

**Comprehensive documentation of how the Policy Network selects in-context examples for mathematical reasoning**

---

## 📋 Table of Contents

1. [Overview](#overview)
2. [Candidate Pool Construction](#candidate-pool-construction)
3. [Policy Network Architecture](#policy-network-architecture)
4. [Example Selection Process](#example-selection-process)
5. [Multi-Objective Reward Function](#multi-objective-reward-function)
6. [Training with PPO](#training-with-ppo)
7. [Inference Time Selection](#inference-time-selection)
8. [Configuration & Hyperparameters](#configuration--hyperparameters)

---

## Overview

The Policy Network is a neural network trained with Reinforcement Learning (PPO) to select optimal in-context examples for mathematical problem solving. Unlike static methods (KATE, CDS, Random), the policy learns from feedback to adaptively select examples that maximize problem-solving accuracy.

### Key Innovation
- **Learned Selection**: Neural network learns which examples help solve which problems
- **Multi-Objective**: Balances correctness, semantic similarity, and diversity
- **Adaptive**: Improves through reinforcement learning with GPT-4o-mini feedback

---

## Candidate Pool Construction

### Step 1: Initial Candidate Generation
**Script**: `generate_candidates.py`

```python
# For each problem in training set:
1. Generate Python solution using FPP (Function Prototype Prompting)
2. Execute code and validate against ground truth
3. Create embedding using OpenAI text-embedding-3-small (1536-D)
4. Store validated (problem, solution, embedding) triplet
```

**Dataset-Specific Pool Sizes** (from `configs/hyperparameters.yaml`):
- **GSM8K**: 20 candidates per problem
- **SVAMP**: 15 candidates per problem
- **TabMWP**: 25 candidates per problem
- **TAT-QA**: 25 candidates per problem
- **FinQA**: 30 candidates per problem

### Step 2: Training-Time Pool Sampling

During training, for each target problem:
```python
# From trainer.py - train_epoch()
available_candidates = [c for c in all_candidates if c != target_problem]
candidate_pool = random.sample(available_candidates, pool_size)
```

**Sampling Rules**:
- ✅ **Exclude self**: Target problem never appears in its own candidate pool
- ✅ **Random sampling**: Uniform random selection without replacement
- ✅ **Fixed pool size**: Consistent pool_size per dataset for fair comparison
- ❌ **No stratification**: Currently no difficulty-based stratification (future improvement)

**Rationale**: Random sampling ensures the policy learns to select from diverse contexts, preventing overfitting to specific candidate orderings.

---

## Policy Network Architecture

### Neural Network Design
**File**: `mint/icrl/policy_network.py`

```
Input Dimension: 1536-D (OpenAI text-embedding-3-small)
Hidden Dimension: 768-D (configurable)
Attention Heads: 8 (configurable)
Dropout: 0.1 (configurable)
```

### Architecture Components

#### 1. Input Projection
```python
problem_h = input_projection(problem_emb)      # [1, 768]
candidate_h = input_projection(candidate_embs)  # [N, 768]
```
Projects 1536-D embeddings to 768-D hidden space for efficient computation.

#### 2. Multi-Head Attention
```python
combined = [problem_h; candidate_h]  # Concatenate
attn_out = MultiheadAttention(query=combined, key=combined, value=combined)
```

**Purpose**: Learn relationships between:
- Problem ↔ Candidates (which examples are relevant?)
- Candidate ↔ Candidate (which combinations work well together?)

**8 Attention Heads**: Each head can focus on different aspects:
- Head 1: Mathematical concepts (algebra, geometry, etc.)
- Head 2: Problem difficulty level
- Head 3: Solution structure similarity
- Head 4-8: Other latent patterns

#### 3. Feed-Forward Network
```python
ffn = Linear(768 → 1536) → GELU → Dropout → Linear(1536 → 768)
```
Non-linear transformation to capture complex selection patterns.

#### 4. Scoring and Softmax
```python
scores = score_projection(candidate_repr)  # [N, 1]
probs = softmax(scores / temperature)      # [N]
```

**Adaptive Temperature**: Learned parameter `τ` controls exploration vs exploitation:
- **High τ** (early training): More uniform distribution, explore diverse examples
- **Low τ** (late training): Sharper distribution, exploit best examples
- **Formula**: `p_i = exp(score_i / τ) / Σ_j exp(score_j / τ)`

---

## Example Selection Process

### Training Time: Stochastic Sampling

```python
# From trainer.py - train_epoch()
probs = policy_net(problem_emb, candidate_embs)  # [N] probability distribution
dist = torch.distributions.Categorical(probs)
chosen_indices = dist.sample(sample_shape=(k,))  # Sample k examples
```

**Selection Strategy**: **Categorical sampling** (stochastic)
- Sample k times from probability distribution
- **Allows duplicates**: Same example can be chosen multiple times if highly relevant
- **Exploration**: Stochastic sampling enables policy to discover new strategies

**Why Stochastic?**
- Enables policy gradient learning (need probability of actions taken)
- Encourages exploration of suboptimal but promising examples
- Reduces variance through multiple samples per problem

### Inference Time: Greedy Selection

```python
# From evaluator.py - evaluate_with_policy()
probs = policy_net(problem_emb, candidate_embs)
top_k_indices = torch.topk(probs, k=k).indices  # Top-k most probable
chosen_examples = [candidates[i] for i in top_k_indices]
```

**Selection Strategy**: **Greedy Top-k** (deterministic)
- Select k examples with highest probabilities
- **No duplicates**: Each example chosen at most once
- **Exploitation**: Use learned policy's best judgment

**Why Greedy?**
- Deterministic for reproducibility
- Maximizes expected performance
- Reflects what policy learned as "best"

### Number of Examples (k)

**Dataset-Specific k values** (from `configs/hyperparameters.yaml`):
- **GSM8K**: k=2 examples
- **SVAMP**: k=2 examples
- **TabMWP**: k=3 examples (more complex, tables needed)
- **TAT-QA**: k=3 examples (financial reasoning)
- **FinQA**: k=3 examples (multi-step financial calculations)

**Rationale**: More complex datasets benefit from more examples showing diverse solution strategies.

---

## Multi-Objective Reward Function

### Reward Components

**Formula** (from `trainer.py - calculate_reward()`):
```
R_total = λ_acc · R_acc + λ_sim · R_sim + λ_div · R_div
```

**Default Weights** (configurable in `configs/hyperparameters.yaml`):
```yaml
reward:
  lambda_accuracy: 0.6   # Correctness is most important
  lambda_similarity: 0.3 # Semantic relevance matters
  lambda_diversity: 0.1  # Some diversity helps
```

### 1. Accuracy Reward (λ_acc = 0.6)

```python
R_acc = 1.0 if GPT_solved_correctly else 0.0
```

**Validation**: GPT-4o-mini generates solution using selected examples, checked against ground truth.

**Dominance**: 60% weight ensures policy prioritizes examples that lead to correct solutions.

### 2. Semantic Similarity Reward (λ_sim = 0.3)

```python
R_sim = cosine_similarity(problem_emb, mean(example_embs))
```

**Range**: [-1, 1], typically [0.5, 0.95] for relevant examples

**Purpose**: 
- Encourages selecting examples semantically related to target problem
- Helps policy learn problem-example relevance patterns
- Useful when accuracy signal is noisy

**Example**:
- Problem: "John has 20 apples..."
- High similarity: Examples about counting, arithmetic
- Low similarity: Examples about geometry, probability

### 3. Diversity Reward (λ_div = 0.1)

```python
if k >= 2:
    R_div = 1.0 - cosine_similarity(example_1_emb, example_2_emb)
else:
    R_div = 0.0
```

**Range**: [0, 1], higher when examples are dissimilar

**Purpose**:
- Prevents selecting k nearly-identical examples
- Encourages showing diverse solution approaches
- Helps with complex problems requiring multiple reasoning strategies

**Example** (k=2):
- Good: One arithmetic example + one word problem example
- Bad: Two nearly-identical arithmetic examples

### Reward Normalization

```python
# Typical reward ranges:
# R_acc: 0 or 1
# R_sim: 0.5 to 0.95
# R_div: 0.0 to 0.4
# R_total: 0.15 to 0.95 (0.6 when correct, < 0.3 when wrong)
```

**Implication**: Policy learns that getting correct answer is ~2x more valuable than just having semantically similar examples.

---

## Training with PPO

### Proximal Policy Optimization (PPO)

**Algorithm**: Industry-standard RL algorithm (used by ChatGPT, GPT-4)

#### PPO Objective
```python
L_PPO = min(
    ratio * advantages,
    clip(ratio, 1-ε, 1+ε) * advantages
)

where:
    ratio = π_new(a|s) / π_old(a|s)  # New policy / Old policy
    ε = 0.2  # Clip parameter (default)
```

**Key Insight**: PPO prevents policy from changing too drastically in one update, ensuring stable learning.

#### Training Loop (per epoch)

```python
# From trainer.py - train_epoch()
for problem in training_data:
    # 1. Sample candidate pool
    candidate_pool = random.sample(all_candidates, pool_size)
    
    # 2. Get old policy probabilities (before update)
    with torch.no_grad():
        old_probs = policy_net(problem_emb, candidate_embs)
    
    # 3. Sample examples using old policy
    chosen_indices = Categorical(old_probs).sample((k,))
    
    # 4. Evaluate with GPT
    is_correct = gpt_solve_with_examples(problem, chosen_examples)
    
    # 5. Calculate reward
    reward = calculate_reward(is_correct, problem_emb, example_embs)
    
    # 6. Get new policy probabilities (after potential update)
    new_probs = policy_net(problem_emb, candidate_embs)
    
    # 7. Calculate advantages (reward - baseline)
    advantages = reward - new_probs.mean()
    
    # 8. Compute PPO loss
    ratio = new_probs[chosen_indices] / old_probs[chosen_indices]
    loss_ppo = -min(ratio * advantages, clip(ratio, 0.8, 1.2) * advantages)
    
    # 9. Add regularization
    loss_kl = KL_divergence(new_probs || old_probs)  # Prevent drastic changes
    loss_contrastive = contrastive_loss(...)          # Representation learning
    
    total_loss = loss_ppo + 0.01 * loss_kl + 0.1 * loss_contrastive
    
    # 10. Backprop and update
    optimizer.zero_grad()
    total_loss.backward()
    clip_grad_norm_(policy_net.parameters(), max_norm=1.0)
    optimizer.step()
```

### Additional Training Techniques

#### 1. Gradient Clipping
```python
clip_grad_norm_(policy_net.parameters(), max_norm=1.0)
```
Prevents exploding gradients, common in RL training.

#### 2. KL Divergence Regularization
```python
loss_kl = KL_divergence(new_policy || old_policy)
total_loss += 0.01 * loss_kl
```
Penalizes large policy changes, similar to PPO clipping.

#### 3. Contrastive Learning
```python
# When problem solved correctly:
positive_examples = selected_examples
negative_examples = random_unselected_examples
loss_contrastive = contrastive_loss(problem, positives, negatives)
```
Improves embedding representations: pull correct examples closer, push incorrect away.

#### 4. Learning Rate Scheduling
```python
scheduler = CosineAnnealingLR(optimizer, T_max=num_epochs)
```
Gradually reduces learning rate for fine-tuning in later epochs.

### Training Hyperparameters

**From `configs/hyperparameters.yaml`**:
```yaml
training:
  optimizer: "Adam"
  default_lr: 3.0e-4
  default_epochs: 10
  eval_frequency: 3  # Evaluate every 3 epochs
  
  ppo:
    clip_epsilon: 0.2
    value_coef: 0.5
    entropy_coef: 0.01
```

**Dataset-Specific Learning Rates**:
- GSM8K: 3e-4 (simple, learns quickly)
- TabMWP: 2e-4 (moderate complexity)
- FinQA: 1e-4 (complex, needs careful tuning)

---

## Inference Time Selection

### Evaluation Pipeline

**Script**: `run_comparison.py` → `evaluator.py`

```python
def evaluate_with_policy(problem, candidates, policy_net, k):
    # 1. Create embeddings
    problem_emb = create_embedding(problem)
    candidate_embs = create_embeddings(candidates)
    
    # 2. Run policy network (GREEDY)
    probs = policy_net(problem_emb, candidate_embs)
    top_k_indices = torch.topk(probs, k=k).indices
    
    # 3. Select top-k examples
    chosen_examples = [candidates[i] for i in top_k_indices]
    
    # 4. Generate FPP prompt
    prompt = create_fpp_prompt(problem, chosen_examples)
    
    # 5. Call GPT-4o-mini
    response = openai_client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.0  # Deterministic
    )
    
    # 6. Extract and validate answer
    predicted_answer = extract_answer(response)
    is_correct = evaluate_result(predicted_answer, ground_truth)
    
    return is_correct, predicted_answer
```

### Comparison with Other Methods

| Method | Selection Strategy | Training Required | Adaptivity |
|--------|-------------------|-------------------|------------|
| **Policy Network** | Learned neural scoring | ✅ Yes (10 epochs) | ✅ High - Learns from feedback |
| **KATE** | k-Nearest Neighbors (cosine) | ❌ No | ❌ Low - Fixed similarity metric |
| **CDS** | Curriculum difficulty ordering | ❌ No | ⚠️ Medium - Uses difficulty heuristic |
| **Random** | Uniform random sampling | ❌ No | ❌ None - Completely random |
| **Zero-Shot** | No examples | ❌ No | ❌ None - No ICL |

### Inference Time Considerations

**Latency**:
- Policy forward pass: ~5-10ms (PyTorch, CPU)
- Embedding creation: ~50-100ms per problem (OpenAI API)
- GPT-4o-mini generation: ~2-5 seconds
- **Total overhead**: < 200ms (negligible vs GPT generation)

**Memory**:
- Policy network: ~15MB (PyTorch model)
- Candidate embeddings: ~6MB for 1000 candidates (1536-D float32)
- **Total**: < 25MB (deployable on edge devices)

**Cost**:
- Training: ~$10-20 per dataset (GPT-4o-mini evaluations)
- Inference: Only GPT generation cost, no extra API calls

---

## Configuration & Hyperparameters

### Complete Configuration Reference

**File**: `configs/hyperparameters.yaml`

```yaml
# Policy Network Architecture
policy_network:
  emb_dim: 1536          # Fixed (OpenAI embedding)
  hidden_dim: 768        # Internal representation size
  num_heads: 8           # Multi-head attention heads
  dropout: 0.1           # Regularization

# Training Settings
training:
  optimizer: "Adam"
  default_lr: 3.0e-4    # Learning rate
  default_epochs: 10
  eval_frequency: 3     # Evaluate every N epochs
  ppo:
    clip_epsilon: 0.2   # PPO clipping parameter
    value_coef: 0.5     # Value loss coefficient
    entropy_coef: 0.01  # Entropy bonus

# Reward Function
reward:
  lambda_accuracy: 0.6   # Correctness weight
  lambda_similarity: 0.3 # Semantic similarity weight
  lambda_diversity: 0.1  # Diversity weight
  normalize: true
  incorrect_penalty: -1.0
  timeout_penalty: -0.5

# Dataset-Specific Settings
datasets:
  GSM8K:
    k: 2                # Number of examples to select
    pool_size: 20       # Candidate pool size during training
    lr: 3.0e-4         # Dataset-specific learning rate
  
  TabMWP:
    k: 3
    pool_size: 25
    lr: 2.0e-4
  
  FinQA:
    k: 3
    pool_size: 30
    lr: 1.0e-4
```

### CLI Overrides

All hyperparameters can be overridden via command line:

```bash
# Override learning rate
python train_policy.py --dataset GSM8K --lr 1e-4

# Override pool size
python train_policy.py --dataset GSM8K --pool-size 30

# Override k
python train_policy.py --dataset GSM8K --k 3

# Override epochs
python train_policy.py --dataset GSM8K --epochs 15
```

### Reproducibility

```yaml
reproducibility:
  default_seed: 42           # Fixed random seed
  deterministic: true        # Force deterministic ops
  seed_python: true          # Python random
  seed_numpy: true           # NumPy
  seed_torch: true           # PyTorch
  seed_transformers: true    # Transformers lib
```

**Usage**:
```bash
python train_policy.py --dataset GSM8K --seed 42
python run_comparison.py --dataset GSM8K --seed 42
```

Same seed → Identical results (assuming same hardware/versions).

---

## Summary: Selection Rules at a Glance

### Training Phase
1. **Candidate Pool**: Random sample of `pool_size` candidates (exclude target)
2. **Selection**: Stochastic sampling from policy distribution (allows duplicates)
3. **Evaluation**: GPT-4o-mini solves problem with selected examples
4. **Reward**: Multi-objective (accuracy 60%, similarity 30%, diversity 10%)
5. **Update**: PPO loss + KL regularization + contrastive learning
6. **Repeat**: Multiple epochs until convergence

### Inference Phase
1. **Candidate Pool**: All available candidates
2. **Selection**: Greedy top-k from policy distribution (no duplicates)
3. **Prompt**: FPP with selected examples
4. **Generation**: GPT-4o-mini produces solution
5. **Validation**: Compare against ground truth

### Key Design Principles
- ✅ **Learned, not heuristic**: Neural network learns from feedback
- ✅ **Multi-objective**: Balances multiple desirable properties
- ✅ **Stable training**: PPO prevents catastrophic updates
- ✅ **Efficient inference**: < 200ms overhead, deployable
- ✅ **Configurable**: All hyperparameters in YAML
- ✅ **Reproducible**: Seed fixing for consistent results

---

## References

- **PPO Paper**: [Proximal Policy Optimization Algorithms](https://arxiv.org/abs/1707.06347) (Schulman et al., 2017)
- **In-Context Learning**: [What Makes Good In-Context Examples for GPT-3?](https://arxiv.org/abs/2101.06804) (Liu et al., 2021)
- **KATE**: [Learning To Retrieve Prompts for In-Context Learning](https://arxiv.org/abs/2112.08633) (Rubin et al., 2021)
- **Function Prototype Prompting**: [Code as Policies: Language Model Programs for Embodied Control](https://arxiv.org/abs/2209.07753) (Liang et al., 2022)

---