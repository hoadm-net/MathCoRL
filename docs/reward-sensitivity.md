# Reward Sensitivity Analysis

Analysis of how reward weight configurations affect Policy Network performance in MathCoRL.

## Objective

Evaluate impact of different reward weight combinations (λ_accuracy, λ_similarity, λ_diversity) on:
- Policy training dynamics
- Final accuracy on mathematical reasoning tasks
- Example selection behavior

## Methodology

### Configurations Tested

| Configuration | λ_acc | λ_sim | λ_div | Rationale |
|---------------|-------|-------|-------|-----------|
| **Baseline** | 0.6 | 0.3 | 0.1 | Current default - balanced with accuracy priority |
| **Accuracy-focused** | 0.9 | 0.05 | 0.05 | Maximize correctness - minimal similarity/diversity |
| **Diversity-focused** | 0.4 | 0.5 | 0.1 | Emphasize semantic similarity - explore varied approaches |
| **Balanced** | 0.5 | 0.25 | 0.25 | Equal consideration of all objectives |
| **Length-penalized** | 0.5 | 0.2 | 0.3 | Prefer diverse examples - reduce redundancy |

### Evaluation Protocol

- **Datasets**: GSM8K, FinQA (representative elementary and financial domains)
- **Training**: 3 epochs per configuration
- **Seed**: Fixed at 42 for reproducibility
- **Samples**: Configurable (50-100 recommended for initial analysis)
- **Metrics**: Accuracy, average reward, training loss

## Usage

### Run Single Configuration

```bash
python scripts/reward_sensitivity.py \
    --dataset GSM8K \
    --config baseline \
    --epochs 3 \
    --samples 50 \
    --seed 42
```

### Run All Configurations

```bash
python scripts/reward_sensitivity.py \
    --dataset GSM8K \
    --all-configs \
    --epochs 3 \
    --samples 100 \
    --seed 42 \
    --plot
```

### Generate Visualizations

```bash
# If plots not generated during analysis
python scripts/plot_sensitivity.py \
    --results results/reward_sensitivity/GSM8K_all_configs_20251215_*.json \
    --output-dir results/reward_sensitivity/plots
```

## Expected Results

### Hypothesis 1: Accuracy Weight Dominance
**Prediction**: Higher λ_acc leads to better final accuracy but slower training convergence.
**Rationale**: Direct reward for correctness provides clearer gradient signal.

### Hypothesis 2: Similarity-Diversity Tradeoff
**Prediction**: Moderate similarity weight (0.2-0.3) balances relevance and exploration.
**Rationale**: Too high similarity may overfit to specific problem types; diversity helps generalization.

### Hypothesis 3: Configuration Stability
**Prediction**: Baseline (0.6, 0.3, 0.1) achieves near-optimal performance.
**Rationale**: Current weights emerged from empirical tuning; major deviations should degrade performance.

## Output Files

Analysis generates:

```
results/reward_sensitivity/
├── GSM8K_baseline.json              # Individual config results
├── GSM8K_accuracy_focused.json
├── GSM8K_diversity_focused.json
├── GSM8K_balanced.json
├── GSM8K_length_penalized.json
├── GSM8K_all_configs_YYYYMMDD_HHMMSS.json  # Combined results
└── plots/
    ├── accuracy_comparison.png      # Bar chart of final accuracies
    ├── training_curves.png          # Loss and reward over epochs
    ├── weight_heatmap.png          # Configuration weights visualization
    └── weight_accuracy_scatter.png  # Correlation analysis
```

## Interpreting Results

### Accuracy Comparison
- **Best configuration**: Highest accuracy indicates optimal weight balance
- **Confidence intervals**: Multiple runs needed for statistical significance
- **Dataset dependency**: Optimal weights may vary by domain

### Training Dynamics
- **Loss convergence**: Faster convergence suggests better reward signal
- **Reward trajectory**: Should increase steadily; plateaus indicate saturation
- **Stability**: Large fluctuations suggest hyperparameter tuning needed

### Weight Sensitivity
- **Strong sensitivity**: Large accuracy changes → weight is critical
- **Weak sensitivity**: Minimal changes → weight less important
- **Interaction effects**: Non-linear relationships between weights

## Next Steps

1. **Statistical Validation**: Run multiple seeds (42, 123, 456, 789, 1024) for confidence intervals
2. **Extended Analysis**: Test on all 5 datasets for generalization
3. **Fine-grained Search**: Grid search around best configuration (e.g., λ_acc ∈ [0.55, 0.65])
4. **Ablation Study**: Single-objective rewards (λ_acc=1.0, others=0.0) as baselines
5. **Adaptive Weights**: Dynamic weight scheduling during training

## Requirements

```bash
pip install matplotlib seaborn  # For plotting
```

Estimated runtime: 2-4 hours for full analysis (5 configs × 2 datasets × 3 epochs).

## References

- Policy Network Architecture: [docs/policy-selection-rules.md](../docs/policy-selection-rules.md)
- PPO Training: Schulman et al. (2017) "Proximal Policy Optimization Algorithms"
- Multi-objective RL: Liu et al. (2014) "Multi-objective reinforcement learning"
