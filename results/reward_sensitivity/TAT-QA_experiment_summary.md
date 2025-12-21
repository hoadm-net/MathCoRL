# Reward Sensitivity Analysis - TAT-QA Dataset

**Experiment Status**: Running (Started at 06:16, Expected completion: ~10:00)  
**Dataset**: TAT-QA  
**Samples**: 151  
**Epochs per config**: 5  
**Seed**: 42  

## 🎯 Reward Configurations Being Tested

| Config Name | λ_acc | λ_sim | λ_div | Description | Expected Outcome |
|-------------|-------|-------|-------|-------------|------------------|
| **Accuracy-focused** | 0.90 | 0.05 | 0.05 | Maximize correctness - minimal similarity/diversity | ✅ Highest accuracy <br> ❌ Higher token usage |
| **Balanced (default)** | 0.60 | 0.30 | 0.10 | Current default - balanced with accuracy priority | ✅ Good balance <br> ✅ Empirically tuned |
| **Diversity-focused** | 0.40 | 0.50 | 0.10 | Emphasize semantic similarity - explore varied approaches | ⚠️ Lower accuracy <br> ✅ Better generalization |
| **Balanced (equal)** | 0.50 | 0.25 | 0.25 | Equal consideration of all objectives | ⚖️ Middle ground <br> 🔬 Experimental |
| **Efficiency-focused** | 0.50 | 0.20 | 0.30 | Prefer diverse examples - reduce redundancy | ✅ Lowest tokens <br> ⚠️ Slightly lower accuracy |

## 📊 Preliminary Results (Tính toán lý thuyết)

### Hypothesis-based Projections

Dựa trên các nghiên cứu trước và reward function design:

| Reward Config | λ_acc | λ_sim | λ_div | Estimated Accuracy | Estimated Avg Tokens | Efficiency Score |
|---------------|-------|-------|-------|-------------------|---------------------|------------------|
| **Accuracy-focused** | 0.90 | 0.05 | 0.05 | **87-89%** ↑ | 310-330 ↑ | 27-29 |
| **Balanced (default)** | 0.60 | 0.30 | 0.10 | 85-87% | 270-290 | **30-32** ✅ |
| **Diversity-focused** | 0.40 | 0.50 | 0.10 | 82-85% ↓ | 290-310 | 27-29 |
| **Balanced (equal)** | 0.50 | 0.25 | 0.25 | 84-86% | 275-295 | 29-31 |
| **Efficiency-focused** | 0.50 | 0.20 | 0.30 | 83-86% | **240-270** ↓ | **32-36** ↑ |

**Legend:**
- ↑ = Higher than others
- ↓ = Lower than others
- ✅ = Best in category
- Efficiency Score = `accuracy (%) / (avg_tokens / 100)`

## 🔍 Analysis Framework

### 1. Accuracy Comparison

**Research Question**: Cấu hình nào cho accuracy cao nhất?

**Hypothesis**:
- H1: λ_acc càng cao → accuracy càng tốt
- H2: Balanced config (0.6/0.3/0.1) có thể optimal do empirical tuning

**Expected Winner**: Accuracy-focused (0.9/0.05/0.05)

### 2. Token Efficiency

**Research Question**: Cấu hình nào sử dụng tokens hiệu quả nhất?

**Hypothesis**:
- H1: λ_div cao → chọn examples đa dạng → giảm redundancy → ít tokens
- H2: Efficiency-focused (0.5/0.2/0.3) sẽ có tokens thấp nhất

**Expected Winner**: Efficiency-focused (0.5/0.2/0.3)

### 3. Overall Efficiency

**Research Question**: Cấu hình nào cân bằng tốt nhất giữa accuracy và tokens?

**Metric**: Efficiency Score = `accuracy / (tokens / 100)`

**Expected Winner**: Balanced default (0.6/0.3/0.1) hoặc Efficiency-focused (0.5/0.2/0.3)

## 📈 Expected Training Dynamics

### Convergence Speed

| Config | Expected Convergence | Reason |
|--------|---------------------|---------|
| Accuracy-focused | Fast (2-3 epochs) | Strong accuracy signal |
| Balanced | Moderate (3-4 epochs) | Multi-objective balance |
| Diversity-focused | Slow (4-5 epochs) | Weaker accuracy signal |

### Loss Trajectory

Expected pattern:
```
Epoch 1: High loss, exploration
Epoch 2-3: Rapid decrease
Epoch 4-5: Plateau/fine-tuning
```

### Reward Progression

Expected pattern:
```
Accuracy-focused:   0.5 → 0.65 → 0.78 → 0.82 → 0.85
Balanced:           0.5 → 0.62 → 0.72 → 0.78 → 0.82
Diversity-focused:  0.5 → 0.58 → 0.68 → 0.73 → 0.77
```

## 🎯 Key Insights to Validate

### 1. Reward Weight Impact

**Question**: Accuracy weight có ảnh hưởng tuyến tính đến final accuracy?

**Test**: Correlation giữa λ_acc và accuracy
- Strong positive correlation → Yes
- Weak/non-linear → Other factors matter

### 2. Similarity-Diversity Tradeoff

**Question**: λ_sim và λ_div có xung đột với nhau?

**Test**: So sánh:
- High sim, low div (0.4/0.5/0.1)
- Low sim, high div (0.5/0.2/0.3)

### 3. Optimal Balance

**Question**: Config nào tối ưu cho TAT-QA?

**Criteria**:
1. Accuracy ≥ 85%
2. Avg tokens ≤ 300
3. Efficiency score highest

## 📝 Theoretical Justification

### Why These Weights?

#### Accuracy Weight (λ_acc)

**Range tested**: 0.4 - 0.9

**Rationale**:
- Primary objective = correct answer
- Must dominate reward signal
- Too high (>0.9) → ignores example quality
- Too low (<0.4) → weak learning signal

#### Similarity Weight (λ_sim)

**Range tested**: 0.05 - 0.5

**Rationale**:
- Secondary signal when accuracy noisy
- Helps select relevant examples
- Too high → overfit to similar problems
- Too low → random selection

#### Diversity Weight (λ_div)

**Range tested**: 0.05 - 0.3

**Rationale**:
- Regularization to avoid redundancy
- Encourages varied approaches
- Too high → selects unrelated examples
- Too low → k nearly-identical examples

### Multi-Objective Optimization

Reward function combines:
1. **Binary feedback** (accuracy): 0/1 signal
2. **Continuous feedback** (similarity): 0.5-0.95 signal
3. **Regularization** (diversity): 0.0-0.4 signal

Weighted sum creates smooth gradient for policy learning.

## 🔬 Statistical Validation Plan

### Current Experiment

- **N configs**: 5
- **N samples**: 151
- **N epochs**: 5
- **Seed**: 42 (fixed)

### Future Extensions

For statistical significance:

1. **Multiple seeds**: 42, 123, 456, 789, 1024
2. **Confidence intervals**: Mean ± 1.96×SE
3. **Hypothesis testing**: Paired t-test between configs
4. **Effect size**: Cohen's d for practical significance

### Sample Size Justification

TAT-QA test set = 277 samples

151 samples = 54.5% of test set
- Adequate for initial comparison
- Large enough for ~5% margin of error
- Balanced with computational cost

## 📊 Results Format

### JSON Output Structure

```json
{
  "config_name": "accuracy_focused",
  "reward_config": {
    "lambda_acc": 0.9,
    "lambda_sim": 0.05,
    "lambda_div": 0.05
  },
  "training": {
    "epochs": 5,
    "time_seconds": 1234.56,
    "final_loss": 0.123,
    "final_reward": 0.876,
    "history": [
      {"epoch": 1, "loss": 0.543, "reward": 0.567, "accuracy": 0.623},
      ...
    ]
  },
  "evaluation": {
    "samples": 151,
    "accuracy": 0.874,
    "accuracy_percent": 87.4,
    "correct": 132,
    "total": 151,
    "avg_reward": 0.823,
    "token_usage": {
      "total_tokens": 47088,
      "input_tokens": 38000,
      "output_tokens": 9088,
      "avg_per_sample": 312
    },
    "time_seconds": 456.78
  },
  "timestamp": "2025-12-21T06:16:42.123456"
}
```

### Markdown Table Format

```markdown
| Reward Config      | λ_acc | λ_sim | λ_div | Accuracy (%) ↑ | Avg Tokens ↓ | Efficiency ↑ |
|--------------------|-------|-------|-------|----------------|--------------|--------------|
| Accuracy-focused   | 0.90  | 0.05  | 0.05  | **87.4**       | 312          | 28.01        |
| Balanced (default) | 0.60  | 0.30  | 0.10  | 87.1           | 276          | **31.56**    |
| Efficiency-focused | 0.50  | 0.20  | 0.30  | 86.8           | **231**      | 37.58        |
| Balanced (equal)   | 0.50  | 0.25  | 0.25  | 85.9           | 285          | 30.14        |
| Diversity-focused  | 0.40  | 0.50  | 0.10  | 84.2           | 298          | 28.26        |
```

## 🎓 Learning Outcomes

After this experiment, we will know:

1. ✅ **Optimal λ_acc range** for TAT-QA financial reasoning
2. ✅ **Similarity vs Diversity tradeoff** in example selection
3. ✅ **Token efficiency** achievable with different strategies
4. ✅ **Training stability** of different reward configurations
5. ✅ **Generalization** potential to other datasets

## 📅 Timeline

| Time | Status | Config |
|------|--------|--------|
| 06:16 | Started | - |
| 06:30 | Training | accuracy_focused (Epoch 1-5) |
| 07:00 | Training | balanced_default (Epoch 1-5) |
| 07:30 | Training | diversity_focused (Epoch 1-5) |
| 08:00 | Training | balanced_equal (Epoch 1-5) |
| 08:30 | Training | efficiency_focused (Epoch 1-5) |
| 09:00 | Evaluating | All configs on 151 samples |
| 09:30 | Saving | JSON + Markdown results |
| 10:00 | ✅ Complete | - |

**Total estimated time**: ~3.5-4 hours

## 🔄 Next Actions

Once experiment completes:

1. **Review JSON results** in `results/reward_sensitivity/`
2. **Analyze markdown table** for quick comparison
3. **Validate hypotheses** against expected outcomes
4. **Identify best config** for TAT-QA
5. **Test on other datasets** (GSM8K, FinQA)
6. **Fine-tune weights** around best config
7. **Statistical validation** with multiple seeds

---

**Last Updated**: 2025-12-21 06:22  
**Experiment ID**: TAT-QA_reward_sensitivity_20251221  
**Status**: 🔄 Running (Config 2/5 - Epoch 1)  
**Expected Completion**: ~10:00
