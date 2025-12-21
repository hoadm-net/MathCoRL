# Reward Sensitivity Experiments - Hướng dẫn

Tài liệu này hướng dẫn cách chạy thực nghiệm sensitivity analysis cho các cấu hình reward khác nhau trong Policy Network training.

## 📊 Mục tiêu

So sánh ảnh hưởng của các trọng số reward (λ_acc, λ_sim, λ_div) đến:
- **Accuracy** - Độ chính xác giải toán
- **Token Usage** - Hiệu quả sử dụng tokens (input + output)
- **Training Dynamics** - Tốc độ học và sự ổn định

## 🎯 Công thức Reward

```
R_total = λ_acc · R_acc + λ_sim · R_sim + λ_div · R_div
```

Trong đó:
- **R_acc**: Binary accuracy (1 nếu đúng, 0 nếu sai)
- **R_sim**: Cosine similarity giữa problem và trung bình examples
- **R_div**: Diversity giữa các examples (1 - similarity)

## 🧪 Các cấu hình được test

| Config Name | λ_acc | λ_sim | λ_div | Mô tả |
|-------------|-------|-------|-------|-------|
| **accuracy_focused** | 0.90 | 0.05 | 0.05 | Tối ưu độ chính xác |
| **balanced_default** | 0.60 | 0.30 | 0.10 | Cân bằng (mặc định) |
| **diversity_focused** | 0.40 | 0.50 | 0.10 | Nhấn mạnh similarity |
| **balanced_equal** | 0.50 | 0.25 | 0.25 | Phân phối đều |
| **efficiency_focused** | 0.50 | 0.20 | 0.30 | Ưu tiên diversity |

## 🚀 Cách chạy thực nghiệm

### Option 1: Full Experiment (Train + Evaluate)

Train policy với từng config và đo accuracy + tokens:

```bash
# Chạy trên TAT-QA với 151 samples
python scripts/reward_sensitivity_experiment.py \
    --dataset TAT-QA \
    --samples 151 \
    --epochs 5 \
    --seed 42

# Chạy trên GSM8K với 200 samples
python scripts/reward_sensitivity_experiment.py \
    --dataset GSM8K \
    --samples 200 \
    --epochs 10 \
    --seed 42

# Chạy chỉ một số configs cụ thể
python scripts/reward_sensitivity_experiment.py \
    --dataset TAT-QA \
    --samples 151 \
    --epochs 5 \
    --configs accuracy_focused balanced_default \
    --seed 42
```

**Thời gian ước tính:**
- 1 config × 5 epochs × 151 samples ≈ 30-45 phút
- 5 configs ≈ 2.5-4 giờ (chạy tuần tự)

### Option 2: Quick Analysis (Evaluate Only)

So sánh các methods có sẵn mà không cần train lại:

```bash
# Quick comparison với 30 samples
python scripts/quick_reward_analysis.py \
    --dataset TAT-QA \
    --samples 30 \
    --methods zero-shot random policy kate cds \
    --seed 42

# Full comparison với 151 samples
python scripts/quick_reward_analysis.py \
    --dataset TAT-QA \
    --samples 151 \
    --seed 42
```

**Thời gian ước tính:**
- 30 samples ≈ 5-10 phút
- 151 samples ≈ 30-45 phút

## 📊 Kết quả Output

### JSON File

Chi tiết đầy đủ mỗi config:

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
    "history": [...]
  },
  "evaluation": {
    "accuracy": 0.874,
    "accuracy_percent": 87.4,
    "correct": 132,
    "total": 151,
    "avg_reward": 0.823,
    "token_usage": {
      "total_tokens": 47088,
      "avg_per_sample": 312
    }
  }
}
```

### Markdown Table

```markdown
| Reward Config | λ_acc | λ_sim | λ_div | Accuracy (%) ↑ | Avg Tokens | Efficiency |
|---------------|-------|-------|-------|----------------|------------|------------|
| Accuracy-focused | 0.90 | 0.05 | 0.05 | **87.4%** | 312 | 28.01 |
| Balanced (default) | 0.60 | 0.30 | 0.10 | 87.1% | 276 | **31.56** |
| Efficiency-focused | 0.50 | 0.20 | 0.30 | 86.8% | **231** | 37.58 |
```

**Efficiency Score**: `accuracy / (avg_tokens / 100)` - cao hơn = tốt hơn

## 📂 File locations

```
results/reward_sensitivity/
├── TAT-QA_reward_sensitivity_20251221_123456.json   # Full data
├── TAT-QA_reward_sensitivity_20251221_123456.md     # Markdown table
├── TAT-QA_quick_analysis_20251221_123456.json       # Quick comparison
├── TAT-QA_quick_analysis_20251221_123456.md         # Quick table
└── experiment_log.txt                               # Training logs
```

## 🔍 Phân tích kết quả

### 1. So sánh Accuracy

Xác định config nào cho accuracy cao nhất:
- Dự đoán: **accuracy_focused** sẽ có accuracy cao nhất
- Nhưng có thể tốn nhiều tokens hơn

### 2. Token Efficiency

Tính efficiency score:
```
Efficiency = accuracy (%) / (avg_tokens / 100)
```

Ví dụ:
- Config A: 87% accuracy, 300 tokens → 87/3 = 29.0
- Config B: 86% accuracy, 250 tokens → 86/2.5 = 34.4 ✅ Better

### 3. Training Stability

Xem loss trajectory trong JSON:
- Giảm đều = stable training
- Fluctuate nhiều = unstable, cần tune hyperparameters

### 4. Accuracy vs Efficiency Tradeoff

Plot scatter:
- X-axis: Avg tokens
- Y-axis: Accuracy
- Ideal: Top-left corner (high accuracy, low tokens)

## 🎯 Kết luận mong đợi

### Giả thuyết:

1. **Accuracy-focused (0.9/0.05/0.05)**:
   - ✅ Accuracy cao nhất
   - ❌ Token usage cao (vì ít quan tâm diversity)
   
2. **Balanced default (0.6/0.3/0.1)**:
   - ✅ Cân bằng tốt
   - ✅ Đã được tune empirically
   
3. **Efficiency-focused (0.5/0.2/0.3)**:
   - ✅ Token usage thấp nhất
   - ❌ Có thể accuracy thấp hơn một chút

### Statistical Significance:

Để kết quả có ý nghĩa thống kê:
- Chạy multiple seeds: 42, 123, 456, 789, 1024
- Tính mean ± std cho mỗi metric
- T-test để so sánh configs

## 📈 Visualizations

Có thể tạo plots từ JSON results:

```python
import json
import matplotlib.pyplot as plt

# Load results
with open('results/reward_sensitivity/TAT-QA_*.json') as f:
    data = json.load(f)

# Plot accuracy vs tokens
configs = [d['config_name'] for d in data]
accuracies = [d['evaluation']['accuracy_percent'] for d in data]
tokens = [d['evaluation']['token_usage']['avg_per_sample'] for d in data]

plt.scatter(tokens, accuracies)
for i, config in enumerate(configs):
    plt.annotate(config, (tokens[i], accuracies[i]))
plt.xlabel('Avg Tokens per Sample')
plt.ylabel('Accuracy (%)')
plt.title('Accuracy vs Token Efficiency')
plt.savefig('accuracy_vs_tokens.png')
```

## 🔄 Next Steps

Sau khi có kết quả:

1. **Analyze best config**: Xác định config tốt nhất cho TAT-QA
2. **Generalize to other datasets**: Test best config trên GSM8K, FinQA
3. **Fine-tune weights**: Grid search quanh config tốt nhất (e.g., λ_acc ∈ [0.55, 0.65])
4. **Ablation study**: Test single-objective (λ_acc=1.0, others=0)
5. **Adaptive weights**: Dynamic weight scheduling during training

## 📚 References

- [mint/icrl/config.py](../mint/icrl/config.py) - RewardConfig class
- [mint/icrl/trainer.py](../mint/icrl/trainer.py) - calculate_reward method
- [docs/policy-selection-rules.md](policy-selection-rules.md) - Policy architecture
- [docs/reward-sensitivity.md](reward-sensitivity.md) - Original sensitivity doc
- [configs/hyperparameters.yaml](../configs/hyperparameters.yaml) - Reward weights config

## 💡 Tips

1. **Start small**: Test với 30-50 samples trước khi chạy full 151
2. **Use seeds**: Luôn set seed để reproducible
3. **Monitor logs**: Theo dõi training progress qua logs
4. **Compare incrementally**: Test 2-3 configs trước, sau đó mở rộng
5. **Save intermediate**: Script tự động save models và results

## ❓ Troubleshooting

**Q: Script bị lỗi "Model not found"?**
A: Policy network chưa được train. Chạy `train_policy.py` trước hoặc dùng quick_analysis.py

**Q: Token tracking không chính xác?**
A: Đảm bảo API tracking đang bật trong .env: `TRACK_API_USAGE=true`

**Q: Experiment chạy quá lâu?**
A: Giảm số epochs (--epochs 3) hoặc samples (--samples 50) cho test nhanh

**Q: Muốn so sánh với baseline methods?**
A: Dùng `quick_reward_analysis.py` để compare với zero-shot, random, kate, cds

