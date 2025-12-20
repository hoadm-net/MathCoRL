# MathCoRL - Research Strategy Summary

**Date**: December 19, 2025  
**Focus**: GSM8K + TAT-QA  
**Timeline**: 5-6 weeks

---

## 🎯 Core Message

> **"You DON'T need expensive GPT/Claude for SOTA reasoning!"**
> 
> Our method (FPP + Policy Network) works effectively on **open-source models**, making advanced mathematical reasoning **accessible to everyone** with a consumer GPU.

---

## 📊 What We're Proving

| # | Claim | How We Prove It |
|---|-------|-----------------|
| 1 | **FPP > CoT/PoT** | Accuracy comparison on GSM8K + TAT-QA |
| 2 | **Policy > Random/Similarity** | Same model, same strategy, different selection method |
| 3 | **⭐ Open-Source Readiness** | Qwen/DeepSeek with FPP+Policy competitive with GPT-4o-mini |
| 4 | **10-20× Cost Savings** | Free local inference vs $0.15-$0.60 per 1M tokens |
| 5 | **Practical Accessibility** | Works on RTX 3090 24GB (consumer GPU) |

---

## 🔬 Experimental Design

### Datasets (2 focused benchmarks)
- **GSM8K**: 7,473 train + 1,319 test (elementary math)
- **TAT-QA**: 2,201 train + 277 test (financial reasoning)

### Models (Primary = Open-Source)
1. **Qwen2.5-Math-7B** (open-source, free) ← PRIMARY
2. **DeepSeek-R1-7B** (open-source, free) ← PRIMARY  
3. GPT-4o-mini (commercial, for cost comparison)

### Selection Methods
1. Zero-shot (no examples)
2. Random-k (k=3)
3. Similarity-based (KATE-style, k=3)
4. **Policy Network** (k=3) ← OUR METHOD

### Prompting Strategies
1. CoT (Chain-of-Thought) - baseline
2. PoT (Program-of-Thought) - baseline
3. **FPP (Function Prototype Prompting)** ← OUR METHOD

### Full Comparison Matrix
- 3 models × 4 selection methods × 3 strategies = **36 configurations**
- Test on 1,319 (GSM8K) + 277 (TAT-QA) = **1,596 samples**

---

## 📅 5-Phase Execution Plan

| Phase | Duration | GPU Hours | Key Deliverable | Status |
|-------|----------|-----------|-----------------|--------|
| **1. Baseline Integration** | Week 1-2 | - | Qwen + DeepSeek providers | ✅ DONE |
| **2. Policy Training** | Week 2-3 | 20h | Trained policies for 2 datasets | 🔜 NEXT |
| **3. Comprehensive Evaluation** | Week 3-4 | 80h | 36 configs × 2 datasets results | ⏳ Pending |
| **4. Ablation Studies** | Week 4-5 | 30h | Pool size, cost, latency analysis | ⏳ Pending |
| **5. Documentation** | Week 5-6 | - | Supplementary materials | ⏳ Pending |

**Total**: 5-6 weeks, ~130 GPU hours, $50-100 API cost

---

## 💡 Key Strategic Decisions

### 1. Why Only 2 Datasets?
✅ GSM8K = industry standard (most cited)  
✅ TAT-QA = domain-specific (financial)  
✅ 70% cost savings, 45% GPU time savings  
✅ Still proves generalization  
✅ Clear story: accessibility + effectiveness

### 2. Why 7B Models?
✅ **Accessibility**: Fits on RTX 3090 24GB (consumer GPU)  
✅ No $30k A100 cluster needed  
✅ Message: "Gaming PC can do SOTA reasoning!"  
✅ Fair comparison (same size across models)

### 3. Why Focus on Open-Source?
✅ **Democratization**: Free for everyone  
✅ **Privacy**: No data sent to external APIs  
✅ **Control**: Full ownership of inference  
✅ **Reproducibility**: No API version changes  
✅ **Economic**: 0 cost vs $100s in API fees

---

## 🎯 Expected Outcomes

### Accuracy Targets
- FPP > CoT/PoT by ≥3% absolute
- Policy > Random/Similarity by ≥2-5%
- Qwen+FPP+Policy within -5% of GPT-4o-mini

### Cost Comparison
```
GPT-4o-mini:  $0.15 / 1M input tokens, $0.60 / 1M output tokens
              ≈ $50-100 for 1,596 samples

Qwen-7B:      $0 (free local inference)
              ≈ $0 for 1,596 samples

Savings:      100% cost reduction + full control
```

### Accessibility Proof
- Setup time: <1 hour
- Hardware: RTX 3090 24GB ($1,500 used)
- No API keys needed
- Works offline

---

## 📈 Success Metrics

| Metric | Target | Evidence |
|--------|--------|----------|
| **Method Effectiveness** | FPP+Policy best | Phase 3 results |
| **Open-Source Viability** | Within -5% of GPT | Phase 3 comparison |
| **Cost Efficiency** | 100% savings | Phase 4 cost analysis |
| **Practical Overhead** | <5% selection time | Phase 4 latency |
| **Accessibility** | RTX 3090 sufficient | Phase 1 integration |

---

## 🚀 Immediate Next Steps

1. **Check if candidates exist:**
   ```bash
   ls -lh /workspace/MathCoRL/candidates/GSM8K.json
   ls -lh /workspace/MathCoRL/candidates/TAT-QA.json
   ```

2. **If missing, generate:**
   ```bash
   python3 generate_candidates.py --dataset GSM8K --method fpp --pool-size 30
   python3 generate_candidates.py --dataset TAT-QA --method fpp --pool-size 30
   ```

3. **Train Policy Networks (Phase 2):**
   ```bash
   python3 train_policy.py --dataset GSM8K --epochs 50 --seed 42
   python3 train_policy.py --dataset TAT-QA --epochs 50 --seed 42
   ```

---

## 📝 Paper Narrative

### Title Ideas
- "Democratizing Mathematical Reasoning: FPP + Policy Networks on Open-Source LLMs"
- "Function Prototype Prompting: Making Advanced Reasoning Accessible"
- "From Enterprise to Open-Source: Effective Mathematical Reasoning without GPT"

### Key Messages
1. **Technical**: FPP + Policy Network outperforms baselines
2. **Practical**: Works on open-source models (Qwen, DeepSeek)
3. **Economic**: 100% cost savings vs GPT/Claude
4. **Social**: Democratizes access to advanced AI reasoning
5. **Reproducible**: Anyone with RTX 3090 can reproduce

### Contribution Summary
- **Method**: FPP (Function Prototype Prompting) - new prompting strategy
- **Method**: Policy Network for intelligent example selection
- **Insight**: These methods work on open-source models (not just GPT)
- **Impact**: Makes SOTA reasoning accessible to everyone

---

## ⚠️ Risk Mitigation

| Risk | Mitigation |
|------|-----------|
| Open-source models underperform | Show competitive results within -5% |
| Reviewers want more datasets | Offer to extend (architecture supports it) |
| GPU unavailable | Use cloud GPU ($200 for 180 hours) |
| Time too short | Prioritize Phase 2-3 (critical path) |

---

## 📊 Comparison: Original vs Streamlined

| Metric | Original | Streamlined | Savings |
|--------|----------|-------------|---------|
| Datasets | 5 | 2 | -60% |
| Test samples | ~32,000 | ~1,600 | -95% |
| GPU hours | 200-300h | 130h | -45% |
| API cost | $250-400 | $50-100 | -70% |
| Storage | 50GB | 32GB | -36% |

**Result**: Faster, cheaper, still scientifically rigorous!

---

## 🎓 Intellectual Contribution

### What's Novel?
1. **FPP**: Function-based prompting with executable prototypes
2. **Policy Network**: Learned example selection (vs heuristics)
3. **Open-Source Validation**: Proving methods generalize beyond GPT
4. **Accessibility Focus**: Making AI research reproducible on consumer hardware

### What's the Impact?
- **Academic**: New prompting + selection methods
- **Practical**: Anyone can run SOTA reasoning
- **Economic**: 100% cost reduction
- **Social**: Democratization of AI capabilities

---

**Last Updated**: December 19, 2025  
**Status**: Phase 1 complete, ready for Phase 2  
**Next Action**: Check candidates → Train policies

---

*"Making advanced mathematical reasoning accessible to everyone with a gaming PC."* 🎮🧠
