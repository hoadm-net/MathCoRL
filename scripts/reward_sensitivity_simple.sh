#!/bin/bash
# Simple Reward Sensitivity Experiment
# Train each config separately with different model names

DATASET="TAT-QA"
EPOCHS=10
SAMPLES=151
SEED=42

echo "=============================================="
echo "REWARD SENSITIVITY EXPERIMENT - Simple Version"
echo "=============================================="
echo "Dataset: $DATASET"
echo "Epochs: $EPOCHS per config"
echo "Evaluation: $SAMPLES samples"
echo "Seed: $SEED"
echo "=============================================="
echo ""

# Config 1: Accuracy-focused (0.9, 0.05, 0.05)
echo ">>> CONFIG 1/5: Accuracy-focused (0.9, 0.05, 0.05)"
python train_policy.py \
    --dataset $DATASET \
    --epochs $EPOCHS \
    --reward-weights "0.9,0.05,0.05" \
    --seed $SEED \
    --overwrite

# Evaluate BEFORE renaming
echo ">>> Evaluating accuracy-focused..."
python run_comparison.py \
    --dataset $DATASET \
    --samples $SAMPLES \
    --methods policy \
    --seed $SEED \
    --save-results \
    --output-dir results/reward_sensitivity

# Save results with specific name
mv results/${DATASET}_comparison_*.json results/reward_sensitivity/${DATASET}_accuracy_focused.json 2>/dev/null || true

# Now backup model with new name
cp models/${DATASET}_policy_best.pt models/${DATASET}_policy_accuracy_focused.pt
cp models/${DATASET}_policy_final.pt models/${DATASET}_policy_accuracy_focused_final.pt

echo ""
echo "=============================================="
echo ""

# Config 2: Balanced default (0.6, 0.3, 0.1)
echo ">>> CONFIG 2/5: Balanced default (0.6, 0.3, 0.1)"
python train_policy.py \
    --dataset $DATASET \
    --epochs $EPOCHS \
    --reward-weights "0.6,0.3,0.1" \
    --seed $SEED \
    --overwrite

echo ">>> Evaluating balanced default..."
python run_comparison.py \
    --dataset $DATASET \
    --samples $SAMPLES \
    --methods policy \
    --seed $SEED \
    --save-results \
    --output-dir results/reward_sensitivity

mv results/${DATASET}_comparison_*.json results/reward_sensitivity/${DATASET}_balanced_default.json 2>/dev/null || true

cp models/${DATASET}_policy_best.pt models/${DATASET}_policy_balanced_default.pt
cp models/${DATASET}_policy_final.pt models/${DATASET}_policy_balanced_default_final.pt

echo ""
echo "=============================================="
echo ""

# Config 3: Diversity-focused (0.4, 0.5, 0.1)
echo ">>> CONFIG 3/5: Diversity-focused (0.4, 0.5, 0.1)"
python train_policy.py \
    --dataset $DATASET \
    --epochs $EPOCHS \
    --reward-weights "0.4,0.5,0.1" \
    --seed $SEED \
    --overwrite

echo ">>> Evaluating diversity-focused..."
python run_comparison.py \
    --dataset $DATASET \
    --samples $SAMPLES \
    --methods policy \
    --seed $SEED \
    --save-results \
    --output-dir results/reward_sensitivity

mv results/${DATASET}_comparison_*.json results/reward_sensitivity/${DATASET}_diversity_focused.json 2>/dev/null || true

cp models/${DATASET}_policy_best.pt models/${DATASET}_policy_diversity_focused.pt
cp models/${DATASET}_policy_final.pt models/${DATASET}_policy_diversity_focused_final.pt

echo ""
echo "=============================================="
echo ""

# Config 4: Balanced equal (0.5, 0.25, 0.25)
echo ">>> CONFIG 4/5: Balanced equal (0.5, 0.25, 0.25)"
python train_policy.py \
    --dataset $DATASET \
    --epochs $EPOCHS \
    --reward-weights "0.5,0.25,0.25" \
    --seed $SEED \
    --overwrite

echo ">>> Evaluating balanced equal..."
python run_comparison.py \
    --dataset $DATASET \
    --samples $SAMPLES \
    --methods policy \
    --seed $SEED \
    --save-results \
    --output-dir results/reward_sensitivity

mv results/${DATASET}_comparison_*.json results/reward_sensitivity/${DATASET}_balanced_equal.json 2>/dev/null || true

cp models/${DATASET}_policy_best.pt models/${DATASET}_policy_balanced_equal.pt
cp models/${DATASET}_policy_final.pt models/${DATASET}_policy_balanced_equal_final.pt

echo ""
echo "=============================================="
echo ""

# Config 5: Efficiency-focused (0.5, 0.2, 0.3)
echo ">>> CONFIG 5/5: Efficiency-focused (0.5, 0.2, 0.3)"
python train_policy.py \
    --dataset $DATASET \
    --epochs $EPOCHS \
    --reward-weights "0.5,0.2,0.3" \
    --seed $SEED \
    --overwrite

echo ">>> Evaluating efficiency-focused..."
python run_comparison.py \
    --dataset $DATASET \
    --samples $SAMPLES \
    --methods policy \
    --seed $SEED \
    --save-results \
    --output-dir results/reward_sensitivity

mv results/${DATASET}_comparison_*.json results/reward_sensitivity/${DATASET}_efficiency_focused.json 2>/dev/null || true

cp models/${DATASET}_policy_best.pt models/${DATASET}_policy_efficiency_focused.pt
cp models/${DATASET}_policy_final.pt models/${DATASET}_policy_efficiency_focused_final.pt

echo ""
echo "=============================================="
echo "ALL CONFIGS COMPLETED!"
echo "=============================================="
echo ""
echo "Results saved to: results/reward_sensitivity/"
echo "Models saved to: models/"
echo ""
echo "To view results:"
echo "  ls -lh results/reward_sensitivity/${DATASET}_*.json"
echo "  ls -lh models/${DATASET}_policy_*.pt"
