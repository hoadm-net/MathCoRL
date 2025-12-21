#!/bin/bash

# Quick test: 1 epoch, 10 samples
DATASET="TAT-QA"
EPOCHS=1
SAMPLES=10
SEED=42

echo "=============================================="
echo "REWARD PIPELINE TEST"
echo "=============================================="
echo "Dataset: $DATASET"
echo "Epochs: $EPOCHS"
echo "Evaluation: $SAMPLES samples"
echo "=============================================="
echo ""

# Test 2 configs to validate workflow

# Config 1: Accuracy-focused
echo ">>> TEST 1/2: Accuracy-focused (0.9, 0.05, 0.05)"
python train_policy.py \
    --dataset $DATASET \
    --epochs $EPOCHS \
    --reward-weights "0.9,0.05,0.05" \
    --seed $SEED \
    --overwrite

echo ">>> Evaluating..."
python run_comparison.py \
    --dataset $DATASET \
    --samples $SAMPLES \
    --methods policy \
    --seed $SEED \
    --save-results \
    --output-dir results/reward_sensitivity

mv results/${DATASET}_comparison_*.json results/reward_sensitivity/${DATASET}_test_accuracy.json 2>/dev/null || true
cp models/${DATASET}_policy_best.pt models/${DATASET}_policy_test_accuracy.pt

echo ""
echo "=============================================="

# Config 2: Balanced default
echo ">>> TEST 2/2: Balanced default (0.6, 0.3, 0.1)"
python train_policy.py \
    --dataset $DATASET \
    --epochs $EPOCHS \
    --reward-weights "0.6,0.3,0.1" \
    --seed $SEED \
    --overwrite

echo ">>> Evaluating..."
python run_comparison.py \
    --dataset $DATASET \
    --samples $SAMPLES \
    --methods policy \
    --seed $SEED \
    --save-results \
    --output-dir results/reward_sensitivity

mv results/${DATASET}_comparison_*.json results/reward_sensitivity/${DATASET}_test_balanced.json 2>/dev/null || true
cp models/${DATASET}_policy_best.pt models/${DATASET}_policy_test_balanced.pt

echo ""
echo "=============================================="
echo "TEST COMPLETED!"
echo "=============================================="
echo "Check results:"
echo "  - results/reward_sensitivity/${DATASET}_test_accuracy.json"
echo "  - results/reward_sensitivity/${DATASET}_test_balanced.json"
echo "=============================================="
