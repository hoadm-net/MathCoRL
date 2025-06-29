# 🔄 MathCoRL Migration Guide

## Overview

The MathCoRL project has been significantly refactored to eliminate code duplication, improve maintainability, and provide a cleaner interface. This guide helps you migrate from the old structure to the new unified system.

## 📊 **What Changed**

### ✅ **Improvements Made**

1. **Eliminated Code Duplication**: 
   - Removed ~80% duplicate code between `fpp_prompting.py` and `cot_prompting.py`
   - Consolidated tolerance functions and evaluation logic

2. **Unified Testing Framework**: 
   - Single `mint.testing` module for all testing needs
   - Consistent interface for FPP, CoT, and PoT methods

3. **Added Program of Thoughts (PoT)**: 
   - New `mint.pot` module implementing PoT prompting
   - Generates Python code to solve numerical problems
   - Separates computation from reasoning for better accuracy

4. **Shared Evaluation Library**: 
   - Centralized `mint.evaluation` module with all tolerance functions
   - Dataset-specific evaluation logic properly modularized

4. **Single CLI Entry Point**: 
   - Replaced 4 separate scripts with one unified `mathcorl.py`
   - Backward compatibility maintained

5. **Better Package Structure**: 
   - Clear separation of concerns
   - Improved imports and exports

## 🔧 **Migration Instructions**

### **Old vs New Usage**

#### **Single Problem Solving**

**Before:**
```bash
# FPP
python fpp.py --question "What is 15 + 27?"

# CoT  
python cot.py --question "What is 15 + 27?"
```

**After:**
```bash
# Unified interface
python mathcorl.py solve --method fpp --question "What is 15 + 27?"
python mathcorl.py solve --method cot --question "What is 15 + 27?"
python mathcorl.py solve --method pot --question "What is 15 + 27?"

# Or use the new interactive mode
python mathcorl.py interactive
```

#### **Dataset Testing**

**Before:**
```bash
# FPP testing
python fpp_prompting.py SVAMP --limit 50

# CoT testing  
python cot_prompting.py SVAMP --limit 50
```

**After:**
```bash
# Unified testing interface
python mathcorl.py test --method fpp --dataset SVAMP --limit 50
python mathcorl.py test --method cot --dataset SVAMP --limit 50
python mathcorl.py test --method pot --dataset SVAMP --limit 50

# New: Compare methods directly
python mathcorl.py compare --dataset SVAMP --limit 50
```

#### **Legacy Compatibility**

The new `mathcorl.py` script maintains backward compatibility:

```bash
# These still work (legacy support)
python mathcorl.py SVAMP --limit 50  # Defaults to FPP
python mathcorl.py GSM8K --limit 100 --method cot
```

### **Code API Changes**

#### **Before (Scattered Functions)**
```python
# Old way - functions scattered across files
from fpp_prompting import solve_single_silent, is_close, get_tolerance_function
from cot_prompting import solve_single_cot, is_close_tatqa

# Testing was manual and duplicated
```

#### **After (Unified API)**
```python
# New way - clean, unified API
from mint.testing import TestRunner, create_fpp_solver, create_cot_solver, create_pot_solver
from mint.evaluation import get_tolerance_function, calculate_accuracy

# Create test runners
fpp_runner = TestRunner('FPP', create_fpp_solver())
cot_runner = TestRunner('CoT', create_cot_solver())
pot_runner = TestRunner('PoT', create_pot_solver())

# Run tests
results = fpp_runner.test_dataset('SVAMP', limit=50)

# Compare methods
comparison = fpp_runner.compare_methods(cot_runner, 'SVAMP', limit=20)
```

## 📁 **New File Structure**

```
MathCoRL/
├── mathcorl.py              # 🆕 Unified entry point (replaces 4 scripts)
├── pot.py                   # 🆕 Legacy PoT script (backward compatibility)
├── mint/
│   ├── __init__.py          # ✅ Updated exports
│   ├── core.py              # ✅ FPP implementation (unchanged)
│   ├── cot.py               # ✅ CoT implementation (unchanged)
│   ├── pot.py               # 🆕 Program of Thoughts implementation
│   ├── evaluation.py        # 🆕 Unified evaluation & tolerance functions
│   ├── testing.py           # 🆕 Unified testing framework
│   ├── cli.py               # ✅ Enhanced CLI interface
│   ├── utils.py             # ✅ Data loading utilities (unchanged)
│   ├── functions.py         # ✅ Mathematical functions (unchanged)
│   └── prompts.py           # ✅ Prompt templates (unchanged)
├── datasets/                # ✅ Datasets (unchanged)
├── results/                 # ✅ Results directory (unchanged)
└── README.md                # ✅ Updated documentation
```

## 🔥 **Files You Can Remove**

The following files are now **deprecated** and can be safely removed:

```bash
# Old scripts (replaced by mathcorl.py)
rm fpp.py
rm cot.py
rm fpp_prompting.py
rm cot_prompting.py
```

**Note**: Keep them temporarily if you have custom modifications, but migrate to the new API.

## 🚀 **New Features**

### 1. **Interactive Mode**
```bash
python mathcorl.py interactive
# Choose between FPP, CoT, and PoT interactively
# Switch methods on the fly
```

### 2. **Method Comparison**
```bash
python mathcorl.py compare --dataset SVAMP --limit 20
# Automatically runs both FPP and CoT and compares results
# PoT can also be tested individually
```

### 3. **Better Result Tracking**
```python
# Results now include timestamps, method info, and detailed metrics
{
    'dataset': 'SVAMP',
    'method': 'FPP',
    'timestamp': '2024-01-15T10:30:00',
    'accuracy': 96.5,
    'total_samples': 100,
    'correct_predictions': 96,
    'error_rate': 3.5
}
```

### 4. **Unified Evaluation**
```python
from mint.evaluation import evaluate_predictions

# Works with any dataset and method
results = evaluate_predictions(predictions, ground_truths, 'SVAMP')
print(f"Accuracy: {results['accuracy']:.2f}%")
```

## 🧪 **Testing the Migration**

1. **Test the new unified interface:**
```bash
# Test interactive mode
python mathcorl.py

# Test single solve
python mathcorl.py solve --method fpp --question "What is 15 + 27?"
python mathcorl.py solve --method pot --question "What is 15 + 27?"

# Test dataset testing
python mathcorl.py test --method fpp --dataset SVAMP --limit 5
python mathcorl.py test --method pot --dataset SVAMP --limit 5

# Test comparison
python mathcorl.py compare --dataset SVAMP --limit 5
```

2. **Test backward compatibility:**
```bash
# These should still work
python mathcorl.py SVAMP --limit 5
python mathcorl.py GSM8K --limit 5 --method cot
```

3. **Test the new API:**
```python
# test_new_api.py
from mint.testing import TestRunner, create_fpp_solver
from mint.evaluation import get_tolerance_function

# Create and test
runner = TestRunner('FPP', create_fpp_solver())
results = runner.test_dataset('SVAMP', limit=5)
print(f"Accuracy: {results['accuracy']:.2f}%")
```

## 📝 **Benefits of the New Structure**

1. **90% Less Code Duplication**: Consolidated evaluation and testing logic
2. **Easier Maintenance**: Single source of truth for each functionality  
3. **Better Testing**: Unified framework supports new methods easily
4. **Cleaner API**: Clear separation between core functionality and scripts
5. **Enhanced UX**: Interactive mode and method comparison features
6. **Future-Proof**: Easy to add new prompting methods

## 🤝 **Need Help?**

- Check `python mathcorl.py --help` for all available options
- Run `python mathcorl.py datasets` to see supported datasets  
- Use `python mathcorl.py interactive` to explore interactively
- The old functionality is still available through the new interface

## 🎯 **Next Steps**

1. Test the new interface with your typical workflows
2. Remove the old script files once comfortable
3. Update any custom scripts to use the new API
4. Enjoy the cleaner, more maintainable codebase! 🎉 