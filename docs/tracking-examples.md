# Practical Usage Guide with API Tracking

This guide demonstrates real-world usage patterns for MathCoRL with comprehensive API cost tracking and optimization.

## 🎯 **Research Workflow Templates**

### **Template 1: Method Comparison Research**

#### **Objective**: Compare prompting methods across mathematical domains

```bash
# Step 1: Test single problems for method validation
python mathcorl.py solve --method fpp --question "Calculate the compound interest on $1000 at 5% annual rate for 3 years"
python mathcorl.py solve --method cot --question "If a train travels 120 km in 1.5 hours, what is its speed in km/h?"
python mathcorl.py solve --method pal --question "Find the average of these numbers: 15, 23, 31, 42, 18"

# Step 2: Small-scale testing (cost-effective exploration)
python mathcorl.py test --method fpp --dataset SVAMP --limit 25
python mathcorl.py test --method cot --dataset GSM8K --limit 25

# Step 3: Monitor costs and optimize
python mathcorl.py stats --hours 1
python mathcorl.py export --format csv

# Step 4: Scale up based on promising results
python mathcorl.py compare --dataset TabMWP --limit 50
python mathcorl.py compare --dataset GSM8K --limit 100

# Step 5: Generate comprehensive comparison
python mathcorl.py chart --type comparison --save
```

**Expected Workflow Cost**: Moderate - progressive scaling helps control budget

### **Template 2: ICL Research Pipeline**

#### **Objective**: Compare example selection strategies with Policy Network

```bash
# Phase 1: Candidate Generation (one-time cost per dataset)
python generate_candidates.py --dataset TAT-QA --n-candidates 100
python generate_candidates.py --dataset GSM8K --n-candidates 50

# Phase 2: Policy Training (one-time training cost)
python train_policy.py --dataset TAT-QA --epochs 3
python train_policy.py --dataset GSM8K --epochs 3

# Phase 3: Small-scale method comparison
python run_comparison.py --dataset TAT-QA --samples 25 --methods policy,kate,random
python run_comparison.py --dataset GSM8K --samples 25 --methods policy,kate,random

# Phase 4: Cost analysis and scaling decision
python mathcorl.py stats --hours 2
python mathcorl.py chart --type cost --save

# Phase 5: Full evaluation (if budget allows)
python run_comparison.py --dataset TAT-QA --samples 150 --save-results
python run_comparison.py --dataset GSM8K --samples 100 --save-results

# Phase 6: Result analysis and documentation
python mathcorl.py export --format json
python mathcorl.py chart --type all --save
```

**Expected Workflow Cost**: Higher initially but amortized across multiple experiments

## 📊 **Real-World Cost Tracking Examples**

### **Example 1: Budget-Conscious Research**

#### **Scenario**: Limited budget research project

```bash
# Step 1: Estimate costs before starting
python mathcorl.py estimate --dataset SVAMP --samples 50 --methods fpp,cot

# Step 2: Start with smallest, cheapest dataset
python mathcorl.py test --method fpp --dataset SVAMP --limit 25

# Step 3: Check costs and success rate
python mathcorl.py stats --hours 1
# Monitor output for success rate and cost per sample

# Step 4: Scale based on performance
if [ success_rate > 80% ]; then
    python mathcorl.py test --method fpp --dataset SVAMP --limit 50
else
    # Optimize method or try different approach
    python mathcorl.py solve --method cot --question "Test problem for debugging"
fi

# Step 5: Export for budget tracking
python mathcorl.py export --format csv
# Use CSV for budget analysis and planning next experiments
```

#### **Budget Management Tips**
- Start with SVAMP (lowest cost per sample)
- Use small sample sizes for exploration
- Monitor `mathcorl.py stats` frequently
- Export data regularly for budget tracking

### **Example 2: Method Efficiency Analysis**

#### **Scenario**: Finding the most cost-effective method

```bash
# Test multiple methods on same dataset
python mathcorl.py test --method fpp --dataset GSM8K --limit 30
python mathcorl.py test --method cot --dataset GSM8K --limit 30
python mathcorl.py test --method pal --dataset GSM8K --limit 30

# Analyze efficiency metrics
python mathcorl.py stats
python mathcorl.py export --format json

# Generate efficiency comparison charts
python mathcorl.py chart --type comparison --save
python mathcorl.py chart --type cost --save

# Sample analysis workflow
python analyze_efficiency.py --input tracking_export.json
```

#### **Efficiency Metrics to Track**
- **Cost per successful solve**: `total_cost / successful_requests`
- **Accuracy per dollar**: `success_rate / average_cost`
- **Token efficiency**: `output_tokens / input_tokens`
- **Time efficiency**: `success_rate / average_time`

### **Example 3: Advanced ICL Cost Analysis**

#### **Scenario**: Comprehensive ICL research with cost optimization

```bash
# Generate candidates (one-time cost)
python generate_candidates.py --dataset TAT-QA --n-candidates 100

# Train policy with cost monitoring
time python train_policy.py --dataset TAT-QA --epochs 3
python mathcorl.py stats --hours 1  # Check training costs

# Progressive evaluation with cost checkpoints
python run_comparison.py --dataset TAT-QA --samples 25 --methods policy,random
python mathcorl.py stats
# Decision point: continue based on initial results and budget

python run_comparison.py --dataset TAT-QA --samples 50 --methods policy,kate,random
python mathcorl.py stats
# Another decision point

python run_comparison.py --dataset TAT-QA --samples 100 --methods all --save-results
python mathcorl.py chart --type all --save
```

#### **Advanced Cost Tracking Workflow**
```python
# Custom cost analysis script
import json
import pandas as pd

# Load tracking data
with open('tracking_export.json') as f:
    tracking_data = json.load(f)

# Create cost analysis DataFrame
df = pd.DataFrame(tracking_data)

# Method efficiency analysis
method_efficiency = df.groupby('method').agg({
    'total_cost': ['sum', 'mean'],
    'success': ['mean', 'count'],
    'execution_time': 'mean',
    'total_tokens': 'mean'
}).round(6)

# Cost per success calculation
method_efficiency['cost_per_success'] = (
    method_efficiency[('total_cost', 'sum')] / 
    method_efficiency[('success', 'count')]
)

print("Method Efficiency Analysis:")
print(method_efficiency)

# Time-based cost analysis
df['timestamp'] = pd.to_datetime(df['timestamp'])
hourly_costs = df.set_index('timestamp').resample('H')['total_cost'].sum()

print("\nHourly Cost Breakdown:")
print(hourly_costs)
```

## 🔧 **Advanced Usage Patterns**

### **Pattern 1: Iterative Method Development**

```bash
# Develop and test new prompting approaches
python mathcorl.py solve --method fpp --question "Test problem" --verbose
# Analyze output and refine approach

python mathcorl.py test --method fpp --dataset SVAMP --limit 10 --verbose
python mathcorl.py stats
# Check performance and costs

# Scale up if promising
python mathcorl.py test --method fpp --dataset SVAMP --limit 50
python mathcorl.py compare --dataset SVAMP --limit 50
```

### **Pattern 2: Cross-Dataset Validation**

```bash
# Test method across different domains
for dataset in GSM8K SVAMP TabMWP; do
    echo "Testing on $dataset"
    python mathcorl.py test --method fpp --dataset $dataset --limit 25
    python mathcorl.py stats --hours 1
done

# Generate cross-dataset comparison
python mathcorl.py chart --type comparison --save
```

### **Pattern 3: Policy Network Optimization**

```bash
# Train with different configurations
python train_policy.py --dataset TAT-QA --epochs 3 --lr 2e-4
python train_policy.py --dataset TAT-QA --epochs 5 --lr 1e-4 --overwrite

# Compare trained models
python run_comparison.py --dataset TAT-QA --samples 50 --methods policy
# Use different trained models for comparison

# Cost-benefit analysis of training configurations
python mathcorl.py stats
python mathcorl.py export --format csv
```

## 🎯 **Best Practices for Cost Optimization**

### **1. Progressive Experimentation**
```bash
# Always start small
python mathcorl.py test --method fpp --dataset SVAMP --limit 10

# Monitor and decide
python mathcorl.py stats
if [ cost_effective ]; then
    python mathcorl.py test --method fpp --dataset SVAMP --limit 50
fi
```

### **2. Method Batching**
```bash
# Test multiple methods together for fair comparison
python mathcorl.py compare --dataset GSM8K --limit 25
# More efficient than individual method testing
```

### **3. Smart Dataset Selection**
```bash
# Development phase: use cheaper datasets
python mathcorl.py test --method fpp --dataset SVAMP --limit 50  # Low cost
python mathcorl.py test --method fpp --dataset GSM8K --limit 50  # Medium cost

# Validation phase: scale to complex datasets
python mathcorl.py test --method fpp --dataset TAT-QA --limit 25  # Higher cost but necessary
```

### **4. Regular Cost Monitoring**
```bash
# Set up monitoring routine
python mathcorl.py stats --hours 1  # Check recent usage
python mathcorl.py stats --hours 24  # Daily summary

# Export for external tracking
python mathcorl.py export --format csv
# Import into spreadsheet for budget management
```

## 🛠️ **Troubleshooting with Cost Awareness**

### **Scenario 1: High API Costs**

#### **Problem**: Unexpected high costs during research

```bash
# Immediate cost audit
python mathcorl.py stats --hours 24
python mathcorl.py export --format csv

# Identify expensive methods
python analyze_costs.py --input tracking_export.csv --identify-expensive

# Cost reduction strategies
# 1. Switch to cheaper model
export DEFAULT_MODEL=gpt-4o-mini  # Instead of gpt-4o

# 2. Reduce sample sizes
python mathcorl.py test --method fpp --dataset TAT-QA --limit 10  # Instead of 100

# 3. Focus on cheaper datasets
python mathcorl.py test --method fpp --dataset SVAMP --limit 50  # Instead of FinQA
```

### **Scenario 2: Policy Training Costs**

#### **Problem**: Policy training consuming too much budget

```bash
# Check training costs
python train_policy.py --dataset TAT-QA --epochs 1 --verbose
python mathcorl.py stats --hours 1

# If too expensive, optimize:
# 1. Reduce candidate pool
python generate_candidates.py --dataset TAT-QA --n-candidates 50  # Instead of 100

# 2. Reduce training epochs
python train_policy.py --dataset TAT-QA --epochs 2  # Instead of 5

# 3. Use smaller datasets for development
python train_policy.py --dataset SVAMP --epochs 3  # Cheaper for testing
```

### **Scenario 3: Model Performance vs. Cost**

#### **Problem**: Need to balance accuracy and costs

```bash
# Compare model costs
python mathcorl.py solve --method fpp --question "Test" --model gpt-4o-mini
python mathcorl.py solve --method fpp --question "Test" --model gpt-4o
python mathcorl.py stats

# A/B test with different models
python mathcorl.py test --method fpp --dataset SVAMP --limit 25 --model gpt-4o-mini
python mathcorl.py test --method fpp --dataset SVAMP --limit 25 --model gpt-4o

# Analyze cost-effectiveness
python mathcorl.py chart --type cost --save
```

## 📈 **Analytics and Reporting**

### **Daily Cost Report Generation**
```bash
# Generate daily research summary
python mathcorl.py stats --hours 24 > daily_report.txt
python mathcorl.py export --format csv
python mathcorl.py chart --type all --save

# Create research progress report
python generate_report.py --tracking-data tracking_export.csv --charts charts/
```

### **Weekly Research Review**
```bash
# Weekly cost and progress analysis
python mathcorl.py stats --hours 168  # 7 days
python mathcorl.py export --format json

# Efficiency analysis
python analyze_weekly_progress.py --tracking-data tracking_export.json
```

### **Publication-Ready Analysis**
```python
# Generate statistics for research papers
def generate_research_summary(tracking_file):
    """Generate research summary for publication"""
    with open(tracking_file) as f:
        data = json.load(f)
    
    # Calculate key metrics
    total_requests = len(data)
    total_cost = sum(entry['total_cost'] for entry in data)
    avg_cost_per_request = total_cost / total_requests
    success_rate = sum(entry['success'] for entry in data) / total_requests
    
    # Method breakdown
    methods = {}
    for entry in data:
        method = entry['method']
        if method not in methods:
            methods[method] = {'requests': 0, 'cost': 0, 'successes': 0}
        methods[method]['requests'] += 1
        methods[method]['cost'] += entry['total_cost']
        methods[method]['successes'] += entry['success']
    
    # Calculate efficiency metrics
    for method, stats in methods.items():
        stats['accuracy'] = stats['successes'] / stats['requests']
        stats['cost_per_success'] = stats['cost'] / max(stats['successes'], 1)
        stats['efficiency'] = stats['accuracy'] / stats['cost'] if stats['cost'] > 0 else 0
    
    return {
        'total_requests': total_requests,
        'total_cost': total_cost,
        'avg_cost_per_request': avg_cost_per_request,
        'overall_success_rate': success_rate,
        'method_breakdown': methods
    }

# Usage
summary = generate_research_summary('tracking_export.json')
print(f"Research conducted {summary['total_requests']} experiments")
print(f"Overall success rate: {summary['overall_success_rate']:.1%}")
print(f"Most efficient method: {max(summary['method_breakdown'], key=lambda x: summary['method_breakdown'][x]['efficiency'])}")
```

## 🎯 **Research Planning Templates**

### **Template: Budget-Constrained Research Project**

#### **Phase 1: Exploration (Budget: Low)**
```bash
# Quick method validation
python mathcorl.py solve --method fpp --question "Simple test problem"
python mathcorl.py solve --method cot --question "Simple test problem"

# Small-scale testing
python mathcorl.py test --method fpp --dataset SVAMP --limit 10
python mathcorl.py stats
```

#### **Phase 2: Development (Budget: Medium)**
```bash
# Method comparison
python mathcorl.py compare --dataset GSM8K --limit 25

# ICL exploration  
python generate_candidates.py --dataset GSM8K --n-candidates 25
python train_policy.py --dataset GSM8K --epochs 2
python run_comparison.py --dataset GSM8K --samples 25 --methods policy,random

python mathcorl.py stats
python mathcorl.py export --format csv
```

#### **Phase 3: Validation (Budget: Higher)**
```bash
# Comprehensive evaluation
python mathcorl.py compare --dataset TAT-QA --limit 50
python run_comparison.py --dataset TAT-QA --samples 100 --save-results

# Generate final analysis
python mathcorl.py chart --type all --save
python mathcorl.py export --format json
```

### **Template: Publication Research Project**

#### **Phase 1: Baseline Establishment**
```bash
# Establish baselines across datasets
for dataset in GSM8K SVAMP TabMWP; do
    python mathcorl.py test --method fpp --dataset $dataset --limit 50
    python mathcorl.py test --method zero-shot --dataset $dataset --limit 50
done
```

#### **Phase 2: Method Innovation**  
```bash
# Develop and test novel approaches
python train_policy.py --dataset TAT-QA --epochs 3
python run_comparison.py --dataset TAT-QA --samples 100 --save-results

# Cross-dataset validation
python run_comparison.py --dataset FinQA --samples 50 --save-results
```

#### **Phase 3: Comprehensive Analysis**
```bash
# Generate publication-quality results
python mathcorl.py export --format json
python generate_publication_analysis.py --data tracking_export.json
python mathcorl.py chart --type all --save
```

---

**💡 Smart Research Management**: This guide demonstrates how to balance research quality with cost efficiency, using MathCoRL's tracking capabilities to make informed decisions about experiment scope, method selection, and budget allocation throughout the research process. 