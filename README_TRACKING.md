# API Usage Tracking in MathCoRL

MathCoRL bây giờ đã được nâng cấp với hệ thống tracking API usage toàn diện, cho phép theo dõi và tối ưa hóa chi phí trong nghiên cứu mathematical reasoning:

- **Input tokens count** - Số tokens gửi đi (**100% chính xác** từ API metadata)
- **Output tokens count** - Số tokens nhận về (**100% chính xác** từ API metadata)
- **Request cost** - Chi phí cho từng request (**100% chính xác** dựa trên actual tokens)
- **Execution time** - Thời gian thực hiện request (**millisecond precision**)
- **Method comparison** - So sánh hiệu quả giữa các phương pháp
- **Visual analytics** - Biểu đồ trực quan cho phân tích chi phí và hiệu suất

### 🎯 **Key Features**
- ✅ **100% Accurate Token Counting** - Sử dụng official OpenAI API response metadata
- ✅ **Real-time Cost Tracking** - Tính toán chi phí chính xác theo pricing mới nhất
- ✅ **Comprehensive Logging** - Log tất cả API calls với full metadata
- ✅ **Multi-method Support** - Track FPP, CoT, PAL, PoT, Zero-Shot, ICRL, Policy Network
- ✅ **Export & Analysis** - Export data ra CSV/JSON cho analysis
- ✅ **Visual Charts** - Tạo biểu đồ so sánh methods và phân tích cost
- ✅ **Cost Optimization** - Recommendations cho việc tối ưa hóa budget

## 🚀 Tính năng

### Automatic Tracking
Tất cả API calls được tự động track cho các phương pháp:
- **FPP** (Function Prototype Prompting)
- **CoT** (Chain-of-Thought)
- **PAL** (Program-aided Language Models)
- **PoT** (Program-of-Thoughts) 
- **Zero-Shot**
- **ICRL** (Policy Network, KATE, CDS, Random selection)
- **Embedding calls** (text-embedding-3-small)

### Detailed Metrics
Mỗi API call được log với thông tin:
```json
{
  "timestamp": "2025-07-10T16:08:13.701495",
  "method": "Policy Network",
  "model": "gpt-4o-mini", 
  "input_tokens": 1847,
  "output_tokens": 156,
  "total_tokens": 2003,
  "input_cost": 0.000277,
  "output_cost": 0.000094,
  "total_cost": 0.000371,
  "execution_time": 2.45,
  "question": "Mathematical reasoning question",
  "context": "Problem context...",
  "success": true,
  "error_message": "",
  "dataset": "Dataset_Name",
  "selected_method": "method_identifier"
}
```

### Cost Calculation
Hỗ trợ pricing tự động cho tất cả OpenAI models:
- **GPT-4o-mini**: $0.00015/$0.0006 per 1K tokens (default - best cost/performance)
- **GPT-4o**: $0.0025/$0.01 per 1K tokens
- **GPT-4-turbo**: $0.01/$0.03 per 1K tokens
- **GPT-3.5-turbo**: $0.0005/$0.0015 per 1K tokens
- **text-embedding-3-small**: $0.00002 per 1K tokens
- **text-embedding-3-large**: $0.00013 per 1K tokens

### Method Efficiency Tracking
Track và so sánh hiệu quả giữa các methods:
```json
{
  "method_comparison": {
    "FPP": {
      "avg_cost_per_request": 0.00089,
      "avg_tokens_per_request": 1456,
      "success_rate": 0.94,
      "cost_per_success": 0.00095
    },
    "Policy Network": {
      "avg_cost_per_request": 0.00124,
      "avg_tokens_per_request": 2103,
      "success_rate": 0.87,
      "cost_per_success": 0.00143
    },
    "KATE": {
      "avg_cost_per_request": 0.00098,
      "avg_tokens_per_request": 1678,
      "success_rate": 0.88,
      "cost_per_success": 0.00111
    }
  }
}
```

## 📊 Usage Analytics

### Real-time Tracking
```python
from mint.tracking import get_tracker

# Get usage summary
tracker = get_tracker()
summary = tracker.get_usage_summary(last_n_hours=24)

print(f"Total cost: ${summary['summary']['total_cost']:.6f}")
print(f"Total tokens: {summary['summary']['total_tokens']}")
print(f"Success rate: {summary['summary']['success_rate']:.1%}")
```

### Method Comparison
```python
# Compare methods by cost/efficiency
by_method = summary['by_method']
for method, stats in by_method.items():
    efficiency = stats['accuracy'] / stats['avg_cost'] if stats['avg_cost'] > 0 else 0
    print(f"{method}: {stats['accuracy']:.1%} accuracy, ${stats['cost']:.6f}, efficiency: {efficiency:.0f}")
```

### Cost Optimization Analysis
```python
# Identify most cost-effective methods
methods_efficiency = []
for method, stats in by_method.items():
    if stats['requests'] > 5:  # Minimum sample size
        cost_per_success = stats['cost'] / max(stats['successful'], 1)
        methods_efficiency.append((method, cost_per_success, stats['accuracy']))

# Sort by cost per successful solve
methods_efficiency.sort(key=lambda x: x[1])
print("Most cost-effective methods:")
for method, cost_per_success, accuracy in methods_efficiency[:3]:
    print(f"  {method}: ${cost_per_success:.6f} per success ({accuracy:.1%} accuracy)")
```

## 🔧 Implementation Details

### Context Manager Pattern
```python
from mint.tracking import track_api_call

with track_api_call("Policy Network", "gpt-4o-mini", question, context) as tracker:
    response = llm.invoke(messages)
    tracker.set_tokens(input_tokens, output_tokens)
    tracker.set_success(is_correct)
    tracker.set_dataset("Dataset_Name")
```

### Token Counting Workflow
```python
# 1. Pre-estimate input tokens (for preview)
input_tokens = count_tokens_openai(prompt, model_name)

# 2. Make API call
response = llm.invoke(messages)

# 3. Extract actual tokens from response metadata
if hasattr(response, 'response_metadata') and 'token_usage' in response.response_metadata:
    actual_input_tokens = response.response_metadata['token_usage']['prompt_tokens']
    output_tokens = response.response_metadata['token_usage']['completion_tokens']
    
    # Use actual tokens (100% accurate)
    input_tokens = actual_input_tokens

# 4. Log with precise token counts
tracker.set_tokens(input_tokens, output_tokens)
```

### Verification Example
```python
# Test để verify token counting accuracy
from langchain_openai import ChatOpenAI
from langchain.schema import HumanMessage

llm = ChatOpenAI(model_name='gpt-4o-mini')
response = llm.invoke([HumanMessage(content='Calculate: 15 + 27')])

# Kiểm tra metadata
print(response.response_metadata['token_usage'])
# Output: {'prompt_tokens': 16, 'completion_tokens': 8, 'total_tokens': 24}

# Cost calculation
input_cost = 16 * 0.00015 / 1000  # $0.000002
output_cost = 8 * 0.0006 / 1000   # $0.000005
total_cost = input_cost + output_cost  # $0.000007
```

### Token Extraction
Tracking tự động extract token counts với **độ chính xác 100%** từ:

#### 1. **API Response Metadata** (Primary - 100% Accurate)
```python
# LangChain tự động extract từ OpenAI API response
response_metadata.token_usage = {
    'prompt_tokens': 16,      # Input tokens (chính xác 100%)
    'completion_tokens': 8,   # Output tokens (chính xác 100%)
    'total_tokens': 24        # Tổng tokens
}
```

#### 2. **Pre-estimation** (Before API Call)
```python
# Ước tính input tokens trước khi gọi API
input_tokens = len(prompt) // 4  # ~4 characters per token
```

#### 3. **Fallback Estimation** (Rare Cases)
```python
# Chỉ dùng khi API không trả metadata (hiếm khi xảy ra)
output_tokens = len(response.content) // 4
```

#### 4. **Accuracy Guarantee**
- ✅ **Input tokens**: 100% chính xác từ `prompt_tokens`
- ✅ **Output tokens**: 100% chính xác từ `completion_tokens`  
- ✅ **Cost calculation**: 100% chính xác dựa trên actual tokens
- ⚠️ **Estimation**: Chỉ dùng cho preview/fallback (~75% accuracy)

## 📊 Visual Analytics & Charts

### Chart Generation
```bash
# Generate all chart types
python mathcorl.py chart --type all --save

# Specific chart types
python mathcorl.py chart --type comparison  # Method comparison
python mathcorl.py chart --type cost        # Cost analysis
python mathcorl.py chart --type time        # Time analysis
python mathcorl.py chart --type tokens      # Token analysis

# Custom time ranges
python mathcorl.py chart --type all --hours 12 --save
```

### Available Chart Types

#### 1. **Method Comparison Chart**
- So sánh **Input Tokens**, **Output Tokens**, **Execution Time**, **Cost** giữa các phương pháp
- Hiển thị trung bình của từng metric
- Dễ dàng nhận biết phương pháp nào hiệu quả nhất

#### 2. **Cost Analysis Chart**
- **Pie Chart**: Phân bổ chi phí theo từng phương pháp
- **Scatter Plot**: Mối quan hệ giữa số tokens và chi phí
- Giúp tối ưa hóa budget

#### 3. **Time Analysis Chart**
- **Box Plot**: Phân bố thời gian thực hiện theo phương pháp
- **Scatter Plot**: Mối quan hệ giữa tokens và thời gian
- Phát hiện bottlenecks

#### 4. **Token Analysis Chart**
- **Stacked Bar**: Input vs Output tokens theo phương pháp
- **Token Efficiency**: Tỷ lệ Output/Input tokens
- **Distribution**: Phân bố tokens của từng phương pháp

### Sample Chart Analysis Structure
Charts provide insights into:
- **Method Efficiency**: Which methods provide best accuracy per cost
- **Token Usage Patterns**: How different methods consume API resources
- **Performance Bottlenecks**: Where time and cost optimizations are needed
- **Budget Planning**: Historical data for future cost estimation

## 📂 Log Files

### Storage Location
- **Default path**: `logs/api_usage.jsonl`
- **Format**: JSON Lines (một JSON object per line)
- **Encoding**: UTF-8
- **Backup**: Automatic backup when clearing logs

### Log Rotation và Management
```python
import os
from datetime import datetime

# Backup current log
current_log = "logs/api_usage.jsonl"
backup_name = f"logs/api_usage_backup_{datetime.now().strftime('%Y%m%d_%H%M%S')}.jsonl"
os.rename(current_log, backup_name)

# Clear logs with backup
python mathcorl.py clear-logs  # Creates backup automatically
```

### Log Analysis
```python
import json
import pandas as pd

# Load logs for analysis
logs = []
with open('logs/api_usage.jsonl', 'r') as f:
    for line in f:
        logs.append(json.loads(line))

# Convert to DataFrame for analysis
df = pd.DataFrame(logs)

# Analysis examples
method_costs = df.groupby('method')['total_cost'].sum()
method_success_rates = df.groupby('method')['success'].mean()
hourly_usage = df.set_index('timestamp').resample('H')['total_cost'].sum()
```

## 🎯 Use Cases & Cost Optimization

### 1. Budget Monitoring
```bash
# Monitor daily spending
python mathcorl.py stats --hours 24

# Sample output structure:
# 💰 Total Cost: Current spending amount
# 📊 Method breakdown: Cost by each method
# 🏆 Most efficient: Best cost-effectiveness ratio
# ⚠️ Budget status: Percentage of daily budget used

# Set budget alerts for cost control
if daily_cost > budget_threshold:
    print(f"⚠️ Daily budget exceeded: ${daily_cost:.4f}")
```

### 2. Method Optimization
```bash
# Find most efficient method for research needs
python mathcorl.py stats
python mathcorl.py export --format csv

# Analysis capabilities:
# - Cost per successful solve
# - Accuracy vs cost trade-offs
# - Time efficiency analysis
```

### 3. Research Cost Planning
```python
# Estimate research costs based on historical data
def estimate_research_cost(dataset, samples, methods, historical_cost_per_sample):
    total_cost = samples * len(methods) * historical_cost_per_sample
    return total_cost

# Example usage for planning
methods = ['policy', 'kate', 'cds', 'random']
estimated_cost = estimate_research_cost('TAT-QA', 150, methods, 0.001)
print(f"Estimated research cost: ${estimated_cost:.4f}")
```

### 4. Model Selection Optimization
```python
# Compare different OpenAI models for cost-effectiveness
model_comparison = {
    'gpt-4o-mini': {'cost_per_1k': 0.15, 'typical_accuracy': 0.86},
    'gpt-4o': {'cost_per_1k': 2.5, 'typical_accuracy': 0.89},
    'gpt-4-turbo': {'cost_per_1k': 10.0, 'typical_accuracy': 0.91}
}

# Calculate cost-effectiveness for informed decision making
for model, stats in model_comparison.items():
    cost_effectiveness = stats['typical_accuracy'] / stats['cost_per_1k']
    print(f"{model}: {cost_effectiveness:.2f} accuracy points per dollar")
```

## 💡 Cost Optimization Best Practices

### 1. **Model Selection**
- ✅ **Use gpt-4o-mini**: Best cost/performance ratio for mathematical reasoning
- ⚠️ **Avoid gpt-4-turbo**: Significantly more expensive for marginal accuracy gains
- 📊 **Monitor accuracy**: Ensure cost savings don't compromise research quality

### 2. **Sample Size Planning**
```bash
# Progressive approach: start small, scale based on results
python run_comparison.py --dataset TAT-QA --samples 10   # Test run
python mathcorl.py stats  # Check costs
python run_comparison.py --dataset TAT-QA --samples 50   # Medium test
python run_comparison.py --dataset TAT-QA --samples 150  # Full evaluation
```

### 3. **Method Selection Strategy**
- **For initial exploration**: Use Zero-shot + Random (lowest cost)
- **For method comparison**: Add FPP + KATE (proven effective)
- **For research validation**: Include Policy Network (novel approach)

### 4. **Token Optimization**
```python
# Strategies to reduce token usage:
# - Use concise function prototypes
# - Limit example length in ICL
# - Remove redundant context
# - Monitor token usage patterns with charts
python mathcorl.py chart --type tokens --save
```

### 5. **Batch Processing**
```bash
# Efficient processing approach
python generate_candidates.py --dataset TAT-QA --n-candidates 200  # One-time cost
python train_policy.py --dataset TAT-QA --epochs 3                 # One-time training
python run_comparison.py --dataset TAT-QA --samples 150            # Batch evaluation
```

## 🛠️ Integration với Research Workflow

### Step 1: Pre-Research Cost Estimation
```bash
# Estimate costs before starting major experiments
python mathcorl.py estimate --dataset TAT-QA --samples 100 --methods all
# Provides estimated cost for planning budget allocation
```

### Step 2: Real-time Monitoring
```bash
# Monitor progress during long-running experiments
python run_comparison.py --dataset TAT-QA --samples 150 &
while [ process_running ]; do
    sleep 300  # Check every 5 minutes
    python mathcorl.py stats --hours 1
done
```

### Step 3: Post-Research Analysis
```bash
# Comprehensive analysis after completion
python mathcorl.py export --format json
python mathcorl.py chart --type all --save

# Generate research cost report for documentation
python analyze_costs.py --input tracking_export.json --output cost_report.pdf
```

### Integration với Publication Pipeline
```python
# Include cost data in research documentation
def generate_cost_summary(tracking_data):
    return {
        "total_api_calls": len(tracking_data),
        "total_cost": sum(entry['total_cost'] for entry in tracking_data),
        "cost_per_sample": calculate_average_cost_per_sample(tracking_data),
        "most_efficient_method": find_most_efficient_method(tracking_data),
        "cost_breakdown": generate_method_breakdown(tracking_data)
    }

# Cost-effectiveness analysis for papers
print(f"Method X achieves Y% accuracy at $Z per sample")
print(f"Cost difference: +W% for -V% accuracy vs. baseline")
```

---

**💰 Smart Cost Management**: Tracking giúp researchers tối ưa hóa budget, so sánh hiệu quả methods, và đưa ra quyết định thông minh về API usage trong mathematical reasoning research! 