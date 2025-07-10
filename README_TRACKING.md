# API Usage Tracking in MathCoRL

MathCoRL bây giờ đã được nâng cấp với hệ thống tracking API usage toàn diện, cho phép theo dõi:

- **Input tokens count** - Số tokens gửi đi (**100% chính xác** từ API metadata)
- **Output tokens count** - Số tokens nhận về (**100% chính xác** từ API metadata)
- **Request cost** - Chi phí cho từng request (**100% chính xác** dựa trên actual tokens)
- **Execution time** - Thời gian thực hiện request (**millisecond precision**)

### 🎯 **Key Features**
- ✅ **100% Accurate Token Counting** - Sử dụng official OpenAI API response metadata
- ✅ **Real-time Cost Tracking** - Tính toán chi phí chính xác theo pricing mới nhất
- ✅ **Comprehensive Logging** - Log tất cả API calls với full metadata
- ✅ **Multi-method Support** - Track FPP, CoT, PAL, PoT, Zero-Shot, ICRL
- ✅ **Export & Analysis** - Export data ra CSV/JSON cho analysis

## 🚀 Tính năng

### Automatic Tracking
Tất cả API calls được tự động track cho các phương pháp:
- **FPP** (Function Prototype Prompting)
- **CoT** (Chain-of-Thought)
- **PAL** (Program-aided Language Models)
- **PoT** (Program-of-Thoughts) 
- **Zero-Shot**
- **ICRL** (Candidate Generation, Evaluation, Embedding)

### Detailed Metrics
Mỗi API call được log với thông tin:
```json
{
  "timestamp": "2025-07-09T16:08:13.701495",
  "method": "FPP",
  "model": "gpt-4o-mini", 
  "input_tokens": 542,
  "output_tokens": 89,
  "total_tokens": 631,
  "input_cost": 0.000081,
  "output_cost": 0.000053,
  "total_cost": 0.000134,
  "execution_time": 1.23,
  "question": "What is 15 × 7?",
  "context": "",
  "success": true,
  "error_message": ""
}
```

### Cost Calculation
Hỗ trợ pricing tự động cho tất cả OpenAI models:
- **GPT-4.1-nano**: $0.0001/$0.0004 per 1K tokens
- **GPT-4.1-mini**: $0.0004/$0.0016 per 1K tokens
- **GPT-4.1**: $0.002/$0.008 per 1K tokens
- **GPT-4o-mini**: $0.00015/$0.0006 per 1K tokens
- **GPT-4o**: $0.0025/$0.01 per 1K tokens  
- **GPT-4-turbo**: $0.01/$0.03 per 1K tokens
- **GPT-3.5-turbo**: $0.0005/$0.0015 per 1K tokens
- **Embedding models**: $0.00002-$0.00013 per 1K tokens

## 📊 Usage Analytics

### Real-time Tracking
```python
from mint.tracking import get_tracker

# Get usage summary
tracker = get_tracker()
summary = tracker.get_usage_summary(last_n_hours=24)

print(f"Total cost: {summary['summary']['total_cost']}")
print(f"Total tokens: {summary['summary']['total_tokens']}")
```

### Method Comparison
```python
# Compare methods by cost/efficiency
by_method = summary['by_method']
for method, stats in by_method.items():
    print(f"{method}: {stats['cost']:.6f} USD, {stats['tokens']} tokens")
```

## 🔧 Implementation Details

### Context Manager Pattern
```python
from mint.tracking import track_api_call

with track_api_call("CoT", "gpt-4o-mini", question, context) as tracker:
    response = llm.invoke(messages)
    tracker.set_tokens(input_tokens, output_tokens)
```

### Token Counting Workflow
```python
# 1. Pre-estimate input tokens (for preview)
input_tokens = count_tokens_openai(prompt, model_name)

# 2. Make API call
response = llm.invoke(messages)

# 3. Extract actual tokens from response metadata
actual_input_tokens, output_tokens = extract_tokens_from_response(response)

# 4. Use actual tokens if available, fallback to estimation
if actual_input_tokens > 0:
    input_tokens = actual_input_tokens  # 100% accurate

# 5. Log with precise token counts
tracker.set_tokens(input_tokens, output_tokens)
```

### Verification Example
```python
# Test để verify token counting accuracy
from langchain_openai import ChatOpenAI
from langchain.schema import HumanMessage

llm = ChatOpenAI(model_name='gpt-4o-mini')
response = llm.invoke([HumanMessage(content='What is 2+2?')])

# Kiểm tra metadata
print(response.response_metadata['token_usage'])
# Output: {'prompt_tokens': 14, 'completion_tokens': 8, 'total_tokens': 22}
```

### Token Extraction
Tracking tự động extract token counts với **độ chính xác 100%** từ:

#### 1. **API Response Metadata** (Primary - 100% Accurate)
```python
# LangChain tự động extract từ OpenAI API response
response_metadata.token_usage = {
    'prompt_tokens': 14,      # Input tokens (chính xác 100%)
    'completion_tokens': 8,   # Output tokens (chính xác 100%)
    'total_tokens': 22        # Tổng tokens
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

## 📂 Log Files

### Storage Location
- **Default path**: `logs/api_usage.jsonl`
- **Format**: JSON Lines (một JSON object per line)
- **Encoding**: UTF-8

### Log Rotation
Logs được append vào file hiện tại. Để rotate logs:
```python
import os
from datetime import datetime

# Backup current log
current_log = "logs/api_usage.jsonl"
backup_name = f"logs/api_usage_{datetime.now().strftime('%Y%m%d_%H%M%S')}.jsonl"
os.rename(current_log, backup_name)
```

## 🎯 Use Cases

### 1. Cost Monitoring
```python
# Monitor daily spending
tracker = get_tracker()
daily_stats = tracker.get_usage_summary(last_n_hours=24)
daily_cost = daily_stats['summary']['total_cost']

if daily_cost > 10.0:  # $10 threshold
    print(f"⚠️ Daily cost exceeded: ${daily_cost:.2f}")
```

### 2. Method Optimization
```python
# Find most efficient method
by_method = summary['by_method']
efficiency = {}
for method, stats in by_method.items():
    if stats['requests'] > 0:
        efficiency[method] = stats['cost'] / stats['requests']

best_method = min(efficiency, key=efficiency.get)
print(f"Most cost-effective: {best_method}")
```

### 3. Performance Analysis
```python
# Analyze execution times
for method, stats in by_method.items():
    avg_time = stats['time'] / stats['requests'] if stats['requests'] > 0 else 0
    print(f"{method}: {avg_time:.2f}s average")
```

## 🛠 Configuration

### Custom Log Location
```python
from mint.tracking import APITracker

# Custom tracker
tracker = APITracker(log_file="custom/path/usage.jsonl")
```

### Environment Variables
```bash
# Optional: Custom model pricing
export CUSTOM_MODEL_PRICING='{"custom-model": {"input": 0.001, "output": 0.002}}'
```

## 📈 Reporting

### Summary Report
```python
tracker.get_usage_summary(last_n_hours=24)
# Returns:
# {
#   "summary": {...},
#   "by_method": {...}, 
#   "by_model": {...}
# }
```

### Token Accuracy Verification
```python
# Verify tracking accuracy với test call
from mint.tracking import get_tracker
import json

tracker = get_tracker()

# Check latest log entry
with open('logs/api_usage.jsonl', 'r') as f:
    lines = f.readlines()
    latest = json.loads(lines[-1])
    
print(f"Latest call:")
print(f"  Method: {latest['method']}")
print(f"  Model: {latest['model']}")
print(f"  Tokens: {latest['input_tokens']} → {latest['output_tokens']}")
print(f"  Cost: ${latest['total_cost']:.6f}")
print(f"  Success: {latest['success']}")
```

### Export Data
```python
import json

# Load all tracking data
with open("logs/api_usage.jsonl", 'r') as f:
    data = [json.loads(line) for line in f if line.strip()]

# Export to CSV
import pandas as pd
df = pd.DataFrame(data)
df.to_csv("usage_report.csv", index=False)
```

## 🔍 Troubleshooting

### Missing Token Counts
Nếu token counts = 0, có thể do:

#### 1. **API Errors**
```bash
# Quota exceeded - tokens = 0, success = false
"error_message": "Error code: 429 - You exceeded your current quota"
```

#### 2. **Network Issues**
```bash
# Connection timeout - tokens = 0, success = false  
"error_message": "Connection timeout"
```

#### 3. **Model Response Issues**
```python
# Rare case: API trả response nhưng không có metadata
if not hasattr(response, 'response_metadata'):
    # Fallback to estimation
    output_tokens = len(response.content) // 4
```

#### 4. **Debug Token Counting**
```python
# Verify token extraction
from mint.tracking import extract_tokens_from_response

response = llm.invoke(messages)
input_tokens, output_tokens = extract_tokens_from_response(response)
print(f"Tokens: {input_tokens} → {output_tokens}")

# Check response metadata
print("Has metadata:", hasattr(response, 'response_metadata'))
if hasattr(response, 'response_metadata'):
    print("Token usage:", response.response_metadata.get('token_usage'))
```

### Cost Calculation Issues
- Kiểm tra model name mapping trong `OPENAI_PRICING`
- Model mới có thể cần update pricing

### Log File Permissions
```bash
# Ensure logs directory is writable
chmod 755 logs/
chmod 644 logs/api_usage.jsonl
```

## 🎉 Benefits

1. **Transparency**: Track exact API costs và usage
2. **Optimization**: Identify most efficient methods
3. **Budgeting**: Monitor và control spending
4. **Debugging**: Detailed error logging
5. **Analytics**: Usage patterns và performance metrics

---

**Note**: Tracking system hoạt động song song với MathCoRL operations mà không ảnh hưởng đến performance hay results. 