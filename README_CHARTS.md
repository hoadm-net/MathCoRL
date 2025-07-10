# 📊 MathCoRL Visualization Charts

MathCoRL bây giờ hỗ trợ tạo biểu đồ trực quan để phân tích hiệu suất các phương pháp prompting.

## 🎯 **Tính năng**

### **1. Method Comparison Chart**
- So sánh **Input Tokens**, **Output Tokens**, **Execution Time**, **Cost** giữa các phương pháp
- Hiển thị trung bình của từng metric
- Dễ dàng nhận biết phương pháp nào hiệu quả nhất

### **2. Cost Analysis Chart**
- **Pie Chart**: Phân bổ chi phí theo từng phương pháp
- **Scatter Plot**: Mối quan hệ giữa số tokens và chi phí
- Giúp tối ưu hóa budget

### **3. Time Analysis Chart**
- **Box Plot**: Phân bố thời gian thực hiện theo phương pháp
- **Scatter Plot**: Mối quan hệ giữa tokens và thời gian
- Phát hiện bottlenecks

### **4. Token Analysis Chart**
- **Stacked Bar**: Input vs Output tokens theo phương pháp
- **Token Efficiency**: Tỷ lệ Output/Input tokens
- **Distribution**: Phân bố tokens của từng phương pháp

## 🚀 **Cách sử dụng**

### **Tạo tất cả biểu đồ**
```bash
python mathcorl.py chart --type all
```

### **Tạo biểu đồ cụ thể**
```bash
# So sánh phương pháp
python mathcorl.py chart --type comparison

# Phân tích chi phí
python mathcorl.py chart --type cost

# Phân tích thời gian
python mathcorl.py chart --type time

# Phân tích tokens
python mathcorl.py chart --type tokens
```

### **Lưu biểu đồ vào file**
```bash
# Lưu tất cả biểu đồ
python mathcorl.py chart --type all --save

# Lưu biểu đồ cụ thể
python mathcorl.py chart --type cost --save
```

### **Tùy chỉnh thời gian**
```bash
# Xem data trong 12 giờ qua
python mathcorl.py chart --hours 12

# Xem data trong 7 ngày qua
python mathcorl.py chart --hours 168
```

## 📁 **Output Files**

Khi sử dụng `--save`, biểu đồ sẽ được lưu trong thư mục `charts/`:

```
charts/
├── method_comparison_20250710_075147.png
├── cost_analysis_20250710_075148.png
├── time_analysis_20250710_075148.png
└── token_analysis_20250710_075148.png
```

## 🎨 **Tùy chỉnh**

### **Dependencies**
```bash
pip install matplotlib seaborn pandas
```

### **Chart Styles**
- **Style**: Seaborn v0.8 với palette "husl"
- **Resolution**: 300 DPI cho quality cao
- **Format**: PNG với bbox_inches='tight'

## 📈 **Ví dụ phân tích**

### **Từ biểu đồ Method Comparison:**
```
Method       Avg Input  Avg Output  Avg Time  Avg Cost
FPP          1,806      33          1.77s     $0.000775
CoT          412        208         4.73s     $0.000497
PAL          154        196         4.02s     $0.000376
PoT          642        109         7.57s     $0.000431
Zero-Shot    32         31          1.87s     $0.000063
```

### **Insights:**
- **Zero-Shot**: Fastest & cheapest cho simple problems
- **FPP**: High input tokens do function definitions
- **CoT**: Balanced reasoning với detailed output
- **PAL**: Best cost/performance ratio
- **PoT**: Slowest nhưng executable code

## 🔧 **Troubleshooting**

### **Font Warnings**
```
UserWarning: Glyph 128202 (\N{BAR CHART}) missing from font(s) Arial.
```
**Solution**: Warnings này không ảnh hưởng functionality. Emoji trong titles sẽ không hiển thị nhưng charts vẫn hoạt động bình thường.

### **No Data**
```
💡 No tracking data found in the last 24 hours.
```
**Solution**: Chạy một vài solve commands trước để có data:
```bash
python mathcorl.py solve --method fpp "What is 2+2?"
python mathcorl.py solve --method cot "What is 2+2?"
```

### **Import Errors**
```
❌ Required libraries not installed.
```
**Solution**: Cài đặt dependencies:
```bash
pip install matplotlib seaborn pandas
```

## 🎯 **Best Practices**

1. **Chạy nhiều tests trước** để có data đa dạng
2. **Sử dụng --save** để lưu biểu đồ cho reports
3. **Tùy chỉnh --hours** để focus vào timeframe cụ thể
4. **Combine với stats command** để có cả text và visual analysis

## 🔗 **Related Commands**

```bash
# Xem stats dạng text
python mathcorl.py stats

# Export raw data
python mathcorl.py export --format json

# Generate charts
python mathcorl.py chart --type all

# Clear old data
python mathcorl.py clear-logs
```

---

**Happy Analyzing!** 📊✨ 