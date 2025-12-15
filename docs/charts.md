# 📊 MathCoRL Visualization Charts

MathCoRL now supports creating visual charts to analyze the performance of prompting methods.

## 🎯 **Features**

### **1. Method Comparison Chart**
- Compare **Input Tokens**, **Output Tokens**, **Execution Time**, **Cost** between methods
- Display average of each metric
- Easily identify which method is most effective

### **2. Cost Analysis Chart**
- **Pie Chart**: Cost distribution by each method
- **Scatter Plot**: Relationship between token count and cost
- Help optimize budget

### **3. Time Analysis Chart**
- **Box Plot**: Execution time distribution by method
- **Scatter Plot**: Relationship between tokens and time
- Identify bottlenecks

### **4. Token Analysis Chart**
- **Stacked Bar**: Input vs Output tokens by method
- **Token Efficiency**: Output/Input token ratio
- **Distribution**: Token distribution of each method

## 🚀 **Usage**

### **Create all charts**
```bash
python mathcorl.py chart --type all
```

### **Create specific charts**
```bash
# Method comparison
python mathcorl.py chart --type comparison

# Cost analysis
python mathcorl.py chart --type cost

# Time analysis
python mathcorl.py chart --type time

# Token analysis
python mathcorl.py chart --type tokens
```

### **Save charts to file**
```bash
# Save all charts
python mathcorl.py chart --type all --save

# Save specific chart
python mathcorl.py chart --type cost --save
```

### **Customize time range**
```bash
# View data from last 12 hours
python mathcorl.py chart --hours 12

# View data from last 7 days
python mathcorl.py chart --hours 168
```

## 📁 **Output Files**

When using `--save`, charts will be saved in the `charts/` directory:

```
charts/
├── method_comparison_20250710_075147.png
├── cost_analysis_20250710_075148.png
├── time_analysis_20250710_075148.png
└── token_analysis_20250710_075148.png
```

## 🎨 **Customization**

### **Dependencies**
```bash
pip install matplotlib seaborn pandas
```

### **Chart Styles**
- **Style**: Seaborn v0.8 with "husl" palette
- **Resolution**: 300 DPI for high quality
- **Format**: PNG with bbox_inches='tight'

## 📈 **Analysis Examples**

### **From Method Comparison chart:**
```
Method       Avg Input  Avg Output  Avg Time  Avg Cost
FPP          1,806      33          1.77s     $0.000775
CoT          412        208         4.73s     $0.000497
PAL          154        196         4.02s     $0.000376
PoT          642        109         7.57s     $0.000431
Zero-Shot    32         31          1.87s     $0.000063
```

### **Insights:**
- **Zero-Shot**: Fastest & cheapest for simple problems
- **FPP**: High input tokens due to function definitions
- **CoT**: Balanced reasoning with detailed output
- **PAL**: Best cost/performance ratio
- **PoT**: Slowest but executable code

## 🔧 **Troubleshooting**

### **Font Warnings**
```
UserWarning: Glyph 128202 (\N{BAR CHART}) missing from font(s) Arial.
```
**Solution**: These warnings don't affect functionality. Emojis in titles won't display but charts will work normally.

### **No Data**
```
💡 No tracking data found in the last 24 hours.
```
**Solution**: Run a few solve commands first to get data:
```bash
python mathcorl.py solve --method fpp "What is 2+2?"
python mathcorl.py solve --method cot "What is 2+2?"
```

### **Import Errors**
```
❌ Required libraries not installed.
```
**Solution**: Install dependencies:
```bash
pip install matplotlib seaborn pandas
```

## 🎯 **Best Practices**

1. **Run multiple tests first** to get diverse data
2. **Use --save** to save charts for reports
3. **Customize --hours** to focus on specific timeframe
4. **Combine with stats command** for both text and visual analysis

## 🔗 **Related Commands**

```bash
# View stats in text format
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