# MathCoRL - Hướng dẫn sử dụng với API Tracking

## 🚀 Cách sử dụng cơ bản

### 1. Giải một bài toán
```bash
# Sử dụng FPP (Function Prototype Prompting)
python mathcorl.py solve --method fpp --question "What is 15 + 27?"

# Sử dụng CoT (Chain-of-Thought)
python mathcorl.py solve --method cot --question "John has 20 apples. He gives 8 to his friend. How many apples does John have left?"

# Sử dụng PoT (Program-of-Thoughts)
python mathcorl.py solve --method pot --question "If I have $100 and spend $35 on groceries, how much do I have left?"

# Sử dụng PAL (Program-aided Language Models)
python mathcorl.py solve --method pal --question "A rectangle has length 8m and width 5m. What is its area?"

# Sử dụng Zero-Shot
python mathcorl.py solve --method zero_shot --question "What is 15 × 7?"
```

### 2. Mode tương tác
```bash
python mathcorl.py interactive
# hoặc chỉ cần
python mathcorl.py
```

### 3. Test trên dataset
```bash
# Test FPP trên 50 bài từ SVAMP
python mathcorl.py test --method fpp --dataset SVAMP --limit 50

# Test CoT trên 100 bài từ GSM8K  
python mathcorl.py test --method cot --dataset GSM8K --limit 100
```

### 4. So sánh các phương pháp
```bash
python mathcorl.py compare --dataset SVAMP --limit 20
```

## 📊 API Tracking - Tính năng mới!

### Xem thống kê usage
```bash
# Xem stats 24 giờ gần nhất
python mathcorl.py stats

# Xem stats 12 giờ gần nhất
python mathcorl.py stats --hours 12

# Xem stats 1 giờ gần nhất
python mathcorl.py stats --hours 1
```

**Output:**
```
📊 API Usage Statistics (Last 24 hours)
==================================================
🔥 OVERVIEW
   Total Requests: 15
   ✅ Successful: 12
   ❌ Failed: 3
   📈 Success Rate: 80.0%
   🔢 Total Tokens: 8,542
   💰 Total Cost: $0.001283
   ⏱️  Avg Time: 2.34s

🔧 BY METHOD
   FPP: 5 requests, 2,142 tokens, $0.000321
   CoT: 4 requests, 3,456 tokens, $0.000518
   PAL: 3 requests, 1,987 tokens, $0.000298
   PoT: 2 requests, 897 tokens, $0.000134
   Zero-Shot: 1 requests, 60 tokens, $0.000012

🤖 BY MODEL
   gpt-4o-mini: 15 requests, 8,542 tokens, $0.001283
```

### Export dữ liệu tracking
```bash
# Export ra CSV
python mathcorl.py export --format csv

# Export ra JSON
python mathcorl.py export --format json
```

### Clear logs tracking
```bash
python mathcorl.py clear-logs
```

### Xem danh sách datasets
```bash
python mathcorl.py datasets
```

## 🎯 Ví dụ thực tế

### Scenario 1: So sánh hiệu quả các phương pháp
```bash
# Test từng method trên cùng dataset
python mathcorl.py test --method fpp --dataset SVAMP --limit 10
python mathcorl.py test --method cot --dataset SVAMP --limit 10
python mathcorl.py test --method pal --dataset SVAMP --limit 10

# Xem thống kê để so sánh
python mathcorl.py stats --hours 1
```

### Scenario 2: Monitor cost khi làm việc
```bash
# Làm việc như bình thường...
python mathcorl.py solve --method fpp --question "Calculate compound interest..."

# Check cost định kỳ
python mathcorl.py stats --hours 1

# Export data để phân tích
python mathcorl.py export --format csv
```

### Scenario 3: Interactive problem solving
```bash
# Start interactive mode
python mathcorl.py interactive

# Trong interactive mode:
# 1. Chọn method (1-5)
# 2. Nhập câu hỏi
# 3. Nhập context (optional)
# 4. Xem kết quả và generated code
# 5. Type 'switch' để đổi method
# 6. Type 'exit' để thoát

# Sau khi xong, check usage
python mathcorl.py stats
```

## 💡 Tips

1. **Cost monitoring**: Check `python mathcorl.py stats` thường xuyên để theo dõi chi phí
2. **Method selection**: 
   - **FPP/PoT/PAL**: Tốt cho bài toán computational
   - **CoT**: Tốt cho bài cần reasoning chi tiết
   - **Zero-Shot**: Fastest, ít cost nhất
3. **Backup logs**: Logs được backup tự động khi clear
4. **Export data**: Dùng CSV để phân tích trong Excel/Google Sheets

## 🔧 Troubleshooting

### Quota exceeded error
```bash
# Check cost hiện tại
python mathcorl.py stats

# Nếu bị limit, chờ reset hoặc upgrade plan
```

### Tracking data không hiển thị
```bash
# Check log file exists
ls -la logs/api_usage.jsonl

# If empty, run some operations first
python mathcorl.py solve --method fpp --question "Test question"
```

---

**Lưu ý**: Tracking hoạt động tự động và không ảnh hưởng đến performance. Tất cả API calls đều được log với đầy đủ thông tin cost, tokens, và timing! 