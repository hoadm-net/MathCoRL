# Claude Integration in MathCoRL

MathCoRL now supports **Claude API** alongside OpenAI! You can use Claude 3.5 Sonnet, Claude 3 Opus, Claude Haiku, and other Anthropic models for mathematical reasoning tasks.

## 🚀 **Quick Setup**

### 1. Install Dependencies

```bash
# Install the additional required package
pip install langchain-anthropic
```

### 2. Configure API Keys

Add your Anthropic API key to your `.env` file:

```bash
# OpenAI API Configuration
OPENAI_API_KEY=your_openai_api_key_here
ANTHROPIC_API_KEY=your_anthropic_api_key_here

# Model Configuration
DEFAULT_MODEL=gpt-4o-mini
ANT_DEFAULT_MODEL=claude-3-5-sonnet-20241022
EMBEDDING_MODEL=text-embedding-3-small
TEMPERATURE=0.1
MAX_TOKENS=1000

# LLM Provider Selection
LLM_PROVIDER=claude  # Options: 'openai', 'claude'
```

### 3. Start Using Claude

```python
from mint.core import FunctionPrototypePrompting, solve_math_problem

# Option 1: Use environment config (set LLM_PROVIDER=claude)
fpp = FunctionPrototypePrompting()
result = fpp.solve("What is 15 + 27?")

# Option 2: Explicitly specify provider
fpp = FunctionPrototypePrompting(provider="claude")
result = fpp.solve("What is 15 + 27?")

# Option 3: Simple function interface
result = solve_math_problem("What is 15 + 27?", provider="claude")
```

## 🧠 **Supported Methods**

All MathCoRL reasoning methods now support Claude:

### Function Prototype Prompting (FPP)
```python
from mint.core import FunctionPrototypePrompting

fpp = FunctionPrototypePrompting(provider="claude")
result = fpp.solve("Calculate the area of a circle with radius 5")
```

### Chain-of-Thought (CoT)
```python
from mint.cot import ChainOfThoughtPrompting

cot = ChainOfThoughtPrompting(provider="claude")
result = cot.solve("John has 20 apples. He gives 8 to his friend. How many are left?")
```

### Program-aided Language Models (PAL)
```python
from mint.pal import ProgramAidedLanguageModel

pal = ProgramAidedLanguageModel(provider="claude")
result = pal.solve("Calculate the average of 10, 20, 30, 40, 50")
```

### Program of Thoughts (PoT)
```python
from mint.pot import ProgramOfThoughtsPrompting

pot = ProgramOfThoughtsPrompting(provider="claude")
result = pot.solve("Solve for x: 2x + 5 = 15")
```

### Zero-Shot
```python
from mint.zero_shot import ZeroShotPrompting

zs = ZeroShotPrompting(provider="claude")
result = zs.solve("What is 25% of 80?")
```

## 🎯 **CLI Usage**

### Single Problem Solving
```bash
# Use Claude for solving problems
python -m mint.cli solve --method fpp --provider claude --question "What is 15 + 27?"
python -m mint.cli solve --method cot --provider claude --question "John has 20 apples..."

# Use OpenAI for comparison
python -m mint.cli solve --method fpp --provider openai --question "What is 15 + 27?"
```

### Dataset Testing
```bash
# Test with Claude
python -m mint.cli test --method fpp --provider claude --dataset SVAMP --limit 50
python -m mint.cli test --method cot --provider claude --dataset GSM8K --limit 100

# Test with OpenAI for comparison
python -m mint.cli test --method fpp --provider openai --dataset SVAMP --limit 50
```

### Interactive Mode
```bash
# Start interactive mode with Claude
python -m mint.cli interactive --provider claude

# Start interactive mode with OpenAI
python -m mint.cli interactive --provider openai
```

## 🤖 **Available Claude Models**

MathCoRL supports all major Claude models:

| Model | Description | Use Case |
|-------|-------------|----------|
| `claude-3-5-sonnet-20241022` | **Latest & Best** - Most capable model | Complex reasoning, coding |
| `claude-3-5-sonnet-20240620` | Previous Sonnet version | General tasks |
| `claude-3-5-haiku-20241022` | Fast and efficient | Quick calculations |
| `claude-3-opus-20240229` | Most powerful (expensive) | Most complex problems |
| `claude-3-sonnet-20240229` | Balanced performance | General mathematical tasks |
| `claude-3-haiku-20240307` | Fastest and cheapest | Simple calculations |

### Configure Model
```bash
# In .env file
ANT_DEFAULT_MODEL=claude-3-5-sonnet-20241022  # Default
ANT_DEFAULT_MODEL=claude-3-opus-20240229      # Most powerful
ANT_DEFAULT_MODEL=claude-3-5-haiku-20241022   # Fastest
```

## 💰 **Cost Tracking**

Claude usage is automatically tracked alongside OpenAI:

```bash
# View usage statistics for both providers
python -m mint.cli stats --hours 24

# Export usage data
python -m mint.cli export --format csv
```

Example output:
```
📊 API Usage Statistics (Last 24 hours)
🔥 OVERVIEW
   Total Requests: 45
   ✅ Successful: 43
   📈 Success Rate: 95.6%
   🔢 Total Tokens: 28,450
   💰 Total Cost: $0.085200

🔧 BY METHOD (Detailed Analysis)
Method       Reqs  Avg Input  Avg Output  Avg Time  Avg Cost
FPP          15    850        120         2.3s      $0.002850
CoT          10    750        180         1.8s      $0.003200
Claude-FPP   12    860        125         2.1s      $0.003100
Claude-CoT   8     780        190         1.9s      $0.003800
```

## 🧪 **Testing Integration**

Run the test script to verify Claude integration:

```bash
python test_claude_integration.py
```

This will test all methods with both OpenAI and Claude to ensure everything works correctly.

## ⚡ **Performance Comparison**

| Aspect | OpenAI GPT-4o-mini | Claude 3.5 Sonnet | Claude 3 Haiku |
|--------|-------------------|-------------------|-----------------|
| **Speed** | ⭐⭐⭐⭐ | ⭐⭐⭐ | ⭐⭐⭐⭐⭐ |
| **Accuracy** | ⭐⭐⭐⭐ | ⭐⭐⭐⭐⭐ | ⭐⭐⭐ |
| **Cost** | ⭐⭐⭐⭐⭐ | ⭐⭐⭐ | ⭐⭐⭐⭐⭐ |
| **Reasoning** | ⭐⭐⭐⭐ | ⭐⭐⭐⭐⭐ | ⭐⭐⭐ |
| **Code Gen** | ⭐⭐⭐⭐ | ⭐⭐⭐⭐⭐ | ⭐⭐⭐ |

## 🔧 **Advanced Configuration**

### Provider-Specific Settings
```python
# Claude with custom settings
claude_fpp = FunctionPrototypePrompting(
    provider="claude",
    model="claude-3-5-sonnet-20241022",
    temperature=0.1,
    max_tokens=1000
)

# OpenAI with custom settings
openai_fpp = FunctionPrototypePrompting(
    provider="openai", 
    model="gpt-4o-mini",
    temperature=0.0,
    max_tokens=800
)
```

### Dynamic Provider Switching
```python
def solve_with_fallback(question: str, primary_provider: str = "claude"):
    """Solve with fallback to alternative provider."""
    try:
        # Try primary provider
        result = solve_math_problem(question, provider=primary_provider)
        if result is not None:
            return result, primary_provider
    except Exception as e:
        print(f"Primary provider failed: {e}")
    
    # Fallback to alternative
    fallback = "openai" if primary_provider == "claude" else "claude"
    try:
        result = solve_math_problem(question, provider=fallback)
        return result, fallback
    except Exception as e:
        print(f"Fallback provider also failed: {e}")
        return None, None

# Usage
result, used_provider = solve_with_fallback("What is 15 + 27?")
print(f"Result: {result} (using {used_provider})")
```

## 🎉 **Why Use Claude?**

### ✅ **Advantages**
- **Better Reasoning**: Claude 3.5 Sonnet often provides more detailed step-by-step reasoning
- **Code Quality**: Excellent at generating clean, well-documented Python code
- **Mathematical Understanding**: Strong performance on complex mathematical problems
- **Latest Technology**: Access to Anthropic's newest models

### ⚠️ **Considerations**
- **Cost**: Generally more expensive than OpenAI GPT-4o-mini
- **Speed**: Slightly slower than GPT-4o-mini for simple calculations
- **API Limits**: Different rate limits compared to OpenAI

## 🛠️ **Troubleshooting**

### Common Issues

**1. ImportError: No module named 'langchain_anthropic'**
```bash
pip install langchain-anthropic
```

**2. AuthenticationError: Invalid API key**
- Check your `ANTHROPIC_API_KEY` in `.env`
- Ensure your Anthropic account has credits
- Verify the API key format

**3. Rate Limit Errors**
- Claude has different rate limits than OpenAI
- Consider using `claude-3-haiku` for faster requests
- Implement retry logic with exponential backoff

**4. Model Not Found**
- Check model name spelling
- Ensure you're using a supported Claude model
- Update `ANT_DEFAULT_MODEL` in `.env`

### Debug Mode
```python
# Enable debug logging
import logging
logging.basicConfig(level=logging.DEBUG)

# Test with detailed output
from mint.core import FunctionPrototypePrompting
fpp = FunctionPrototypePrompting(provider="claude")
result = fpp.solve_detailed("Test problem")
print(result)
```

## 📚 **Next Steps**

1. **Try Different Models**: Experiment with various Claude models for your use cases
2. **Compare Performance**: Use both providers and compare accuracy/cost
3. **Monitor Usage**: Track API costs and optimize your usage patterns
4. **Contribute**: Help improve Claude integration by reporting issues or submitting PRs

---

🚀 **Happy Mathematical Reasoning with Claude!** 

For questions or issues, please check the main project documentation or create an issue on GitHub. 