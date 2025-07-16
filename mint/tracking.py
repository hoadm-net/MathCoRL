"""
API Usage Tracking for MathCoRL

This module provides utilities to track and log API usage metrics including:
- Input tokens count
- Output tokens count  
- Request cost calculation
- Request execution time
"""

import time
import json
import logging
from datetime import datetime
from typing import Dict, Any, Optional, Union
from dataclasses import dataclass, asdict
from pathlib import Path

logger = logging.getLogger(__name__)

# OpenAI pricing (per 1000 tokens) - Updated January 2025
# Based on official OpenAI pricing as of January 2025
# Source: https://openai.com/api/pricing/
OPENAI_PRICING = {
    # GPT-4.1 models (Latest - Released April 2025)
    "gpt-4.1": {"input": 0.002, "output": 0.008},  # Official: $2.00/$8.00 per 1M tokens
    "gpt-4.1-mini": {"input": 0.0004, "output": 0.0016},  # Official: $0.40/$1.60 per 1M tokens
    "gpt-4.1-nano": {"input": 0.0001, "output": 0.0004},  # Official: $0.10/$0.40 per 1M tokens
    
    # GPT-4 models
    "gpt-4": {"input": 0.03, "output": 0.06},
    "gpt-4-32k": {"input": 0.06, "output": 0.12},
    "gpt-4-turbo": {"input": 0.01, "output": 0.03},
    "gpt-4-turbo-preview": {"input": 0.01, "output": 0.03},
    "gpt-4o": {"input": 0.0025, "output": 0.01},  # Updated: $2.50/$10.00 per 1M tokens
    "gpt-4o-mini": {"input": 0.00015, "output": 0.0006},  # Official: $0.15/$0.60 per 1M tokens
    
    # GPT-3.5 models
    "gpt-3.5-turbo": {"input": 0.0005, "output": 0.0015},
    "gpt-3.5-turbo-16k": {"input": 0.003, "output": 0.004},
    "gpt-3.5-turbo-instruct": {"input": 0.0015, "output": 0.002},
    
    # Embedding models
    "text-embedding-3-small": {"input": 0.00002, "output": 0.0},
    "text-embedding-3-large": {"input": 0.00013, "output": 0.0},
    "text-embedding-ada-002": {"input": 0.0001, "output": 0.0},
}

# Claude pricing (per 1000 tokens) - Updated January 2025
# Based on official Anthropic pricing as of January 2025
# Source: https://www.anthropic.com/pricing
CLAUDE_PRICING = {
    # Claude 3.5 Sonnet (Latest)
    "claude-3-5-sonnet-20241022": {"input": 0.003, "output": 0.015},  # Official: $3.00/$15.00 per 1M tokens
    "claude-3-5-sonnet-20240620": {"input": 0.003, "output": 0.015},  # Official: $3.00/$15.00 per 1M tokens
    
    # Claude 3.5 Haiku
    "claude-3-5-haiku-20241022": {"input": 0.001, "output": 0.005},  # Official: $1.00/$5.00 per 1M tokens
    
    # Claude 3 Opus
    "claude-3-opus-20240229": {"input": 0.015, "output": 0.075},  # Official: $15.00/$75.00 per 1M tokens
    
    # Claude 3 Sonnet
    "claude-3-sonnet-20240229": {"input": 0.003, "output": 0.015},  # Official: $3.00/$15.00 per 1M tokens
    
    # Claude 3 Haiku
    "claude-3-haiku-20240307": {"input": 0.00025, "output": 0.00125},  # Official: $0.25/$1.25 per 1M tokens
    
    # Claude 2.1
    "claude-2.1": {"input": 0.008, "output": 0.024},  # Official: $8.00/$24.00 per 1M tokens
    
    # Claude 2.0
    "claude-2.0": {"input": 0.008, "output": 0.024},  # Official: $8.00/$24.00 per 1M tokens
    
    # Claude Instant
    "claude-instant-1.2": {"input": 0.0008, "output": 0.0024},  # Official: $0.80/$2.40 per 1M tokens
}

# Combined pricing dictionary for easy lookup
MODEL_PRICING = {**OPENAI_PRICING, **CLAUDE_PRICING}


@dataclass
class APIUsageLog:
    """Data class for API usage tracking."""
    timestamp: str
    method: str  # CoT, PAL, PoT, FPP, etc.
    model: str
    input_tokens: int
    output_tokens: int
    total_tokens: int
    input_cost: float
    output_cost: float
    total_cost: float
    execution_time: float  # in seconds
    question: str
    context: str
    success: bool
    error_message: str = ""
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return asdict(self)


class APITracker:
    """
    API usage tracker for MathCoRL.
    
    Tracks tokens, cost, and execution time for all API calls.
    """
    
    def __init__(self, log_file: Optional[str] = None):
        """
        Initialize the API tracker.
        
        Args:
            log_file: Path to log file (optional, defaults to logs/api_usage.jsonl)
        """
        from .config import LOG_DIR
        
        if log_file is None:
            log_file = LOG_DIR / "api_usage.jsonl"
        
        self.log_file = Path(log_file)
        self.log_file.parent.mkdir(parents=True, exist_ok=True)
        
        logger.info(f"API tracker initialized. Logging to: {self.log_file}")
    
    def calculate_cost(self, model: str, input_tokens: int, output_tokens: int) -> tuple[float, float, float]:
        """
        Calculate cost for API usage for both OpenAI and Claude models.
        
        Args:
            model: Model name
            input_tokens: Number of input tokens
            output_tokens: Number of output tokens
            
        Returns:
            Tuple of (input_cost, output_cost, total_cost)
        """
        # Normalize model name for pricing lookup
        model_key = model.lower()
        
        # Handle OpenAI model name variations
        if "gpt-4.1-nano" in model_key:
            model_key = "gpt-4.1-nano"
        elif "gpt-4.1-mini" in model_key:
            model_key = "gpt-4.1-mini"
        elif "gpt-4.1" in model_key:
            model_key = "gpt-4.1"
        elif "gpt-4o-mini" in model_key:
            model_key = "gpt-4o-mini"
        elif "gpt-4o" in model_key:
            model_key = "gpt-4o"
        elif "gpt-4-turbo" in model_key:
            model_key = "gpt-4-turbo"
        elif "gpt-4" in model_key and "32k" in model_key:
            model_key = "gpt-4-32k"
        elif "gpt-4" in model_key:
            model_key = "gpt-4"
        elif "gpt-3.5-turbo-16k" in model_key:
            model_key = "gpt-3.5-turbo-16k"
        elif "gpt-3.5-turbo-instruct" in model_key:
            model_key = "gpt-3.5-turbo-instruct"
        elif "gpt-3.5-turbo" in model_key:
            model_key = "gpt-3.5-turbo"
        elif "text-embedding-3-small" in model_key:
            model_key = "text-embedding-3-small"
        elif "text-embedding-3-large" in model_key:
            model_key = "text-embedding-3-large"
        elif "text-embedding-ada-002" in model_key:
            model_key = "text-embedding-ada-002"
        # Handle Claude model name variations
        elif "claude-3-5-sonnet-20241022" in model_key:
            model_key = "claude-3-5-sonnet-20241022"
        elif "claude-3-5-sonnet-20240620" in model_key:
            model_key = "claude-3-5-sonnet-20240620"
        elif "claude-3-5-haiku-20241022" in model_key:
            model_key = "claude-3-5-haiku-20241022"
        elif "claude-3-opus-20240229" in model_key:
            model_key = "claude-3-opus-20240229"
        elif "claude-3-sonnet-20240229" in model_key:
            model_key = "claude-3-sonnet-20240229"
        elif "claude-3-haiku-20240307" in model_key:
            model_key = "claude-3-haiku-20240307"
        elif "claude-2.1" in model_key:
            model_key = "claude-2.1"
        elif "claude-2.0" in model_key:
            model_key = "claude-2.0"
        elif "claude-instant-1.2" in model_key:
            model_key = "claude-instant-1.2"
        
        # Get pricing from combined pricing dictionary
        if model_key not in MODEL_PRICING:
            logger.warning(f"Pricing not found for model: {model}. Using gpt-4o-mini pricing as fallback.")
            model_key = "gpt-4o-mini"
        
        pricing = MODEL_PRICING[model_key]
        
        # Calculate costs (pricing is per 1000 tokens)
        input_cost = (input_tokens / 1000) * pricing["input"]
        output_cost = (output_tokens / 1000) * pricing["output"]
        total_cost = input_cost + output_cost
        
        return input_cost, output_cost, total_cost
    
    def log_usage(self, 
                  method: str,
                  model: str,
                  input_tokens: int,
                  output_tokens: int,
                  execution_time: float,
                  question: str,
                  context: str = "",
                  success: bool = True,
                  error_message: str = "") -> APIUsageLog:
        """
        Log API usage data.
        
        Args:
            method: Method used (CoT, PAL, PoT, FPP, etc.)
            model: Model name
            input_tokens: Number of input tokens
            output_tokens: Number of output tokens
            execution_time: Execution time in seconds
            question: Problem question
            context: Problem context
            success: Whether the request was successful
            error_message: Error message if failed
            
        Returns:
            APIUsageLog object
        """
        # Calculate costs
        input_cost, output_cost, total_cost = self.calculate_cost(model, input_tokens, output_tokens)
        
        # Create log entry
        log_entry = APIUsageLog(
            timestamp=datetime.now().isoformat(),
            method=method,
            model=model,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            total_tokens=input_tokens + output_tokens,
            input_cost=input_cost,
            output_cost=output_cost,
            total_cost=total_cost,
            execution_time=execution_time,
            question=question[:200] + "..." if len(question) > 200 else question,  # Truncate long questions
            context=context[:200] + "..." if len(context) > 200 else context,  # Truncate long context
            success=success,
            error_message=error_message
        )
        
        # Write to log file
        try:
            with open(self.log_file, 'a', encoding='utf-8') as f:
                f.write(json.dumps(log_entry.to_dict(), ensure_ascii=False) + '\n')
        except Exception as e:
            logger.error(f"Failed to write to log file: {e}")
        
        # Log summary to console
        logger.info(f"API Call - Method: {method}, Model: {model}, "
                   f"Tokens: {input_tokens}→{output_tokens} ({input_tokens + output_tokens} total), "
                   f"Cost: ${total_cost:.6f}, Time: {execution_time:.2f}s")
        
        return log_entry
    
    def get_usage_summary(self, last_n_hours: int = 24) -> Dict[str, Any]:
        """
        Get usage summary for the last N hours.
        
        Args:
            last_n_hours: Number of hours to look back
            
        Returns:
            Dictionary with usage statistics
        """
        try:
            if not self.log_file.exists():
                return {"error": "No usage data found"}
            
            # Read logs
            logs = []
            cutoff_time = datetime.now().timestamp() - (last_n_hours * 3600)
            
            with open(self.log_file, 'r', encoding='utf-8') as f:
                for line in f:
                    if line.strip():
                        log_data = json.loads(line.strip())
                        log_time = datetime.fromisoformat(log_data['timestamp']).timestamp()
                        if log_time >= cutoff_time:
                            logs.append(log_data)
            
            if not logs:
                return {"message": f"No usage data in the last {last_n_hours} hours"}
            
            # Calculate summary
            total_requests = len(logs)
            successful_requests = sum(1 for log in logs if log['success'])
            total_tokens = sum(log['total_tokens'] for log in logs)
            total_cost = sum(log['total_cost'] for log in logs)
            avg_execution_time = sum(log['execution_time'] for log in logs) / total_requests
            
            # Method breakdown with detailed metrics
            method_stats = {}
            for log in logs:
                method = log['method']
                if method not in method_stats:
                    method_stats[method] = {
                        'requests': 0, 
                        'tokens': 0, 
                        'cost': 0.0,
                        'input_tokens': 0,
                        'output_tokens': 0,
                        'execution_time': 0.0
                    }
                method_stats[method]['requests'] += 1
                method_stats[method]['tokens'] += log['total_tokens']
                method_stats[method]['cost'] += log['total_cost']
                method_stats[method]['input_tokens'] += log['input_tokens']
                method_stats[method]['output_tokens'] += log['output_tokens']
                method_stats[method]['execution_time'] += log['execution_time']
            
            # Calculate averages for each method
            for method, stats in method_stats.items():
                if stats['requests'] > 0:
                    stats['avg_input_tokens'] = stats['input_tokens'] / stats['requests']
                    stats['avg_output_tokens'] = stats['output_tokens'] / stats['requests']
                    stats['avg_time'] = stats['execution_time'] / stats['requests']
                    stats['avg_cost'] = stats['cost'] / stats['requests']
                else:
                    stats['avg_input_tokens'] = 0
                    stats['avg_output_tokens'] = 0
                    stats['avg_time'] = 0
                    stats['avg_cost'] = 0
            
            # Model breakdown
            model_stats = {}
            for log in logs:
                model = log['model']
                if model not in model_stats:
                    model_stats[model] = {'requests': 0, 'tokens': 0, 'cost': 0.0}
                model_stats[model]['requests'] += 1
                model_stats[model]['tokens'] += log['total_tokens']
                model_stats[model]['cost'] += log['total_cost']
            
            return {
                "time_period": f"Last {last_n_hours} hours",
                "summary": {
                    "total_requests": total_requests,
                    "successful_requests": successful_requests,
                    "success_rate": f"{(successful_requests/total_requests)*100:.1f}%",
                    "total_tokens": total_tokens,
                    "total_cost": f"${total_cost:.6f}",
                    "avg_execution_time": f"{avg_execution_time:.2f}s"
                },
                "by_method": method_stats,
                "by_model": model_stats
            }
            
        except Exception as e:
            logger.error(f"Error generating usage summary: {e}")
            return {"error": str(e)}


# Global tracker instance
_global_tracker = None


def get_tracker() -> APITracker:
    """Get the global API tracker instance."""
    global _global_tracker
    if _global_tracker is None:
        _global_tracker = APITracker()
    return _global_tracker


def track_api_call(method: str, model: str, question: str, context: str = ""):
    """
    Decorator/context manager for tracking API calls.
    
    Usage as context manager:
        with track_api_call("CoT", "gpt-4o-mini", question, context) as tracker:
            response = llm.invoke(messages)
            tracker.set_tokens(input_tokens, output_tokens)
    """
    return APICallTracker(method, model, question, context)


class APICallTracker:
    """Context manager for tracking individual API calls."""
    
    def __init__(self, method: str, model: str, question: str, context: str = ""):
        self.method = method
        self.model = model
        self.question = question
        self.context = context
        self.start_time = None
        self.input_tokens = 0
        self.output_tokens = 0
        self.success = True
        self.error_message = ""
        self.tracker = get_tracker()
    
    def __enter__(self):
        self.start_time = time.time()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        execution_time = time.time() - self.start_time
        
        if exc_type is not None:
            self.success = False
            self.error_message = str(exc_val)
        
        # Log the usage
        self.tracker.log_usage(
            method=self.method,
            model=self.model,
            input_tokens=self.input_tokens,
            output_tokens=self.output_tokens,
            execution_time=execution_time,
            question=self.question,
            context=self.context,
            success=self.success,
            error_message=self.error_message
        )
    
    def set_tokens(self, input_tokens: int, output_tokens: int):
        """Set token counts after receiving response."""
        self.input_tokens = input_tokens
        self.output_tokens = output_tokens


def count_tokens_openai(text: str, model: str = "gpt-4o-mini") -> int:
    """
    Estimate token count for OpenAI models.
    
    This is a rough estimation. For exact counts, you'd need tiktoken library.
    """
    # Rough estimation: ~4 characters per token for most models
    return len(text) // 4


def count_tokens_claude(text: str, model: str = "claude-3-5-sonnet-20241022") -> int:
    """
    Estimate token count for Claude models.
    
    This is a rough estimation based on Claude's tokenization.
    Claude uses a similar tokenization to OpenAI models.
    """
    # Rough estimation: ~4 characters per token for most models
    # Claude tends to be slightly more efficient, so we use the same ratio
    return len(text) // 4


def count_tokens_universal(text: str, model: str) -> int:
    """
    Universal token counter that works for both OpenAI and Claude models.
    
    Args:
        text: Text to count tokens for
        model: Model name (determines which counting method to use)
        
    Returns:
        Estimated token count
    """
    model_lower = model.lower()
    
    if "claude" in model_lower:
        return count_tokens_claude(text, model)
    else:
        return count_tokens_openai(text, model)


def extract_tokens_from_response(response) -> tuple[int, int]:
    """
    Extract token counts from LangChain response (supports both OpenAI and Claude).
    
    Args:
        response: LangChain response object
        
    Returns:
        Tuple of (input_tokens, output_tokens)
    """
    try:
        # Check if response has usage information (OpenAI format)
        if hasattr(response, 'response_metadata') and 'token_usage' in response.response_metadata:
            usage = response.response_metadata['token_usage']
            return usage.get('prompt_tokens', 0), usage.get('completion_tokens', 0)
        
        # Check for Claude format (Anthropic)
        if hasattr(response, 'response_metadata') and 'usage' in response.response_metadata:
            usage = response.response_metadata['usage']
            input_tokens = usage.get('input_tokens', 0)
            output_tokens = usage.get('output_tokens', 0)
            return input_tokens, output_tokens
        
        # Fallback: estimate tokens based on content
        if hasattr(response, 'content') and response.content:
            # Try to determine model from response metadata
            model = "unknown"
            if hasattr(response, 'response_metadata'):
                model = response.response_metadata.get('model', 'unknown')
            
            output_tokens = count_tokens_universal(response.content, model)
            return 0, output_tokens  # Can't get input tokens without prompt
        
        return 0, 0
        
    except Exception as e:
        logger.error(f"Error extracting tokens from response: {e}")
        return 0, 0 