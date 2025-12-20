"""
MathCoRL Model Providers

This package contains provider implementations for different LLM backends:
- OpenAI (via LangChain)
- Claude (via LangChain)
- HuggingFace (local models like DeepSeek-R1, Qwen2.5-Math)
"""

from .huggingface_provider import (
    HuggingFaceProvider,
    DeepSeekR1Provider,
    QwenMathProvider
)

__all__ = [
    'HuggingFaceProvider',
    'DeepSeekR1Provider',
    'QwenMathProvider'
]
