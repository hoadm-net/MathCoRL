"""
HuggingFace Provider for Local Model Inference

Supports local models like:
- DeepSeek-R1-Distill-Qwen-7B
- Qwen2.5-Math-7B-Instruct
- Qwen2.5-Math-72B-Instruct

Features:
- GPU acceleration with CUDA
- 8-bit quantization for memory efficiency
- Backward compatible - doesn't modify existing OpenAI/Claude code
"""

import logging
import torch
from typing import Optional, Dict, Any, List
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    GenerationConfig
)

logger = logging.getLogger(__name__)


class HuggingFaceProvider:
    """
    Provider for HuggingFace models with local inference.
    
    Designed to be backward compatible - does NOT modify existing
    OpenAI/Claude infrastructure. Creates parallel implementation.
    
    Example:
        >>> provider = HuggingFaceProvider(
        ...     model_name="deepseek-ai/DeepSeek-R1-Distill-Qwen-7B",
        ...     device="cuda",
        ...     load_in_8bit=True
        ... )
        >>> response = provider.generate("What is 15 + 27?")
        >>> print(response)
    """
    
    def __init__(
        self,
        model_name: str,
        device: str = "cuda",
        load_in_8bit: bool = True,
        max_memory: Optional[Dict[int, str]] = None,
        temperature: float = 0.0,
        max_new_tokens: int = 1000,
        cache_dir: Optional[str] = None
    ):
        """
        Initialize HuggingFace model provider.
        
        Args:
            model_name: HuggingFace model identifier (e.g., "deepseek-ai/DeepSeek-R1-Distill-Qwen-7B")
            device: Device to use ("cuda", "cpu", or specific like "cuda:0")
            load_in_8bit: Whether to use 8-bit quantization (saves memory)
            max_memory: Maximum memory per GPU (e.g., {0: "20GB"})
            temperature: Sampling temperature (0.0 = deterministic)
            max_new_tokens: Maximum tokens to generate
            cache_dir: Directory to cache downloaded models
        """
        self.model_name = model_name
        self.device = device if torch.cuda.is_available() else "cpu"
        self.temperature = temperature
        self.max_new_tokens = max_new_tokens
        
        logger.info(f"Initializing HuggingFace provider: {model_name}")
        logger.info(f"Device: {self.device}, 8-bit: {load_in_8bit}")
        
        # Configure quantization for memory efficiency
        quantization_config = None
        device_map_value = None
        
        if load_in_8bit and self.device != "cpu":
            quantization_config = BitsAndBytesConfig(
                load_in_8bit=True,
                llm_int8_threshold=6.0,
                llm_int8_has_fp16_weight=False
            )
            # IMPORTANT: Do NOT use device_map with 8-bit quantization
            # The quantization process automatically places the model on GPU
            device_map_value = None
            logger.info("Using 8-bit quantization (model will auto-place on GPU)")
        elif self.device == "cuda":
            # Use device_map for non-quantized CUDA models
            device_map_value = "auto"
            logger.info("Using device_map='auto' for GPU placement")
        else:
            # CPU mode - no device_map
            device_map_value = None
        
        # Load tokenizer
        logger.info("Loading tokenizer...")
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_name,
            trust_remote_code=True,
            cache_dir=cache_dir
        )
        
        # Set pad token if not exists
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        
        # Load model
        logger.info("Loading model (this may take a few minutes)...")
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            quantization_config=quantization_config,
            device_map=device_map_value,
            trust_remote_code=True,
            torch_dtype=torch.float16 if self.device != "cpu" else torch.float32,
            max_memory=max_memory,
            cache_dir=cache_dir
        )
        
        # Note: Do NOT call .to() when using quantization or device_map="auto"
        # The model is already on the correct device
        
        self.model.eval()
        
        # Configure generation
        self.generation_config = GenerationConfig(
            max_new_tokens=max_new_tokens,
            temperature=temperature if temperature > 0 else 0.001,  # Avoid exact 0
            do_sample=temperature > 0,
            top_p=0.95 if temperature > 0 else 1.0,
            pad_token_id=self.tokenizer.pad_token_id,
            eos_token_id=self.tokenizer.eos_token_id
        )
        
        logger.info(f"Model loaded successfully on {self.device}")
        
        # Get model info
        if torch.cuda.is_available() and self.device == "cuda":
            memory_allocated = torch.cuda.memory_allocated() / 1024**3
            logger.info(f"GPU memory allocated: {memory_allocated:.2f} GB")
    
    def generate(
        self,
        prompt: str,
        temperature: Optional[float] = None,
        max_new_tokens: Optional[int] = None,
        return_full_text: bool = False
    ) -> str:
        """
        Generate text from prompt using the loaded model.
        
        Args:
            prompt: Input text prompt
            temperature: Override default temperature
            max_new_tokens: Override default max tokens
            return_full_text: If True, return prompt + generated text; else only generated
            
        Returns:
            Generated text string
        """
        # Prepare generation config
        gen_config = GenerationConfig(
            max_new_tokens=max_new_tokens or self.max_new_tokens,
            temperature=temperature or self.temperature if (temperature or self.temperature) > 0 else 0.001,
            do_sample=(temperature or self.temperature) > 0,
            top_p=0.95 if (temperature or self.temperature) > 0 else 1.0,
            pad_token_id=self.tokenizer.pad_token_id,
            eos_token_id=self.tokenizer.eos_token_id
        )
        
        # Tokenize input
        inputs = self.tokenizer(
            prompt,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=2048
        )
        
        # Move to device
        inputs = {k: v.to(self.model.device) for k, v in inputs.items()}
        
        # Generate
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                generation_config=gen_config
            )
        
        # Decode
        generated_text = self.tokenizer.decode(
            outputs[0],
            skip_special_tokens=True
        )
        
        # Remove prompt if requested
        if not return_full_text and generated_text.startswith(prompt):
            generated_text = generated_text[len(prompt):].strip()
        
        return generated_text
    
    def count_tokens(self, text: str) -> int:
        """
        Count tokens in text using model's tokenizer.
        
        Args:
            text: Input text
            
        Returns:
            Number of tokens
        """
        tokens = self.tokenizer.encode(text, add_special_tokens=False)
        return len(tokens)
    
    def get_model_info(self) -> Dict[str, Any]:
        """
        Get information about the loaded model.
        
        Returns:
            Dictionary with model information
        """
        info = {
            "model_name": self.model_name,
            "device": str(self.model.device),
            "dtype": str(self.model.dtype),
            "temperature": self.temperature,
            "max_new_tokens": self.max_new_tokens
        }
        
        if torch.cuda.is_available() and self.device == "cuda":
            info["gpu_memory_allocated_gb"] = torch.cuda.memory_allocated() / 1024**3
            info["gpu_memory_reserved_gb"] = torch.cuda.memory_reserved() / 1024**3
        
        return info
    
    def __repr__(self) -> str:
        """String representation."""
        return f"HuggingFaceProvider(model={self.model_name}, device={self.device})"


class DeepSeekR1Provider(HuggingFaceProvider):
    """
    Specialized provider for DeepSeek-R1 models.
    
    Pre-configured for optimal DeepSeek-R1 inference with:
    - Recommended temperature and sampling settings
    - Appropriate prompt formatting
    - Optimized generation parameters
    
    Example:
        >>> provider = DeepSeekR1Provider(model_variant="7B")
        >>> response = provider.solve_math_problem("What is 15 + 27?")
    """
    
    # Model variants
    MODELS = {
        "7B": "deepseek-ai/DeepSeek-R1-Distill-Qwen-7B",
        "1.5B": "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B",
        "14B": "deepseek-ai/DeepSeek-R1-Distill-Llama-14B",
        "8B": "deepseek-ai/DeepSeek-R1-Distill-Llama-8B"
    }
    
    def __init__(
        self,
        model_variant: str = "7B",
        device: str = "cuda",
        load_in_8bit: bool = True,
        temperature: float = 0.0,
        max_new_tokens: int = 1000,
        cache_dir: Optional[str] = None
    ):
        """
        Initialize DeepSeek-R1 provider.
        
        Args:
            model_variant: Model size ("7B", "1.5B", "14B", "8B")
            device: Device to use ("cuda" or "cpu")
            load_in_8bit: Use 8-bit quantization
            temperature: Sampling temperature
            max_new_tokens: Maximum tokens to generate
            cache_dir: Model cache directory
        """
        if model_variant not in self.MODELS:
            raise ValueError(
                f"Invalid model variant: {model_variant}. "
                f"Choose from: {list(self.MODELS.keys())}"
            )
        
        model_name = self.MODELS[model_variant]
        self.variant = model_variant
        
        super().__init__(
            model_name=model_name,
            device=device,
            load_in_8bit=load_in_8bit,
            temperature=temperature,
            max_new_tokens=max_new_tokens,
            cache_dir=cache_dir
        )
    
    def solve_math_problem(
        self,
        question: str,
        context: str = "",
        method: str = "fpp"
    ) -> str:
        """
        Solve mathematical problem using DeepSeek-R1.
        
        Args:
            question: Mathematical question
            context: Additional context (optional)
            method: Prompting method ("fpp", "cot", "pot", "pal")
            
        Returns:
            Model response with solution
        """
        # Format prompt based on method
        if method == "fpp":
            prompt = self._format_fpp_prompt(question, context)
        elif method == "cot":
            prompt = self._format_cot_prompt(question, context)
        elif method == "pot":
            prompt = self._format_pot_prompt(question, context)
        elif method == "pal":
            prompt = self._format_pal_prompt(question, context)
        else:
            # Default: simple prompt
            prompt = f"{context}\n\n{question}" if context else question
        
        return self.generate(prompt)
    
    def _format_fpp_prompt(self, question: str, context: str = "") -> str:
        """Format prompt for FPP method."""
        context_section = f"Context: {context}\n\n" if context else ""
        
        prompt = f"""{context_section}Solve this mathematical problem step by step. Write Python code using the provided functions.

Problem: {question}

Solution (write Python code):
```python
def solve():
    # Your solution here
    pass
```"""
        return prompt
    
    def _format_cot_prompt(self, question: str, context: str = "") -> str:
        """Format prompt for CoT method."""
        context_section = f"Context: {context}\n\n" if context else ""
        
        prompt = f"""{context_section}Solve this mathematical problem step by step, showing your reasoning.

Problem: {question}

Solution:
Let me think through this step by step:

Step 1:"""
        return prompt
    
    def _format_pot_prompt(self, question: str, context: str = "") -> str:
        """Format prompt for PoT method."""
        context_section = f"Context: {context}\n\n" if context else ""
        
        prompt = f"""{context_section}Solve this mathematical problem by writing a Python program.

Problem: {question}

Program:
```python
# Mathematical solution
"""
        return prompt
    
    def _format_pal_prompt(self, question: str, context: str = "") -> str:
        """Format prompt for PAL method."""
        context_section = f"Context: {context}\n\n" if context else ""
        
        prompt = f"""{context_section}Solve this problem using program-aided reasoning.

Problem: {question}

Solution:
# Let me break this down and write code to solve it
```python
"""
        return prompt
    
    def __repr__(self) -> str:
        """String representation."""
        return f"DeepSeekR1Provider(variant={self.variant}, device={self.device})"


class QwenMathProvider(HuggingFaceProvider):
    """
    Specialized provider for Qwen2.5-Math models.
    
    Pre-configured for optimal Qwen2.5-Math inference with:
    - Recommended temperature and sampling settings
    - Appropriate prompt formatting for mathematical reasoning
    - Optimized generation parameters
    
    Qwen2.5-Math models are specifically trained for mathematical problem solving
    and provide strong performance on mathematical reasoning tasks.
    
    Example:
        >>> provider = QwenMathProvider(model_variant="7B")
        >>> response = provider.solve_math_problem("What is 15 + 27?")
    """
    
    # Model variants
    MODELS = {
        "7B": "Qwen/Qwen2.5-Math-7B-Instruct",
        "72B": "Qwen/Qwen2.5-Math-72B-Instruct",
        "1.5B": "Qwen/Qwen2.5-Math-1.5B-Instruct",  # If available
    }
    
    def __init__(
        self,
        model_variant: str = "7B",
        device: str = "cuda",
        load_in_8bit: bool = True,
        temperature: float = 0.0,
        max_new_tokens: int = 1000,
        cache_dir: Optional[str] = None
    ):
        """
        Initialize Qwen2.5-Math provider.
        
        Args:
            model_variant: Model size ("7B", "72B", "1.5B")
            device: Device to use ("cuda" or "cpu")
            load_in_8bit: Use 8-bit quantization (recommended for 7B+)
            temperature: Sampling temperature (0.0 = deterministic)
            max_new_tokens: Maximum tokens to generate
            cache_dir: Model cache directory
        """
        if model_variant not in self.MODELS:
            raise ValueError(
                f"Invalid model variant: {model_variant}. "
                f"Choose from: {list(self.MODELS.keys())}"
            )
        
        model_name = self.MODELS[model_variant]
        self.variant = model_variant
        
        # Adjust quantization based on model size
        if model_variant == "72B":
            logger.warning(
                "Qwen2.5-Math-72B requires significant GPU memory (40GB+). "
                "Ensure you have adequate resources or consider using 7B variant."
            )
        
        super().__init__(
            model_name=model_name,
            device=device,
            load_in_8bit=load_in_8bit,
            temperature=temperature,
            max_new_tokens=max_new_tokens,
            cache_dir=cache_dir
        )
    
    def solve_math_problem(
        self,
        question: str,
        context: str = "",
        method: str = "fpp"
    ) -> str:
        """
        Solve mathematical problem using Qwen2.5-Math.
        
        Qwen2.5-Math models are trained specifically for mathematical reasoning,
        so they work well with structured prompting approaches.
        
        Args:
            question: Mathematical question
            context: Additional context (optional)
            method: Prompting method ("fpp", "cot", "pot", "pal")
            
        Returns:
            Model response with solution
        """
        # Format prompt based on method
        if method == "fpp":
            prompt = self._format_fpp_prompt(question, context)
        elif method == "cot":
            prompt = self._format_cot_prompt(question, context)
        elif method == "pot":
            prompt = self._format_pot_prompt(question, context)
        elif method == "pal":
            prompt = self._format_pal_prompt(question, context)
        else:
            # Default: simple prompt
            prompt = f"{context}\n\n{question}" if context else question
        
        return self.generate(prompt)
    
    def _format_fpp_prompt(self, question: str, context: str = "") -> str:
        """
        Format prompt for FPP (Function Prototype Prompting) method.
        
        Qwen2.5-Math benefits from clear structure and step-by-step guidance.
        """
        context_section = f"Context: {context}\n\n" if context else ""
        
        prompt = f"""{context_section}Solve this mathematical problem step by step. You can write Python code to help solve it.

Problem: {question}

Solution:
Let me solve this step by step.

Step 1: Understand the problem
Step 2: Set up the solution approach
Step 3: Calculate the answer

```python
def solve():
    # Solution code
    pass
```"""
        return prompt
    
    def _format_cot_prompt(self, question: str, context: str = "") -> str:
        """
        Format prompt for CoT (Chain-of-Thought) method.
        
        Qwen2.5-Math excels at step-by-step reasoning.
        """
        context_section = f"Context: {context}\n\n" if context else ""
        
        prompt = f"""{context_section}Solve this mathematical problem step by step, showing your reasoning clearly.

Problem: {question}

Solution:
Let me think through this carefully:

Step 1: What information do we have?
Step 2: What do we need to find?
Step 3: How can we solve this?
Step 4: Let's calculate:"""
        return prompt
    
    def _format_pot_prompt(self, question: str, context: str = "") -> str:
        """
        Format prompt for PoT (Program of Thoughts) method.
        
        Structured algorithmic approach works well with Qwen2.5-Math.
        """
        context_section = f"Context: {context}\n\n" if context else ""
        
        prompt = f"""{context_section}Solve this mathematical problem by writing a clear Python program.

Problem: {question}

Solution (Python program):
```python
# Step-by-step solution

# 1. Parse the problem

# 2. Set up variables

# 3. Perform calculations

# 4. Return answer
answer = None  # Calculate here
print(f"Answer: {{answer}}")
```"""
        return prompt
    
    def _format_pal_prompt(self, question: str, context: str = "") -> str:
        """
        Format prompt for PAL (Program-aided Language) method.
        
        Combines natural language reasoning with code execution.
        """
        context_section = f"Context: {context}\n\n" if context else ""
        
        prompt = f"""{context_section}Solve this problem using both reasoning and code.

Problem: {question}

Solution:
# First, let me understand what we need to find:
# [Your analysis here]

# Now let's write code to solve it:
```python
# Mathematical solution
"""
        return prompt
    
    def __repr__(self) -> str:
        """String representation."""
        return f"QwenMathProvider(variant={self.variant}, device={self.device})"
