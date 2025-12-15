"""
Core Function Prototype Prompting implementation.

This module contains the main FPP class for mathematical problem solving.
"""

import os
import logging
from typing import Optional, Union, Dict, Any
from dotenv import load_dotenv

from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage

from .prompts import create_fpp_prompt, create_problem_prompt
from .utils import clean_code, execute_code, evaluate_result

# Load environment variables
load_dotenv()

# Configure logging
logger = logging.getLogger(__name__)


class FunctionPrototypePrompting:
    """
    Function Prototype Prompting (FPP) for mathematical problem solving.
    
    Simple interface:
    - Input: question (str), context (str, optional)
    - Output: result (numerical answer)
    
    Supports ablation variants:
    - Policy types: retrieval, random
    - Manual prompts (no policy)
    - Custom function prototypes
    - Unconstrained function usage
    """
    
    def __init__(self, 
                 api_key: Optional[str] = None, 
                 model: Optional[str] = None,
                 temperature: float = 0.0,
                 max_tokens: int = 1000,
                 provider: Optional[str] = None):
        """Initialize Function Prototype Prompting solver.
        
        Supports dual LLM providers (OpenAI, Claude) with automatic configuration
        from environment variables or YAML config.
        
        Args:
            api_key: API key override. Defaults to OPENAI_API_KEY or ANTHROPIC_API_KEY env var.
            model: Model name override. Defaults to config value based on provider.
            temperature: Sampling temperature for response generation. Default: 0.0 (deterministic).
            max_tokens: Maximum tokens in LLM response. Default: 1000.
            provider: LLM provider selection ('openai' or 'claude'). Defaults to LLM_PROVIDER env var or 'openai'.
            
        Example:
            >>> fpp = FunctionPrototypePrompting(provider="claude")
            >>> result = fpp.solve("What is 15 + 27?")
            >>> print(result)  # 42
        """
        from .config import load_config, create_llm_client, get_current_model_name
        
        # Load configuration
        config = load_config()
        
        # Determine provider
        self.provider = provider or config['provider']
        
        # Setup model parameters
        self.model = model or get_current_model_name(self.provider)
        self.temperature = temperature or config['temperature']
        self.max_tokens = max_tokens or config['max_tokens']
        
        # Initialize LangChain LLM client
        self.llm = create_llm_client(
            provider=self.provider,
            model=self.model,
            api_key=api_key,
            temperature=self.temperature,
            max_tokens=self.max_tokens
        )
        
        # Ablation variant attributes
        self._custom_prototypes = None
        self._variant_type = "baseline"
        self._use_policy = True
        self._policy_type = "retrieval"
        self._manual_prompt = False
        self._unconstrained = False
        
        # Setup LangSmith if configured
        self._setup_langsmith()
        
        logger.info(f"FPP initialized with {self.provider} provider, model: {self.model}")
    
    def _setup_langsmith(self):
        """Setup LangSmith tracing if configured."""
        if (os.getenv("LANGCHAIN_TRACING_V2", "false").lower() == "true" and 
            os.getenv("LANGCHAIN_API_KEY")):
            os.environ["LANGCHAIN_TRACING_V2"] = "true"
            os.environ["LANGCHAIN_ENDPOINT"] = os.getenv("LANGCHAIN_ENDPOINT", "https://api.smith.langchain.com")
            os.environ["LANGCHAIN_PROJECT"] = os.getenv("LANGCHAIN_PROJECT", "MathCoRL-FPP")
            logger.info(f"LangSmith tracing enabled for project: {os.getenv('LANGCHAIN_PROJECT')}")
        else:
            logger.info("LangSmith tracing disabled")
    
    def solve(self, question: str, context: str = "") -> Union[float, int, None]:
        """Solve mathematical question using structured code generation.
        
        Generates Python code with predefined function prototypes, executes it,
        and returns the numerical answer.
        
        Args:
            question: Natural language math question.
            context: Additional context (tables, financial data, etc.). Optional.
            
        Returns:
            Numerical answer (int/float) or None if execution failed.
            
        Example:
            >>> fpp = FunctionPrototypePrompting()
            >>> result = fpp.solve("John has 20 apples. He gives 8 away. How many left?")
            >>> print(result)  # 12
        """
        try:
            # Create prompt based on variant type
            if self._manual_prompt or not self._use_policy:
                prompt = self._create_manual_prompt(question, context)
            elif self._custom_prototypes:
                prompt = self._create_custom_prompt(question, context)
            else:
                prompt = create_fpp_prompt(question, context)
            
            # Generate code using LLM
            raw_code = self._call_llm(prompt, question, context)
            if not raw_code:
                logger.error("No response received from LLM")
                return None

            # Clean and execute code
            cleaned_code = clean_code(raw_code)
            result, error = execute_code(cleaned_code)
            
            if error:
                logger.error(f"Code execution error: {error}")
                return None

            return result
            
        except Exception as e:
            logger.error(f"Error in solve(): {e}")
            return None
    
    def solve_problem(self, problem: Dict, context: str = "") -> Union[float, int, None]:
        """
        Solve a problem dictionary (e.g., from SVAMP dataset).
        
        Args:
            problem: Problem dictionary with 'Body' and 'Question' keys
            context: Optional context information
            
        Returns:
            Numerical result or None if solving failed
        """
        try:
            # Create prompt from problem dictionary
            prompt = create_problem_prompt(problem, context)
            
            # Generate code using LLM
            question_text = problem.get('Question', '') if isinstance(problem, dict) else str(problem)
            raw_code = self._call_llm(prompt, question_text, context)
            if not raw_code:
                logger.error("No response received from LLM")
                return None
            
            # Clean and execute code
            cleaned_code = clean_code(raw_code)
            result, error = execute_code(cleaned_code)
            
            if error:
                logger.error(f"Code execution error: {error}")
                return None
            
            return result
            
        except Exception as e:
            logger.error(f"Error in solve_problem(): {e}")
            return None
    
    def solve_detailed(self, question: str, context: str = "") -> Dict[str, Any]:
        """
        Solve with detailed output including generated code and execution info.
        
        Args:
            question: Mathematical question to solve
            context: Optional context information
            
        Returns:
            Dictionary with detailed results
        """
        result = {
            'question': question,
            'context': context,
            'result': None,
            'code': '',
            'error': '',
            'success': False,
            'variant_type': getattr(self, '_variant_type', 'baseline'),
            'policy_type': getattr(self, '_policy_type', 'retrieval'),
            'use_policy': getattr(self, '_use_policy', True)
        }
        
        try:
            # Create prompt based on variant type
            if self._manual_prompt or not self._use_policy:
                prompt = self._create_manual_prompt(question, context)
            elif self._custom_prototypes:
                prompt = self._create_custom_prompt(question, context)
            else:
                prompt = create_fpp_prompt(question, context)
            
            # Generate code using LLM
            raw_code = self._call_llm(prompt, question, context)
            if not raw_code:
                result['error'] = "No response received from LLM"
                return result
            
            # Clean code
            cleaned_code = clean_code(raw_code)
            result['code'] = cleaned_code
            
            # Execute code
            exec_result, error = execute_code(cleaned_code)
            
            if error:
                result['error'] = error
            else:
                result['result'] = exec_result
                result['success'] = True
            
        except Exception as e:
            result['error'] = str(e)
        
        return result
    
    def _create_custom_prompt(self, question: str, context: str = "") -> str:
        """Create custom prompt with specific function prototypes for ablation variants."""
        if not self._custom_prototypes:
            return create_fpp_prompt(question, context)
        
        # Use custom prototypes instead of standard ones
        base_prompt = f"""You are a mathematical problem solver. Use the provided function prototypes to solve the problem.

Available Function Prototypes:
{self._custom_prototypes}

Context: {context}
Question: {question}

Please write Python code using ONLY the functions defined above to solve this problem. 
The code should end with a statement that prints or returns the final numerical answer.

Code:
```python"""
        
        return base_prompt
    
    def _create_manual_prompt(self, question: str, context: str = "") -> str:
        """Create manual prompt without policy-based function selection."""
        manual_prompt = f"""You are a mathematical problem solver. Solve the following problem step by step using basic Python operations.

Context: {context}
Question: {question}

Please write Python code to solve this problem. You can use standard Python operations and basic mathematical functions.
The code should end with a statement that prints or returns the final numerical answer.

Code:
```python"""
        
        return manual_prompt
    
    def _apply_policy_constraints(self, prompt: str) -> str:
        """Apply policy-based constraints to function selection."""
        if not self._use_policy:
            return prompt
        
        if self._policy_type == "random":
            # Add instruction for random function selection
            constraint = "\nNote: Use a random selection of available functions for this problem.\n"
            return prompt + constraint
        elif self._policy_type == "retrieval":
            # Add instruction for retrieval-based function selection
            constraint = "\nNote: Use the most relevant functions based on the problem type.\n"
            return prompt + constraint
        
        return prompt
    
    def _call_llm(self, prompt: str, question: str = "", context: str = "") -> str:
        """
        Call LLM API to generate code with tracking (supports OpenAI and Claude).
        
        Args:
            prompt: Formatted prompt string
            question: Original question for tracking
            context: Original context for tracking
            
        Returns:
            Generated code or empty string if failed
        """
        from .tracking import track_api_call, extract_tokens_from_response, count_tokens_universal
        
        try:
            with track_api_call("FPP", self.model, question, context) as tracker:
                messages = [
                    SystemMessage(content="You are an expert Python programmer specialized in mathematical problem solving."),
                    HumanMessage(content=prompt)
                ]
                
                # Estimate input tokens using universal counter
                system_content = "You are an expert Python programmer specialized in mathematical problem solving."
                input_tokens = count_tokens_universal(system_content + prompt, self.model)
                
                response = self.llm.invoke(messages)
                
                # Extract token counts from response
                actual_input_tokens, output_tokens = extract_tokens_from_response(response)
                if actual_input_tokens > 0:
                    input_tokens = actual_input_tokens
                
                tracker.set_tokens(input_tokens, output_tokens)
                
                return response.content.strip()
            
        except Exception as e:
            logger.error(f"Error calling LLM API: {e}")
            return ""


# Simple function interface for quick usage
def solve_math_problem(question: str, context: str = "", provider: str = None, **kwargs) -> Union[float, int, None]:
    """
    Simple function to solve a math problem using FPP.
    
    Args:
        question: Mathematical question to solve
        context: Optional context information
        provider: LLM provider ('openai', 'claude', optional - will use config default)
        **kwargs: Additional arguments for FPP initialization
        
    Returns:
        Numerical result or None if solving failed
    """
    try:
        fpp = FunctionPrototypePrompting(provider=provider, **kwargs)
        return fpp.solve(question, context)
    except Exception as e:
        logger.error(f"Error in solve_math_problem(): {e}")
        return None 