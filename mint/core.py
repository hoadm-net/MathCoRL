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
    """
    
    def __init__(self, 
                 api_key: Optional[str] = None, 
                 model: Optional[str] = None,
                 temperature: float = 0.0,
                 max_tokens: int = 1000):
        """
        Initialize FPP with LangChain OpenAI.
        
        Args:
            api_key: OpenAI API key (optional, will use environment variable)
            model: Model name (default: gpt-3.5-turbo)
            temperature: Sampling temperature (default: 0.0)
            max_tokens: Maximum tokens in response (default: 1000)
        """
        # Setup API key
        self.api_key = api_key or os.getenv("OPENAI_API_KEY")
        if not self.api_key:
            raise ValueError("OpenAI API key not found. Set OPENAI_API_KEY environment variable or pass api_key parameter.")
        
        # Setup model parameters
        self.model = model or os.getenv("DEFAULT_MODEL", "gpt-3.5-turbo")
        self.temperature = temperature or float(os.getenv("TEMPERATURE", "0.0"))
        self.max_tokens = max_tokens or int(os.getenv("MAX_TOKENS", "1000"))
        
        # Initialize LangChain ChatOpenAI
        self.llm = ChatOpenAI(
            api_key=self.api_key,
            model=self.model,
            temperature=self.temperature,
            max_tokens=self.max_tokens
        )
        
        # Setup LangSmith if configured
        self._setup_langsmith()
        
        logger.info(f"FPP initialized with model: {self.model}")
    
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
        """
        Solve a mathematical question using Function Prototype Prompting.
        
        Args:
            question: Mathematical question to solve
            context: Optional context information
            
        Returns:
            Numerical result or None if solving failed
        """
        try:
            # Create prompt
            prompt = create_fpp_prompt(question, context)
            
            # Generate code using LLM
            raw_code = self._call_llm(prompt)
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
            raw_code = self._call_llm(prompt)
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
            'success': False
        }
        
        try:
            # Create prompt
            prompt = create_fpp_prompt(question, context)
            
            # Generate code using LLM
            raw_code = self._call_llm(prompt)
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
    
    def _call_llm(self, prompt: str) -> str:
        """
        Call LLM API to generate code.
        
        Args:
            prompt: Formatted prompt string
            
        Returns:
            Generated code or empty string if failed
        """
        try:
            messages = [
                SystemMessage(content="You are an expert Python programmer specialized in mathematical problem solving."),
                HumanMessage(content=prompt)
            ]
            
            response = self.llm.invoke(messages)
            return response.content.strip()
            
        except Exception as e:
            logger.error(f"Error calling LLM API: {e}")
            return ""


# Simple function interface for quick usage
def solve_math_problem(question: str, context: str = "", **kwargs) -> Union[float, int, None]:
    """
    Simple function to solve a math problem using FPP.
    
    Args:
        question: Mathematical question to solve
        context: Optional context information
        **kwargs: Additional arguments for FPP initialization
        
    Returns:
        Numerical result or None if solving failed
    """
    try:
        fpp = FunctionPrototypePrompting(**kwargs)
        return fpp.solve(question, context)
    except Exception as e:
        logger.error(f"Error in solve_math_problem(): {e}")
        return None 