"""
Zero-Shot Prompting for Mathematical Problem Solving.

This module provides zero-shot prompting capabilities for solving mathematical problems.
Zero-shot prompting asks the model to solve problems directly without examples or 
step-by-step reasoning guidance.
"""

import re
import logging
from typing import Dict, Any, Optional, Union
from langchain_openai import ChatOpenAI
from langchain.schema import HumanMessage

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ZeroShotPrompting:
    """
    Zero-Shot Prompting system for mathematical problem solving.
    
    This class implements zero-shot prompting, which directly asks the language model
    to solve mathematical problems without providing examples or reasoning templates.
    """
    
    def __init__(self, model_name: str = None, temperature: float = None):
        """
        Initialize the Zero-Shot prompting system.
        
        Args:
            model_name: The OpenAI model to use (defaults to config value)
            temperature: Temperature for response generation (defaults to config value)
        """
        from .config import load_config
        config = load_config()
        
        self.model_name = model_name or config['model']
        self.temperature = temperature if temperature is not None else config['temperature']
        
        # Setup LangSmith if configured
        self._setup_langsmith()
        
        self.llm = ChatOpenAI(
            model_name=self.model_name,
            temperature=self.temperature
        )
        
        # Simple zero-shot template
        self.zero_shot_template = """
{context_section}Solve this mathematical problem and provide the numerical answer:

{question}

Answer:"""
    
    def solve(self, question: str, context: str = "", show_reasoning: bool = True) -> Dict[str, Any]:
        """
        Solve a mathematical problem using Zero-Shot prompting.
        
        Args:
            question: The mathematical question to solve
            context: Additional context for the problem
            show_reasoning: Whether to show the reasoning process
            
        Returns:
            Dictionary containing the result, reasoning, and metadata
        """
        try:
            # Prepare context section
            context_section = f"Context: {context}\n\n" if context.strip() else ""
            
            # Create the prompt
            prompt = self.zero_shot_template.format(
                context_section=context_section,
                question=question
            )
            
            if show_reasoning:
                print(f"🎯 Zero-Shot Prompt:\n{prompt}\n")
            
            # Get response from LLM with tracking
            from .tracking import track_api_call, extract_tokens_from_response, count_tokens_openai
            
            with track_api_call("Zero-Shot", self.model_name, question, context) as tracker:
                messages = [HumanMessage(content=prompt)]
                
                # Estimate input tokens
                input_tokens = count_tokens_openai(prompt, self.model_name)
                
                response = self.llm.invoke(messages)
                reasoning = response.content
                
                # Extract token counts from response
                actual_input_tokens, output_tokens = extract_tokens_from_response(response)
                if actual_input_tokens > 0:
                    input_tokens = actual_input_tokens
                
                tracker.set_tokens(input_tokens, output_tokens)
            
            if show_reasoning:
                print(f"🤖 Zero-Shot Response:\n{reasoning}\n")
            
            # Extract the final numerical answer
            result = self._extract_answer(reasoning)
            
            if show_reasoning:
                print(f"📊 Final Answer: {result}")
            
            return {
                'result': result,
                'reasoning': reasoning,
                'question': question,
                'context': context,
                'model': self.model_name,
                'method': 'Zero-Shot'
            }
            
        except Exception as e:
            logger.error(f"Error in Zero-Shot solving: {e}")
            return {
                'result': None,
                'reasoning': f"Error: {str(e)}",
                'question': question,
                'context': context,
                'model': self.model_name,
                'method': 'Zero-Shot'
            }
    
    def solve_silent(self, question: str, context: str = "") -> Dict[str, Any]:
        """
        Solve a problem without showing the reasoning process.
        
        Args:
            question: The mathematical question to solve
            context: Additional context for the problem
            
        Returns:
            Dictionary containing the result and reasoning
        """
        return self.solve(question, context, show_reasoning=False)
    
    def _extract_answer(self, reasoning: str) -> Optional[float]:
        """
        Extract the numerical answer from the reasoning text.
        
        Args:
            reasoning: The full reasoning text from the model
            
        Returns:
            The extracted numerical answer, or None if not found
        """
        # Common patterns for answers in zero-shot responses
        answer_patterns = [
            r"(?:answer|result|solution)(?:\s*is)?\s*:?\s*([+-]?\d*\.?\d+)",
            r"([+-]?\d*\.?\d+)\s*(?:is the answer|is the result|is the solution)",
            r"(?:the answer is|answer is)\s*([+-]?\d*\.?\d+)",
            r"(?:equals?|=)\s*([+-]?\d*\.?\d+)",
            r"([+-]?\d*\.?\d+)\s*$",  # Number at end of response
            r"^([+-]?\d*\.?\d+)",     # Number at start of response
            r"(\d+)\s*\+\s*(\d+)\s*=\s*(\d+)",  # For simple arithmetic like "15 + 27 = 42"
            r"final(?:\s+answer)?\s*:?\s*([+-]?\d*\.?\d+)",
            r"(?:therefore|so|thus|hence)\s*,?\s*([+-]?\d*\.?\d+)",
        ]
        
        # Try to find answer patterns (case insensitive)
        reasoning_lower = reasoning.lower()
        
        # Special handling for simple arithmetic patterns
        arithmetic_match = re.search(r'(\d+)\s*\+\s*(\d+)\s*=\s*(\d+)', reasoning)
        if arithmetic_match:
            return float(arithmetic_match.group(3))
        
        for pattern in answer_patterns:
            matches = re.findall(pattern, reasoning_lower, re.MULTILINE | re.IGNORECASE)
            if matches:
                try:
                    # For tuple matches (like arithmetic), take the last number
                    if isinstance(matches[0], tuple):
                        return float(matches[-1][-1])
                    else:
                        # Take the last match (most likely to be the final answer)
                        return float(matches[-1])
                except ValueError:
                    continue
        
        # If no pattern matches, try to find the last number in the response
        numbers = re.findall(r'([+-]?\d+\.?\d*)', reasoning)
        if numbers:
            try:
                # Take the last number found
                return float(numbers[-1])
            except ValueError:
                pass
        
        logger.warning(f"Could not extract numerical answer from reasoning: {reasoning[:200]}...")
        return None
    
    def _setup_langsmith(self):
        """Setup LangSmith tracing if configured."""
        import os
        if (os.getenv("LANGCHAIN_TRACING_V2", "false").lower() == "true" and 
            os.getenv("LANGCHAIN_API_KEY")):
            os.environ["LANGCHAIN_TRACING_V2"] = "true"
            os.environ["LANGCHAIN_ENDPOINT"] = os.getenv("LANGCHAIN_ENDPOINT", "https://api.smith.langchain.com")
            os.environ["LANGCHAIN_PROJECT"] = os.getenv("LANGCHAIN_PROJECT", "MathCoRL-ZeroShot")
            logger.info(f"LangSmith tracing enabled for project: {os.getenv('LANGCHAIN_PROJECT')}")
        else:
            logger.info("LangSmith tracing disabled")


def solve_with_zero_shot(question: str, context: str = "", model_name: str = None) -> Optional[float]:
    """
    Convenience function to solve a problem using Zero-Shot prompting.
    
    Args:
        question: The mathematical question to solve
        context: Additional context for the problem
        model_name: The OpenAI model to use (defaults to config value)
        
    Returns:
        The numerical answer, or None if the problem couldn't be solved
    """
    from .config import load_config
    if model_name is None:
        config = load_config()
        model_name = config['model']
        
    zs = ZeroShotPrompting(model_name=model_name)
    result = zs.solve_silent(question, context)
    return result.get('result') 