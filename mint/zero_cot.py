"""
Zero-Shot Chain-of-Thought (Zero-CoT) Prompting

Implementation of "Large Language Models are Zero-Shot Reasoners" (Kojima et al., 2022)
This approach uses simple "Let's think step by step" prompt without any examples.
"""

import re
import logging
from typing import Dict, Any, Optional
from langchain_openai import ChatOpenAI
from langchain_core.prompts import PromptTemplate
from langchain_core.messages import HumanMessage

logger = logging.getLogger(__name__)


class ZeroShotCoTPrompting:
    """
    Zero-Shot Chain-of-Thought prompting implementation.
    
    Uses the simple "Let's think step by step" approach without requiring
    any few-shot examples, making it more efficient and generalizable.
    """
    
    def __init__(self, model_name: str = None, temperature: float = None, provider: str = None):
        """
        Initialize the Zero-Shot CoT prompting system.
        
        Args:
            model_name: The model to use (defaults to config value)
            temperature: Temperature for response generation (defaults to config value)
            provider: LLM provider ('openai', 'claude', optional)
        """
        from .config import load_config, create_llm_client, get_current_model_name
        config = load_config()
        
        self.provider = provider or config['provider']
        self.model_name = model_name or get_current_model_name(self.provider)
        self.temperature = temperature if temperature is not None else config['temperature']
        
        # Setup LangSmith if configured
        self._setup_langsmith()
        
        self.llm = create_llm_client(
            provider=self.provider,
            model=self.model_name,
            temperature=self.temperature
        )
        
        # Zero-Shot CoT prompt template
        self.zero_cot_template = PromptTemplate(
            input_variables=["context", "question"],
            template="""You are an expert mathematician. 

{context_section}

Problem: {question}

Let's think step by step."""
        )
    
    def _setup_langsmith(self):
        """Setup LangSmith tracing if configured."""
        import os
        if (os.getenv("LANGCHAIN_TRACING_V2", "false").lower() == "true" and 
            os.getenv("LANGCHAIN_API_KEY")):
            os.environ["LANGCHAIN_TRACING_V2"] = "true"
            os.environ["LANGCHAIN_ENDPOINT"] = os.getenv("LANGCHAIN_ENDPOINT", "https://api.smith.langchain.com")
            os.environ["LANGCHAIN_PROJECT"] = os.getenv("LANGCHAIN_PROJECT", "MathCoRL-FPP")
            logger.info(f"LangSmith tracing enabled for project: {os.getenv('LANGCHAIN_PROJECT')}")
        else:
            logger.info("LangSmith tracing disabled")
    
    def solve(self, question: str, context: str = "", show_reasoning: bool = True) -> Dict[str, Any]:
        """
        Solve a mathematical problem using Zero-Shot CoT prompting.
        
        Args:
            question: The mathematical question to solve
            context: Additional context for the problem
            show_reasoning: Whether to show the reasoning steps
            
        Returns:
            Dictionary containing the result, reasoning, and metadata
        """
        try:
            # Prepare context section
            context_section = f"Context: {context}\n" if context.strip() else ""
            
            # Create the prompt
            prompt = self.zero_cot_template.format(
                context_section=context_section,
                question=question
            )
            
            if show_reasoning:
                print(f"🤖 Zero-CoT Prompt:\n{prompt}\n")
            
            # Get response from LLM with tracking
            from .tracking import track_api_call, extract_tokens_from_response, count_tokens_universal
            
            with track_api_call("Zero-CoT", self.model_name, question, context) as tracker:
                messages = [HumanMessage(content=prompt)]
                
                # Estimate input tokens
                input_tokens = count_tokens_universal(prompt, self.model_name)
                
                response = self.llm.invoke(messages)
                reasoning = response.content
                
                # Extract token counts from response
                actual_input_tokens, output_tokens = extract_tokens_from_response(response)
                if actual_input_tokens > 0:
                    input_tokens = actual_input_tokens
                
                tracker.set_tokens(input_tokens, output_tokens)
            
            # Extract the final answer
            final_answer = self._extract_answer(reasoning)
            
            result = {
                'question': question,
                'context': context,
                'reasoning': reasoning,
                'answer': final_answer,
                'method': 'Zero-CoT',
                'model': self.model_name,
                'success': True
            }
            
            if show_reasoning:
                print(f"🧠 Zero-CoT Reasoning:\n{reasoning}\n")
                print(f"📊 Final Answer: {final_answer}")
            
            return result
            
        except Exception as e:
            logger.error(f"Error in Zero-CoT solve: {e}")
            return {
                'question': question,
                'context': context,
                'reasoning': '',
                'answer': None,
                'method': 'Zero-CoT',
                'model': self.model_name,
                'success': False,
                'error': str(e)
            }
    
    def solve_problem(self, problem: Dict, context: str = "") -> Any:
        """
        Solve a problem dictionary (e.g., from datasets).
        
        Args:
            problem: Problem dictionary with 'Body' and 'Question' keys
            context: Optional context information
            
        Returns:
            Numerical result or None if solving failed
        """
        try:
            if isinstance(problem, dict):
                # Handle different dataset formats
                if 'Question' in problem:
                    question = problem['Question']
                    if 'Body' in problem:
                        problem_context = problem['Body']
                    else:
                        problem_context = context
                elif 'question' in problem:
                    question = problem['question']
                    problem_context = context
                else:
                    question = str(problem)
                    problem_context = context
            else:
                question = str(problem)
                problem_context = context
            
            result = self.solve(question, problem_context, show_reasoning=False)
            return result['answer']
            
        except Exception as e:
            logger.error(f"Error in solve_problem: {e}")
            return None
    
    def _extract_answer(self, reasoning: str) -> Optional[float]:
        """
        Extract the final numerical answer from the reasoning text.
        
        Args:
            reasoning: The step-by-step reasoning text
            
        Returns:
            The extracted numerical answer or None if not found
        """
        try:
            # Common patterns for final answers - improved patterns
            patterns = [
                r"(?:final answer|answer|result).*?(?:is|:|=)\s*\*\*([+-]?\d*\.?\d+)\*\*",  # **30**
                r"(?:final answer|answer|result).*?(?:is|:|=)\s*([+-]?\d*\.?\d+)",
                r"therefore.*?([+-]?\d*\.?\d+)",
                r"so.*?([+-]?\d*\.?\d+)",
                r"thus.*?([+-]?\d*\.?\d+)",
                r"=\s*([+-]?\d*\.?\d+)(?:\s|$)",
                r"(?:is|equals?)\s*\*\*([+-]?\d*\.?\d+)\*\*",  # is **30**
                r"(?:is|equals?)\s*([+-]?\d*\.?\d+)(?:\*\*)?\.?\s*$"  # is 30. or is 30**
            ]
            
            reasoning_lower = reasoning.lower()
            
            for pattern in patterns:
                matches = re.findall(pattern, reasoning_lower, re.IGNORECASE)
                if matches:
                    try:
                        # Get the last match (most likely to be the final answer)
                        answer = float(matches[-1])
                        return answer
                    except ValueError:
                        continue
            
            # If no pattern matches, try to find numbers at the end
            lines = reasoning.strip().split('\n')
            for line in reversed(lines):
                # Look for numbers in bold or at end of sentence
                bold_numbers = re.findall(r'\*\*([+-]?\d*\.?\d+)\*\*', line)
                if bold_numbers:
                    try:
                        return float(bold_numbers[-1])
                    except ValueError:
                        continue
                        
                # Look for any number at end of line
                numbers = re.findall(r'([+-]?\d*\.?\d+)(?:\*\*)?\.?\s*$', line)
                if numbers:
                    try:
                        return float(numbers[-1])
                    except ValueError:
                        continue
            
            logger.warning("Could not extract numerical answer from reasoning")
            return None
            
        except Exception as e:
            logger.error(f"Error extracting answer: {e}")
            return None


def solve_math_problem_zero_cot(question: str, context: str = "", provider: str = None, **kwargs) -> Optional[float]:
    """
    Simple function to solve a math problem using Zero-Shot CoT.
    
    Args:
        question: Mathematical question to solve
        context: Optional context information
        provider: LLM provider ('openai', 'claude', optional)
        **kwargs: Additional arguments for ZeroShotCoTPrompting initialization
        
    Returns:
        Numerical result or None if solving failed
    """
    try:
        zero_cot = ZeroShotCoTPrompting(provider=provider, **kwargs)
        result = zero_cot.solve(question, context, show_reasoning=False)
        return result['answer']
    except Exception as e:
        logger.error(f"Error in solve_math_problem_zero_cot: {e}")
        return None