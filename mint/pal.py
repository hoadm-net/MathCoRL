"""
PAL (Program-aided Language Models) for Mathematical Problem Solving.

This module implements PAL (Program-aided Language Models), which combines 
natural language reasoning with program synthesis to solve mathematical problems.
PAL first generates reasoning steps, then writes executable code to compute the answer.
"""

import re
import logging
from typing import Dict, Any, Optional, Union
from langchain_openai import ChatOpenAI
from langchain.schema import HumanMessage
from .utils import execute_code

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ProgramAidedLanguageModel:
    """
    PAL (Program-aided Language Models) system for mathematical problem solving.
    
    This class implements PAL, which combines natural language reasoning with 
    program synthesis. It first generates reasoning steps about the problem,
    then writes Python code to solve the computational parts.
    """
    
    def __init__(self, model_name: str = None, temperature: float = None, provider: str = None):
        """
        Initialize the PAL system.
        
        Args:
            model_name: The model to use (defaults to config value based on provider)
            temperature: Temperature for response generation (defaults to config value)
            provider: LLM provider ('openai', 'claude', optional - will use config default)
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
        
        # PAL template that combines reasoning and code
        self.pal_template = """
{context_section}Let's solve this step by step using both reasoning and code.

Problem: {question}

Solution:
Let me think through this problem step by step, then write code to compute the answer.

# Reasoning:
# First, I'll analyze what the problem is asking for and break it down into steps.
# Then I'll write Python code to perform the calculations.

# Code:
```python
# Let me solve this step by step with code

# [Your reasoning and code here]

# Final calculation
answer = # final result
print(f"The answer is: {{answer}}")
```

Please provide both the reasoning steps and the Python code to solve this problem.
"""
    
    def solve(self, question: str, context: str = "", show_reasoning: bool = True) -> Dict[str, Any]:
        """
        Solve a mathematical problem using PAL (Program-aided Language Models).
        
        Args:
            question: The mathematical question to solve
            context: Additional context for the problem
            show_reasoning: Whether to show the reasoning and code generation process
            
        Returns:
            Dictionary containing the result, reasoning, code, and metadata
        """
        try:
            # Prepare context section
            context_section = f"Context: {context}\n\n" if context.strip() else ""
            
            # Create the prompt
            prompt = self.pal_template.format(
                context_section=context_section,
                question=question
            )
            
            if show_reasoning:
                print(f"🧠 PAL Prompt:\n{prompt}\n")
            
            # Get response from LLM with tracking
            from .tracking import track_api_call, extract_tokens_from_response, count_tokens_universal
            
            with track_api_call("PAL", self.model_name, question, context) as tracker:
                messages = [HumanMessage(content=prompt)]
                
                # Estimate input tokens
                input_tokens = count_tokens_universal(prompt, self.model_name)
                
                response = self.llm.invoke(messages)
                full_response = response.content
                
                # Extract token counts from response
                actual_input_tokens, output_tokens = extract_tokens_from_response(response)
                if actual_input_tokens > 0:
                    input_tokens = actual_input_tokens
                
                tracker.set_tokens(input_tokens, output_tokens)
            
            if show_reasoning:
                print(f"🤖 PAL Response:\n{full_response}\n")
            
            # Extract reasoning and code separately
            reasoning = self._extract_reasoning(full_response)
            code = self._extract_code(full_response)
            
            if show_reasoning:
                if reasoning:
                    print(f"💭 Reasoning:\n{reasoning}\n")
                if code:
                    print(f"🐍 Generated Code:\n{code}\n")
            
            # Execute the code
            execution_result = None
            result = None
            
            if code:
                execution_result = self._execute_code_safely(code, show_reasoning)
                result = self._extract_answer_from_execution(execution_result, code)
            
            # If no result from code execution, try to extract from text
            if result is None:
                result = self._extract_answer_from_text(full_response)
            
            if show_reasoning:
                print(f"📊 Final Answer: {result}")
            
            return {
                'result': result,
                'reasoning': reasoning,
                'code': code,
                'full_response': full_response,
                'execution_result': execution_result,
                'question': question,
                'context': context,
                'model': self.model_name,
                'method': 'PAL'
            }
            
        except Exception as e:
            logger.error(f"Error in PAL solving: {e}")
            return {
                'result': None,
                'reasoning': f"Error: {str(e)}",
                'code': None,
                'full_response': None,
                'execution_result': None,
                'question': question,
                'context': context,
                'model': self.model_name,
                'method': 'PAL'
            }
    
    def solve_silent(self, question: str, context: str = "") -> Dict[str, Any]:
        """
        Solve a problem without showing the reasoning process.
        
        Args:
            question: The mathematical question to solve
            context: Additional context for the problem
            
        Returns:
            Dictionary containing the result and other information
        """
        return self.solve(question, context, show_reasoning=False)
    
    def _extract_reasoning(self, full_response: str) -> str:
        """
        Extract the reasoning part from the full response.
        
        Args:
            full_response: The complete response from the model
            
        Returns:
            The reasoning text, or empty string if not found
        """
        # Look for reasoning section
        reasoning_patterns = [
            r"# Reasoning:(.*?)# Code:",
            r"Reasoning:(.*?)Code:",
            r"Let me think(.*?)```python",
            r"Step by step:(.*?)```python",
            r"Analysis:(.*?)```python"
        ]
        
        for pattern in reasoning_patterns:
            match = re.search(pattern, full_response, re.DOTALL | re.IGNORECASE)
            if match:
                reasoning = match.group(1).strip()
                # Clean up the reasoning text
                reasoning = re.sub(r'^#\s*', '', reasoning, flags=re.MULTILINE)
                return reasoning
        
        # If no specific reasoning section found, try to extract text before code
        code_start = re.search(r'```python', full_response, re.IGNORECASE)
        if code_start:
            reasoning = full_response[:code_start.start()].strip()
            return reasoning
        
        # Fallback: return first part of response
        lines = full_response.split('\n')
        reasoning_lines = []
        for line in lines:
            if '```' in line:
                break
            reasoning_lines.append(line)
        
        return '\n'.join(reasoning_lines).strip()
    
    def _extract_code(self, full_response: str) -> str:
        """
        Extract Python code from the response.
        
        Args:
            full_response: The complete response from the model
            
        Returns:
            The extracted Python code, or empty string if not found
        """
        # Look for code blocks marked with ```python
        python_blocks = re.findall(r'```python\s*\n(.*?)\n```', full_response, re.DOTALL)
        if python_blocks:
            return python_blocks[0].strip()
        
        # Look for code blocks marked with just ```
        code_blocks = re.findall(r'```\s*\n(.*?)\n```', full_response, re.DOTALL)
        if code_blocks:
            return code_blocks[0].strip()
        
        # Look for lines that start with common Python patterns after "Code:" or "#Code:"
        code_match = re.search(r'(?:# Code:|Code:)(.*)', full_response, re.DOTALL | re.IGNORECASE)
        if code_match:
            code_section = code_match.group(1).strip()
            # Extract Python-like lines
            lines = code_section.split('\n')
            code_lines = []
            for line in lines:
                if re.match(r'^\s*[#]?.*[=]', line) or re.match(r'^\s*(print|import|def|if|for|while)', line):
                    code_lines.append(line)
                elif line.strip() and not line.strip().startswith('```'):
                    code_lines.append(line)
            
            if code_lines:
                return '\n'.join(code_lines).strip()
        
        return ""
    
    def _execute_code_safely(self, code: str, show_output: bool = True) -> Dict[str, Any]:
        """
        Execute Python code safely and capture the result.
        
        Args:
            code: The Python code to execute
            show_output: Whether to print the execution output
            
        Returns:
            Dictionary containing execution results
        """
        try:
            # execute_code returns (result, error_message) tuple
            result_value, error_message = execute_code(code)
            
            # Capture stdout for output
            import io
            import sys
            from contextlib import redirect_stdout
            
            output = ""
            locals_dict = {}
            
            try:
                # Re-execute to capture stdout and locals
                f = io.StringIO()
                namespace = {}
                
                # Import the execution namespace
                from .functions import get_execution_namespace
                namespace.update(get_execution_namespace())
                
                with redirect_stdout(f):
                    exec(code, namespace, namespace)
                
                output = f.getvalue()
                locals_dict = {k: v for k, v in namespace.items() 
                              if not k.startswith('__') and not callable(v)}
                
            except Exception:
                # If re-execution fails, just use what we have
                pass
            
            execution_result = {
                'output': output,
                'error': error_message,
                'locals': locals_dict,
                'result': result_value
            }
            
            if show_output:
                print(f"🚀 Code Execution:")
                if output:
                    print(output)
                if error_message:
                    print(f"❌ Execution Error: {error_message}")
                print()
            
            return execution_result
            
        except Exception as e:
            error_msg = f"Execution failed: {str(e)}"
            logger.error(error_msg)
            
            if show_output:
                print(f"❌ Execution Error: {error_msg}\n")
            
            return {
                'output': '',
                'error': error_msg,
                'locals': {},
                'result': None
            }
    
    def _extract_answer_from_execution(self, execution_result: Dict[str, Any], code: str) -> Optional[Union[int, float]]:
        """
        Extract the numerical answer from code execution results.
        
        Args:
            execution_result: Results from code execution
            code: The original code that was executed
            
        Returns:
            The extracted numerical answer, or None if not found
        """
        # First try to get direct 'result' from execution
        if 'result' in execution_result and execution_result['result'] is not None:
            try:
                return float(execution_result['result'])
            except (ValueError, TypeError):
                pass
        
        # Then try to get 'answer' variable from execution locals
        if 'locals' in execution_result and execution_result['locals']:
            locals_dict = execution_result['locals']
            
            # Look for 'answer' variable first (PAL commonly uses this)
            if 'answer' in locals_dict:
                try:
                    return float(locals_dict['answer'])
                except (ValueError, TypeError):
                    pass
            
            # Look for other potential answer variables
            for var_name in ['result', 'final_answer', 'solution', 'total', 'ans']:
                if var_name in locals_dict:
                    try:
                        return float(locals_dict[var_name])
                    except (ValueError, TypeError):
                        continue
        
        # Try to extract from output text
        if 'output' in execution_result and execution_result['output']:
            output = execution_result['output']
            
            # Look for "The answer is: X" pattern (common in PAL)
            answer_match = re.search(r'the answer is:?\s*([+-]?\d*\.?\d+)', output, re.IGNORECASE)
            if answer_match:
                try:
                    return float(answer_match.group(1))
                except ValueError:
                    pass
            
            # Look for other patterns
            patterns = [
                r'answer:?\s*([+-]?\d*\.?\d+)',
                r'result:?\s*([+-]?\d*\.?\d+)',
                r'final answer:?\s*([+-]?\d*\.?\d+)'
            ]
            
            for pattern in patterns:
                match = re.search(pattern, output, re.IGNORECASE)
                if match:
                    try:
                        return float(match.group(1))
                    except ValueError:
                        pass
            
            # Look for last number in the output
            numbers = re.findall(r'([+-]?\d*\.?\d+)', output)
            if numbers:
                try:
                    return float(numbers[-1])
                except ValueError:
                    pass
        
        logger.warning(f"Could not extract numerical answer from execution result: {execution_result}")
        return None
    
    def _extract_answer_from_text(self, text: str) -> Optional[float]:
        """
        Extract numerical answer from text when code execution fails.
        
        Args:
            text: The text to extract answer from
            
        Returns:
            The extracted numerical answer, or None if not found
        """
        # Common patterns for final answers
        answer_patterns = [
            r"(?:final answer|answer|result|solution)(?:\s*is)?\s*:?\s*([+-]?\d*\.?\d+)",
            r"([+-]?\d*\.?\d+)\s*(?:is the answer|is the result|is the solution)",
            r"(?:the answer is|answer is)\s*([+-]?\d*\.?\d+)",
            r"(?:equals?|=)\s*([+-]?\d*\.?\d+)",
            r"therefore,?\s*([+-]?\d*\.?\d+)",
            r"so,?\s*([+-]?\d*\.?\d+)"
        ]
        
        # Try to find answer patterns (case insensitive)
        text_lower = text.lower()
        
        for pattern in answer_patterns:
            matches = re.findall(pattern, text_lower, re.MULTILINE | re.IGNORECASE)
            if matches:
                try:
                    # Take the last match (most likely to be the final answer)
                    return float(matches[-1])
                except ValueError:
                    continue
        
        # Last resort: find the last number in the text
        numbers = re.findall(r'([+-]?\d+\.?\d*)', text)
        if numbers:
            try:
                return float(numbers[-1])
            except ValueError:
                pass
        
        logger.warning(f"Could not extract numerical answer from text: {text[:200]}...")
        return None
    
    def _setup_langsmith(self):
        """Setup LangSmith tracing if configured."""
        import os
        if (os.getenv("LANGCHAIN_TRACING_V2", "false").lower() == "true" and 
            os.getenv("LANGCHAIN_API_KEY")):
            os.environ["LANGCHAIN_TRACING_V2"] = "true"
            os.environ["LANGCHAIN_ENDPOINT"] = os.getenv("LANGCHAIN_ENDPOINT", "https://api.smith.langchain.com")
            os.environ["LANGCHAIN_PROJECT"] = os.getenv("LANGCHAIN_PROJECT", "MathCoRL-PAL")
            logger.info(f"LangSmith tracing enabled for project: {os.getenv('LANGCHAIN_PROJECT')}")
        else:
            logger.info("LangSmith tracing disabled")


def solve_with_pal(question: str, context: str = "", model_name: str = None) -> Optional[float]:
    """
    Convenience function to solve a problem using PAL (Program-aided Language Models).
    
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
        
    pal = ProgramAidedLanguageModel(model_name=model_name)
    result = pal.solve_silent(question, context)
    return result.get('result') 