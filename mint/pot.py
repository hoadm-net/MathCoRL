"""
Program of Thoughts (PoT) Prompting for Mathematical Problem Solving

This module implements Program of Thoughts prompting, which disentangles computation 
from reasoning by having the model generate Python code to solve numerical problems.

Based on the paper "Program of Thoughts Prompting: Disentangling Computation from 
Reasoning for Numerical Reasoning Tasks".
"""

import re
import ast
import logging
import traceback
from typing import Dict, Any, Optional, Union
from langchain_openai import ChatOpenAI
from langchain.prompts import PromptTemplate
from langchain.schema import HumanMessage

from .utils import execute_code

logger = logging.getLogger(__name__)


class ProgramOfThoughtsPrompting:
    """
    Program of Thoughts prompting implementation for mathematical problem solving.
    
    PoT prompting asks the model to generate Python code that solves the problem,
    then executes the code to get the numerical result.
    """
    
    def __init__(self, model_name: str = None, temperature: float = None):
        """
        Initialize the Program of Thoughts prompting system.
        
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
        
        # PoT prompt template with few-shot examples
        self.pot_template = PromptTemplate(
            input_variables=["context", "question"],
            template="""You are an expert Python programmer and mathematician. Generate Python code to solve the following mathematical problem.

Write clear, executable Python code that:
1. Reads the problem carefully
2. Implements the solution step-by-step
3. Prints intermediate steps for clarity
4. Returns the final numerical answer

Here are some examples:

Example 1:
Problem: Janet's ducks lay 16 eggs per day. She eats 3 for breakfast and bakes muffins with 4. She sells the remainder at $2 per egg. How much does she make daily?

Code:
```python
# Janet's daily egg problem
eggs_per_day = 16
eggs_eaten = 3
eggs_for_muffins = 4
price_per_egg = 2

print(f"Total eggs per day: {{eggs_per_day}}")
print(f"Eggs eaten: {{eggs_eaten}}")
print(f"Eggs used for muffins: {{eggs_for_muffins}}")

# Calculate eggs left to sell
eggs_to_sell = eggs_per_day - eggs_eaten - eggs_for_muffins
print(f"Eggs left to sell: {{eggs_to_sell}}")

# Calculate daily earnings
daily_earnings = eggs_to_sell * price_per_egg
print(f"Daily earnings: ${{daily_earnings}}")

answer = daily_earnings
print(f"Final answer: {{answer}}")
```

Example 2:
Problem: A school has 569 girls and 236 boys. How many more girls than boys does the school have?

Code:
```python
# School population problem
girls = 569
boys = 236

print(f"Number of girls: {{girls}}")
print(f"Number of boys: {{boys}}")

# Calculate the difference
difference = girls - boys
print(f"Difference: {{difference}}")

answer = difference
print(f"Final answer: {{answer}}")
```

Example 3:
Problem: John has 20 apples. He gives 1/4 of them to his friend and eats 3. How many apples does John have left?

Code:
```python
# John's apple problem
import math

initial_apples = 20
fraction_given = 1/4
apples_eaten = 3

print(f"Initial apples: {{initial_apples}}")

# Calculate apples given to friend
apples_given = initial_apples * fraction_given
print(f"Apples given to friend: {{apples_given}}")

# Calculate remaining apples
apples_left = initial_apples - apples_given - apples_eaten
print(f"Apples left: {{apples_left}}")

answer = apples_left
print(f"Final answer: {{answer}}")
```

Now solve this problem:

{context_section}

Problem: {question}

Generate Python code to solve this step by step. Make sure to:
- Use clear variable names
- Add print statements to show your work
- Store the final numerical answer in a variable called 'answer'
- Use proper mathematical operations

Code:
```python
# Your solution here
"""
        )
    
    def solve(self, question: str, context: str = "", show_reasoning: bool = True) -> Dict[str, Any]:
        """
        Solve a mathematical problem using Program of Thoughts prompting.
        
        Args:
            question: The mathematical question to solve
            context: Additional context for the problem
            show_reasoning: Whether to show the generated code and execution
            
        Returns:
            Dictionary containing the result, code, execution output, and metadata
        """
        try:
            # Prepare context section
            context_section = f"Context: {context}\n" if context.strip() else ""
            
            # Create the prompt
            prompt = self.pot_template.format(
                context_section=context_section,
                question=question
            )
            
            if show_reasoning:
                print(f"🤖 PoT Prompt:\n{prompt}\n")
            
            # Get response from LLM with tracking
            from .tracking import track_api_call, extract_tokens_from_response, count_tokens_openai
            
            with track_api_call("PoT", self.model_name, question, context) as tracker:
                messages = [HumanMessage(content=prompt)]
                
                # Estimate input tokens
                input_tokens = count_tokens_openai(prompt, self.model_name)
                
                response = self.llm.invoke(messages)
                generated_code = response.content
                
                # Extract token counts from response
                actual_input_tokens, output_tokens = extract_tokens_from_response(response)
                if actual_input_tokens > 0:
                    input_tokens = actual_input_tokens
                
                tracker.set_tokens(input_tokens, output_tokens)
            
            if show_reasoning:
                print(f"💻 Generated Code:\n{generated_code}\n")
            
            # Extract and clean the Python code
            code = self._extract_code(generated_code)
            
            if show_reasoning:
                print(f"🔧 Cleaned Code:\n{code}\n")
            
            # Execute the code
            execution_result = self._execute_code_safely(code, show_reasoning)
            
            # Extract the final answer
            result = self._extract_answer_from_execution(execution_result, code)
            
            if show_reasoning:
                print(f"📊 Final Answer: {result}")
            
            return {
                'result': result,
                'code': code,
                'generated_response': generated_code,
                'execution_output': execution_result.get('output', ''),
                'execution_error': execution_result.get('error', ''),
                'question': question,
                'context': context,
                'model': self.model_name,
                'method': 'Program-of-Thoughts'
            }
            
        except Exception as e:
            logger.error(f"Error in PoT solving: {e}")
            return {
                'result': None,
                'code': '',
                'generated_response': '',
                'execution_output': '',
                'execution_error': f"Error: {str(e)}",
                'question': question,
                'context': context,
                'model': self.model_name,
                'method': 'Program-of-Thoughts'
            }
    
    def solve_silent(self, question: str, context: str = "") -> Dict[str, Any]:
        """
        Solve a problem without showing the code generation and execution process.
        
        Args:
            question: The mathematical question to solve
            context: Additional context for the problem
            
        Returns:
            Dictionary containing the result and code
        """
        return self.solve(question, context, show_reasoning=False)
    
    def _extract_code(self, generated_text: str) -> str:
        """
        Extract Python code from the generated response.
        
        Args:
            generated_text: The full generated response from the LLM
            
        Returns:
            The extracted Python code
        """
        # Look for code blocks marked with ```python
        python_blocks = re.findall(r'```python\s*\n(.*?)\n```', generated_text, re.DOTALL)
        if python_blocks:
            return python_blocks[0].strip()
        
        # Look for code blocks marked with ```
        code_blocks = re.findall(r'```\s*\n(.*?)\n```', generated_text, re.DOTALL)
        if code_blocks:
            return code_blocks[0].strip()
        
        # Look for lines that start with common Python patterns
        lines = generated_text.split('\n')
        code_lines = []
        in_code_section = False
        
        for line in lines:
            # Start collecting code when we see Python-like patterns
            if any(pattern in line for pattern in ['=', 'print(', 'import ', 'def ', 'if ', 'for ', 'while ']):
                in_code_section = True
            
            if in_code_section:
                # Skip empty lines and comments that don't look like code
                if line.strip() and not line.strip().startswith('#'):
                    code_lines.append(line)
                elif line.strip() == '':
                    code_lines.append(line)
        
        if code_lines:
            return '\n'.join(code_lines)
        
        # Fallback: return the whole response
        return generated_text.strip()
    
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
            
            # Look for 'answer' variable
            if 'answer' in locals_dict:
                try:
                    return float(locals_dict['answer'])
                except (ValueError, TypeError):
                    pass
            
            # Look for other potential answer variables
            for var_name in ['result', 'final_answer', 'solution', 'total', 'answer']:
                if var_name in locals_dict:
                    try:
                        return float(locals_dict[var_name])
                    except (ValueError, TypeError):
                        continue
        
        # Try to extract from output text
        if 'output' in execution_result and execution_result['output']:
            output = execution_result['output']
            
            # Look for "Final answer: X" pattern
            final_answer_match = re.search(r'final answer:?\s*([+-]?\d*\.?\d+)', output, re.IGNORECASE)
            if final_answer_match:
                try:
                    return float(final_answer_match.group(1))
                except ValueError:
                    pass
            
            # Look for "Answer: X" pattern
            answer_match = re.search(r'answer:?\s*([+-]?\d*\.?\d+)', output, re.IGNORECASE)
            if answer_match:
                try:
                    return float(answer_match.group(1))
                except ValueError:
                    pass
            
            # Look for last number in the output
            numbers = re.findall(r'([+-]?\d*\.?\d+)', output)
            if numbers:
                try:
                    return float(numbers[-1])
                except ValueError:
                    pass
        
        # Try to extract from the code itself (static analysis)
        # Look for lines like "answer = ..."
        answer_pattern = re.search(r'answer\s*=\s*([^#\n]+)', code)
        if answer_pattern:
            try:
                # Try to evaluate the expression
                expr = answer_pattern.group(1).strip()
                # Simple numeric expressions only
                if re.match(r'^[+-]?\d*\.?\d+$', expr):
                    return float(expr)
            except:
                pass
        
        logger.warning(f"Could not extract numerical answer from execution result: {execution_result}")
        return None
    
    def _setup_langsmith(self):
        """Setup LangSmith tracing if configured."""
        import os
        if (os.getenv("LANGCHAIN_TRACING_V2", "false").lower() == "true" and 
            os.getenv("LANGCHAIN_API_KEY")):
            os.environ["LANGCHAIN_TRACING_V2"] = "true"
            os.environ["LANGCHAIN_ENDPOINT"] = os.getenv("LANGCHAIN_ENDPOINT", "https://api.smith.langchain.com")
            os.environ["LANGCHAIN_PROJECT"] = os.getenv("LANGCHAIN_PROJECT", "MathCoRL-PoT")
            logger.info(f"LangSmith tracing enabled for project: {os.getenv('LANGCHAIN_PROJECT')}")
        else:
            logger.info("LangSmith tracing disabled")


def solve_with_pot(question: str, context: str = "", model_name: str = None) -> Optional[float]:
    """
    Convenience function to solve a problem using Program of Thoughts prompting.
    
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
        
    pot = ProgramOfThoughtsPrompting(model_name=model_name)
    result = pot.solve_silent(question, context)
    return result.get('result') 