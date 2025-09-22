"""
CoMAT: Chain of Mathematically Annotated Thought Prompting

Implementation based on "CoMAT: Chain of Mathematically Annotated Thought Improves Mathematical Reasoning" (Leang et al., 2024)
This approach enhances reasoning through two stages: Symbolic Conversion and Reasoning Execution.
"""

import re
import json
import logging
from typing import Dict, Any, List, Optional, Tuple
from langchain_openai import ChatOpenAI
from langchain.prompts import PromptTemplate
from langchain.schema import HumanMessage

logger = logging.getLogger(__name__)


class CoMATPrompting:
    """
    Chain of Mathematically Annotated Thought (CoMAT) prompting implementation.
    
    CoMAT enhances mathematical reasoning through two-stage process:
    1. Symbolic Conversion: Convert natural language to symbolic form
    2. Reasoning Execution: Derive answers from symbolic representations
    """
    
    def __init__(self, model_name: str = None, temperature: float = None, provider: str = None):
        """
        Initialize the CoMAT prompting system.
        
        Args:
            model_name: The model to use
            temperature: Temperature for response generation
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
        
        # Stage 1: Symbolic Conversion Template
        self.symbolic_conversion_template = PromptTemplate(
            input_variables=["context", "question"],
            template="""You are a mathematical expert. Your task is to convert a natural language mathematical problem into symbolic mathematical form.

{context_section}

Problem: {question}

Please convert this problem into symbolic mathematical notation by following these steps:

1. Identify all mathematical quantities and variables
2. Express relationships using mathematical symbols and equations
3. Define any variables or parameters clearly
4. Present the problem in a structured symbolic form

Symbolic Conversion:

Variables:
- [Define all variables and their meanings]

Given Information:
- [List all given values and constraints in symbolic form]

Mathematical Relationships:
- [Express relationships as equations or inequalities]

Target:
- [State what needs to be found in symbolic form]

Symbolic Form: [Present the complete symbolic representation]"""
        )
        
        # Stage 2: Reasoning Execution Template
        self.reasoning_execution_template = PromptTemplate(
            input_variables=["question", "symbolic_form", "context"],
            template="""You are a mathematical expert. Given the symbolic representation of a mathematical problem, solve it step by step using mathematical reasoning.

Original Problem: {question}
{context_section}

Symbolic Representation:
{symbolic_form}

Now solve this problem step by step using the symbolic form:

Mathematical Reasoning:

Step 1: [Start with the symbolic form and identify the solution approach]

Step 2: [Apply mathematical operations or formulas]

Step 3: [Continue the logical progression]

[Continue with additional steps as needed]

Final Calculation:
[Show the final computation leading to the answer]

Answer: [Provide the numerical result]

Verification:
[Verify the answer makes sense in the original context]"""
        )
    
    def _setup_langsmith(self):
        """Setup LangSmith tracing if configured."""
        import os
        if (os.getenv("LANGCHAIN_TRACING_V2", "false").lower() == "true" and 
            os.getenv("LANGCHAIN_API_KEY")):
            os.environ["LANGCHAIN_TRACING_V2"] = "true"
            os.environ["LANGCHAIN_ENDPOINT"] = os.getenv("LANGCHAIN_ENDPOINT", "https://api.smith.langchain.com")
            os.environ["LANGCHAIN_PROJECT"] = os.getenv("LANGCHAIN_PROJECT", "MathCoRL-CoMAT")
            logger.info(f"LangSmith tracing enabled for project: {os.getenv('LANGCHAIN_PROJECT')}")
        else:
            logger.info("LangSmith tracing disabled")
    
    def symbolic_conversion(self, question: str, context: str = "") -> str:
        """
        Stage 1: Convert natural language problem to symbolic form.
        
        Args:
            question: The mathematical question
            context: Additional context
            
        Returns:
            Symbolic representation of the problem
        """
        try:
            context_section = f"Context: {context}\n" if context.strip() else ""
            
            prompt = self.symbolic_conversion_template.format(
                context_section=context_section,
                question=question
            )
            
            from .tracking import track_api_call, extract_tokens_from_response, count_tokens_universal
            
            with track_api_call("CoMAT-Symbolic", self.model_name, question, context) as tracker:
                messages = [HumanMessage(content=prompt)]
                
                input_tokens = count_tokens_universal(prompt, self.model_name)
                response = self.llm.invoke(messages)
                
                actual_input_tokens, output_tokens = extract_tokens_from_response(response)
                if actual_input_tokens > 0:
                    input_tokens = actual_input_tokens
                
                tracker.set_tokens(input_tokens, output_tokens)
            
            return response.content.strip()
            
        except Exception as e:
            logger.error(f"Error in symbolic conversion: {e}")
            return f"Unable to convert to symbolic form: {str(e)}"
    
    def reasoning_execution(self, question: str, symbolic_form: str, context: str = "") -> str:
        """
        Stage 2: Execute mathematical reasoning from symbolic form.
        
        Args:
            question: The original question
            symbolic_form: Symbolic representation from Stage 1
            context: Additional context
            
        Returns:
            Step-by-step mathematical reasoning and solution
        """
        try:
            context_section = f"Context: {context}\n" if context.strip() else ""
            
            prompt = self.reasoning_execution_template.format(
                question=question,
                symbolic_form=symbolic_form,
                context_section=context_section
            )
            
            from .tracking import track_api_call, extract_tokens_from_response, count_tokens_universal
            
            with track_api_call("CoMAT-Reasoning", self.model_name, question, context) as tracker:
                messages = [HumanMessage(content=prompt)]
                
                input_tokens = count_tokens_universal(prompt, self.model_name)
                response = self.llm.invoke(messages)
                
                actual_input_tokens, output_tokens = extract_tokens_from_response(response)
                if actual_input_tokens > 0:
                    input_tokens = actual_input_tokens
                
                tracker.set_tokens(input_tokens, output_tokens)
            
            return response.content.strip()
            
        except Exception as e:
            logger.error(f"Error in reasoning execution: {e}")
            return f"Unable to execute reasoning: {str(e)}"
    
    def solve(self, question: str, context: str = "", show_reasoning: bool = True) -> Dict[str, Any]:
        """
        Solve a mathematical problem using CoMAT prompting.
        
        Args:
            question: The mathematical question to solve
            context: Additional context for the problem
            show_reasoning: Whether to show the reasoning steps
            
        Returns:
            Dictionary containing the result, reasoning, and metadata
        """
        try:
            if show_reasoning:
                print(f"🔬 Starting CoMAT analysis for: {question}\n")
            
            # Stage 1: Symbolic Conversion
            symbolic_form = self.symbolic_conversion(question, context)
            if show_reasoning:
                print(f"📊 Symbolic Conversion:\n{symbolic_form}\n")
            
            # Stage 2: Reasoning Execution
            reasoning = self.reasoning_execution(question, symbolic_form, context)
            if show_reasoning:
                print(f"🧠 Mathematical Reasoning:\n{reasoning}\n")
            
            # Extract the final answer
            final_answer = self._extract_answer(reasoning)
            
            # Combine reasoning
            full_reasoning = f"Symbolic Conversion:\n{symbolic_form}\n\nReasoning Execution:\n{reasoning}"
            
            result = {
                'question': question,
                'context': context,
                'reasoning': full_reasoning,
                'symbolic_conversion': symbolic_form,
                'reasoning_execution': reasoning,
                'answer': final_answer,
                'method': 'CoMAT',
                'model': self.model_name,
                'success': True
            }
            
            if show_reasoning:
                print(f"📊 Final Answer: {final_answer}")
            
            return result
            
        except Exception as e:
            logger.error(f"Error in CoMAT solve: {e}")
            return {
                'question': question,
                'context': context,
                'reasoning': '',
                'answer': None,
                'method': 'CoMAT',
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
            # Enhanced patterns for mathematical reasoning
            patterns = [
                r"(?:final answer|answer|result).*?(?:is|:|=)\s*\*\*([+-]?\d*\.?\d+)\*\*",  # **30**
                r"(?:final answer|answer|result).*?(?:is|:|=)\s*([+-]?\d*\.?\d+)",
                r"answer:\s*([+-]?\d*\.?\d+)",  # Answer: 42
                r"john has\s*\*\*([+-]?\d*\.?\d+)\s*apples\*\*",  # John has **17 apples**
                r"has\s*\*\*([+-]?\d*\.?\d+)\s*apples\*\*",  # has **17 apples**
                r"therefore.*?([+-]?\d*\.?\d+)",
                r"so.*?([+-]?\d*\.?\d+)",
                r"thus.*?([+-]?\d*\.?\d+)",
                r"=\s*([+-]?\d*\.?\d+)(?:\s|$)",
                r"(?:is|equals?)\s*\*\*([+-]?\d*\.?\d+)\*\*",  # is **30**
                r"(?:is|equals?)\s*([+-]?\d*\.?\d+)(?:\*\*)?\.?\s*$",  # is 30. or is 30**
                r"final calculation.*?([+-]?\d*\.?\d+)",  # Final Calculation: 42
                r"verification.*?([+-]?\d*\.?\d+)"  # Verification: 42
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
                        
                # Look for answer patterns in the line
                answer_patterns = [
                    r'answer.*?([+-]?\d*\.?\d+)',
                    r'result.*?([+-]?\d*\.?\d+)',
                    r'=\s*([+-]?\d*\.?\d+)'
                ]
                
                for pattern in answer_patterns:
                    matches = re.findall(pattern, line.lower())
                    if matches:
                        try:
                            return float(matches[-1])
                        except ValueError:
                            continue
            
            logger.warning("Could not extract numerical answer from reasoning")
            return None
            
        except Exception as e:
            logger.error(f"Error extracting answer: {e}")
            return None


def solve_math_problem_comat(question: str, context: str = "", provider: str = None, **kwargs) -> Optional[float]:
    """
    Simple function to solve a math problem using CoMAT.
    
    Args:
        question: Mathematical question to solve
        context: Optional context information
        provider: LLM provider ('openai', 'claude', optional)
        **kwargs: Additional arguments for CoMATPrompting initialization
        
    Returns:
        Numerical result or None if solving failed
    """
    try:
        comat = CoMATPrompting(provider=provider, **kwargs)
        result = comat.solve(question, context, show_reasoning=False)
        return result['answer']
    except Exception as e:
        logger.error(f"Error in solve_math_problem_comat: {e}")
        return None