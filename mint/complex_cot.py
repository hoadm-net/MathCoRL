"""
Complex Chain-of-Thought (Complex-CoT) Prompting

Implementation of advanced multi-step reasoning for complex mathematical problems.
This approach uses sophisticated prompting strategies for challenging mathematical tasks.
"""

import re
import json
import logging
from typing import Dict, Any, List, Optional, Tuple
from langchain_openai import ChatOpenAI
from langchain_core.prompts import PromptTemplate
from langchain_core.messages import HumanMessage

logger = logging.getLogger(__name__)


class ComplexCoTPrompting:
    """
    Complex Chain-of-Thought prompting implementation.
    
    Uses advanced reasoning strategies for complex mathematical problems,
    including decomposition, verification, and multi-stage solving.
    """
    
    def __init__(self, model_name: str = None, temperature: float = None, provider: str = None):
        """
        Initialize the Complex CoT prompting system.
        
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
        
        # Complex CoT prompt templates
        self.analysis_template = PromptTemplate(
            input_variables=["context", "question"],
            template="""You are an expert mathematician. Analyze this mathematical problem carefully.

{context_section}

Problem: {question}

First, let's analyze the problem structure:
1. What type of mathematical problem is this?
2. What are the key variables and given information?
3. What mathematical concepts or formulas are needed?
4. What is the final goal?

Analysis:"""
        )
        
        self.decomposition_template = PromptTemplate(
            input_variables=["context", "question", "analysis"],
            template="""Based on the analysis, let's break down this problem into manageable steps.

Problem: {question}
{context_section}

Analysis: {analysis}

Now, decompose this problem into clear, logical steps:
1. [Step 1 description]
2. [Step 2 description]
3. [Step 3 description]
...

Decomposition:"""
        )
        
        self.solution_template = PromptTemplate(
            input_variables=["context", "question", "analysis", "decomposition"],
            template="""Now let's solve the problem step by step using our analysis and decomposition.

Problem: {question}
{context_section}

Analysis: {analysis}

Decomposition: {decomposition}

Solution:
Let's work through each step carefully:"""
        )
        
        self.verification_template = PromptTemplate(
            input_variables=["question", "solution", "answer"],
            template="""Let's verify our solution to ensure it's correct.

Original Problem: {question}

Our Solution: {solution}

Our Answer: {answer}

Verification:
1. Check if our approach is mathematically sound
2. Verify each calculation step
3. Check if the answer makes sense in context
4. Consider alternative approaches

Verification Result:"""
        )
    
    def _setup_langsmith(self):
        """Setup LangSmith tracing if configured."""
        import os
        if (os.getenv("LANGCHAIN_TRACING_V2", "false").lower() == "true" and 
            os.getenv("LANGCHAIN_API_KEY")):
            os.environ["LANGCHAIN_TRACING_V2"] = "true"
            os.environ["LANGCHAIN_ENDPOINT"] = os.getenv("LANGCHAIN_ENDPOINT", "https://api.smith.langchain.com")
            os.environ["LANGCHAIN_PROJECT"] = os.getenv("LANGCHAIN_PROJECT", "MathCoRL-Complex-CoT")
            logger.info(f"LangSmith tracing enabled for project: {os.getenv('LANGCHAIN_PROJECT')}")
        else:
            logger.info("LangSmith tracing disabled")
    
    def analyze_problem(self, question: str, context: str = "") -> str:
        """
        Analyze the mathematical problem structure.
        
        Args:
            question: The mathematical question
            context: Additional context
            
        Returns:
            Problem analysis text
        """
        try:
            context_section = f"Context: {context}\n" if context.strip() else ""
            
            prompt = self.analysis_template.format(
                context_section=context_section,
                question=question
            )
            
            from .tracking import track_api_call, extract_tokens_from_response, count_tokens_universal
            
            with track_api_call("Complex-CoT-Analysis", self.model_name, question, context) as tracker:
                messages = [HumanMessage(content=prompt)]
                
                input_tokens = count_tokens_universal(prompt, self.model_name)
                response = self.llm.invoke(messages)
                
                actual_input_tokens, output_tokens = extract_tokens_from_response(response)
                if actual_input_tokens > 0:
                    input_tokens = actual_input_tokens
                
                tracker.set_tokens(input_tokens, output_tokens)
            
            return response.content.strip()
            
        except Exception as e:
            logger.error(f"Error in problem analysis: {e}")
            return "Unable to analyze problem structure."
    
    def decompose_problem(self, question: str, analysis: str, context: str = "") -> str:
        """
        Decompose the problem into logical steps.
        
        Args:
            question: The mathematical question
            analysis: Problem analysis from previous step
            context: Additional context
            
        Returns:
            Problem decomposition text
        """
        try:
            context_section = f"Context: {context}\n" if context.strip() else ""
            
            prompt = self.decomposition_template.format(
                context_section=context_section,
                question=question,
                analysis=analysis
            )
            
            from .tracking import track_api_call, extract_tokens_from_response, count_tokens_universal
            
            with track_api_call("Complex-CoT-Decomposition", self.model_name, question, context) as tracker:
                messages = [HumanMessage(content=prompt)]
                
                input_tokens = count_tokens_universal(prompt, self.model_name)
                response = self.llm.invoke(messages)
                
                actual_input_tokens, output_tokens = extract_tokens_from_response(response)
                if actual_input_tokens > 0:
                    input_tokens = actual_input_tokens
                
                tracker.set_tokens(input_tokens, output_tokens)
            
            return response.content.strip()
            
        except Exception as e:
            logger.error(f"Error in problem decomposition: {e}")
            return "Unable to decompose problem into steps."
    
    def solve_step_by_step(self, question: str, analysis: str, decomposition: str, context: str = "") -> str:
        """
        Solve the problem using the analysis and decomposition.
        
        Args:
            question: The mathematical question
            analysis: Problem analysis
            decomposition: Problem decomposition
            context: Additional context
            
        Returns:
            Step-by-step solution
        """
        try:
            context_section = f"Context: {context}\n" if context.strip() else ""
            
            prompt = self.solution_template.format(
                context_section=context_section,
                question=question,
                analysis=analysis,
                decomposition=decomposition
            )
            
            from .tracking import track_api_call, extract_tokens_from_response, count_tokens_universal
            
            with track_api_call("Complex-CoT-Solution", self.model_name, question, context) as tracker:
                messages = [HumanMessage(content=prompt)]
                
                input_tokens = count_tokens_universal(prompt, self.model_name)
                response = self.llm.invoke(messages)
                
                actual_input_tokens, output_tokens = extract_tokens_from_response(response)
                if actual_input_tokens > 0:
                    input_tokens = actual_input_tokens
                
                tracker.set_tokens(input_tokens, output_tokens)
            
            return response.content.strip()
            
        except Exception as e:
            logger.error(f"Error in step-by-step solving: {e}")
            return "Unable to solve problem step by step."
    
    def verify_solution(self, question: str, solution: str, answer: str) -> str:
        """
        Verify the solution for correctness.
        
        Args:
            question: The original question
            solution: The step-by-step solution
            answer: The extracted answer
            
        Returns:
            Verification result
        """
        try:
            prompt = self.verification_template.format(
                question=question,
                solution=solution,
                answer=answer
            )
            
            from .tracking import track_api_call, extract_tokens_from_response, count_tokens_universal
            
            with track_api_call("Complex-CoT-Verification", self.model_name, question, "") as tracker:
                messages = [HumanMessage(content=prompt)]
                
                input_tokens = count_tokens_universal(prompt, self.model_name)
                response = self.llm.invoke(messages)
                
                actual_input_tokens, output_tokens = extract_tokens_from_response(response)
                if actual_input_tokens > 0:
                    input_tokens = actual_input_tokens
                
                tracker.set_tokens(input_tokens, output_tokens)
            
            return response.content.strip()
            
        except Exception as e:
            logger.error(f"Error in solution verification: {e}")
            return "Unable to verify solution."
    
    def solve(self, question: str, context: str = "", show_reasoning: bool = True, 
              verify_answer: bool = True) -> Dict[str, Any]:
        """
        Solve a mathematical problem using Complex CoT prompting.
        
        Args:
            question: The mathematical question to solve
            context: Additional context for the problem
            show_reasoning: Whether to show the reasoning steps
            verify_answer: Whether to verify the final answer
            
        Returns:
            Dictionary containing the result, reasoning, and metadata
        """
        try:
            if show_reasoning:
                print(f"🔍 Starting Complex CoT analysis for: {question}\n")
            
            # Step 1: Analyze the problem
            analysis = self.analyze_problem(question, context)
            if show_reasoning:
                print(f"📊 Problem Analysis:\n{analysis}\n")
            
            # Step 2: Decompose the problem
            decomposition = self.decompose_problem(question, analysis, context)
            if show_reasoning:
                print(f"🔧 Problem Decomposition:\n{decomposition}\n")
            
            # Step 3: Solve step by step
            solution = self.solve_step_by_step(question, analysis, decomposition, context)
            if show_reasoning:
                print(f"🧠 Step-by-Step Solution:\n{solution}\n")
            
            # Extract the final answer
            final_answer = self._extract_answer(solution)
            
            # Step 4: Verify the solution (optional)
            verification = ""
            if verify_answer and final_answer is not None:
                verification = self.verify_solution(question, solution, str(final_answer))
                if show_reasoning:
                    print(f"✅ Solution Verification:\n{verification}\n")
            
            # Combine all reasoning
            full_reasoning = f"Analysis:\n{analysis}\n\nDecomposition:\n{decomposition}\n\nSolution:\n{solution}"
            if verification:
                full_reasoning += f"\n\nVerification:\n{verification}"
            
            result = {
                'question': question,
                'context': context,
                'reasoning': full_reasoning,
                'analysis': analysis,
                'decomposition': decomposition,
                'solution': solution,
                'verification': verification,
                'answer': final_answer,
                'method': 'Complex-CoT',
                'model': self.model_name,
                'verified': bool(verification),
                'success': True
            }
            
            if show_reasoning:
                print(f"📊 Final Answer: {final_answer}")
            
            return result
            
        except Exception as e:
            logger.error(f"Error in Complex-CoT solve: {e}")
            return {
                'question': question,
                'context': context,
                'reasoning': '',
                'answer': None,
                'method': 'Complex-CoT',
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
            
            result = self.solve(question, problem_context, show_reasoning=False, verify_answer=False)
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
                r"(?:is|equals?)\s*([+-]?\d*\.?\d+)(?:\*\*)?\.?\s*$",  # is 30. or is 30**
                r"width.*?([+-]?\d*\.?\d+).*?meters",  # Extract width from dimension problems
                r"length.*?([+-]?\d*\.?\d+).*?meters"   # Extract length from dimension problems
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


def solve_math_problem_complex_cot(question: str, context: str = "", provider: str = None, **kwargs) -> Optional[float]:
    """
    Simple function to solve a math problem using Complex CoT.
    
    Args:
        question: Mathematical question to solve
        context: Optional context information
        provider: LLM provider ('openai', 'claude', optional)
        **kwargs: Additional arguments for ComplexCoTPrompting initialization
        
    Returns:
        Numerical result or None if solving failed
    """
    try:
        complex_cot = ComplexCoTPrompting(provider=provider, **kwargs)
        result = complex_cot.solve(question, context, show_reasoning=False, verify_answer=False)
        return result['answer']
    except Exception as e:
        logger.error(f"Error in solve_math_problem_complex_cot: {e}")
        return None