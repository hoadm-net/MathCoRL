"""
Chain-of-Thought (CoT) Prompting for Mathematical Problem Solving

This module implements Chain-of-Thought prompting, which encourages the model
to show its reasoning steps before providing the final answer.
"""

import re
import logging
from typing import Dict, Any, Optional
from langchain_openai import ChatOpenAI
from langchain.prompts import PromptTemplate
from langchain.schema import HumanMessage

logger = logging.getLogger(__name__)


class ChainOfThoughtPrompting:
    """
    Chain-of-Thought prompting implementation for mathematical problem solving.
    
    CoT prompting asks the model to work through problems step by step,
    showing its reasoning process before arriving at the final answer.
    """
    
    def __init__(self, model_name: str = "gpt-4", temperature: float = 0.0):
        """
        Initialize the Chain-of-Thought prompting system.
        
        Args:
            model_name: The OpenAI model to use
            temperature: Temperature for response generation
        """
        self.model_name = model_name
        self.temperature = temperature
        self.llm = ChatOpenAI(
            model_name=model_name,
            temperature=temperature
        )
        
        # CoT prompt template with few-shot examples
        self.cot_template = PromptTemplate(
            input_variables=["context", "question"],
            template="""You are an expert mathematician. Solve the following problem step by step.

Think through the problem carefully and show your reasoning process. Break down the problem into smaller steps and explain each step clearly.

Here are some examples of how to solve math problems step by step:

Example 1:
Problem: Janet's ducks lay 16 eggs per day. She eats 3 for breakfast and bakes muffins with 4. She sells the remainder at $2 per egg. How much does she make daily?

Solution:
Step 1: Identify what we need to find.
We need to find how much money Janet makes daily from selling eggs.

Step 2: Calculate how many eggs are left to sell.
- Total eggs per day: 16
- Eggs Janet eats: 3
- Eggs used for muffins: 4
- Eggs left to sell: 16 - 3 - 4 = 9 eggs

Step 3: Calculate daily earnings.
- Price per egg: $2
- Total earnings: 9 × $2 = $18

Therefore, Janet makes $18 daily.

Example 2:
Problem: A school has 569 girls and 236 boys. How many more girls than boys does the school have?

Solution:
Step 1: Identify what we need to find.
We need to find the difference between the number of girls and boys.

Step 2: Calculate the difference.
- Number of girls: 569
- Number of boys: 236
- Difference: 569 - 236 = 333

Therefore, the school has 333 more girls than boys.

Now solve this problem:

{context_section}

Problem: {question}

Solution:
Step 1: Identify what we need to find.
Step 2: Identify the given information and set up the calculation.
Step 3: Perform the calculations step by step.
Step 4: State the final answer clearly.

Let me work through this:"""
        )
    
    def solve(self, question: str, context: str = "", show_reasoning: bool = True) -> Dict[str, Any]:
        """
        Solve a mathematical problem using Chain-of-Thought prompting.
        
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
            prompt = self.cot_template.format(
                context_section=context_section,
                question=question
            )
            
            if show_reasoning:
                print(f"🤖 CoT Prompt:\n{prompt}\n")
            
            # Get response from LLM
            messages = [HumanMessage(content=prompt)]
            response = self.llm.invoke(messages)
            reasoning = response.content
            
            if show_reasoning:
                print(f"🧠 CoT Reasoning:\n{reasoning}\n")
            
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
                'method': 'Chain-of-Thought'
            }
            
        except Exception as e:
            logger.error(f"Error in CoT solving: {e}")
            return {
                'result': None,
                'reasoning': f"Error: {str(e)}",
                'question': question,
                'context': context,
                'model': self.model_name,
                'method': 'Chain-of-Thought'
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
        # Common patterns for final answers in CoT reasoning
        answer_patterns = [
            r"(?:final answer|answer|result|solution)(?:\s*is)?\s*:?\s*([+-]?\d*\.?\d+)",
            r"therefore,?\s*([+-]?\d*\.?\d+)",
            r"so,?\s*([+-]?\d*\.?\d+)",
            r"the answer is\s*([+-]?\d*\.?\d+)",
            r"equals?\s*([+-]?\d*\.?\d+)",
            r"=\s*([+-]?\d*\.?\d+)(?:\s*$|\s*\.|\s*,)",
            r"([+-]?\d*\.?\d+)\s*(?:is the answer|is the final answer|is the result)",
            r"total(?:\s*is)?\s*([+-]?\d*\.?\d+)",
            r"(\d+)\s*\+\s*(\d+)\s*=\s*(\d+)",  # For simple arithmetic like "15 + 27 = 42"
            r"answer:\s*([+-]?\d*\.?\d+)",
            r"solution:\s*([+-]?\d*\.?\d+)",
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
        
        # If no pattern matches, try to find numbers near common conclusion words
        conclusion_patterns = [
            r"(?:therefore|so|thus|hence|final|answer|result|solution|total).*?([+-]?\d+\.?\d*)",
            r"([+-]?\d+\.?\d*).*?(?:is the answer|is the result|is the solution)"
        ]
        
        for pattern in conclusion_patterns:
            matches = re.findall(pattern, reasoning_lower, re.MULTILINE | re.IGNORECASE)
            if matches:
                try:
                    return float(matches[-1])
                except ValueError:
                    continue
        
        # Last resort: find the last number in the text
        numbers = re.findall(r'([+-]?\d+\.?\d*)', reasoning)
        if numbers:
            try:
                return float(numbers[-1])
            except ValueError:
                pass
        
        logger.warning(f"Could not extract numerical answer from reasoning: {reasoning[:200]}...")
        return None


def solve_with_cot(question: str, context: str = "", model_name: str = "gpt-4") -> Optional[float]:
    """
    Convenience function for solving math problems with Chain-of-Thought prompting.
    
    Args:
        question: The mathematical question to solve
        context: Additional context for the problem
        model_name: The OpenAI model to use
        
    Returns:
        The numerical result, or None if solving failed
    """
    cot = ChainOfThoughtPrompting(model_name=model_name)
    result = cot.solve_silent(question, context)
    return result.get('result') 