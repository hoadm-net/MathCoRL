"""
Auto-CoT: Automatic Chain-of-Thought Prompting

Implementation based on "Automatic Chain of Thought Prompting in Large Language Models" (Zhang et al., 2022)
This approach automatically generates diverse examples and clusters them for few-shot demonstrations.
"""

import re
import json
import random
import logging
from typing import Dict, Any, List, Optional, Tuple
from langchain_openai import ChatOpenAI
from langchain.prompts import PromptTemplate
from langchain.schema import HumanMessage
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans

logger = logging.getLogger(__name__)


class AutoCoTPrompting:
    """
    Auto-CoT prompting implementation.
    
    Automatically generates diverse reasoning examples and uses clustering
    to select representative demonstrations for few-shot prompting.
    """
    
    def __init__(self, model_name: str = None, temperature: float = None, provider: str = None, 
                 num_clusters: int = 4, examples_per_cluster: int = 1):
        """
        Initialize the Auto-CoT prompting system.
        
        Args:
            model_name: The model to use
            temperature: Temperature for response generation
            provider: LLM provider ('openai', 'claude', optional)
            num_clusters: Number of clusters for example diversity
            examples_per_cluster: Number of examples to select per cluster
        """
        from .config import load_config, create_llm_client, get_current_model_name
        config = load_config()
        
        self.provider = provider or config['provider']
        self.model_name = model_name or get_current_model_name(self.provider)
        self.temperature = temperature if temperature is not None else config['temperature']
        self.num_clusters = num_clusters
        self.examples_per_cluster = examples_per_cluster
        
        # Setup LangSmith if configured
        self._setup_langsmith()
        
        self.llm = create_llm_client(
            provider=self.provider,
            model=self.model_name,
            temperature=self.temperature
        )
        
        # Auto-CoT prompt template
        self.auto_cot_template = PromptTemplate(
            input_variables=["examples", "context", "question"],
            template="""You are an expert mathematician. Here are some examples of step-by-step mathematical reasoning:

{examples}

{context_section}

Problem: {question}

Let's solve this step by step:"""
        )
        
        # Example generation template
        self.example_generation_template = PromptTemplate(
            input_variables=["question_type", "difficulty"],
            template="""Generate a diverse {question_type} math problem with {difficulty} difficulty.
            
Problem: [Generate a specific mathematical question]
Let's think step by step:
[Provide detailed step-by-step reasoning]
Answer: [Final numerical answer]

Generate only one complete example."""
        )
    
    def _setup_langsmith(self):
        """Setup LangSmith tracing if configured."""
        import os
        if (os.getenv("LANGCHAIN_TRACING_V2", "false").lower() == "true" and 
            os.getenv("LANGCHAIN_API_KEY")):
            os.environ["LANGCHAIN_TRACING_V2"] = "true"
            os.environ["LANGCHAIN_ENDPOINT"] = os.getenv("LANGCHAIN_ENDPOINT", "https://api.smith.langchain.com")
            os.environ["LANGCHAIN_PROJECT"] = os.getenv("LANGCHAIN_PROJECT", "MathCoRL-Auto-CoT")
            logger.info(f"LangSmith tracing enabled for project: {os.getenv('LANGCHAIN_PROJECT')}")
        else:
            logger.info("LangSmith tracing disabled")
    
    def generate_diverse_examples(self, question: str, num_examples: int = 4) -> List[str]:
        """
        Generate diverse mathematical examples using automatic generation.
        
        Args:
            question: The target question to generate examples for
            num_examples: Number of examples to generate
            
        Returns:
            List of generated example strings
        """
        try:
            # Analyze question type and difficulty
            question_type, difficulty = self._analyze_question(question)
            
            examples = []
            variations = [
                ("basic", "easy"),
                ("intermediate", "medium"),
                ("advanced", "medium"),
                ("complex", "hard")
            ]
            
            for i in range(num_examples):
                var_type, var_diff = variations[i % len(variations)]
                
                # Generate example with tracking
                from .tracking import track_api_call, extract_tokens_from_response, count_tokens_universal
                
                prompt = self.example_generation_template.format(
                    question_type=f"{question_type} {var_type}",
                    difficulty=var_diff
                )
                
                with track_api_call("Auto-CoT-Gen", self.model_name, prompt, "") as tracker:
                    messages = [HumanMessage(content=prompt)]
                    
                    input_tokens = count_tokens_universal(prompt, self.model_name)
                    response = self.llm.invoke(messages)
                    
                    actual_input_tokens, output_tokens = extract_tokens_from_response(response)
                    if actual_input_tokens > 0:
                        input_tokens = actual_input_tokens
                    
                    tracker.set_tokens(input_tokens, output_tokens)
                
                example = response.content.strip()
                if example:
                    examples.append(example)
            
            return examples
            
        except Exception as e:
            logger.error(f"Error generating examples: {e}")
            # Fallback to predefined examples
            return self._get_fallback_examples()
    
    def _analyze_question(self, question: str) -> Tuple[str, str]:
        """
        Analyze question type and difficulty.
        
        Args:
            question: The mathematical question
            
        Returns:
            Tuple of (question_type, difficulty)
        """
        question_lower = question.lower()
        
        # Determine question type
        if any(word in question_lower for word in ['percentage', 'percent', '%']):
            question_type = "percentage"
        elif any(word in question_lower for word in ['ratio', 'proportion']):
            question_type = "ratio"
        elif any(word in question_lower for word in ['algebra', 'equation', 'solve for']):
            question_type = "algebra"
        elif any(word in question_lower for word in ['geometry', 'area', 'volume', 'perimeter']):
            question_type = "geometry"
        elif any(word in question_lower for word in ['profit', 'loss', 'cost', 'price', 'finance']):
            question_type = "financial"
        else:
            question_type = "arithmetic"
        
        # Determine difficulty based on complexity
        complexity_indicators = len(re.findall(r'\d+', question))
        if complexity_indicators <= 2:
            difficulty = "easy"
        elif complexity_indicators <= 4:
            difficulty = "medium"
        else:
            difficulty = "hard"
        
        return question_type, difficulty
    
    def cluster_examples(self, examples: List[str]) -> List[str]:
        """
        Cluster examples for diversity and select representatives.
        
        Args:
            examples: List of generated examples
            
        Returns:
            List of selected diverse examples
        """
        try:
            if len(examples) <= self.num_clusters:
                return examples
            
            # Extract problem statements for clustering
            problems = []
            for example in examples:
                # Extract the problem part (before "Let's think step by step")
                problem_match = re.search(r'Problem:\s*(.*?)(?:\n|Let\'s think)', example, re.DOTALL)
                if problem_match:
                    problems.append(problem_match.group(1).strip())
                else:
                    problems.append(example[:100])  # Fallback to first 100 chars
            
            # Vectorize problems using TF-IDF
            vectorizer = TfidfVectorizer(max_features=100, stop_words='english')
            problem_vectors = vectorizer.fit_transform(problems)
            
            # Cluster problems
            n_clusters = min(self.num_clusters, len(examples))
            kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
            cluster_labels = kmeans.fit_predict(problem_vectors)
            
            # Select representative examples from each cluster
            selected_examples = []
            for cluster_id in range(n_clusters):
                cluster_indices = [i for i, label in enumerate(cluster_labels) if label == cluster_id]
                if cluster_indices:
                    # Select the example closest to cluster center
                    cluster_center = kmeans.cluster_centers_[cluster_id]
                    distances = [
                        np.linalg.norm(problem_vectors[i].toarray().flatten() - cluster_center)
                        for i in cluster_indices
                    ]
                    best_idx = cluster_indices[np.argmin(distances)]
                    selected_examples.append(examples[best_idx])
            
            return selected_examples[:self.num_clusters]
            
        except Exception as e:
            logger.error(f"Error in clustering: {e}")
            # Fallback to random selection
            return random.sample(examples, min(self.num_clusters, len(examples)))
    
    def _get_fallback_examples(self) -> List[str]:
        """Get predefined fallback examples."""
        return [
            """Problem: A store has 120 apples. If 25% are red apples, how many red apples are there?
Let's think step by step:
1. We need to find 25% of 120 apples
2. 25% = 25/100 = 0.25
3. 0.25 × 120 = 30
Answer: 30""",
            
            """Problem: If 3x + 5 = 20, what is the value of x?
Let's think step by step:
1. We have the equation 3x + 5 = 20
2. Subtract 5 from both sides: 3x = 20 - 5 = 15
3. Divide both sides by 3: x = 15 ÷ 3 = 5
Answer: 5""",
            
            """Problem: A rectangle has length 8 cm and width 6 cm. What is its area?
Let's think step by step:
1. Area of rectangle = length × width
2. Length = 8 cm, width = 6 cm
3. Area = 8 × 6 = 48 square cm
Answer: 48""",
            
            """Problem: John bought 5 books for $12 each. How much did he spend in total?
Let's think step by step:
1. Number of books = 5
2. Price per book = $12
3. Total cost = 5 × $12 = $60
Answer: 60"""
        ]
    
    def solve(self, question: str, context: str = "", show_reasoning: bool = True, 
              generate_new_examples: bool = True) -> Dict[str, Any]:
        """
        Solve a mathematical problem using Auto-CoT prompting.
        
        Args:
            question: The mathematical question to solve
            context: Additional context for the problem
            show_reasoning: Whether to show the reasoning steps
            generate_new_examples: Whether to generate new examples or use cached ones
            
        Returns:
            Dictionary containing the result, reasoning, and metadata
        """
        try:
            # Generate or get examples
            if generate_new_examples:
                examples = self.generate_diverse_examples(question)
                selected_examples = self.cluster_examples(examples)
            else:
                selected_examples = self._get_fallback_examples()
            
            # Format examples
            examples_text = "\n\n".join(selected_examples)
            
            # Prepare context section
            context_section = f"Context: {context}\n" if context.strip() else ""
            
            # Create the prompt
            prompt = self.auto_cot_template.format(
                examples=examples_text,
                context_section=context_section,
                question=question
            )
            
            if show_reasoning:
                print(f"🤖 Auto-CoT Prompt (with {len(selected_examples)} examples):\n{prompt}\n")
            
            # Get response from LLM with tracking
            from .tracking import track_api_call, extract_tokens_from_response, count_tokens_universal
            
            with track_api_call("Auto-CoT", self.model_name, question, context) as tracker:
                messages = [HumanMessage(content=prompt)]
                
                input_tokens = count_tokens_universal(prompt, self.model_name)
                response = self.llm.invoke(messages)
                reasoning = response.content
                
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
                'method': 'Auto-CoT',
                'model': self.model_name,
                'examples_used': len(selected_examples),
                'success': True
            }
            
            if show_reasoning:
                print(f"🧠 Auto-CoT Reasoning:\n{reasoning}\n")
                print(f"📊 Final Answer: {final_answer}")
            
            return result
            
        except Exception as e:
            logger.error(f"Error in Auto-CoT solve: {e}")
            return {
                'question': question,
                'context': context,
                'reasoning': '',
                'answer': None,
                'method': 'Auto-CoT',
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
            
            result = self.solve(question, problem_context, show_reasoning=False, 
                              generate_new_examples=False)  # Use cached examples for speed
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
                r"\\boxed\{([+-]?\d*\.?\d+)\}"  # \boxed{16}
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
                        
                # Look for boxed numbers
                boxed_numbers = re.findall(r'\\boxed\{([+-]?\d*\.?\d+)\}', line)
                if boxed_numbers:
                    try:
                        return float(boxed_numbers[-1])
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


def solve_math_problem_auto_cot(question: str, context: str = "", provider: str = None, **kwargs) -> Optional[float]:
    """
    Simple function to solve a math problem using Auto-CoT.
    
    Args:
        question: Mathematical question to solve
        context: Optional context information
        provider: LLM provider ('openai', 'claude', optional)
        **kwargs: Additional arguments for AutoCoTPrompting initialization
        
    Returns:
        Numerical result or None if solving failed
    """
    try:
        auto_cot = AutoCoTPrompting(provider=provider, **kwargs)
        result = auto_cot.solve(question, context, show_reasoning=False, generate_new_examples=False)
        return result['answer']
    except Exception as e:
        logger.error(f"Error in solve_math_problem_auto_cot: {e}")
        return None