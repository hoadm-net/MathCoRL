"""
Meta Prompting Framework for MathCoRL

Implementation of advanced meta-prompting techniques that automatically optimize
prompting strategies for mathematical problem solving. This framework combines:

1. Problem Type Classification and Method Selection
2. Self-Improving Prompt Optimization with Feedback Loops  
3. Reflection Memory for Cross-Problem Learning

Based on research from:
- REMO: Reflection-Enhanced Meta-Optimization (arXiv:2508.18749)
- Promptomatix: Automatic Prompt Optimization (arXiv:2507.14241)
- PCF: Polymorphic Combinatorial Frameworks (arXiv:2508.01581)
"""

import json
import logging
import os
import re
import signal
from datetime import datetime
from typing import Dict, Any, List, Optional, Tuple, Union
from dataclasses import dataclass, asdict
from collections import defaultdict
import hashlib

try:
    from langchain_core.messages import HumanMessage
except ImportError:
    try:
        from langchain_core.messages import HumanMessage
    except ImportError:
        # Fallback for testing without langchain
        class HumanMessage:
            def __init__(self, content):
                self.content = content

logger = logging.getLogger(__name__)


@dataclass
class PromptOptimizationResult:
    """Result of prompt optimization process."""
    original_prompt: str
    optimized_prompt: str
    performance_score: float
    optimization_iterations: int
    method_used: str
    timestamp: str
    problem_type: str


@dataclass
class ReflectionEntry:
    """Entry in the reflection memory system."""
    problem_hash: str
    problem_type: str
    method_used: str
    prompt_used: str
    success: bool
    performance_score: float
    failure_reason: Optional[str]
    timestamp: str
    optimization_applied: bool


class ProblemTypeClassifier:
    """
    Classifies mathematical problems into types for optimal method selection.
    """
    
    def __init__(self):
        self.problem_types = {
            'arithmetic': ['addition', 'subtraction', 'multiplication', 'division', 'basic math'],
            'algebra': ['equation', 'variable', 'solve for', 'linear', 'quadratic'],
            'geometry': ['area', 'perimeter', 'volume', 'triangle', 'circle', 'rectangle'],
            'word_problem': ['has', 'give', 'total', 'left', 'cost', 'price', 'time'],
            'percentage': ['percent', '%', 'percentage', 'discount', 'interest'],
            'fractions': ['fraction', 'half', 'quarter', 'third', 'parts'],
            'statistics': ['average', 'mean', 'median', 'mode', 'probability'],
            'combinatorics': ['combination', 'permutation', 'arrange', 'choose']
        }
        
        self.method_preferences = {
            'arithmetic': ['fpp', 'comat', 'cot'],
            'algebra': ['comat', 'complex_cot', 'pot'],
            'geometry': ['comat', 'fpp', 'complex_cot'],
            'word_problem': ['cot', 'auto_cot', 'comat'],
            'percentage': ['fpp', 'pot', 'comat'],
            'fractions': ['fpp', 'cot', 'comat'],
            'statistics': ['pot', 'fpp', 'comat'],
            'combinatorics': ['pot', 'complex_cot', 'comat']
        }
    
    def classify_problem(self, question: str) -> Tuple[str, float]:
        """
        Classify a mathematical problem into type.
        
        Args:
            question: The mathematical question
            
        Returns:
            Tuple of (problem_type, confidence_score)
        """
        question_lower = question.lower()
        
        type_scores = defaultdict(float)
        
        # Count keyword matches for each type
        for problem_type, keywords in self.problem_types.items():
            for keyword in keywords:
                if keyword in question_lower:
                    type_scores[problem_type] += 1
        
        if not type_scores:
            return 'word_problem', 0.5  # Default type
        
        # Find the type with highest score
        best_type = max(type_scores, key=type_scores.get)
        max_score = type_scores[best_type]
        
        # Calculate confidence (normalize by total possible matches)
        total_keywords = len(self.problem_types[best_type])
        confidence = min(max_score / total_keywords, 1.0)
        
        return best_type, confidence
    
    def recommend_methods(self, problem_type: str, top_k: int = 3) -> List[str]:
        """
        Recommend best methods for a problem type.
        
        Args:
            problem_type: Type of mathematical problem
            top_k: Number of methods to recommend
            
        Returns:
            List of recommended method names
        """
        if problem_type in self.method_preferences:
            return self.method_preferences[problem_type][:top_k]
        else:
            return ['comat', 'cot', 'fpp']  # Default fallback


class PromptOptimizer:
    """
    Self-improving prompt optimization system with feedback loops.
    """
    
    def __init__(self, model_name: str = None, provider: str = None):
        from .config import load_config, create_llm_client, get_current_model_name
        config = load_config()
        
        self.provider = provider or config['provider']
        self.model_name = model_name or get_current_model_name(self.provider)
        
        self.llm = create_llm_client(
            provider=self.provider,
            model=self.model_name,
            temperature=0.7  # Higher creativity for prompt generation
        )
        
        # Optimization templates
        self.optimization_prompt = """
You are an expert prompt engineer specializing in mathematical problem solving. 
Your task is to optimize prompts for better performance on mathematical reasoning tasks.

Current Method: {method}
Problem Type: {problem_type}
Current Prompt Performance: {performance_score:.2f}

Original Prompt:
{original_prompt}

Previous Attempts and Results:
{previous_attempts}

Based on the performance feedback and mathematical reasoning best practices, 
generate an improved version of this prompt that will likely perform better.

Focus on:
1. Clarity and specificity for mathematical reasoning
2. Structured thinking approaches
3. Error prevention and verification steps
4. Problem-type specific optimizations

Optimized Prompt:
"""
        
        self.reflection_prompt = """
Analyze the following mathematical problem solving attempt and provide insights:

Problem: {question}
Method Used: {method}
Success: {success}
Performance Score: {performance_score:.2f}
Prompt Used: {prompt_used}

Provide analysis on:
1. Why this approach worked/failed
2. What could be improved in the prompt
3. Key patterns for future optimization
4. Method-specific recommendations

Analysis:
"""
    
    def optimize_prompt(self, original_prompt: str, method: str, problem_type: str, 
                       performance_history: List[Dict], max_iterations: int = 3) -> str:
        """
        Optimize a prompt based on performance feedback.
        
        Args:
            original_prompt: The original prompt to optimize
            method: The reasoning method being used
            problem_type: Type of mathematical problem
            performance_history: History of previous attempts
            max_iterations: Maximum optimization iterations
            
        Returns:
            Optimized prompt
        """
        try:
            current_prompt = original_prompt
            best_prompt = original_prompt
            best_score = 0.0
            
            # Extract performance info from history
            if performance_history:
                avg_score = sum(h.get('score', 0) for h in performance_history) / len(performance_history)
                previous_attempts = "\n".join([
                    f"Attempt {i+1}: Score {h.get('score', 0):.2f}, Success: {h.get('success', False)}"
                    for i, h in enumerate(performance_history[-3:])  # Last 3 attempts
                ])
            else:
                avg_score = 0.5
                previous_attempts = "No previous attempts available."
            
            for iteration in range(max_iterations):
                optimization_request = self.optimization_prompt.format(
                    method=method,
                    problem_type=problem_type,
                    performance_score=avg_score,
                    original_prompt=current_prompt,
                    previous_attempts=previous_attempts
                )
                
                from .tracking import track_api_call, extract_tokens_from_response, count_tokens_universal
                
                with track_api_call("MetaPrompt-Optimize", self.model_name, "prompt_optimization", method) as tracker:
                    messages = [HumanMessage(content=optimization_request)]
                    
                    input_tokens = count_tokens_universal(optimization_request, self.model_name)
                    response = self.llm.invoke(messages)
                    
                    actual_input_tokens, output_tokens = extract_tokens_from_response(response)
                    if actual_input_tokens > 0:
                        input_tokens = actual_input_tokens
                    
                    tracker.set_tokens(input_tokens, output_tokens)
                
                # Extract optimized prompt
                optimized_prompt = self._extract_optimized_prompt(response.content.strip())
                
                if optimized_prompt and optimized_prompt != current_prompt:
                    current_prompt = optimized_prompt
                    logger.info(f"Prompt optimization iteration {iteration + 1} completed")
                else:
                    logger.info("No further optimization possible")
                    break
            
            return current_prompt
            
        except Exception as e:
            logger.error(f"Error in prompt optimization: {e}")
            return original_prompt
    
    def _extract_optimized_prompt(self, response: str) -> str:
        """Extract the optimized prompt from LLM response."""
        # Look for prompt after "Optimized Prompt:" marker
        lines = response.split('\n')
        
        prompt_started = False
        optimized_lines = []
        
        for line in lines:
            if 'optimized prompt:' in line.lower():
                prompt_started = True
                # Check if prompt is on the same line
                parts = line.split(':', 1)
                if len(parts) > 1 and parts[1].strip():
                    optimized_lines.append(parts[1].strip())
                continue
            
            if prompt_started:
                # Stop at empty line or new section
                if line.strip() == '' and optimized_lines:
                    break
                if line.strip():
                    optimized_lines.append(line.strip())
        
        if optimized_lines:
            return '\n'.join(optimized_lines)
        
        # Fallback: return the entire response if no clear prompt found
        return response.strip()
    
    def analyze_performance(self, question: str, method: str, success: bool, 
                          performance_score: float, prompt_used: str) -> str:
        """
        Analyze performance for reflection and learning.
        
        Args:
            question: The mathematical question
            method: Method used
            success: Whether the solution was correct
            performance_score: Performance score (0-1)
            prompt_used: The prompt that was used
            
        Returns:
            Analysis insights
        """
        try:
            analysis_request = self.reflection_prompt.format(
                question=question,
                method=method,
                success=success,
                performance_score=performance_score,
                prompt_used=prompt_used
            )
            
            from .tracking import track_api_call, extract_tokens_from_response, count_tokens_universal
            
            with track_api_call("MetaPrompt-Reflect", self.model_name, "performance_analysis", method) as tracker:
                messages = [HumanMessage(content=analysis_request)]
                
                input_tokens = count_tokens_universal(analysis_request, self.model_name)
                response = self.llm.invoke(messages)
                
                actual_input_tokens, output_tokens = extract_tokens_from_response(response)
                if actual_input_tokens > 0:
                    input_tokens = actual_input_tokens
                
                tracker.set_tokens(input_tokens, output_tokens)
            
            return response.content.strip()
            
        except Exception as e:
            logger.error(f"Error in performance analysis: {e}")
            return f"Analysis failed: {str(e)}"


class ReflectionMemory:
    """
    Memory system for storing optimization history and learning patterns.
    """
    
    def __init__(self, memory_file: str = "logs/meta_prompting_memory.jsonl"):
        self.memory_file = memory_file
        self.ensure_memory_file()
        
        # In-memory cache for recent entries
        self.cache = []
        self.cache_size = 100
        self._load_recent_entries()
    
    def ensure_memory_file(self):
        """Ensure memory file and directory exist."""
        os.makedirs(os.path.dirname(self.memory_file), exist_ok=True)
        if not os.path.exists(self.memory_file):
            with open(self.memory_file, 'w') as f:
                pass  # Create empty file
    
    def _load_recent_entries(self):
        """Load recent entries into cache."""
        try:
            if os.path.exists(self.memory_file):
                with open(self.memory_file, 'r') as f:
                    lines = f.readlines()
                    
                # Load last cache_size entries
                for line in lines[-self.cache_size:]:
                    if line.strip():
                        entry_dict = json.loads(line.strip())
                        entry = ReflectionEntry(**entry_dict)
                        self.cache.append(entry)
                        
        except Exception as e:
            logger.error(f"Error loading memory cache: {e}")
            self.cache = []
    
    def add_entry(self, problem_hash: str, problem_type: str, method_used: str,
                  prompt_used: str, success: bool, performance_score: float,
                  failure_reason: Optional[str] = None, optimization_applied: bool = False):
        """
        Add a new reflection entry to memory.
        
        Args:
            problem_hash: Hash of the problem
            problem_type: Type of mathematical problem
            method_used: Method that was used
            prompt_used: The prompt that was used
            success: Whether the solution was correct
            performance_score: Performance score (0-1)
            failure_reason: Reason for failure if applicable
            optimization_applied: Whether prompt optimization was applied
        """
        entry = ReflectionEntry(
            problem_hash=problem_hash,
            problem_type=problem_type,
            method_used=method_used,
            prompt_used=prompt_used,
            success=success,
            performance_score=performance_score,
            failure_reason=failure_reason,
            timestamp=datetime.now().isoformat(),
            optimization_applied=optimization_applied
        )
        
        # Add to cache
        self.cache.append(entry)
        if len(self.cache) > self.cache_size:
            self.cache.pop(0)  # Remove oldest entry
        
        # Append to file
        try:
            with open(self.memory_file, 'a') as f:
                f.write(json.dumps(asdict(entry)) + '\n')
        except Exception as e:
            logger.error(f"Error writing to memory file: {e}")
    
    def get_performance_history(self, method: str, problem_type: str, limit: int = 10) -> List[Dict]:
        """
        Get performance history for a specific method and problem type.
        
        Args:
            method: The reasoning method
            problem_type: Type of mathematical problem
            limit: Maximum number of entries to return
            
        Returns:
            List of performance history entries
        """
        matching_entries = []
        
        for entry in reversed(self.cache):  # Most recent first
            if entry.method_used == method and entry.problem_type == problem_type:
                matching_entries.append({
                    'success': entry.success,
                    'score': entry.performance_score,
                    'timestamp': entry.timestamp,
                    'optimization_applied': entry.optimization_applied
                })
                
                if len(matching_entries) >= limit:
                    break
        
        return matching_entries
    
    def get_successful_patterns(self, problem_type: str, min_score: float = 0.8) -> List[Dict]:
        """
        Get successful prompt patterns for a problem type.
        
        Args:
            problem_type: Type of mathematical problem
            min_score: Minimum performance score to consider
            
        Returns:
            List of successful patterns
        """
        successful_patterns = []
        
        for entry in self.cache:
            if (entry.problem_type == problem_type and 
                entry.success and 
                entry.performance_score >= min_score):
                
                successful_patterns.append({
                    'method': entry.method_used,
                    'prompt': entry.prompt_used,
                    'score': entry.performance_score,
                    'timestamp': entry.timestamp
                })
        
        # Sort by performance score (descending)
        successful_patterns.sort(key=lambda x: x['score'], reverse=True)
        
        return successful_patterns
    
    def get_failure_analysis(self, method: str, problem_type: str) -> Dict:
        """
        Analyze failure patterns for a method and problem type.
        
        Args:
            method: The reasoning method
            problem_type: Type of mathematical problem
            
        Returns:
            Failure analysis summary
        """
        failures = []
        total_attempts = 0
        
        for entry in self.cache:
            if entry.method_used == method and entry.problem_type == problem_type:
                total_attempts += 1
                if not entry.success:
                    failures.append({
                        'reason': entry.failure_reason,
                        'score': entry.performance_score,
                        'timestamp': entry.timestamp
                    })
        
        failure_rate = len(failures) / total_attempts if total_attempts > 0 else 0
        
        return {
            'total_attempts': total_attempts,
            'failures': len(failures),
            'failure_rate': failure_rate,
            'common_failures': failures[-5:],  # Last 5 failures
            'avg_failure_score': sum(f['score'] for f in failures) / len(failures) if failures else 0
        }


def create_problem_hash(question: str, context: str = "") -> str:
    """Create a hash for a mathematical problem."""
    problem_text = f"{question.strip()}{context.strip()}"
    return hashlib.md5(problem_text.encode()).hexdigest()


class TimeoutError(Exception):
    """Custom timeout exception."""
    pass


def timeout_handler(signum, frame):
    """Signal handler for timeout."""
    raise TimeoutError("Method execution timed out")


def with_timeout(timeout_seconds: int):
    """Decorator to add timeout to function execution."""
    def decorator(func):
        def wrapper(*args, **kwargs):
            # Set up the signal handler
            old_handler = signal.signal(signal.SIGALRM, timeout_handler)
            signal.alarm(timeout_seconds)
            
            try:
                result = func(*args, **kwargs)
                return result
            except TimeoutError:
                logger.warning(f"Function {func.__name__} timed out after {timeout_seconds} seconds")
                return None
            finally:
                # Clean up
                signal.alarm(0)
                signal.signal(signal.SIGALRM, old_handler)
        return wrapper
    return decorator


class MetaPrompting:
    """
    Main Meta Prompting framework that orchestrates all components.
    """
    
    def __init__(self, model_name: str = None, provider: str = None):
        self.classifier = ProblemTypeClassifier()
        self.optimizer = PromptOptimizer(model_name, provider)
        self.memory = ReflectionMemory()
        
        from .config import load_config, get_current_model_name
        config = load_config()
        
        self.provider = provider or config['provider']
        self.model_name = model_name or get_current_model_name(self.provider)
        
        # Available methods in MathCoRL
        self.available_methods = [
            'fpp', 'cot', 'zero_cot', 'auto_cot', 'complex_cot', 
            'comat', 'pot', 'zero_shot', 'pal'
        ]
        
        logger.info(f"Meta Prompting initialized with {self.provider} {self.model_name}")
    
    def solve(self, question: str, context: str = "", show_reasoning: bool = True, 
              enable_optimization: bool = True, method_timeout: int = 30) -> Dict[str, Any]:
        """
        Solve a mathematical problem using meta-prompting approach.
        
        Args:
            question: The mathematical question
            context: Additional context
            show_reasoning: Whether to show reasoning steps
            enable_optimization: Whether to apply prompt optimization
            method_timeout: Timeout in seconds for each method attempt
            
        Returns:
            Dictionary containing the result and meta-information
        """
        problem_hash = create_problem_hash(question, context)
        
        if show_reasoning:
            print(f"🧠 Starting Meta Prompting analysis for: {question}\n")
        
        # Step 1: Classify problem type
        problem_type, confidence = self.classifier.classify_problem(question)
        if show_reasoning:
            print(f"📊 Problem Classification: {problem_type} (confidence: {confidence:.2f})")
        
        # Step 2: Get method recommendations (limit to 2 for faster testing)
        recommended_methods = self.classifier.recommend_methods(problem_type, top_k=2)
        if show_reasoning:
            print(f"🎯 Recommended Methods: {', '.join(recommended_methods)}")
        
        # Step 3: Try methods in order of recommendation
        best_result = None
        best_score = 0.0
        
        for method in recommended_methods:
            if show_reasoning:
                print(f"\n🔍 Trying method: {method.upper()}")
            
            try:
                # Get performance history for this method and problem type
                performance_history = self.memory.get_performance_history(method, problem_type)
                
                # Solve using the method with timeout
                result = self._solve_with_method_timeout(method, question, context, 
                                                       show_reasoning=False, timeout=method_timeout)
                
                if result and result.get('success', False):
                    # Try to extract a better answer from reasoning
                    better_answer = self._extract_better_answer(result)
                    if better_answer is not None:
                        result['answer'] = better_answer
                    
                    # Calculate performance score
                    performance_score = self._calculate_performance_score(result)
                    
                    if show_reasoning:
                        print(f"✅ {method.upper()} succeeded with score: {performance_score:.2f}")
                    
                    # Store in memory
                    self.memory.add_entry(
                        problem_hash=problem_hash,
                        problem_type=problem_type,
                        method_used=method,
                        prompt_used=result.get('prompt_used', ''),
                        success=True,
                        performance_score=performance_score,
                        optimization_applied=False
                    )
                    
                    if performance_score > best_score:
                        best_result = result
                        best_score = performance_score
                        best_result['meta_info'] = {
                            'problem_type': problem_type,
                            'classification_confidence': confidence,
                            'method_selected': method,
                            'performance_score': performance_score,
                            'optimization_applied': False
                        }
                        
                        # If score is high enough, return immediately
                        if performance_score >= 0.9:
                            break
                
                else:
                    if show_reasoning:
                        print(f"❌ {method.upper()} failed")
                    
                    # Store failure in memory
                    self.memory.add_entry(
                        problem_hash=problem_hash,
                        problem_type=problem_type,
                        method_used=method,
                        prompt_used=result.get('prompt_used', '') if result else '',
                        success=False,
                        performance_score=0.0,
                        failure_reason=result.get('error', 'Unknown error') if result else 'Method failed',
                        optimization_applied=False
                    )
                    
            except Exception as e:
                logger.error(f"Error trying method {method}: {e}")
                if show_reasoning:
                    print(f"❌ {method.upper()} error: {e}")
        
        if best_result:
            if show_reasoning:
                print(f"\n🎯 Best Result: {best_result['meta_info']['method_selected'].upper()} "
                     f"(score: {best_score:.2f})")
                if best_result.get('reasoning'):
                    print(f"\n💭 Reasoning:\n{best_result['reasoning']}")
                print(f"\n📊 Final Answer: {best_result.get('answer', 'N/A')}")
            
            return best_result
        else:
            # All methods failed
            if show_reasoning:
                print("❌ All recommended methods failed")
            
            return {
                'question': question,
                'context': context,
                'answer': None,
                'success': False,
                'method': 'meta_prompting',
                'meta_info': {
                    'problem_type': problem_type,
                    'classification_confidence': confidence,
                    'all_methods_failed': True
                },
                'error': 'All recommended methods failed'
            }
    
    def _solve_with_method_timeout(self, method: str, question: str, context: str, 
                                  show_reasoning: bool = False, timeout: int = 30) -> Dict[str, Any]:
        """Solve using a specific method with timeout."""
        @with_timeout(timeout)
        def solve_method():
            return self._solve_with_method(method, question, context, show_reasoning)
        
        result = solve_method()
        if result is None:
            return {
                'success': False,
                'error': f'Method {method} timed out after {timeout} seconds',
                'method': method
            }
        return result
    
    def _solve_with_method(self, method: str, question: str, context: str, show_reasoning: bool = False) -> Dict[str, Any]:
        """Solve using a specific method."""
        try:
            if method == 'fpp':
                from .core import FunctionPrototypePrompting
                solver = FunctionPrototypePrompting(provider=self.provider)
                return solver.solve_detailed(question, context)
            
            elif method == 'cot':
                from .cot import ChainOfThoughtPrompting
                solver = ChainOfThoughtPrompting(provider=self.provider)
                return solver.solve(question, context, show_reasoning=show_reasoning)
            
            elif method == 'zero_cot':
                from .zero_cot import ZeroShotCoTPrompting
                solver = ZeroShotCoTPrompting(provider=self.provider)
                return solver.solve(question, context, show_reasoning=show_reasoning)
            
            elif method == 'auto_cot':
                from .auto_cot import AutoCoTPrompting
                solver = AutoCoTPrompting(provider=self.provider)
                return solver.solve(question, context, show_reasoning=show_reasoning, generate_new_examples=False)
            
            elif method == 'complex_cot':
                from .complex_cot import ComplexCoTPrompting
                solver = ComplexCoTPrompting(provider=self.provider)
                return solver.solve(question, context, show_reasoning=show_reasoning)
            
            elif method == 'comat':
                from .comat import CoMATPrompting
                solver = CoMATPrompting(provider=self.provider)
                return solver.solve(question, context, show_reasoning=show_reasoning)
            
            elif method == 'pot':
                from .pot import ProgramOfThoughtsPrompting
                solver = ProgramOfThoughtsPrompting(provider=self.provider)
                return solver.solve(question, context, show_reasoning=show_reasoning)
            
            elif method == 'zero_shot':
                from .zero_shot import ZeroShotPrompting
                solver = ZeroShotPrompting(provider=self.provider)
                return solver.solve(question, context, show_reasoning=show_reasoning)
            
            elif method == 'pal':
                from .pal import ProgramAidedLanguageModel
                solver = ProgramAidedLanguageModel(provider=self.provider)
                return solver.solve(question, context, show_reasoning=show_reasoning)
            
            else:
                raise ValueError(f"Unknown method: {method}")
                
        except Exception as e:
            logger.error(f"Error solving with method {method}: {e}")
            return {
                'success': False,
                'error': str(e),
                'method': method
            }
    
    def _calculate_performance_score(self, result: Dict[str, Any]) -> float:
        """Calculate a performance score for a result."""
        score = 0.0
        
        # Base score for success
        if result.get('success', False):
            score += 0.6
        
        # Bonus for having an answer
        if result.get('answer') is not None:
            score += 0.2
        
        # Bonus for having reasoning
        if result.get('reasoning'):
            score += 0.1
            
        # Bonus for detailed reasoning (longer explanations)
        reasoning_length = len(result.get('reasoning', ''))
        if reasoning_length > 200:
            score += 0.1
        
        return min(score, 1.0)
    
    def _extract_better_answer(self, result: Dict[str, Any]) -> float:
        """
        Extract a better numerical answer from result with reasoning.
        
        This function improves on individual method extraction by looking for
        final answers more intelligently.
        """
        answer = result.get('answer')
        reasoning = result.get('reasoning', '')
        
        if answer is not None and reasoning:
            # Try to extract a better answer from reasoning
            try:
                # Look for boxed answers first (highest priority)
                boxed_pattern = r'\\boxed\{([+-]?\d*\.?\d+)\}'
                boxed_matches = re.findall(boxed_pattern, reasoning)
                if boxed_matches:
                    return float(boxed_matches[-1])
                
                # Look for dollar amounts first (high priority for financial problems)
                dollar_patterns = [
                    r'\$\s*([+-]?\d+(?:\.\d+)?)\s*(?:million|billion|thousand)?',
                    r'([+-]?\d+(?:\.\d+)?)\s*(?:million|billion|thousand)?\s*dollars?',
                ]
                
                for pattern in dollar_patterns:
                    matches = re.findall(pattern, reasoning, re.IGNORECASE)
                    if matches:
                        # Return the last (most likely final answer) dollar amount
                        return float(matches[-1])
                
                # Look for "Answer:" or "Final answer:" patterns but be more specific
                answer_patterns = [
                    r'(?:final\s+)?answer[:\s]+.*?(?:is|of)\s*\$?\s*([+-]?\d*\.?\d+)',
                    r'(?:the\s+)?(?:final\s+)?(?:answer|result)\s+(?:is|=)\s*\$?\s*([+-]?\d*\.?\d+)',
                    r'(?:final\s+)?answer[:\s]+([+-]?\d*\.?\d+)',
                ]
                
                reasoning_lower = reasoning.lower()
                for pattern in answer_patterns:
                    matches = re.findall(pattern, reasoning_lower, re.IGNORECASE)
                    if matches:
                        # Filter out years (numbers > 1900 and < 2100)
                        for match in reversed(matches):
                            try:
                                num = float(match)
                                if not (1900 < num < 2100):  # Not a year
                                    return num
                            except ValueError:
                                continue
                
                # Look for equations at the end of reasoning
                lines = reasoning.strip().split('\n')
                for line in reversed(lines[-3:]):  # Check last 3 lines
                    # Look for standalone numbers or equations
                    if '=' in line:
                        eq_matches = re.findall(r'=\s*\$?\s*([+-]?\d*\.?\d+)', line)
                        if eq_matches:
                            return float(eq_matches[-1])
                    
                    # Look for numbers in parentheses or bold
                    paren_matches = re.findall(r'\(([+-]?\d*\.?\d+)\)', line)
                    if paren_matches:
                        return float(paren_matches[-1])
                    
                    bold_matches = re.findall(r'\*\*([+-]?\d*\.?\d+)\*\*', line)
                    if bold_matches:
                        return float(bold_matches[-1])
                
                # If all else fails, return the original answer
                return float(answer) if answer is not None else None
                
            except (ValueError, TypeError):
                # Fall back to original answer
                pass
        
        return float(answer) if answer is not None else None
    
    def solve_problem(self, problem: Union[Dict, str], context: str = "") -> Any:
        """
        Solve a problem dictionary (e.g., from datasets).
        
        Args:
            problem: Problem dictionary or string
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
            return result.get('answer')
            
        except Exception as e:
            logger.error(f"Error in solve_problem: {e}")
            return None


def solve_math_problem_meta(question: str, context: str = "", provider: str = None, **kwargs) -> Optional[float]:
    """
    Simple function to solve a math problem using meta prompting.
    
    Args:
        question: Mathematical question to solve
        context: Optional context information
        provider: LLM provider ('openai', 'claude', optional)
        **kwargs: Additional arguments for MetaPrompting initialization
        
    Returns:
        Numerical result or None if solving failed
    """
    try:
        meta_prompting = MetaPrompting(provider=provider, **kwargs)
        result = meta_prompting.solve(question, context, show_reasoning=False)
        return result.get('answer')
    except Exception as e:
        logger.error(f"Error in solve_math_problem_meta: {e}")
        return None