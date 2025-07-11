"""
Problem Evaluator for Ablation Studies.
Handles evaluation of individual problems with different function sets.
"""

from typing import Dict, Any
import logging

from mint.core import FunctionPrototypePrompting
from mint.evaluation import get_tolerance_function
from mint.namespace_manager import NamespaceManager
from mint.ablation_config import FunctionPrototypeConfig

logger = logging.getLogger(__name__)


class ProblemEvaluator:
    """Evaluates individual problems for ablation studies."""
    
    def __init__(self, namespace_manager: NamespaceManager = None):
        """Initialize problem evaluator.
        
        Args:
            namespace_manager: Manager for function namespaces
        """
        self.namespace_manager = namespace_manager or NamespaceManager()
    
    def evaluate_problem(self, solver: FunctionPrototypePrompting, 
                        problem: Dict, dataset: str) -> Dict[str, Any]:
        """Evaluate a single problem with given solver.
        
        Args:
            solver: The solver to use
            problem: Problem data
            dataset: Dataset name
            
        Returns:
            Evaluation results dictionary
        """
        try:
            # Extract problem data
            problem_data = self._extract_problem_data(problem, dataset)
            
            # Solve the problem
            result = self._solve_problem(solver, problem_data)
            
            # Evaluate correctness
            is_correct = self._evaluate_correctness(result, problem_data, dataset)
            
            return {
                'problem_id': problem.get('id', ''),
                'question': problem_data['question'],
                'correct_answer': problem_data['correct_answer'],
                'predicted_answer': result['result'],
                'is_correct': is_correct,
                'success': result['success'],
                'error': result.get('error', ''),
                'code': result.get('code', '')
            }
            
        except Exception as e:
            logger.error(f"Error evaluating problem: {e}")
            return self._create_error_result(problem, str(e))
    
    def _extract_problem_data(self, problem: Dict, dataset: str) -> Dict[str, Any]:
        """Extract problem data based on dataset configuration.
        
        Args:
            problem: Raw problem data
            dataset: Dataset name
            
        Returns:
            Structured problem data
        """
        config = FunctionPrototypeConfig.get_dataset_config(dataset)
        
        return {
            'correct_answer': float(problem.get(config['ground_truth_field'], 0)),
            'question': problem.get(config['question_field'], ''),
            'context': problem.get(config['context_field'], '')
        }
    
    def _solve_problem(self, solver: FunctionPrototypePrompting, 
                      problem_data: Dict[str, Any]) -> Dict[str, Any]:
        """Solve a problem using the given solver.
        
        Args:
            solver: Solver instance
            problem_data: Problem data
            
        Returns:
            Solution result
        """
        question_text = problem_data['question']
        context = problem_data['context']
        
        # Check if using enhanced functions
        if self.namespace_manager.is_enhanced_solver(solver):
            return self._solve_with_enhanced_functions(solver, question_text, context)
        else:
            # Use default solve method for original functions
            return solver.solve_detailed(question_text, context)
    
    def _solve_with_enhanced_functions(self, solver: FunctionPrototypePrompting,
                                     question_text: str, context: str) -> Dict[str, Any]:
        """Solve problem using enhanced functions namespace.
        
        Args:
            solver: Solver instance
            question_text: Question text
            context: Problem context
            
        Returns:
            Solution result
        """
        custom_prototypes = solver._custom_prototypes
        
        with self.namespace_manager.enhanced_execution_context(solver, custom_prototypes) as namespace:
            # Get the generated code
            result = solver.solve_detailed(question_text, context)
            
            # Re-execute with enhanced namespace if we have code
            if result['code']:
                from mint.utils import clean_code
                cleaned_code = clean_code(result['code'])
                result_value, error = self.namespace_manager.execute_with_enhanced_namespace(cleaned_code)
                
                result = {
                    'result': result_value,
                    'success': not error,  # True if error is empty/None
                    'error': error or '',
                    'code': cleaned_code
                }
        
        return result
    
    def _evaluate_correctness(self, result: Dict[str, Any], 
                            problem_data: Dict[str, Any], dataset: str) -> bool:
        """Evaluate if the result is correct.
        
        Args:
            result: Solver result
            problem_data: Problem data
            dataset: Dataset name
            
        Returns:
            True if correct, False otherwise
        """
        if not result['success'] or result['result'] is None:
            return False
        
        correct_answer = problem_data['correct_answer']
        predicted_answer = result['result']
        
        if dataset == 'FinQA':
            # Use FinQA specific evaluation
            from mint.utils import FinQA_generate_candidates
            candidates = FinQA_generate_candidates(predicted_answer)
            return correct_answer in candidates
        else:
            # Use standard tolerance function
            tolerance_fn = get_tolerance_function(dataset)
            return tolerance_fn(predicted_answer, correct_answer)
    
    def _create_error_result(self, problem: Dict, error_msg: str) -> Dict[str, Any]:
        """Create error result dictionary.
        
        Args:
            problem: Problem data
            error_msg: Error message
            
        Returns:
            Error result dictionary
        """
        return {
            'problem_id': problem.get('id', ''),
            'question': problem.get('question', ''),
            'correct_answer': None,
            'predicted_answer': None,
            'is_correct': False,
            'success': False,
            'error': error_msg,
            'code': ''
        } 