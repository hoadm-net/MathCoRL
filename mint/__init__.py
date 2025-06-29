"""
MINT - Mathematical Intelligence Library

A library for mathematical problem solving using Function Prototype Prompting (FPP)
and Chain-of-Thought (CoT) prompting methods.
"""

from .core import FunctionPrototypePrompting, solve_math_problem
from .cot import ChainOfThoughtPrompting, solve_with_cot
from .functions import get_execution_namespace
from .prompts import create_fpp_prompt, create_problem_prompt, load_function_prototypes, load_fpp_template
from .utils import load_svamp_dataset, clean_code, execute_code, evaluate_result
from .config import load_config

# New unified modules
from .evaluation import (
    is_close, is_close_tatqa, is_close_finqa, get_tolerance_function,
    calculate_accuracy, evaluate_predictions
)
from .testing import TestRunner, DatasetLoader, create_fpp_solver, create_cot_solver

__version__ = "0.1.0"
__author__ = "MathCoRL Team"

__all__ = [
    # Main classes and functions
    "FunctionPrototypePrompting",
    "solve_math_problem",
    "ChainOfThoughtPrompting",
    "solve_with_cot",
    
    # Function utilities
    "get_execution_namespace",
    
    # Prompt utilities
    "create_fpp_prompt",
    "create_problem_prompt", 
    "load_function_prototypes",
    "load_fpp_template",
    
    # Data and execution utilities
    "load_svamp_dataset",
    "clean_code",
    "execute_code",
    "evaluate_result",
    "load_config",
    
    # Evaluation utilities
    "is_close",
    "is_close_tatqa", 
    "is_close_finqa",
    "get_tolerance_function",
    "calculate_accuracy",
    "evaluate_predictions",
    
    # Testing framework
    "TestRunner",
    "DatasetLoader",
    "create_fpp_solver",
    "create_cot_solver"
]
