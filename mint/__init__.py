"""
MINT - Mathematical Intelligence Library

A library for mathematical problem solving using Function Prototype Prompting (FPP).
"""

from .core import FunctionPrototypePrompting, solve_math_problem
from .functions import get_execution_namespace
from .prompts import create_fpp_prompt, create_problem_prompt, load_function_prototypes, load_fpp_template
from .utils import load_svamp_dataset, clean_code, execute_code, evaluate_result

__version__ = "0.1.0"
__author__ = "MathCoRL Team"

__all__ = [
    # Main classes and functions
    "FunctionPrototypePrompting",
    "solve_math_problem",
    
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
]
