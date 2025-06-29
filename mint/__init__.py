"""
MINT - Mathematical Intelligence Library

A library for mathematical problem solving using Function Prototype Prompting (FPP),
Chain-of-Thought (CoT), Program of Thoughts (PoT), Zero-Shot, and PAL prompting methods.
"""

from .core import FunctionPrototypePrompting, solve_math_problem
from .cot import ChainOfThoughtPrompting, solve_with_cot
from .pot import ProgramOfThoughtsPrompting, solve_with_pot
from .zero_shot import ZeroShotPrompting, solve_with_zero_shot
from .pal import ProgramAidedLanguageModel, solve_with_pal
from .functions import get_execution_namespace
from .prompts import create_fpp_prompt, create_problem_prompt, load_function_prototypes, load_fpp_template
from .utils import load_svamp_dataset, clean_code, execute_code, evaluate_result
from .config import load_config

# New unified modules
from .evaluation import (
    is_close, is_close_tatqa, is_close_finqa, get_tolerance_function,
    calculate_accuracy, evaluate_predictions
)
from .testing import TestRunner, DatasetLoader, create_fpp_solver, create_cot_solver, create_pot_solver, create_zero_shot_solver, create_pal_solver

__version__ = "0.4.0"
__author__ = "MathCoRL Team"

__all__ = [
    # Main classes and functions
    "FunctionPrototypePrompting",
    "solve_math_problem",
    "ChainOfThoughtPrompting",
    "solve_with_cot",
    "ProgramOfThoughtsPrompting",
    "solve_with_pot",
    "ZeroShotPrompting",
    "solve_with_zero_shot",
    "ProgramAidedLanguageModel",
    "solve_with_pal",
    
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
    "create_cot_solver",
    "create_pot_solver",
    "create_zero_shot_solver",
    "create_pal_solver"
]
