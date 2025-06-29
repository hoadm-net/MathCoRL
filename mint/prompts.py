"""
Prompt management for Function Prototype Prompting.

This module handles loading and formatting prompts and function prototypes.
"""

import os
from pathlib import Path
from typing import Dict, Optional


def get_templates_dir() -> str:
    """Get the templates directory path."""
    current_dir = Path(__file__).parent.parent
    return os.path.join(current_dir, "templates")


def load_function_prototypes() -> str:
    """Load function prototypes from template file."""
    templates_dir = get_templates_dir()
    prototypes_path = os.path.join(templates_dir, "function_prototypes.txt")
    
    try:
        with open(prototypes_path, 'r', encoding='utf-8') as f:
            return f.read()
    except FileNotFoundError:
        raise FileNotFoundError(f"Function prototypes file not found: {prototypes_path}")


def load_fpp_template() -> str:
    """Load FPP template from file."""
    templates_dir = get_templates_dir()
    template_path = os.path.join(templates_dir, "fpp.txt")
    
    try:
        with open(template_path, 'r', encoding='utf-8') as f:
            return f.read()
    except FileNotFoundError:
        raise FileNotFoundError(f"FPP template file not found: {template_path}")


def create_fpp_prompt(question: str, context: str = "") -> str:
    """
    Create FPP prompt from question and optional context.
    
    Args:
        question: The mathematical question to solve
        context: Optional context information
        
    Returns:
        Formatted prompt string
    """
    # Load templates
    function_prototypes = load_function_prototypes()
    fpp_template = load_fpp_template()
    
    # Format FPP template with function prototypes
    prompt = fpp_template.format(function_prototypes=function_prototypes)
    
    # Add context if provided
    if context:
        prompt += f"\n\n# CONTEXT\n{context}\n"
    
    # Add problem
    prompt += f"\n\n# PROBLEM\n{question}\n"
    prompt += "\n# SOLUTION\nGenerate Python code step by step:"
    
    return prompt


def create_problem_prompt(problem: Dict, context: str = "") -> str:
    """
    Create prompt for a problem dictionary (e.g., from SVAMP dataset).
    
    Args:
        problem: Problem dictionary with 'Body' and 'Question' keys
        context: Optional context information
        
    Returns:
        Formatted prompt string
    """
    # Create problem text
    problem_text = f"{problem['Body']} {problem['Question']}"
    
    return create_fpp_prompt(problem_text, context) 