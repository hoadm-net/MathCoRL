"""
Prompt management for Function Prototype Prompting using LangChain templates.

This module handles loading and formatting prompts and function prototypes.
"""

import os
from pathlib import Path
from typing import Dict, Optional, List
from langchain.prompts import PromptTemplate


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


def load_template(template_name: str) -> str:
    """Load template content from file."""
    templates_dir = get_templates_dir()
    template_path = os.path.join(templates_dir, f"{template_name}.txt")
    
    try:
        with open(template_path, 'r', encoding='utf-8') as f:
            return f.read()
    except FileNotFoundError:
        raise FileNotFoundError(f"Template file not found: {template_path}")


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
    fpp_template_content = load_template("fpp")
    
    # Create LangChain template
    template = PromptTemplate(
        template=fpp_template_content,
        input_variables=["function_prototypes"]
    )
    
    # Format FPP template with function prototypes
    prompt = template.format(function_prototypes=function_prototypes)
    
    # Add context if provided
    if context:
        prompt += f"\n\n# CONTEXT\n{context}\n"
    
    # Add problem
    prompt += f"\n\n# PROBLEM\n{question}\n"
    prompt += "\n# SOLUTION\nGenerate Python code step by step:"
    
    return prompt


def create_fpp_with_examples_prompt(question: str, examples: List[Dict], context: str = "") -> str:
    """
    Create FPP prompt with few-shot examples.
    
    Args:
        question: The mathematical question to solve
        examples: List of example dicts with 'question' and 'code' keys
        context: Optional context information
        
    Returns:
        Formatted prompt string
    """
    # Load templates
    function_prototypes = load_function_prototypes()
    template_content = load_template("fpp_with_examples")
    
    # Format examples
    examples_str = ""
    for idx, ex in enumerate(examples):
        examples_str += f"""
# EXAMPLE {idx + 1}
Question: {ex['question']}
{ex['code']}

"""
    
    # Format context section
    context_section = ""
    if context:
        context_text = context
        if isinstance(context_text, list):
            context_text = ' '.join(str(item) for item in context_text if str(item).strip() and str(item) != '.')
        context_section = f"Context: {context_text}\n"
    
    # Create LangChain template
    template = PromptTemplate(
        template=template_content,
        input_variables=["function_prototypes", "examples", "context_section", "question"]
    )
    
    # Format template
    prompt = template.format(
        function_prototypes=function_prototypes,
        examples=examples_str.strip(),
        context_section=context_section,
        question=question
    )
    
    return prompt


def create_code_generation_prompt(question: str, answer: float, explanation: str, context: str = "") -> str:
    """
    Create prompt for code generation from problem + answer + explanation.
    
    Args:
        question: The mathematical question to solve
        answer: Ground truth answer
        explanation: Explanation/equation
        context: Optional context information
        
    Returns:
        Formatted prompt string
    """
    # Load templates
    function_prototypes = load_function_prototypes()
    template_content = load_template("code_generation")
    
    # Format context section
    context_section = ""
    if context:
        context_section = f"Context: {context}\n"
    
    # Create LangChain template
    template = PromptTemplate(
        template=template_content,
        input_variables=["context_section", "question", "answer", "explanation", "function_prototypes"]
    )
    
    # Format template
    prompt = template.format(
        context_section=context_section,
        question=question,
        answer=answer,
        explanation=explanation,
        function_prototypes=function_prototypes
    )
    
    return prompt


def create_policy_evaluation_prompt(question: str, examples: List[Dict], context: str = "") -> str:
    """
    Create prompt for policy network evaluation with selected examples.
    
    Args:
        question: The mathematical question to solve
        examples: List of example dicts selected by policy network
        context: Optional context information
        
    Returns:
        Formatted prompt string
    """
    # Load templates
    function_prototypes = load_function_prototypes()
    template_content = load_template("policy_evaluation")
    
    # Format examples section
    examples_section = ""
    for idx, ex in enumerate(examples):
        examples_section += f"""
# EXAMPLE {idx + 1}
Question: {ex['question']}
{ex['code']}

"""
    
    # Format problem section  
    problem_section = ""
    if context:
        context_text = context
        if isinstance(context_text, list):
            context_text = ' '.join(str(item) for item in context_text if str(item).strip() and str(item) != '.')
        problem_section = f"Context: {context_text}\nQuestion: {question}"
    else:
        problem_section = f"Question: {question}"
    
    # Create LangChain template
    template = PromptTemplate(
        template=template_content,
        input_variables=["function_prototypes", "examples_section", "problem_section"]
    )
    
    # Format template
    prompt = template.format(
        function_prototypes=function_prototypes,
        examples_section=examples_section.strip(),
        problem_section=problem_section
    )
    
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