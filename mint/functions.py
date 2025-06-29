"""
Mathematical functions for Function Prototype Prompting.

This module contains all predefined mathematical functions that can be used
in generated code for solving mathematical problems.
"""

import math
from typing import List, Union, Any


def add(a: float, b: float) -> float:
    """Add two numbers together."""
    return a + b


def sub(a: float, b: float) -> float:
    """Subtract second number from first number."""
    return a - b


def mul(a: float, b: float) -> float:
    """Multiply two numbers."""
    return a * b


def div(a: float, b: float) -> float:
    """Divide first number by second number."""
    return a / b if b != 0 else float('inf')


def mod(a: float, b: float) -> float:
    """Get remainder of division."""
    return a % b if b != 0 else 0


def power(a: float, b: float) -> float:
    """Raise first number to the power of second number."""
    return a ** b


# Use Python built-in functions directly
# min, max, abs, round, sum are built-in functions


def floor(a: float) -> int:
    """Round down to nearest integer."""
    return int(a)


def ceil(a: float) -> int:
    """Round up to nearest integer."""
    return int(a) + (1 if a > int(a) else 0)


def mean(numbers: List[float]) -> float:
    """Calculate the arithmetic mean of a list of numbers."""
    return sum(numbers) / len(numbers) if numbers else 0


def median(numbers: List[float]) -> float:
    """Calculate the median of a list of numbers."""
    if not numbers:
        return 0
    sorted_nums = sorted(numbers)
    n = len(sorted_nums)
    if n % 2 == 0:
        return (sorted_nums[n//2 - 1] + sorted_nums[n//2]) / 2
    else:
        return sorted_nums[n//2]


def mode(numbers: List[float]) -> float:
    """Find the most frequently occurring value."""
    if not numbers:
        return 0
    return max(set(numbers), key=numbers.count)


def count(numbers: List[Any]) -> int:
    """Count the number of elements in a list."""
    return len(numbers)


def percentage(part: float, whole: float) -> float:
    """Calculate what percentage part is of whole."""
    return (part / whole) * 100 if whole != 0 else 0


def greater_than(a: float, b: float) -> bool:
    """Check if a is greater than b."""
    return a > b


def less_than(a: float, b: float) -> bool:
    """Check if a is less than b."""
    return a < b


def equal(a: float, b: float) -> bool:
    """Check if a equals b."""
    return abs(a - b) < 1e-9


def gcd(a: int, b: int) -> int:
    """Calculate greatest common divisor."""
    a, b = abs(int(a)), abs(int(b))
    while b:
        a, b = b, a % b
    return a


def lcm(a: int, b: int) -> int:
    """Calculate least common multiple."""
    a, b = abs(int(a)), abs(int(b))
    if a == 0 or b == 0:
        return 0
    return (a * b) // gcd(a, b)


def get_execution_namespace() -> dict:
    """
    Get namespace dictionary with all mathematical functions.
    
    Returns:
        Dictionary mapping function names to functions
    """
    import builtins
    
    return {
        # Basic arithmetic
        'add': add,
        'sub': sub,
        'mul': mul,
        'div': div,
        'mod': mod,
        'pow': power,  # Use custom power function to avoid conflict with built-in pow
        
        # Python built-in functions
        'min': builtins.min,
        'max': builtins.max,
        'abs': builtins.abs,
        'round': builtins.round,
        'sum': builtins.sum,
        'len': builtins.len,
        
        # Custom math functions
        'floor': floor,
        'ceil': ceil,
        
        # List operations  
        'mean': mean,
        'median': median,
        'mode': mode,
        'count': count,
        
        # Utility functions
        'percentage': percentage,
        'greater_than': greater_than,
        'less_than': less_than,
        'equal': equal,
        
        # Math utilities
        'gcd': gcd,
        'lcm': lcm,
    } 