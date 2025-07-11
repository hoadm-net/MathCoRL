"""
Function Registry for Enhanced Functions.
Organizes and categorizes mathematical functions for ablation studies.
"""

import math
from collections import defaultdict
from typing import Dict, List, Callable, Any
import builtins


class FunctionRegistry:
    """Registry for organizing mathematical functions by category."""
    
    def __init__(self):
        """Initialize function registry."""
        self._functions = defaultdict(dict)
        self._register_all_functions()
    
    def _register_all_functions(self):
        """Register all functions by category."""
        self._register_basic_math()
        self._register_financial()
        self._register_table()
        self._register_advanced_math()
    
    def _register_basic_math(self):
        """Register basic mathematical functions."""
        category = 'basic_math'
        
        def add(a: float, b: float) -> float:
            """Add two numbers together."""
            return float(a) + float(b)
        
        def sub(a: float, b: float) -> float:
            """Subtract second number from first number."""
            return float(a) - float(b)
        
        def mul(a: float, b: float) -> float:
            """Multiply two numbers."""
            return float(a) * float(b)
        
        def div(a: float, b: float) -> float:
            """Divide first number by second number."""
            return float(a) / float(b) if b != 0 else float('inf')
        
        def pow_func(a: float, b: float) -> float:
            """Raise first number to the power of second number."""
            return float(a) ** float(b)
        
        def sum_func(numbers: list) -> float:
            """Calculate the sum of a list of numbers."""
            return float(builtins.sum(float(x) for x in numbers))
        
        def mean(numbers: list) -> float:
            """Calculate the arithmetic mean of a list of numbers."""
            if not numbers:
                return 0.0
            return sum_func(numbers) / len(numbers)
        
        def percentage(part: float, whole: float) -> float:
            """Calculate what percentage part is of whole."""
            if whole == 0:
                return 0.0
            return (float(part) / float(whole)) * 100.0
        
        # Register basic math functions
        self._functions[category].update({
            'add': add,
            'sub': sub,
            'mul': mul,
            'div': div,
            'pow': pow_func,  # Avoid conflict with built-in pow
            'sum': sum_func,  # Avoid conflict with built-in sum
            'mean': mean,
            'percentage': percentage
        })
    
    def _register_financial(self):
        """Register financial calculation functions."""
        category = 'financial'
        
        def calculate_growth_rate(initial: float, final: float) -> float:
            """Calculate growth rate percentage."""
            if initial == 0:
                return 0.0
            return ((final - initial) / initial) * 100.0
        
        def calculate_percentage_change(old_value: float, new_value: float) -> float:
            """Calculate percentage change between two values."""
            if old_value == 0:
                return 0.0
            return ((new_value - old_value) / old_value) * 100.0
        
        def calculate_compound_growth_rate(start_value: float, end_value: float, years: float) -> float:
            """Calculate compound annual growth rate (CAGR)."""
            if start_value == 0 or years == 0:
                return 0.0
            return ((end_value / start_value) ** (1 / years) - 1) * 100
        
        def calculate_ratio(numerator: float, denominator: float) -> float:
            """Calculate ratio between two numbers."""
            if denominator == 0:
                return 0.0
            return numerator / denominator
        
        def calculate_net_change(current: float, previous: float) -> float:
            """Calculate net change between two values."""
            return current - previous
        
        self._functions[category].update({
            'calculate_growth_rate': calculate_growth_rate,
            'calculate_percentage_change': calculate_percentage_change,
            'calculate_compound_growth_rate': calculate_compound_growth_rate,
            'calculate_ratio': calculate_ratio,
            'calculate_net_change': calculate_net_change
        })
    
    def _register_table(self):
        """Register table processing functions."""
        category = 'table'
        
        def get_cell(table: list, row: int, col: int) -> float:
            """Get value from specific cell in table."""
            try:
                return float(table[row][col])
            except (IndexError, ValueError, TypeError):
                return 0.0
        
        def get_column(table: list, col: int) -> list:
            """Get entire column from table."""
            try:
                return [float(row[col]) for row in table]
            except (IndexError, ValueError, TypeError):
                return []
        
        def sum_column(table: list, col: int) -> float:
            """Calculate sum of values in a column."""
            column_values = get_column(table, col)
            return self._functions['basic_math']['sum'](column_values)
        
        def average_column(table: list, col: int) -> float:
            """Calculate average of values in a column."""
            column_values = get_column(table, col)
            return self._functions['basic_math']['mean'](column_values)
        
        self._functions[category].update({
            'get_cell': get_cell,
            'get_column': get_column,
            'sum_column': sum_column,
            'average_column': average_column
        })
    
    def _register_advanced_math(self):
        """Register advanced mathematical functions."""
        category = 'advanced_math'
        
        def min_func(*args: float) -> float:
            """Find the minimum value among given numbers."""
            return float(builtins.min(args))
        
        def max_func(*args: float) -> float:
            """Find the maximum value among given numbers."""
            return float(builtins.max(args))
        
        def floor_func(a: float) -> int:
            """Round down to nearest integer."""
            return math.floor(float(a))
        
        def ceil_func(a: float) -> int:
            """Round up to nearest integer."""
            return math.ceil(float(a))
        
        def median(numbers: list) -> float:
            """Calculate the median of a list of numbers."""
            if not numbers:
                return 0.0
            sorted_nums = sorted([float(x) for x in numbers])
            n = len(sorted_nums)
            if n % 2 == 0:
                return (sorted_nums[n//2 - 1] + sorted_nums[n//2]) / 2.0
            else:
                return sorted_nums[n//2]
        
        self._functions[category].update({
            'min': min_func,  # Avoid conflict with built-in min
            'max': max_func,  # Avoid conflict with built-in max
            'floor': floor_func,
            'ceil': ceil_func,
            'median': median
        })
    
    def get_functions_by_category(self, category: str) -> Dict[str, Callable]:
        """Get all functions in a specific category.
        
        Args:
            category: Function category name
            
        Returns:
            Dictionary of function name to function object
        """
        return self._functions.get(category, {})
    
    def get_all_functions(self) -> Dict[str, Callable]:
        """Get all registered functions.
        
        Returns:
            Dictionary of all function name to function object
        """
        all_functions = {}
        for category_functions in self._functions.values():
            all_functions.update(category_functions)
        return all_functions
    
    def get_function_names_by_category(self) -> Dict[str, List[str]]:
        """Get function names organized by category.
        
        Returns:
            Dictionary mapping category to list of function names
        """
        return {
            category: list(functions.keys()) 
            for category, functions in self._functions.items()
        }
    
    def create_namespace(self, categories: List[str] = None) -> Dict[str, Any]:
        """Create namespace with functions from specified categories.
        
        Args:
            categories: List of categories to include. If None, include all.
            
        Returns:
            Namespace dictionary
        """
        if categories is None:
            return self.get_all_functions()
        
        namespace = {}
        for category in categories:
            namespace.update(self.get_functions_by_category(category))
        return namespace 