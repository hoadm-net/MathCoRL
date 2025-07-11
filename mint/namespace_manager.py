"""
Namespace Manager for Enhanced Functions in Ablation Studies.
Handles loading and execution of different function namespaces.
"""

import importlib
from typing import Dict, Any, Optional
from contextlib import contextmanager
import logging

logger = logging.getLogger(__name__)


class NamespaceManager:
    """Manages function namespaces for ablation studies."""
    
    def __init__(self, enhanced_functions_module: str = "mint.enhanced_functions"):
        """Initialize namespace manager.
        
        Args:
            enhanced_functions_module: Module containing enhanced functions
        """
        self.enhanced_functions_module = enhanced_functions_module
        self._enhanced_namespace_cache = None
    
    def get_enhanced_namespace(self) -> Dict[str, Any]:
        """Get namespace with enhanced functions.
        
        Returns:
            Dictionary mapping function names to function objects
        """
        if self._enhanced_namespace_cache is None:
            self._enhanced_namespace_cache = self._load_enhanced_namespace()
        return self._enhanced_namespace_cache
    
    def _load_enhanced_namespace(self) -> Dict[str, Any]:
        """Load enhanced functions into namespace."""
        try:
            enhanced_functions = importlib.import_module(self.enhanced_functions_module)
            namespace = {}
            
            for name in dir(enhanced_functions):
                if not name.startswith('_'):
                    attr = getattr(enhanced_functions, name)
                    if callable(attr):
                        namespace[name] = attr
            
            logger.info(f"Loaded {len(namespace)} enhanced functions")
            return namespace
            
        except ImportError as e:
            logger.error(f"Failed to import enhanced functions module: {e}")
            return {}
    
    @contextmanager
    def enhanced_execution_context(self, solver, custom_prototypes: str):
        """Context manager for enhanced function execution.
        
        Args:
            solver: The solver object to modify
            custom_prototypes: Custom function prototypes to use
            
        Yields:
            Namespace dictionary for enhanced functions
        """
        from mint import prompts
        
        # Store original function
        original_load_function_prototypes = prompts.load_function_prototypes
        
        # Create custom loader
        def custom_load_function_prototypes():
            return custom_prototypes
        
        try:
            # Replace with custom loader
            prompts.load_function_prototypes = custom_load_function_prototypes
            
            # Yield enhanced namespace
            yield self.get_enhanced_namespace()
            
        finally:
            # Restore original function
            prompts.load_function_prototypes = original_load_function_prototypes
    
    def execute_with_enhanced_namespace(self, code: str) -> tuple[Any, Optional[str]]:
        """Execute code with enhanced functions namespace.
        
        Args:
            code: Python code to execute
            
        Returns:
            Tuple of (result, error_message)
        """
        from mint.utils import execute_code_with_namespace
        
        namespace = self.get_enhanced_namespace()
        return execute_code_with_namespace(code, namespace)
    
    def is_enhanced_solver(self, solver) -> bool:
        """Check if solver uses enhanced functions.
        
        Args:
            solver: Solver object to check
            
        Returns:
            True if solver uses enhanced functions
        """
        return (hasattr(solver, '_prototype_type') and 
                solver._prototype_type != 'original')
    
    def clear_cache(self):
        """Clear cached namespace."""
        self._enhanced_namespace_cache = None 