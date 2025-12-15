"""
Reproducibility utilities for MathCoRL.

Ensures consistent results across runs by setting seeds for all random number generators.
"""

import os
import random
import warnings
from typing import Optional

import numpy as np


def set_seed(seed: int = 42, deterministic: bool = True) -> None:
    """
    Set random seeds for reproducibility across all libraries.
    
    Args:
        seed: Random seed to use (default: 42)
        deterministic: If True, enforces deterministic algorithms in PyTorch
                      (may impact performance)
    
    Note:
        This sets seeds for:
        - Python's built-in random module
        - NumPy
        - PyTorch (if available)
        - Transformers (if available)
        - CUDA operations (if available)
    """
    # Set Python random seed
    random.seed(seed)
    
    # Set NumPy seed
    np.random.seed(seed)
    
    # Set environment variable for Python hash seed
    os.environ['PYTHONHASHSEED'] = str(seed)
    
    # Set PyTorch seeds if available
    try:
        import torch
        torch.manual_seed(seed)
        
        if torch.cuda.is_available():
            torch.cuda.manual_seed(seed)
            torch.cuda.manual_seed_all(seed)  # For multi-GPU
            
        if deterministic:
            # Make PyTorch operations deterministic
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
            
            # Set environment variables for additional determinism
            os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'
            
            # Enable deterministic algorithms (PyTorch 1.8+)
            if hasattr(torch, 'use_deterministic_algorithms'):
                try:
                    torch.use_deterministic_algorithms(True)
                except RuntimeError as e:
                    warnings.warn(
                        f"Could not enable deterministic algorithms: {e}. "
                        "Some operations may still be non-deterministic."
                    )
    except ImportError:
        pass  # PyTorch not installed
    
    # Set Transformers seed if available
    try:
        import transformers
        transformers.set_seed(seed)
    except ImportError:
        pass  # Transformers not installed


def get_seed_from_args(args) -> int:
    """
    Extract seed from argparse arguments with fallback.
    
    Args:
        args: Argparse namespace object
        
    Returns:
        Seed value (from args.seed or default 42)
    """
    return getattr(args, 'seed', 42)


def add_seed_argument(parser) -> None:
    """
    Add --seed argument to argparse parser.
    
    Args:
        parser: ArgumentParser instance
    """
    parser.add_argument(
        '--seed',
        type=int,
        default=42,
        help='Random seed for reproducibility (default: 42)'
    )


def seed_context(seed: int = 42, deterministic: bool = True):
    """
    Context manager for temporary seed setting.
    
    Usage:
        with seed_context(123):
            # All random operations here use seed 123
            result = some_random_function()
        # Original random state restored after context
    
    Args:
        seed: Random seed to use
        deterministic: Whether to enforce deterministic PyTorch operations
    """
    # Save current random states
    python_state = random.getstate()
    numpy_state = np.random.get_state()
    
    torch_state = None
    torch_cuda_state = None
    torch_deterministic = None
    torch_benchmark = None
    
    try:
        import torch
        torch_state = torch.get_rng_state()
        if torch.cuda.is_available():
            torch_cuda_state = torch.cuda.get_rng_state_all()
        torch_deterministic = torch.backends.cudnn.deterministic
        torch_benchmark = torch.backends.cudnn.benchmark
    except ImportError:
        pass
    
    try:
        # Set new seed
        set_seed(seed, deterministic)
        yield
    finally:
        # Restore original states
        random.setstate(python_state)
        np.random.set_state(numpy_state)
        
        if torch_state is not None:
            import torch
            torch.set_rng_state(torch_state)
            if torch_cuda_state is not None and torch.cuda.is_available():
                torch.cuda.set_rng_state_all(torch_cuda_state)
            torch.backends.cudnn.deterministic = torch_deterministic
            torch.backends.cudnn.benchmark = torch_benchmark


def is_reproducible() -> dict:
    """
    Check if current environment supports full reproducibility.
    
    Returns:
        Dictionary with reproducibility status for each component
    """
    status = {
        'python_random': True,
        'numpy': True,
        'torch': False,
        'torch_cuda': False,
        'transformers': False,
    }
    
    try:
        import torch
        status['torch'] = True
        status['torch_cuda'] = torch.cuda.is_available()
    except ImportError:
        pass
    
    try:
        import transformers
        status['transformers'] = True
    except ImportError:
        pass
    
    return status


if __name__ == '__main__':
    # Test reproducibility
    print("Testing reproducibility...")
    
    # Check what's available
    status = is_reproducible()
    print("\nComponent availability:")
    for component, available in status.items():
        symbol = "✓" if available else "✗"
        print(f"  {symbol} {component}")
    
    # Test seed setting
    print("\nTesting seed=123:")
    set_seed(123)
    print(f"  Python random: {random.random():.10f}")
    print(f"  NumPy random: {np.random.random():.10f}")
    
    try:
        import torch
        print(f"  PyTorch random: {torch.rand(1).item():.10f}")
    except ImportError:
        print("  PyTorch: Not installed")
    
    print("\nResetting seed=123:")
    set_seed(123)
    print(f"  Python random: {random.random():.10f}")
    print(f"  NumPy random: {np.random.random():.10f}")
    
    try:
        import torch
        print(f"  PyTorch random: {torch.rand(1).item():.10f}")
    except ImportError:
        pass
    
    print("\n✓ Reproducibility test complete!")
