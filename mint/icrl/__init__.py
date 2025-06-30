"""
In-Context Reinforcement Learning (ICRL) Module

This module implements the ICRL pipeline for enhancing mathematical problem solving:

Step 1: Candidate Generation
- Generate high-quality training candidates với FPP approach
- Support multiple datasets: SVAMP, GSM8K, TabMWP, TAT-QA, FinQA
- Comprehensive data processing và validation

Step 2: Policy Network Training  
- Train neural policy for intelligent example selection
- Multi-head attention architecture với contrastive learning
- PPO-based optimization với multi-objective reward
- Comprehensive evaluation system (Policy vs Random)

Step 3-5: To be implemented in future versions
"""

# Step 1 components (always available)
from .candidate_generator import CandidateGenerator

# Step 2 components (require PyTorch)
def _lazy_import_step2():
    """Lazy import Step 2 components to avoid PyTorch dependency issues"""
    try:
        from .policy_network import PolicyNetwork, ppo_loss, contrastive_loss
        from .evaluator import PolicyNetworkEvaluator
        from .trainer import PolicyNetworkTrainer
        return {
            'PolicyNetwork': PolicyNetwork,
            'PolicyNetworkEvaluator': PolicyNetworkEvaluator, 
            'PolicyNetworkTrainer': PolicyNetworkTrainer,
            'ppo_loss': ppo_loss,
            'contrastive_loss': contrastive_loss
        }
    except ImportError as e:
        return None

# Try to import Step 2 components
_step2_components = _lazy_import_step2()

# Dynamic __all__ based on available components
__all__ = ['CandidateGenerator']

if _step2_components:
    # Add Step 2 components to module globals
    globals().update(_step2_components)
    __all__.extend([
        'PolicyNetwork',
        'PolicyNetworkEvaluator', 
        'PolicyNetworkTrainer',
        'ppo_loss',
        'contrastive_loss'
    ])

def __getattr__(name):
    """Provide helpful error messages for missing Step 2 components"""
    if name in ['PolicyNetwork', 'PolicyNetworkEvaluator', 'PolicyNetworkTrainer', 'ppo_loss', 'contrastive_loss']:
        if _step2_components is None:
            raise ImportError(
                f"Component '{name}' requires PyTorch. "
                "Please install PyTorch: pip install torch torchvision torchaudio"
            )
        else:
            return _step2_components[name]
    raise AttributeError(f"module '{__name__}' has no attribute '{name}'") 