"""
In-Context Reinforcement Learning (ICRL) Module

This module implements the In-Context RL approach for enhancing FPP
with intelligently selected few-shot examples.

Steps:
1. Generate candidates from training set
2. Build Policy Network  
3. Train Policy Network
4. Use trained PN to select examples
5. Enhance FPP with selected examples
"""

from .candidate_generator import CandidateGenerator

__all__ = [
    "CandidateGenerator",
] 