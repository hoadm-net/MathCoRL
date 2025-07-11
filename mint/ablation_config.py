"""
Configuration module for ablation studies.
Centralizes all configuration settings and function prototype mappings.
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List
import os


@dataclass
class FunctionPrototypeConfig:
    """Configuration for function prototype files."""
    
    # Function prototype file mappings
    PROTOTYPE_FILES = {
        'original': 'templates/function_prototypes.txt',
        'financial': 'templates/function_prototypes_fin.txt', 
        'table': 'templates/function_prototypes_tbl.txt',
        'all': 'templates/function_prototypes_all.txt'
    }
    
    # Dataset configurations for ablation studies
    DATASET_CONFIGS = {
        'FinQA': {
            'test_file': 'datasets/FinQA/test.json',
            'original_functions': 'original',
            'enhanced_functions': 'financial',
            'ground_truth_field': 'ground_truth',
            'question_field': 'question',
            'context_field': 'context'
        },
        'TabMWP': {
            'test_file': 'datasets/TabMWP/test.json', 
            'original_functions': 'original',
            'enhanced_functions': 'table',
            'ground_truth_field': 'ground_truth',
            'question_field': 'question',
            'context_field': 'context'
        }
    }
    
    @classmethod
    def get_prototype_file_path(cls, prototype_type: str) -> str:
        """Get file path for prototype type."""
        if prototype_type not in cls.PROTOTYPE_FILES:
            raise ValueError(f"Unknown prototype type: {prototype_type}")
        return cls.PROTOTYPE_FILES[prototype_type]
    
    @classmethod
    def validate_prototype_files(cls) -> List[str]:
        """Validate that all prototype files exist."""
        missing_files = []
        for prototype_type, file_path in cls.PROTOTYPE_FILES.items():
            if not os.path.exists(file_path):
                missing_files.append(f"{prototype_type}: {file_path}")
        return missing_files
    
    @classmethod
    def get_dataset_config(cls, dataset: str) -> Dict:
        """Get configuration for specific dataset."""
        if dataset not in cls.DATASET_CONFIGS:
            raise ValueError(f"Unsupported dataset: {dataset}")
        return cls.DATASET_CONFIGS[dataset]


@dataclass 
class AblationStudyConfig:
    """Configuration for ablation study execution."""
    
    # Default model settings
    DEFAULT_MODEL: str = "gpt-4o-mini"
    DEFAULT_TEMPERATURE: float = 0.1
    
    # Execution settings
    DEFAULT_SEED: int = 42
    DEFAULT_SAMPLES: int = 10
    MAX_SAMPLES: int = 1000
    
    # Output settings
    RESULTS_DIR: str = "results"
    LOG_LEVEL: str = "INFO"
    
    # Enhanced functions namespace module
    ENHANCED_FUNCTIONS_MODULE: str = "mint.enhanced_functions"
    
    @classmethod
    def get_model(cls, model: str = None) -> str:
        """Get model name with fallback to environment and default."""
        return model or os.getenv("DEFAULT_MODEL", cls.DEFAULT_MODEL)
    
    @classmethod
    def validate_samples(cls, n_samples: int) -> int:
        """Validate and clamp sample count."""
        if n_samples <= 0:
            raise ValueError("Number of samples must be positive")
        if n_samples > cls.MAX_SAMPLES:
            print(f"Warning: Clamping samples to maximum {cls.MAX_SAMPLES}")
            return cls.MAX_SAMPLES
        return n_samples 