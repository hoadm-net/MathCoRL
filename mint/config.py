"""
Configuration for MINT - Mathematical Intelligence Library.
"""

import os
from pathlib import Path
from dotenv import load_dotenv
from typing import Optional, List
from openai import OpenAI
import logging

# Load environment variables
load_dotenv()

# Project root directory
PROJECT_ROOT = Path(__file__).parent.parent

# Data and template directories
DATA_DIR = PROJECT_ROOT / "datasets"
TPL_DIR = PROJECT_ROOT / "templates"
RESULT_DIR = PROJECT_ROOT / "results"
MODEL_DIR = PROJECT_ROOT / "models"
LOG_DIR = PROJECT_ROOT / "logs"

# Create directories if they don't exist
for directory in [RESULT_DIR, MODEL_DIR, LOG_DIR]:
    directory.mkdir(exist_ok=True)

logger = logging.getLogger(__name__)


def load_config():
    """Load configuration from environment variables."""
    return {
        'openai_api_key': os.getenv('OPENAI_API_KEY'),
        'model': os.getenv('DEFAULT_MODEL', 'gpt-4o-mini'),
        'embedding_model': os.getenv('EMBEDDING_MODEL', 'text-embedding-3-small'),
        'temperature': float(os.getenv('TEMPERATURE', '0.0')),
        'max_tokens': int(os.getenv('MAX_TOKENS', '1000')),
        'langchain_tracing': os.getenv('LANGCHAIN_TRACING_V2', 'false').lower() == 'true',
        'langchain_api_key': os.getenv('LANGCHAIN_API_KEY'),
        'langchain_project': os.getenv('LANGCHAIN_PROJECT', 'MathCoRL-FPP'),
        'log_level': os.getenv('LOG_LEVEL', 'INFO'),
        'debug_mode': os.getenv('DEBUG_MODE', 'false').lower() == 'true',
    }


def create_standardized_embedding(context: Optional[str], question: str, 
                                openai_client: Optional[OpenAI] = None) -> Optional[List[float]]:
    """
    Create standardized embedding for mathematical problems.
    
    This function ensures consistent embedding format across all components:
    - Candidate generation
    - Policy network training  
    - Policy network inference
    
    Args:
        context: Problem context (can be None for datasets like GSM8K)
        question: Problem question
        openai_client: OpenAI client (creates new one if None)
        
    Returns:
        Embedding vector or None if failed
    """
    try:
        # Standardized text format - matches candidate_generator.py logic
        if context and context.strip():
            embedding_text = f"{context.strip()}\n\n{question.strip()}"
        else:
            embedding_text = question.strip()
        
        if not embedding_text:
            logger.warning("Empty text for embedding")
            return None
        
        # Initialize client if not provided
        if openai_client is None:
            config = load_config()
            openai_client = OpenAI(api_key=config['openai_api_key'])
        
        # Create embedding with standardized model
        config = load_config()
        response = openai_client.embeddings.create(
            model=config['embedding_model'],
            input=embedding_text
        )
        
        return response.data[0].embedding
        
    except Exception as e:
        logger.error(f"Standardized embedding creation failed: {e}")
        return None


def get_data_dir():
    """Get datasets directory path."""
    return str(DATA_DIR)


def get_templates_dir():
    """Get templates directory path.""" 
    return str(TPL_DIR)


def get_results_dir():
    """Get results directory path."""
    return str(RESULT_DIR)

