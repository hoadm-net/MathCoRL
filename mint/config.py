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
        'anthropic_api_key': os.getenv('ANTHROPIC_API_KEY'),
        'model': os.getenv('DEFAULT_MODEL', 'gpt-4o-mini'),
        'anthropic_model': os.getenv('ANT_DEFAULT_MODEL', 'claude-3-5-sonnet-20241022'),
        'embedding_model': os.getenv('EMBEDDING_MODEL', 'text-embedding-3-small'),
        'temperature': float(os.getenv('TEMPERATURE', '0.0')),
        'max_tokens': int(os.getenv('MAX_TOKENS', '1000')),
        'provider': os.getenv('LLM_PROVIDER', 'openai').lower(),  # 'openai' or 'claude'
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


def create_llm_client(provider: str = None, model: str = None, **kwargs):
    """
    Create LLM client based on provider selection.
    
    Args:
        provider: 'openai' or 'claude' (defaults to config)
        model: Model name (defaults to config based on provider)
        **kwargs: Additional arguments for LLM initialization
        
    Returns:
        LLM client (LangChain compatible)
    """
    config = load_config()
    
    # Use provided provider or default from config
    provider = provider or config['provider']
    
    if provider == 'claude' or provider == 'anthropic':
        try:
            from langchain_anthropic import ChatAnthropic
            
            # Use provided model or default Claude model
            model = model or config['anthropic_model']
            
            # Get API key
            api_key = kwargs.get('api_key') or config['anthropic_api_key']
            if not api_key:
                raise ValueError("Anthropic API key not found. Set ANTHROPIC_API_KEY environment variable.")
            
            return ChatAnthropic(
                api_key=api_key,
                model=model,
                temperature=kwargs.get('temperature', config['temperature']),
                max_tokens=kwargs.get('max_tokens', config['max_tokens'])
            )
            
        except ImportError:
            logger.error("langchain_anthropic not installed. Install with: pip install langchain-anthropic")
            raise
            
    else:  # Default to OpenAI
        from langchain_openai import ChatOpenAI
        
        # Use provided model or default OpenAI model
        model = model or config['model']
        
        # Get API key
        api_key = kwargs.get('api_key') or config['openai_api_key']
        if not api_key:
            raise ValueError("OpenAI API key not found. Set OPENAI_API_KEY environment variable.")
        
        return ChatOpenAI(
            api_key=api_key,
            model=model,
            temperature=kwargs.get('temperature', config['temperature']),
            max_tokens=kwargs.get('max_tokens', config['max_tokens'])
        )


def get_current_model_name(provider: str = None):
    """
    Get the current model name based on provider.
    
    Args:
        provider: 'openai' or 'claude' (defaults to config)
        
    Returns:
        Model name string
    """
    config = load_config()
    provider = provider or config['provider']
    
    if provider == 'claude' or provider == 'anthropic':
        return config['anthropic_model']
    else:
        return config['model']

