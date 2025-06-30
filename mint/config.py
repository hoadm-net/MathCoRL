"""
Configuration for MINT - Mathematical Intelligence Library.
"""

import os
from pathlib import Path
from dotenv import load_dotenv

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


def load_config():
    """Load configuration from environment variables."""
    return {
        'openai_api_key': os.getenv('OPENAI_API_KEY'),
        'model': os.getenv('DEFAULT_MODEL', 'gpt-4o-mini'),
        'temperature': float(os.getenv('TEMPERATURE', '0.0')),
        'max_tokens': int(os.getenv('MAX_TOKENS', '1000')),
        'langchain_tracing': os.getenv('LANGCHAIN_TRACING_V2', 'false').lower() == 'true',
        'langchain_api_key': os.getenv('LANGCHAIN_API_KEY'),
        'langchain_project': os.getenv('LANGCHAIN_PROJECT', 'MathCoRL-FPP'),
        'log_level': os.getenv('LOG_LEVEL', 'INFO'),
        'debug_mode': os.getenv('DEBUG_MODE', 'false').lower() == 'true',
    }


def get_data_dir():
    """Get datasets directory path."""
    return str(DATA_DIR)


def get_templates_dir():
    """Get templates directory path.""" 
    return str(TPL_DIR)


def get_results_dir():
    """Get results directory path."""
    return str(RESULT_DIR)

