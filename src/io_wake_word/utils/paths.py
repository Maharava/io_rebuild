"""
Path utilities for Io wake word detection
"""
import os
from pathlib import Path
from typing import Optional, Union

# Base directory for all Io data
IO_DIR_ENV = "IO_WAKE_WORD_DIR"
DEFAULT_IO_DIR = Path.home() / ".io_wake_word"

def get_base_dir() -> Path:
    """Get the base directory for Io data
    
    Returns:
        Path to the base directory
    """
    base_dir = os.environ.get(IO_DIR_ENV)
    if base_dir:
        return Path(base_dir)
    return DEFAULT_IO_DIR

def get_config_dir() -> Path:
    """Get the directory for configuration files
    
    Returns:
        Path to the configuration directory
    """
    config_dir = get_base_dir() / "config"
    config_dir.mkdir(parents=True, exist_ok=True)
    return config_dir

def get_models_dir() -> Path:
    """Get the directory for model files
    
    Returns:
        Path to the models directory
    """
    models_dir = get_base_dir() / "models"
    models_dir.mkdir(parents=True, exist_ok=True)
    return models_dir

def get_data_dir() -> Path:
    """Get the directory for training data
    
    Returns:
        Path to the data directory
    """
    data_dir = get_base_dir() / "training_data"
    data_dir.mkdir(parents=True, exist_ok=True)
    return data_dir

def get_logs_dir() -> Path:
    """Get the directory for log files
    
    Returns:
        Path to the logs directory
    """
    logs_dir = get_base_dir() / "logs"
    logs_dir.mkdir(parents=True, exist_ok=True)
    return logs_dir

def resolve_model_path(model_path_str: Union[str, Path]) -> Path:
    """Resolve a model path string to an actual file path
    
    This function handles both absolute paths and model names.
    If just a filename is provided, it will look in the default models directory.
    
    Args:
        model_path_str: Model path string or filename
        
    Returns:
        Resolved Path object
    """
    # Convert to string if it's already a Path
    if isinstance(model_path_str, Path):
        model_path_str = str(model_path_str)
    
    # Convert to Path
    model_path = Path(model_path_str)
    
    # If the path exists as provided, use it
    if model_path.exists():
        return model_path
    
    # If it's just a filename, look in the default models directory
    if not model_path.is_absolute() and "/" not in str(model_path) and "\\" not in str(model_path):
        default_path = get_models_dir() / model_path
        if default_path.exists():
            print(f"Found model at default location: {default_path}")
            return default_path
    
    # Return the original path (will fail with appropriate error)
    return model_path

def ensure_app_directories() -> None:
    """Create all necessary application directories"""
    # Create all directories
    get_config_dir()
    get_models_dir()
    get_data_dir()
    get_logs_dir()
    
    # Create subdirectories for training data
    wake_word_dir = get_data_dir() / "wake_word"
    wake_word_dir.mkdir(exist_ok=True)
    
    negative_dir = get_data_dir() / "negative"
    negative_dir.mkdir(exist_ok=True)
    
    # Create diagnostics directory
    diagnostics_dir = get_base_dir() / "training_diagnostics"
    diagnostics_dir.mkdir(exist_ok=True)