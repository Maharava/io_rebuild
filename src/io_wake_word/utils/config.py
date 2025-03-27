"""
Configuration utility for Io wake word engine
"""
import json
import logging
import os
from pathlib import Path
from typing import Any, Dict, Optional

from io_wake_word.utils.paths import get_config_dir

logger = logging.getLogger("io_wake_word.utils")

class Config:
    """Config manager for Io wake word engine"""
    
    @staticmethod
    def get_path() -> Path:
        """Get path to configuration file
        
        Returns:
            Path to the configuration file
        """
        config_dir = get_config_dir()
        return config_dir / "config.json"
    
    @staticmethod
    def load() -> Dict[str, Any]:
        """Load configuration from file
        
        Returns:
            Configuration dictionary
        """
        config_file = Config.get_path()
        
        try:
            if config_file.exists():
                with open(config_file, 'r') as f:
                    config = json.load(f)
                logger.info(f"Configuration loaded from {config_file}")
                return Config.validate(config)
            else:
                logger.warning(f"Configuration file not found: {config_file}")
                return Config.create_default()
        except json.JSONDecodeError as e:
            logger.error(f"Invalid JSON in config file: {e}")
            return Config.create_default()
        except Exception as e:
            logger.error(f"Error loading configuration: {e}")
            return Config.create_default()
    
    @staticmethod
    def save(config: Dict[str, Any]) -> bool:
        """Save configuration to file
        
        Args:
            config: Configuration dictionary
            
        Returns:
            True if saved successfully, False otherwise
        """
        config_file = Config.get_path()
        
        try:
            # Validate before saving
            config = Config.validate(config)
            
            # Create parent directory if needed
            config_file.parent.mkdir(parents=True, exist_ok=True)
            
            # Write configuration atomically
            temp_file = config_file.with_suffix('.tmp')
            with open(temp_file, 'w') as f:
                json.dump(config, f, indent=4)
            
            # On Windows, we need to remove the target file first
            if os.name == 'nt' and config_file.exists():
                os.remove(config_file)
                
            # Rename temp file to actual config file
            os.rename(temp_file, config_file)
            
            logger.info(f"Configuration saved to {config_file}")
            return True
        except Exception as e:
            logger.error(f"Error saving configuration: {e}")
            return False
    
    @staticmethod
    def create_default() -> Dict[str, Any]:
        """Create default configuration
        
        Returns:
            Default configuration dictionary
        """
        default_config = {
            "audio_device": None,  # Will be selected on first run
            "sample_rate": 16000,
            "frame_size": 512,
            "model_path": None,    # Will be selected on first run
            "threshold": 0.85,
            "debounce_time": 3.0,  # seconds
            "action": {
                "type": "notification",
                "params": {"message": "Wake word detected!"}
            },
            "autostart": False,
            "minimize_on_close": True,
            # Feature extraction parameters
            "feature_extraction": {
                "n_mfcc": 13,
                "n_fft": 2048,
                "hop_length": 160
            }
        }
        
        # Save the default configuration
        Config.save(default_config)
        
        return default_config
    
    @staticmethod
    def validate(config: Dict[str, Any]) -> Dict[str, Any]:
        """Validate configuration and fix any issues
        
        Args:
            config: Configuration dictionary
            
        Returns:
            Validated configuration dictionary
        """
        required_keys = {
            "audio_device": None,
            "sample_rate": 16000,
            "frame_size": 512,
            "model_path": None,
            "threshold": 0.85,
            "debounce_time": 3.0,
            "action": {
                "type": "notification",
                "params": {"message": "Wake word detected!"}
            },
            "autostart": False,
            "minimize_on_close": True,
            "feature_extraction": {
                "n_mfcc": 13,
                "n_fft": 2048,
                "hop_length": 160
            }
        }
        
        # Check for missing keys and set defaults
        for key, default_value in required_keys.items():
            if key not in config:
                config[key] = default_value
        
        # Check action substructure
        if "action" in config:
            if not isinstance(config["action"], dict):
                config["action"] = required_keys["action"]
            elif "type" not in config["action"]:
                config["action"]["type"] = "notification"
            elif "params" not in config["action"]:
                config["action"]["params"] = {"message": "Wake word detected!"}
        
        # Check feature extraction parameters
        if "feature_extraction" in config:
            if not isinstance(config["feature_extraction"], dict):
                config["feature_extraction"] = required_keys["feature_extraction"]
            else:
                # Ensure all required subkeys exist
                for subkey, default_subvalue in required_keys["feature_extraction"].items():
                    if subkey not in config["feature_extraction"]:
                        config["feature_extraction"][subkey] = default_subvalue
        
        # Ensure threshold is within valid range
        if "threshold" in config:
            config["threshold"] = max(0.5, min(0.99, float(config["threshold"])))
        
        # Ensure debounce time is within valid range
        if "debounce_time" in config:
            config["debounce_time"] = max(0.5, min(10.0, float(config["debounce_time"])))
        
        # Validate model path
        if config.get("model_path") and not os.path.exists(config["model_path"]):
            logger.warning(f"Model path does not exist: {config['model_path']}")
            
            # Check if the file exists in the models directory
            from io_wake_word.utils.paths import get_models_dir
            model_name = Path(config["model_path"]).name
            alternative_path = get_models_dir() / model_name
            
            if alternative_path.exists():
                config["model_path"] = str(alternative_path)
                logger.info(f"Found model at alternative path: {alternative_path}")
            else:
                logger.warning("Model not found, resetting model_path to None")
                config["model_path"] = None
        
        return config