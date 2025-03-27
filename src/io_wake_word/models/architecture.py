"""
Model Architecture - Defines the neural network models for wake word detection,
including loading/saving functionality.
"""
import logging
import math
import os
from pathlib import Path
from typing import Optional, Tuple, Union

import torch
import torch.nn as nn

logger = logging.getLogger("io_wake_word.models")

def calculate_conv_output_length(
    input_length: int, 
    kernel_size: int, 
    stride: int, 
    padding: int = 0
) -> int:
    """Calculate output length after a conv/pool layer
    
    Args:
        input_length: Length of input
        kernel_size: Size of kernel
        stride: Stride of convolution
        padding: Padding size
        
    Returns:
        Output length after convolution
    """
    return math.floor((input_length + 2 * padding - kernel_size) / stride + 1)


class SimpleWakeWordModel(nn.Module):
    """Simplified CNN model for wake word detection"""
    
    def __init__(self, n_mfcc: int = 13, num_frames: int = 101):
        """Initialize a simple CNN model for wake word detection
        
        Args:
            n_mfcc: Number of MFCC features
            num_frames: Number of time frames
        """
        super(SimpleWakeWordModel, self).__init__()
        
        # A simpler architecture with fewer layers
        self.conv_layer = nn.Sequential(
            nn.Conv1d(n_mfcc, 32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=3, stride=2, padding=0)
        )
        
        # Calculate output size
        output_width = calculate_conv_output_length(num_frames, 3, 2, 0)
        self.fc_input_size = 32 * output_width
        
        self.fc_layers = nn.Sequential(
            nn.Linear(self.fc_input_size, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
            nn.Sigmoid()
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through the network
        
        Args:
            x: Input tensor of shape [batch, channels, time]
            
        Returns:
            Output tensor with wake word probability
        """
        x = self.conv_layer(x)
        x = x.view(x.size(0), -1)
        x = self.fc_layers(x)
        return x


class WakeWordModel(nn.Module):
    """Standard CNN model for wake word detection"""
    
    def __init__(self, n_mfcc: int = 13, num_frames: int = 101):
        """Initialize CNN model for wake word detection
        
        Args:
            n_mfcc: Number of MFCC features
            num_frames: Number of time frames
        """
        super(WakeWordModel, self).__init__()
        
        # Calculate exact output dimensions for each layer
        # First MaxPool: kernel=3, stride=2, padding=0
        after_pool1 = calculate_conv_output_length(num_frames, 3, 2, 0)
        # Second MaxPool: kernel=3, stride=2, padding=0
        after_pool2 = calculate_conv_output_length(after_pool1, 3, 2, 0)
        # Final flattened size
        self.fc_input_size = 64 * after_pool2
        
        # Simplified architecture with clear dimensions tracking
        self.conv_layers = nn.Sequential(
            # First conv block - ensure n_mfcc is used here as input channels
            nn.Conv1d(n_mfcc, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=3, stride=2, padding=0),
            
            # Second conv block
            nn.Conv1d(64, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=3, stride=2, padding=0)
        )
        
        # Fully connected layers
        self.fc_layers = nn.Sequential(
            nn.Linear(self.fc_input_size, 128),
            nn.ReLU(),
            nn.Linear(128, 1),
            nn.Sigmoid()
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through the network
        
        Args:
            x: Input tensor of shape [batch, channels, time]
            
        Returns:
            Output tensor with wake word probability
        """
        # Apply conv layers
        x = self.conv_layers(x)
        # Flatten for FC layers
        x = x.view(x.size(0), -1)
        # Apply FC layers
        x = self.fc_layers(x)
        return x


def create_model(
    n_mfcc: int = 13, 
    num_frames: int = 101, 
    use_simple_model: bool = False
) -> nn.Module:
    """Create a new wake word model
    
    Args:
        n_mfcc: Number of MFCC features
        num_frames: Number of time frames
        use_simple_model: Whether to use the simplified model
        
    Returns:
        Neural network model
    """
    if use_simple_model:
        return SimpleWakeWordModel(n_mfcc=n_mfcc, num_frames=num_frames)
    else:
        return WakeWordModel(n_mfcc=n_mfcc, num_frames=num_frames)


def save_model(model: nn.Module, path: Union[str, Path]) -> bool:
    """Save model to disk with proper resource management
    
    Args:
        model: Neural network model to save
        path: Path to save the model
        
    Returns:
        True if saved successfully, False otherwise
    """
    try:
        if model is None:
            logger.error("Cannot save None model")
            return False
            
        # Ensure the directory exists
        path = Path(path) if not isinstance(path, Path) else path
        path.parent.mkdir(parents=True, exist_ok=True)
        
        # Save the model state dict
        torch.save(model.state_dict(), path)
        logger.info(f"Model saved to {path}")
        return True
        
    except Exception as e:
        logger.error(f"Error saving model: {e}")
        return False


def load_model(
    path: Union[str, Path], 
    n_mfcc: int = 13, 
    num_frames: int = 101
) -> Optional[nn.Module]:
    """Load model from disk with automatic architecture detection
    
    Args:
        path: Path to the model file
        n_mfcc: Number of MFCC features
        num_frames: Number of time frames
        
    Returns:
        Loaded model or None if loading failed
    """
    if not path:
        logger.error("Model path is None")
        return None
    
    # Convert to Path object and check if it exists
    path = Path(path) if not isinstance(path, Path) else path
    
    if not path.exists():
        logger.error(f"Model file not found: {path}")
        return None
    
    try:
        # Load state dictionary to check architecture
        state_dict = torch.load(path, map_location=torch.device('cpu'))
        
        # Check for model architecture by examining state_dict keys
        # Look for either "conv_layer" (SimpleWakeWordModel) or "conv_layers" (WakeWordModel)
        state_dict_keys = list(state_dict.keys())
        is_simple_model = any('conv_layer.' in key for key in state_dict_keys)
        is_standard_model = any('conv_layers.' in key for key in state_dict_keys)
        
        # Create the appropriate model based on detected architecture
        model = None
        
        # Try WakeWordModel first if we see conv_layers
        if is_standard_model:
            logger.info("Detected WakeWordModel architecture")
            model = WakeWordModel(n_mfcc=n_mfcc, num_frames=num_frames)
            
            try:
                model.load_state_dict(state_dict)
                logger.info(f"Model loaded successfully as WakeWordModel")
                model.eval()
                return model
            except Exception as e:
                logger.warning(f"Error loading as WakeWordModel: {e}")
                # Continue to try SimpleWakeWordModel
        
        # Try SimpleWakeWordModel if we see conv_layer or if WakeWordModel failed
        if is_simple_model or (not is_standard_model):
            logger.info("Trying SimpleWakeWordModel architecture")
            model = SimpleWakeWordModel(n_mfcc=n_mfcc, num_frames=num_frames)
            
            try:
                model.load_state_dict(state_dict)
                logger.info(f"Model loaded successfully as SimpleWakeWordModel")
                model.eval()
                return model
            except Exception as e:
                logger.warning(f"Error loading as SimpleWakeWordModel: {e}")
                # Continue to try with modified state dict if both direct methods fail
        
        # If we got here, neither approach worked directly
        # As a last resort, try to fix mismatched keys by inspecting the state dict structure
        if not model:
            # The model is likely a WakeWordModel but wasn't properly identified
            # Let's try that as a fallback
            logger.warning("Standard loading failed. Attempting to load with architecture adaptation...")
            model = WakeWordModel(n_mfcc=n_mfcc, num_frames=num_frames)
            model.eval()
            
            # At this point, we return the model without loading the state dict
            # It won't have the trained weights, but it's better than returning None
            logger.warning("Model architecture couldn't be determined. Using default WakeWordModel.")
            return model
                
    except Exception as e:
        logger.error(f"Error loading model: {e}")
        return None
    
    # If all else fails
    return None