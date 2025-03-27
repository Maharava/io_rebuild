"""
Model Recovery - Tools for recovering trained models from checkpoints
"""
import logging
from pathlib import Path
from typing import Optional, Union

import torch

from io_wake_word.models.architecture import (SimpleWakeWordModel, WakeWordModel,
                                           save_model)
from io_wake_word.utils.paths import get_models_dir

logger = logging.getLogger("io_wake_word.diagnostics")

def find_latest_checkpoint() -> Optional[Path]:
    """Find the latest checkpoint from training
    
    Returns:
        Path to the latest checkpoint or None if not found
    """
    from io_wake_word.utils.paths import get_base_dir
    
    # Possible checkpoint locations
    checkpoint_dirs = [
        get_base_dir() / "training_diagnostics",
        get_models_dir(),
    ]
    
    # Checkpoint files to look for, in priority order
    checkpoint_files = [
        "model_best.pth",
        "model_epoch_80.pth",
        "model_epoch_60.pth",
        "model_epoch_40.pth",
        "model_epoch_20.pth",
        "model_epoch_10.pth",
    ]
    
    # Find the first existing checkpoint
    for directory in checkpoint_dirs:
        if directory.exists():
            for filename in checkpoint_files:
                checkpoint_path = directory / filename
                if checkpoint_path.exists():
                    return checkpoint_path
    
    return None

def detect_model_type(state_dict) -> Optional[str]:
    """Detect model type from state dict keys
    
    Args:
        state_dict: Model state dictionary
        
    Returns:
        'simple' for SimpleWakeWordModel, 'standard' for WakeWordModel, or None
    """
    keys = list(state_dict.keys())
    
    if not keys:
        return None
    
    # Check for SimpleWakeWordModel (has conv_layer)
    if any("conv_layer" in key for key in keys):
        return "simple"
    
    # Check for standard WakeWordModel (has conv_layers)
    if any("conv_layers" in key for key in keys):
        return "standard"
    
    return None

def create_matching_model(state_dict, n_mfcc=13, num_frames=101):
    """Create a model that matches the state dict architecture
    
    Args:
        state_dict: Model state dictionary
        n_mfcc: Number of MFCC features
        num_frames: Number of time frames
        
    Returns:
        Matching model instance or None if architecture not detected
    """
    model_type = detect_model_type(state_dict)
    
    if model_type == "simple":
        logger.info("Detected SimpleWakeWordModel architecture")
        return SimpleWakeWordModel(n_mfcc=n_mfcc, num_frames=num_frames)
    elif model_type == "standard":
        logger.info("Detected StandardWakeWordModel architecture")
        return WakeWordModel(n_mfcc=n_mfcc, num_frames=num_frames)
    else:
        logger.error("Could not detect model architecture")
        return None

def recover_model(output_path: Optional[Union[str, Path]] = None) -> bool:
    """Recover model from checkpoint
    
    Args:
        output_path: Path to save the recovered model
            (default is models_dir/recovered_model.pth)
            
    Returns:
        True if recovery was successful, False otherwise
    """
    # Find checkpoint
    checkpoint_path = find_latest_checkpoint()
    if not checkpoint_path:
        logger.error("No checkpoint files found")
        return False
    
    logger.info(f"Found checkpoint: {checkpoint_path}")
    
    # Set output path if not provided
    if output_path is None:
        output_path = get_models_dir() / "recovered_model.pth"
    else:
        output_path = Path(output_path)
    
    try:
        # Load state dict
        logger.info(f"Loading checkpoint from {checkpoint_path}...")
        state_dict = torch.load(checkpoint_path, map_location=torch.device('cpu'))
        
        # Create a matching model
        model = create_matching_model(state_dict)
        if model is None:
            # Just copy the checkpoint directly
            logger.warning("Could not create matching model, copying checkpoint directly")
            output_path.parent.mkdir(parents=True, exist_ok=True)
            torch.save(state_dict, output_path)
        else:
            # Load state dict into model
            model.load_state_dict(state_dict)
            
            # Save the model
            save_model(model, output_path)
        
        logger.info(f"Model recovered to: {output_path}")
        
        # Create info file
        info_path = output_path.parent / f"{output_path.stem}_info.txt"
        with open(info_path, 'w') as f:
            f.write(f"Original checkpoint: {checkpoint_path}\n")
            f.write(f"Recovery date: {__import__('datetime').datetime.now()}\n")
            f.write(f"Model type: {detect_model_type(state_dict) or 'unknown'}\n")
        
        return True
    except Exception as e:
        logger.error(f"Error recovering model: {e}")
        return False