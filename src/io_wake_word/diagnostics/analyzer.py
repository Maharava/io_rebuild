"""
Data Analysis - Tools for analyzing training data and debugging models
"""
import logging
import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import librosa
import numpy as np
import torch

logger = logging.getLogger("io_wake_word.diagnostics")

def analyze_audio_files(
    audio_dir: Union[str, Path], 
    limit: Optional[int] = None
) -> Dict[str, Any]:
    """Analyze audio files in a directory
    
    Args:
        audio_dir: Directory containing audio files
        limit: Maximum number of files to analyze
        
    Returns:
        Dictionary with analysis results
    """
    logger.info(f"Analyzing audio files in {audio_dir}")
    
    # Get all WAV files
    audio_dir = Path(audio_dir)
    wav_files = list(audio_dir.glob("*.wav"))
    if limit:
        wav_files = wav_files[:limit]
    
    if not wav_files:
        logger.warning(f"No WAV files found in {audio_dir}")
        return {"files_found": 0}
    
    logger.info(f"Analyzing {len(wav_files)} WAV files")
    
    # Track audio statistics
    durations = []
    energies = []
    sample_rates = []
    channels = []
    
    # Process each file
    for file_path in wav_files:
        try:
            # Load audio
            y, sr = librosa.load(str(file_path), sr=None)
            
            # Calculate statistics
            duration = len(y) / sr
            energy = np.mean(y**2)
            
            durations.append(duration)
            energies.append(energy)
            sample_rates.append(sr)
            channels.append(1 if len(y.shape) == 1 else y.shape[1])
            
        except Exception as e:
            logger.error(f"Error processing {file_path}: {e}")
    
    # Calculate summary statistics
    if durations:
        results = {
            "files_analyzed": len(durations),
            "mean_duration": float(np.mean(durations)),
            "min_duration": float(np.min(durations)),
            "max_duration": float(np.max(durations)),
            "mean_energy": float(np.mean(energies)),
            "min_energy": float(np.min(energies)),
            "max_energy": float(np.max(energies)),
            "sample_rates": list(set(sample_rates)),
            "has_multichannel": any(c > 1 for c in channels),
        }
        
        logger.info(f"Analysis complete: {results}")
        return results
    else:
        logger.warning("No valid audio files were processed")
        return {"files_analyzed": 0}

def analyze_feature_extraction(
    audio_file: Union[str, Path]
) -> Dict[str, Any]:
    """Analyze feature extraction from an audio file
    
    Args:
        audio_file: Path to audio file
        
    Returns:
        Dictionary with analysis results
    """
    from io_wake_word.audio.features import FeatureExtractor
    
    logger.info(f"Analyzing feature extraction from {audio_file}")
    
    try:
        # Load audio
        y, sr = librosa.load(str(audio_file), sr=16000)
        
        # Create feature extractor
        extractor = FeatureExtractor()
        
        # Add audio to buffer
        extractor.audio_buffer = y
        
        # Extract features
        features = extractor.extract(np.array([]))
        
        if features is not None:
            # Feature statistics
            mean_val = float(np.mean(features))
            std_val = float(np.std(features))
            min_val = float(np.min(features))
            max_val = float(np.max(features))
            
            results = {
                "feature_shape": list(features.shape),
                "mean": mean_val,
                "std": std_val,
                "min": min_val,
                "max": max_val,
                "success": True,
            }
            
            logger.info(f"Feature extraction successful: {results}")
            return results
        else:
            logger.warning(f"Failed to extract features from {audio_file}")
            return {"success": False, "error": "Feature extraction returned None"}
            
    except Exception as e:
        logger.error(f"Error analyzing features for {audio_file}: {e}")
        return {"success": False, "error": str(e)}

def analyze_model(
    model_path: Union[str, Path]
) -> Dict[str, Any]:
    """Analyze a model file
    
    Args:
        model_path: Path to model file
        
    Returns:
        Dictionary with analysis results
    """
    import torch
    from io_wake_word.models.architecture import load_model
    
    logger.info(f"Analyzing model: {model_path}")
    
    try:
        # Try loading the model
        model = load_model(model_path)
        
        if model is None:
            return {"success": False, "error": "Failed to load model"}
        
        # Get model structure
        model_type = model.__class__.__name__
        
        # Count parameters
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        
        results = {
            "success": True,
            "model_type": model_type,
            "total_params": total_params,
            "trainable_params": trainable_params,
            "layers": [],
        }
        
        # Get layer information
        for name, module in model.named_children():
            if hasattr(module, "children") and len(list(module.children())) > 0:
                for subname, submodule in module.named_children():
                    layer_name = f"{name}.{subname}"
                    layer_type = submodule.__class__.__name__
                    layer_params = sum(p.numel() for p in submodule.parameters())
                    results["layers"].append({
                        "name": layer_name,
                        "type": layer_type,
                        "params": layer_params,
                    })
            else:
                layer_type = module.__class__.__name__
                layer_params = sum(p.numel() for p in module.parameters())
                results["layers"].append({
                    "name": name,
                    "type": layer_type,
                    "params": layer_params,
                })
        
        logger.info(f"Model analysis complete: {model_type}, {total_params} parameters")
        return results
        
    except Exception as e:
        logger.error(f"Error analyzing model: {e}")
        return {"success": False, "error": str(e)}

def analyze_detection_performance(
    model_path: Union[str, Path],
    test_files: List[Union[str, Path]],
    ground_truth: List[bool]
) -> Dict[str, Any]:
    """Analyze detection performance on a test set
    
    Args:
        model_path: Path to model file
        test_files: List of audio file paths
        ground_truth: List of expected detection results (True/False)
        
    Returns:
        Dictionary with performance metrics
    """
    from io_wake_word.models.detector import WakeWordDetector
    from io_wake_word.audio.features import FeatureExtractor
    
    if len(test_files) != len(ground_truth):
        logger.error("Mismatch between test files and ground truth")
        return {"success": False, "error": "Mismatch between test files and ground truth"}
    
    try:
        # Load model
        detector = WakeWordDetector(model_path=model_path)
        extractor = FeatureExtractor()
        
        # Track results
        true_positives = 0
        false_positives = 0
        true_negatives = 0
        false_negatives = 0
        confidences = []
        
        # Process each file
        for i, file_path in enumerate(test_files):
            expected = ground_truth[i]
            
            # Load audio
            y, sr = librosa.load(str(file_path), sr=16000)
            
            # Extract features
            extractor.clear_buffer()
            extractor.audio_buffer = y
            features = extractor.extract(np.array([]))
            
            if features is not None:
                # Detect wake word
                detected, confidence = detector.detect(features)
                confidences.append(confidence)
                
                # Update metrics
                if detected and expected:
                    true_positives += 1
                elif detected and not expected:
                    false_positives += 1
                elif not detected and expected:
                    false_negatives += 1
                else:  # not detected and not expected
                    true_negatives += 1
            else:
                logger.warning(f"Failed to extract features from {file_path}")
                # Count as false negative if expected to detect
                if expected:
                    false_negatives += 1
                else:
                    true_negatives += 1
        
        # Calculate metrics
        total = len(test_files)
        accuracy = (true_positives + true_negatives) / total if total > 0 else 0
        precision = true_positives / (true_positives + false_positives) if (true_positives + false_positives) > 0 else 0
        recall = true_positives / (true_positives + false_negatives) if (true_positives + false_negatives) > 0 else 0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
        
        results = {
            "success": True,
            "total_files": total,
            "true_positives": true_positives,
            "false_positives": false_positives,
            "true_negatives": true_negatives,
            "false_negatives": false_negatives,
            "accuracy": accuracy,
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "mean_confidence": float(np.mean(confidences)) if confidences else 0,
        }
        
        logger.info(f"Detection performance: accuracy={accuracy:.4f}, precision={precision:.4f}, recall={recall:.4f}, f1={f1:.4f}")
        return results
        
    except Exception as e:
        logger.error(f"Error analyzing detection performance: {e}")
        return {"success": False, "error": str(e)}