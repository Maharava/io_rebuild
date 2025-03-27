"""
Wake Word Detector - Provides a clean interface for detecting wake words
in audio streams with configurable callbacks.
"""
import collections
import logging
import threading
import time
from pathlib import Path
from typing import Callable, List, Optional, Tuple, Union

import numpy as np
import torch

from io_wake_word.models.architecture import load_model
from io_wake_word.utils.paths import resolve_model_path

logger = logging.getLogger("io_wake_word.models")

class WakeWordDetector:
    """Real-time wake word detector with smoothing and callback support"""
    
    def __init__(
        self,
        model_path: Optional[Union[str, Path]] = None,
        threshold: float = 0.85,
        window_size: int = 5,
        debounce_time: float = 2.0,
        required_streak: int = 2,
    ):
        """Initialize detector with given model and parameters
        
        Args:
            model_path: Path to the model file
            threshold: Detection threshold (0.0-1.0)
            window_size: Number of predictions to average
            debounce_time: Seconds to wait between detections
            required_streak: Required number of consecutive high confidence frames
        """
        self.model_path = model_path
        self.threshold = threshold
        self.window_size = window_size
        self.debounce_time = debounce_time
        self.required_streak = required_streak
        
        # Store recent predictions for smoothing with timestamps
        self.recent_predictions = collections.deque(maxlen=window_size)
        
        # Detection debouncing
        self.last_detection_time = 0
        
        # Consecutive frames counter
        self.high_confidence_streak = 0
        
        # Thread safety
        self.lock = threading.Lock()
        
        # Callback registry
        self.callbacks: List[Callable[[float], None]] = []
        
        # Load the model if provided
        self.model = None
        if model_path:
            self.load_model(model_path)
    
    def load_model(self, model_path: Union[str, Path]) -> bool:
        """Load a wake word model
        
        Args:
            model_path: Path to the model file
            
        Returns:
            True if loaded successfully, False otherwise
        """
        if not model_path:
            logger.error("Model path is None, cannot load model")
            return False
            
        try:
            # Resolve model path (check default directory if needed)
            resolved_path = resolve_model_path(model_path)
            
            if not resolved_path.exists():
                logger.error(f"Model file not found: {model_path}")
                return False
                
            # Load model, specifying n_mfcc=13
            new_model = load_model(resolved_path, n_mfcc=13, num_frames=101)
            
            if new_model:
                # Switch to evaluation mode
                new_model.eval()
                
                # Update model
                with self.lock:
                    self.model = new_model
                    self.recent_predictions.clear()
                    self.high_confidence_streak = 0
                
                logger.info(f"Model loaded successfully from {resolved_path}")
                return True
            else:
                logger.error("Failed to load model (returned None)")
                return False
                
        except Exception as e:
            logger.error(f"Error loading model: {str(e)}")
            return False
    
    def set_threshold(self, threshold: float) -> None:
        """Update detection threshold
        
        Args:
            threshold: New threshold (0.0-1.0)
        """
        with self.lock:
            self.threshold = max(0.0, min(1.0, threshold))
        logger.info(f"Detection threshold set to {self.threshold}")
    
    def register_callback(self, callback: Callable[[float], None]) -> None:
        """Register a callback for detection events
        
        Args:
            callback: Function that takes confidence value as parameter
        """
        if callback not in self.callbacks:
            self.callbacks.append(callback)
            logger.debug(f"Registered callback {callback}")
    
    def unregister_callback(self, callback: Callable[[float], None]) -> None:
        """Unregister a previously registered callback
        
        Args:
            callback: The callback to remove
        """
        if callback in self.callbacks:
            self.callbacks.remove(callback)
            logger.debug(f"Unregistered callback {callback}")
    
    def detect(self, features: np.ndarray) -> Tuple[bool, float]:
        """Detect wake word in audio features with robust filtering
        
        Args:
            features: Audio features extracted from audio frame
            
        Returns:
            Tuple of (is_detected, confidence)
        """
        # Check if features is None (indicates silence)
        if features is None:
            self.high_confidence_streak = 0
            return False, 0.0
            
        # First check if model is loaded
        with self.lock:
            if self.model is None:
                return False, 0.0
            
            current_threshold = self.threshold
        
        # Convert numpy array to torch tensor
        if isinstance(features, np.ndarray):
            features = torch.from_numpy(features).float()
        
        # Validate feature dimensions
        if len(features.shape) != 3:
            logger.error(f"Invalid feature shape: {features.shape}")
            return False, 0.0
        
        try:
            # Make prediction
            with torch.no_grad():
                outputs = self.model(features)
                confidence = outputs.item()
            
            # Process high confidence scores
            if confidence > current_threshold:
                self.high_confidence_streak += 1
            else:
                self.high_confidence_streak = 0
            
            # Add to recent predictions
            current_time = time.time()
            with self.lock:
                self.recent_predictions.append((confidence, current_time))
                recent_preds = list(self.recent_predictions)
            
            # Process predictions with time-weighting
            if len(recent_preds) >= 3:
                # Calculate time-weighted average
                total_weight = 0
                weighted_sum = 0
                
                latest_time = recent_preds[-1][1]
                
                for conf, timestamp in recent_preds:
                    time_diff = latest_time - timestamp
                    weight = np.exp(-time_diff * 2)
                    
                    weighted_sum += conf * weight
                    total_weight += weight
                
                avg_confidence = weighted_sum / total_weight if total_weight > 0 else 0
                
                # Count high confidence predictions
                high_conf_count = sum(1 for conf, _ in recent_preds if conf > current_threshold)
                
                # Check debounce time
                time_since_last = current_time - self.last_detection_time
                can_trigger = time_since_last > self.debounce_time
                
                # Final detection decision
                is_detected = (
                    avg_confidence > current_threshold and
                    high_conf_count >= min(3, len(recent_preds)) and
                    self.high_confidence_streak >= self.required_streak and
                    can_trigger
                )
                
                # Update last detection time if triggered
                if is_detected:
                    self.last_detection_time = current_time
                    self.high_confidence_streak = 0
                    
                    # Call callbacks
                    for callback in self.callbacks:
                        try:
                            callback(avg_confidence)
                        except Exception as e:
                            logger.error(f"Error in callback: {e}")
                    
                    logger.info(f"Wake word detected with confidence: {avg_confidence:.4f}")
                
                return is_detected, avg_confidence
            else:
                return False, confidence
                
        except Exception as e:
            logger.error(f"Error during inference: {str(e)}")
            return False, 0.0
    
    def process_frame(self, audio_frame: np.ndarray, features: Optional[np.ndarray] = None) -> Tuple[bool, float]:
        """Process an audio frame for wake word detection
        
        This is a convenience method that extracts features if not provided
        
        Args:
            audio_frame: Raw audio frame
            features: Pre-extracted features (optional)
            
        Returns:
            Tuple of (is_detected, confidence)
        """
        from io_wake_word.audio.features import FeatureExtractor
        
        # Use provided features or extract them
        if features is not None:
            return self.detect(features)
        
        # Create a feature extractor and process the frame
        extractor = FeatureExtractor()
        features = extractor.extract(audio_frame)
        
        # Detect wake word
        return self.detect(features)