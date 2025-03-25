"""
Feature Extraction - Converts audio frames to MFCC features for neural network processing
"""
import logging
from typing import Dict, Optional, Union

import librosa
import numpy as np

logger = logging.getLogger("io_wake_word.audio")

class FeatureExtractor:
    """Extract MFCC features from audio frames"""
    
    def __init__(
        self,
        sample_rate: int = 16000,
        frame_size: int = 512,
        n_mfcc: int = 13,
        n_fft: int = 2048,
        hop_length: int = 160,
        use_cache: bool = True,
        max_cache_size: int = 100,
    ):
        """Initialize feature extractor with given parameters
        
        Args:
            sample_rate: Audio sample rate in Hz
            frame_size: Number of samples per frame
            n_mfcc: Number of MFCC coefficients
            n_fft: FFT window size
            hop_length: Hop length for feature extraction
            use_cache: Whether to cache features for repeated audio
            max_cache_size: Maximum number of items in the cache
        """
        self.sample_rate = sample_rate
        self.frame_size = frame_size
        self.n_mfcc = n_mfcc
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.use_cache = use_cache
        
        # Feature cache for frequently used audio
        self.feature_cache = {}
        self.max_cache_size = max_cache_size
        
        # Number of frames to generate 1 second of context
        self.num_frames = 101
        
        # Running buffer for audio context
        self.audio_buffer = np.zeros(0)
        
        # Energy threshold for silence detection
        self.energy_threshold = 0.005
        
        # Number of silence frames in a row
        self.silence_counter = 0
        self.max_silence_frames = 5
    
    def extract(self, audio_frame: np.ndarray) -> Optional[np.ndarray]:
        """Extract MFCC features from audio frame
        
        Args:
            audio_frame: Audio samples as numpy array
            
        Returns:
            Feature array in shape [batch, channels, features, time] or None if silent
        """
        # Add current frame to buffer
        self.audio_buffer = np.append(self.audio_buffer, audio_frame)
        
        # We need at least 1 second of audio
        min_samples = self.sample_rate + self.frame_size
        
        if len(self.audio_buffer) < min_samples:
            return None
        
        # Keep only the most recent audio
        if len(self.audio_buffer) > min_samples * 1.2:
            self.audio_buffer = self.audio_buffer[-min_samples:]
        
        # Check if audio is silent
        energy = np.mean(self.audio_buffer**2)
        if energy < self.energy_threshold:
            self.silence_counter += 1
            if self.silence_counter > self.max_silence_frames:
                return None  # Don't process silent audio
        else:
            self.silence_counter = 0
        
        # Calculate buffer hash for cache lookup
        if self.use_cache:
            buffer_hash = hash(self.audio_buffer.tobytes())
            
            # Check cache first
            if buffer_hash in self.feature_cache:
                return self.feature_cache[buffer_hash]
        
        # Extract MFCCs
        try:
            # Calculate MFCCs
            mfccs = librosa.feature.mfcc(
                y=self.audio_buffer, 
                sr=self.sample_rate,
                n_mfcc=self.n_mfcc,
                n_fft=self.n_fft,
                hop_length=self.hop_length
            )
            
            # Ensure consistent length
            if mfccs.shape[1] < self.num_frames:
                # Pad if too short
                pad_width = self.num_frames - mfccs.shape[1]
                mfccs = np.pad(mfccs, ((0, 0), (0, pad_width)))
            elif mfccs.shape[1] > self.num_frames:
                # Truncate if too long
                mfccs = mfccs[:, -self.num_frames:]
            
            # Apply normalization (feature-wise for better robustness)
            for i in range(mfccs.shape[0]):
                feature_mean = np.mean(mfccs[i])
                feature_std = np.std(mfccs[i])
                if feature_std > 1e-6:  # Prevent division by zero
                    mfccs[i] = (mfccs[i] - feature_mean) / feature_std
            
            # Reshape for the model [batch, channels, features, time]
            features = np.expand_dims(mfccs, axis=0)
            
            # Cache the result
            if self.use_cache:
                self._update_cache(buffer_hash, features)
            
            return features
            
        except Exception as e:
            logger.error(f"Error extracting features: {str(e)}")
            return None
    
    def _update_cache(self, key, value) -> None:
        """Update feature cache with size limit"""
        if not self.use_cache:
            return
            
        # Add to cache
        self.feature_cache[key] = value
        
        # Remove oldest items if cache is too large
        if len(self.feature_cache) > self.max_cache_size:
            # Get oldest items (first items in the cache)
            oldest_key = next(iter(self.feature_cache))
            del self.feature_cache[oldest_key]
    
    def clear_buffer(self) -> None:
        """Clear the audio buffer"""
        self.audio_buffer = np.zeros(0)
        self.silence_counter = 0
    
    def clear_cache(self) -> None:
        """Clear the feature cache"""
        self.feature_cache.clear()
    
    def get_info(self) -> Dict:
        """Return information about the feature extractor"""
        return {
            "n_mfcc": self.n_mfcc,
            "n_fft": self.n_fft,
            "hop_length": self.hop_length,
            "num_frames": self.num_frames,
            "buffer_length": len(self.audio_buffer),
            "cache_size": len(self.feature_cache),
        }