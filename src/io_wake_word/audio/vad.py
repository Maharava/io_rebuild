"""
Voice Activity Detection - Filters out silent audio frames to improve
wake word detection accuracy and reduce processing load.
"""
import logging
from typing import List, Optional, Tuple

import numpy as np

logger = logging.getLogger("io_wake_word.audio")

class VoiceActivityDetector:
    """Energy-based voice activity detector with adaptive thresholding"""
    
    def __init__(
        self,
        sample_rate: int = 16000,
        frame_duration_ms: int = 30,
        threshold_energy: float = 0.0001,
        threshold_zero_crossings: int = 10,
        use_adaptive_threshold: bool = True,
        speech_hangover: int = 8,
        silence_hangover: int = 3,
    ):
        """Initialize voice activity detector
        
        Args:
            sample_rate: Audio sample rate in Hz
            frame_duration_ms: Frame duration in milliseconds
            threshold_energy: Minimum energy threshold for speech
            threshold_zero_crossings: Minimum zero crossings for speech
            use_adaptive_threshold: Whether to adapt threshold based on audio
            speech_hangover: Frames to continue detection after energy drops
            silence_hangover: Frames to wait before switching to silence
        """
        self.sample_rate = sample_rate
        self.frame_size = int(sample_rate * frame_duration_ms / 1000)
        self.threshold_energy = threshold_energy
        self.threshold_zero_crossings = threshold_zero_crossings
        
        # Adaptive threshold parameters
        self.use_adaptive_threshold = use_adaptive_threshold
        self.adaptive_energy_threshold = threshold_energy
        self.noise_level = threshold_energy
        
        # Sliding window for energy history
        self.energy_history: List[float] = []
        self.history_size = 100
        
        # State machine for smoother transitions
        self.speech_mode = False
        self.speech_frame_count = 0
        self.silence_frame_count = 0
        
        # Hangover counters for smoother detection
        self.speech_hangover = speech_hangover
        self.silence_hangover = silence_hangover
    
    def is_speech(self, audio_frame: np.ndarray) -> bool:
        """Determine if audio frame contains speech
        
        Args:
            audio_frame: Audio samples as numpy array
            
        Returns:
            True if speech detected, False otherwise
        """
        try:
            # Calculate frame energy
            energy = np.mean(audio_frame**2)
            
            # Calculate zero crossing rate
            zero_crossings = np.sum(np.abs(np.diff(np.signbit(audio_frame).astype(int))))
            
            # Update adaptive threshold if enabled
            if self.use_adaptive_threshold:
                self._update_adaptive_threshold(energy)
            
            # Raw speech detection
            raw_speech_detected = (energy > self.adaptive_energy_threshold and 
                                  zero_crossings > self.threshold_zero_crossings)
            
            # State machine with hangover for smoother detection
            if self.speech_mode:
                if raw_speech_detected:
                    # Continue in speech mode, reset silence counter
                    self.silence_frame_count = 0
                    self.speech_frame_count += 1
                    return True
                else:
                    # Possible end of speech, increment silence counter
                    self.silence_frame_count += 1
                    
                    # Stay in speech mode during hangover
                    if self.silence_frame_count < self.speech_hangover:
                        return True
                    else:
                        # Switch to silence mode
                        self.speech_mode = False
                        self.speech_frame_count = 0
                        return False
            else:
                if raw_speech_detected:
                    # Possible start of speech, increment speech counter
                    self.speech_frame_count += 1
                    
                    # Switch to speech mode if enough consecutive speech frames
                    if self.speech_frame_count > self.silence_hangover:
                        self.speech_mode = True
                        self.silence_frame_count = 0
                        return True
                    else:
                        return False
                else:
                    # Definite silence
                    self.speech_frame_count = 0
                    return False
                    
        except Exception as e:
            logger.error(f"Error in VAD processing: {e}")
            return True  # Default to speech on error to avoid missing potential wake words
    
    def _update_adaptive_threshold(self, current_energy: float) -> None:
        """Update adaptive threshold based on recent audio levels
        
        Args:
            current_energy: Energy of the current audio frame
        """
        # Add current energy to history
        self.energy_history.append(current_energy)
        
        # Keep history size limited
        if len(self.energy_history) > self.history_size:
            self.energy_history.pop(0)
        
        # Need enough history for reliable estimation
        if len(self.energy_history) < 10:
            return
        
        # Calculate noise level from the lowest 20% of energies
        sorted_energies = sorted(self.energy_history)
        noise_count = max(1, int(len(sorted_energies) * 0.2))
        recent_noise_level = sum(sorted_energies[:noise_count]) / noise_count
        
        # Calculate speech level from the highest 10% of energies
        speech_count = max(1, int(len(sorted_energies) * 0.1))
        speech_level = sum(sorted_energies[-speech_count:]) / speech_count
        
        # Smooth update of noise level
        self.noise_level = 0.95 * self.noise_level + 0.05 * recent_noise_level
        
        # Set adaptive threshold based on noise and speech levels
        if speech_level > self.noise_level * 2:
            # Clear speech/noise separation, set threshold in between
            self.adaptive_energy_threshold = self.noise_level * 1.5
        else:
            # Poor separation, use more conservative threshold
            self.adaptive_energy_threshold = self.noise_level * 2.5
        
        # Ensure minimum threshold
        self.adaptive_energy_threshold = max(self.threshold_energy, self.adaptive_energy_threshold)
    
    def reset(self) -> None:
        """Reset the detector state"""
        self.energy_history = []
        self.speech_mode = False
        self.speech_frame_count = 0
        self.silence_frame_count = 0
        self.adaptive_energy_threshold = self.threshold_energy
        self.noise_level = self.threshold_energy