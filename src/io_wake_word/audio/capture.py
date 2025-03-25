"""
Audio Capture - Handles microphone input and buffering with both callback
and synchronous interfaces for flexibility.
"""
import collections
import logging
import threading
import wave
from contextlib import contextmanager
from typing import Callable, Dict, List, Optional, Union

import numpy as np
import pyaudio

from io_wake_word.audio.vad import VoiceActivityDetector

logger = logging.getLogger("io_wake_word.audio")

class AudioCapture:
    """Thread-safe audio capture with support for both streaming and callback interfaces"""
    
    def __init__(
        self,
        device_index: Optional[int] = None,
        sample_rate: int = 16000,
        frame_size: int = 512,
        callback: Optional[Callable] = None,
        use_vad: bool = True,
    ):
        """Initialize audio capture with PyAudio
        
        Args:
            device_index: Index of the audio device to use, or None for default
            sample_rate: Audio sample rate in Hz
            frame_size: Number of samples per frame
            callback: Optional callback function for streaming mode
            use_vad: Whether to use voice activity detection
        """
        self.device_index = device_index
        self.sample_rate = sample_rate
        self.frame_size = frame_size
        self.callback = callback
        self.use_vad = use_vad
        
        # Create circular buffer (2 seconds of audio)
        buffer_frames = int(2 * sample_rate / frame_size)
        self.buffer = collections.deque(maxlen=buffer_frames)
        
        # Create VAD for filtering silent frames
        if use_vad:
            self.vad = VoiceActivityDetector(sample_rate=sample_rate)
        
        # Lock for thread safety
        self.lock = threading.Lock()
        
        # Stream state
        self.stream = None
        self.pyaudio = None
        self.is_running = False
        
        # Audio level tracking
        self.current_audio_level = 0.0
    
    def list_devices(self) -> List[Dict]:
        """List available audio input devices
        
        Returns:
            List of dictionaries with device information
        """
        try:
            p = pyaudio.PyAudio()
            devices = []
            
            for i in range(p.get_device_count()):
                device_info = p.get_device_info_by_index(i)
                if device_info["maxInputChannels"] > 0:
                    devices.append({
                        "index": i,
                        "name": device_info["name"],
                        "channels": device_info["maxInputChannels"],
                        "sample_rate": int(device_info["defaultSampleRate"])
                    })
            
            p.terminate()
            return devices
        except Exception as e:
            logger.error(f"Error listing audio devices: {e}")
            return []
    
    def _audio_callback(self, in_data, frame_count, time_info, status):
        """PyAudio callback function for streaming audio data"""
        if status:
            logger.warning(f"Audio callback status: {status}")
        
        try:
            # Convert bytes to numpy array (16-bit audio)
            audio_data = np.frombuffer(in_data, dtype=np.int16).astype(np.float32)
            
            # Normalize to [-1.0, 1.0]
            audio_data = audio_data / 32768.0
            
            # Calculate audio level for visualization
            self.current_audio_level = float(np.abs(audio_data).mean())
            
            # Apply automatic gain control (simple normalization)
            if np.abs(audio_data).max() > 0:
                audio_data = audio_data / np.abs(audio_data).max() * 0.9
            
            # Check if audio contains voice (if VAD is enabled)
            if not self.use_vad or self.vad.is_speech(audio_data):
                # Add to buffer with thread safety
                with self.lock:
                    self.buffer.append(audio_data)
                
                # Process the audio if callback is set
                if self.callback:
                    self.callback(audio_data)
        except Exception as e:
            logger.error(f"Error in audio callback: {e}")
        
        # Continue the stream
        return (None, pyaudio.paContinue)
    
    def start(self):
        """Start audio capture with proper resource management"""
        if self.is_running:
            return
        
        try:
            self.pyaudio = pyaudio.PyAudio()
            
            # If no device index specified, use default
            if self.device_index is None:
                try:
                    self.device_index = self.pyaudio.get_default_input_device_info()["index"]
                    logger.info(f"Using default audio device with index {self.device_index}")
                except Exception as e:
                    logger.error(f"Could not get default device: {e}. Using device 0.")
                    self.device_index = 0
            
            # Open audio stream in callback mode
            self.stream = self.pyaudio.open(
                format=pyaudio.paInt16,
                channels=1,
                rate=self.sample_rate,
                input=True,
                input_device_index=self.device_index,
                frames_per_buffer=self.frame_size,
                stream_callback=self._audio_callback
            )
            
            self.is_running = True
            logger.info(f"Audio capture started on device {self.device_index}")
        except Exception as e:
            logger.error(f"Error starting audio capture: {e}")
            self._cleanup_resources()
    
    def stop(self):
        """Stop audio capture with proper resource cleanup"""
        if not self.is_running:
            return
        
        self.is_running = False
        self._cleanup_resources()
        logger.info("Audio capture stopped")
    
    def _cleanup_resources(self):
        """Clean up PyAudio resources properly"""
        try:
            if self.stream:
                if self.stream.is_active():
                    self.stream.stop_stream()
                self.stream.close()
                self.stream = None
        except Exception as e:
            logger.error(f"Error closing stream: {e}")
        
        try:
            if self.pyaudio:
                self.pyaudio.terminate()
                self.pyaudio = None
        except Exception as e:
            logger.error(f"Error terminating PyAudio: {e}")
    
    def get_buffer(self) -> np.ndarray:
        """Get the current audio buffer (thread-safe)
        
        Returns:
            Numpy array of audio samples
        """
        with self.lock:
            # Convert deque to numpy array
            if len(self.buffer) > 0:
                return np.concatenate(list(self.buffer))
            else:
                return np.array([], dtype=np.float32)
    
    def get_audio_level(self) -> float:
        """Get current audio level for visualization
        
        Returns:
            Float between 0.0 and 1.0 representing audio level
        """
        return self.current_audio_level
    
    @contextmanager
    def stream(self):
        """Context manager for streaming audio frames
        
        Example:
            with audio_capture.stream() as stream:
                for frame in stream:
                    process_frame(frame)
        """
        # Start capture if not already running
        if not self.is_running:
            self.start()
        
        # Create a generator for streaming frames
        frame_queue = collections.deque()
        
        def frame_callback(audio_frame):
            frame_queue.append(audio_frame)
        
        # Store original callback and replace with ours
        original_callback = self.callback
        self.callback = frame_callback
        
        try:
            # Generator that yields frames as they become available
            def frame_generator():
                while self.is_running:
                    if frame_queue:
                        yield frame_queue.popleft()
                    else:
                        import time
                        time.sleep(0.01)  # Prevent CPU spinning
            
            # Yield the generator
            yield frame_generator()
        finally:
            # Restore original callback
            self.callback = original_callback
    
    def save_sample(self, filename: str, duration: float = 3.0) -> bool:
        """Record a sample to a WAV file for testing
        
        Args:
            filename: Path to save the WAV file
            duration: Duration to record in seconds
            
        Returns:
            True if successful, False otherwise
        """
        if self.is_running:
            logger.warning("Cannot save sample while capture is running")
            return False
        
        p = None
        stream = None
        
        try:
            p = pyaudio.PyAudio()
            stream = p.open(
                format=pyaudio.paInt16,
                channels=1,
                rate=self.sample_rate,
                input=True,
                input_device_index=self.device_index,
                frames_per_buffer=self.frame_size
            )
            
            logger.info(f"Recording {duration} second sample to {filename}")
            
            frames = []
            for i in range(0, int(self.sample_rate / self.frame_size * duration)):
                data = stream.read(self.frame_size)
                frames.append(data)
            
            logger.info("Finished recording")
            
            # Save to WAV file
            with wave.open(filename, 'wb') as wf:
                wf.setnchannels(1)
                wf.setsampwidth(p.get_sample_size(pyaudio.paInt16))
                wf.setframerate(self.sample_rate)
                wf.writeframes(b''.join(frames))
            
            return True
        
        except Exception as e:
            logger.error(f"Error saving audio sample: {e}")
            return False
        
        finally:
            if stream:
                stream.stop_stream()
                stream.close()
            
            if p:
                p.terminate()
    
    def __del__(self):
        """Destructor to ensure resources are released"""
        self.stop()