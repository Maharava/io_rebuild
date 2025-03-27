"""
Training panel for Io wake word engine UI
"""
import logging
import os
import threading
import time
import tkinter as tk
from pathlib import Path
from tkinter import filedialog
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import customtkinter as ctk
import librosa
import numpy as np
import pyaudio
import wave

from io_wake_word.utils.paths import get_data_dir
from io_wake_word.models.trainer import WakeWordTrainer

logger = logging.getLogger("io_app.ui")

class RecordingThread(threading.Thread):
    """Thread for recording audio samples"""
    
    def __init__(self, filename: Union[str, Path], duration: float, sample_rate: int = 16000):
        """Initialize recording thread
        
        Args:
            filename: Path to save the recording
            duration: Recording duration in seconds
            sample_rate: Audio sample rate in Hz
        """
        super().__init__()
        self.filename = str(filename)
        self.duration = duration
        self.sample_rate = sample_rate
        self.is_running = False
        self.audio_level = 0.0
        self.daemon = True
        self.error = None
        self._stop_event = threading.Event()
        self._start_time = None
    
    def run(self):
        """Record audio for specified duration"""
        self.is_running = True
        p = None
        stream = None
        
        try:
            # Record start time for progress tracking
            self._start_time = time.time()
            
            p = pyaudio.PyAudio()
            stream = p.open(
                format=pyaudio.paInt16,
                channels=1,
                rate=self.sample_rate,
                input=True,
                frames_per_buffer=1024
            )
            
            # Calculate total frames needed
            chunk_size = 1024
            total_chunks = int((self.sample_rate * self.duration) / chunk_size)
            
            logger.info(f"Starting recording to {self.filename} for {self.duration}s ({total_chunks} chunks)")
            
            frames = []
            for i in range(total_chunks):
                if self._stop_event.is_set():
                    break
                    
                data = stream.read(chunk_size, exception_on_overflow=False)
                frames.append(data)
                
                # Calculate audio level for visualization
                try:
                    audio_data = np.frombuffer(data, dtype=np.int16).astype(np.float32)
                    max_value = np.abs(audio_data).max() if len(audio_data) > 0 else 0
                    self.audio_level = min(1.0, max_value / 32768)
                except Exception as e:
                    logger.debug(f"Error calculating audio level: {e}")
            
            # Ensure parent directory exists
            directory = os.path.dirname(self.filename)
            os.makedirs(directory, exist_ok=True)
            
            # Save to WAV file if we have frames
            if frames and not self._stop_event.is_set():
                with wave.open(self.filename, 'wb') as wf:
                    wf.setnchannels(1)  # Mono
                    wf.setsampwidth(2)  # 16-bit (2 bytes)
                    wf.setframerate(self.sample_rate)
                    wf.writeframes(b''.join(frames))
                logger.info(f"Successfully saved recording to {self.filename}")
                
        except Exception as e:
            logger.error(f"Error recording audio: {e}")
            self.error = str(e)
        finally:
            # Clean up resources
            if stream:
                try:
                    stream.stop_stream()
                    stream.close()
                except Exception as e:
                    logger.error(f"Error closing audio stream: {e}")
            
            if p:
                try:
                    p.terminate()
                except Exception as e:
                    logger.error(f"Error terminating PyAudio: {e}")
            
            self.is_running = False
            self.audio_level = 0.0
    
    def stop(self):
        """Stop recording"""
        logger.debug("Stopping recording")
        self._stop_event.set()


class TrainingThread(threading.Thread):
    """Thread for training wake word model"""
    
    def __init__(
        self, 
        wake_word_files: List[Path], 
        negative_files: List[Path], 
        model_name: str, 
        progress_callback: Optional[Callable[[str, int], None]] = None
    ):
        """Initialize training thread
        
        Args:
            wake_word_files: List of wake word audio files
            negative_files: List of negative audio files
            model_name: Name for the trained model
            progress_callback: Callback for progress updates
        """
        super().__init__()
        self.wake_word_files = wake_word_files
        self.negative_files = negative_files
        self.model_name = model_name
        self.progress_callback = progress_callback
        self.result = {"success": False}
        self.daemon = True
    
    def run(self):
        """Run training process"""
        try:
            # Create trainer
            trainer = WakeWordTrainer()
            
            # Prepare data
            if self.progress_callback:
                self.progress_callback("Preparing training data...", 10)
            
            train_loader, val_loader = trainer.prepare_data(
                self.wake_word_files, self.negative_files
            )
            
            # Train model
            if self.progress_callback:
                self.progress_callback("Training model...", 30)
            
            model = trainer.train(
                train_loader, 
                val_loader, 
                progress_callback=self.progress_callback
            )
            
            # Check if model training failed
            if model is None:
                self.result = {
                    "success": False,
                    "error": "Training failed - returned None model"
                }
                if self.progress_callback:
                    self.progress_callback("Error: Training failed to produce a valid model", -1)
                return
            
            # Save model
            if self.progress_callback:
                self.progress_callback("Saving model...", 90)
            
            # Create models directory
            models_dir = Path.home() / ".io_wake_word" / "models"
            models_dir.mkdir(parents=True, exist_ok=True)
            
            model_path = str(models_dir / self.model_name)
            
            try:
                # Try trainer's save method first
                from io_wake_word.models.architecture import save_model
                save_model(model, model_path)
                
                self.result = {
                    "success": True,
                    "model_path": model_path
                }
                
                if self.progress_callback:
                    self.progress_callback("Training complete!", 100)
                
            except Exception as e:
                logger.error(f"Error saving model: {e}")
                self.result = {
                    "success": False,
                    "error": f"Failed to save model: {str(e)}"
                }
                
                if self.progress_callback:
                    self.progress_callback(f"Error: {str(e)}", -1)
            
        except Exception as e:
            logger.error(f"Error training model: {e}")
            
            if self.progress_callback:
                self.progress_callback(f"Error: {e}", -1)
            
            self.result = {
                "success": False,
                "error": str(e)
            }


class TrainingPanel(ctk.CTkFrame):
    """Training panel for Io wake word engine"""
    
    def __init__(
        self, 
        parent, 
        config: Dict[str, Any], 
        on_model_trained: Optional[Callable[[Dict[str, Any]], None]] = None
    ):
        """Initialize training panel
        
        Args:
            parent: Parent widget
            config: Configuration dictionary
            on_model_trained: Callback for when model training completes
        """
        super().__init__(parent)
        
        self.config = config
        self.on_model_trained = on_model_trained
        self.recording_thread = None
        self.training_thread = None
        
        # Training data directories
        self.data_dir = get_data_dir()
        self.wake_word_dir = self.data_dir / "wake_word"
        self.negative_dir = self.data_dir / "negative"
        
        # Create directories if they don't exist
        for directory in [self.data_dir, self.wake_word_dir, self.negative_dir]:
            directory.mkdir(parents=True, exist_ok=True)
        
        # Load existing samples
        self._refresh_files()
        
        # Create UI elements
        self._create_ui()
    
    def _refresh_files(self):
        """Refresh file lists"""
        # Ensure directories exist
        self.wake_word_dir.mkdir(parents=True, exist_ok=True)
        self.negative_dir.mkdir(parents=True, exist_ok=True)
        
        # Refresh file lists with explicit conversion to list
        self.wake_word_files = list(self.wake_word_dir.glob("*.wav"))
        self.negative_files = list(self.negative_dir.glob("*.wav"))
        
        # Log found files
        logger.info(f"Found {len(self.wake_word_files)} wake word files and {len(self.negative_files)} background files")
    
    def _create_ui(self):
        """Create training UI elements"""
        # Configure grid
        self.grid_columnconfigure(0, weight=1)
        self.grid_rowconfigure(3, weight=1)
        
        # Title
        title = ctk.CTkLabel(
            self, 
            text="Training Panel",
            font=ctk.CTkFont(size=20, weight="bold"),
            text_color="#00FFFF"
        )
        title.grid(row=0, column=0, padx=20, pady=(20, 10))
        
        # Instructions
        instructions = ctk.CTkLabel(
            self,
            text="Train your wake word model by recording samples or adding existing WAV files.",
            wraplength=600
        )
        instructions.grid(row=1, column=0, padx=20, pady=(0, 10))
        
        # Main content (scrollable)
        content = ctk.CTkScrollableFrame(self)
        content.grid(row=2, column=0, padx=20, pady=10, sticky="nsew")
        content.grid_columnconfigure(0, weight=1)
        
        # Data directories section
        dir_frame = ctk.CTkFrame(content)
        dir_frame.grid(row=0, column=0, padx=10, pady=10, sticky="ew")
        dir_frame.grid_columnconfigure(1, weight=1)
        
        ctk.CTkLabel(
            dir_frame,
            text="Training Data Directories",
            font=ctk.CTkFont(size=16, weight="bold")
        ).grid(row=0, column=0, columnspan=3, padx=10, pady=(10, 5), sticky="w")
        
        # Wake word directory
        ctk.CTkLabel(
            dir_frame,
            text="Wake Word Directory:"
        ).grid(row=1, column=0, padx=10, pady=(10, 5), sticky="w")
        
        self.wake_dir_label = ctk.CTkLabel(
            dir_frame,
            text=str(self.wake_word_dir),
            anchor="w"
        )
        self.wake_dir_label.grid(row=1, column=1, padx=10, pady=(10, 5), sticky="ew")
        
        wake_dir_btn = ctk.CTkButton(
            dir_frame,
            text="Open",
            command=lambda: self._open_directory(self.wake_word_dir),
            width=60,
            fg_color="#444444",
            hover_color="#666666"
        )
        wake_dir_btn.grid(row=1, column=2, padx=10, pady=(10, 5))
        
        # Negative directory
        ctk.CTkLabel(
            dir_frame,
            text="Background Directory:"
        ).grid(row=2, column=0, padx=10, pady=(10, 5), sticky="w")
        
        self.neg_dir_label = ctk.CTkLabel(
            dir_frame,
            text=str(self.negative_dir),
            anchor="w"
        )
        self.neg_dir_label.grid(row=2, column=1, padx=10, pady=(10, 5), sticky="ew")
        
        neg_dir_btn = ctk.CTkButton(
            dir_frame,
            text="Open",
            command=lambda: self._open_directory(self.negative_dir),
            width=60,
            fg_color="#444444",
            hover_color="#666666"
        )
        neg_dir_btn.grid(row=2, column=2, padx=10, pady=(10, 5))
        
        # Import section
        import_frame = ctk.CTkFrame(dir_frame)
        import_frame.grid(row=3, column=0, columnspan=3, padx=10, pady=10, sticky="ew")
        
        ctk.CTkLabel(
            import_frame,
            text="Import WAV Files",
            font=ctk.CTkFont(size=14)
        ).grid(row=0, column=0, columnspan=2, padx=10, pady=5, sticky="w")
        
        import_wake_btn = ctk.CTkButton(
            import_frame,
            text="Import Wake Word Files",
            command=lambda: self._import_files("wake"),
            fg_color="#00AAAA",
            hover_color="#008888"
        )
        import_wake_btn.grid(row=1, column=0, padx=10, pady=5)
        
        import_neg_btn = ctk.CTkButton(
            import_frame,
            text="Import Background Files",
            command=lambda: self._import_files("background"),
            fg_color="#00AAAA",
            hover_color="#008888"
        )
        import_neg_btn.grid(row=1, column=1, padx=10, pady=5)
        
        refresh_btn = ctk.CTkButton(
            import_frame,
            text="Refresh File Lists",
            command=self._update_file_ui,
            fg_color="#555555",
            hover_color="#777777"
        )
        refresh_btn.grid(row=2, column=0, columnspan=2, padx=10, pady=5)
        
        # Wake word recordings section
        wake_frame = ctk.CTkFrame(content)
        wake_frame.grid(row=1, column=0, padx=10, pady=10, sticky="ew")
        wake_frame.grid_columnconfigure(0, weight=1)
        
        ctk.CTkLabel(
            wake_frame, 
            text="Step 1: Record Wake Word Samples",
            font=ctk.CTkFont(size=16, weight="bold")
        ).grid(row=0, column=0, columnspan=2, padx=10, pady=(10, 5), sticky="w")
        
        self.wake_count = ctk.CTkLabel(
            wake_frame,
            text=f"Current samples: {len(self.wake_word_files)}"
        )
        self.wake_count.grid(row=1, column=0, padx=10, pady=(0, 5), sticky="w")
        
        ctk.CTkLabel(
            wake_frame,
            text="Speak your wake word clearly when recording"
        ).grid(row=2, column=0, padx=10, pady=(0, 5), sticky="w")
        
        wake_btn_frame = ctk.CTkFrame(wake_frame, fg_color="transparent")
        wake_btn_frame.grid(row=3, column=0, padx=10, pady=(0, 10), sticky="ew")
        
        self.record_wake_btn = ctk.CTkButton(
            wake_btn_frame,
            text="Record Wake Word (2s)",
            command=lambda: self._start_recording("wake"),
            fg_color="#00AAAA",
            hover_color="#008888"
        )
        self.record_wake_btn.pack(side="left", padx=(0, 5))
        
        self.stop_wake_btn = ctk.CTkButton(
            wake_btn_frame,
            text="Stop",
            command=self._stop_recording,
            state="disabled",
            fg_color="#AA5555",
            hover_color="#883333"
        )
        self.stop_wake_btn.pack(side="left", padx=5)
        
        self.play_wake_btn = ctk.CTkButton(
            wake_btn_frame,
            text="Play",
            command=lambda: self._play_audio(self.wake_word_files[-1] if self.wake_word_files else None),
            state="disabled" if not self.wake_word_files else "normal",
            fg_color="#555555",
            hover_color="#777777"
        )
        self.play_wake_btn.pack(side="left", padx=5)
        
        self.delete_wake_btn = ctk.CTkButton(
            wake_btn_frame,
            text="Delete Last",
            command=lambda: self._delete_last("wake"),
            state="disabled" if not self.wake_word_files else "normal",
            fg_color="#555555",
            hover_color="#777777"
        )
        self.delete_wake_btn.pack(side="left", padx=5)
        
        # Background noise recordings section
        bg_frame = ctk.CTkFrame(content)
        bg_frame.grid(row=2, column=0, padx=10, pady=10, sticky="ew")
        bg_frame.grid_columnconfigure(0, weight=1)
        
        ctk.CTkLabel(
            bg_frame, 
            text="Step 2: Record Background Noise",
            font=ctk.CTkFont(size=16, weight="bold")
        ).grid(row=0, column=0, columnspan=2, padx=10, pady=(10, 5), sticky="w")
        
        self.bg_count = ctk.CTkLabel(
            bg_frame,
            text=f"Current samples: {len(self.negative_files)}"
        )
        self.bg_count.grid(row=1, column=0, padx=10, pady=(0, 5), sticky="w")
        
        ctk.CTkLabel(
            bg_frame,
            text="Record typical background sounds, music, speech, etc."
        ).grid(row=2, column=0, padx=10, pady=(0, 5), sticky="w")
        
        bg_btn_frame = ctk.CTkFrame(bg_frame, fg_color="transparent")
        bg_btn_frame.grid(row=3, column=0, padx=10, pady=(0, 10), sticky="ew")
        
        self.record_bg_btn = ctk.CTkButton(
            bg_btn_frame,
            text="Record Background (5s)",
            command=lambda: self._start_recording("background"),
            fg_color="#00AAAA",
            hover_color="#008888"
        )
        self.record_bg_btn.pack(side="left", padx=(0, 5))
        
        self.stop_bg_btn = ctk.CTkButton(
            bg_btn_frame,
            text="Stop",
            command=self._stop_recording,
            state="disabled",
            fg_color="#AA5555",
            hover_color="#883333"
        )
        self.stop_bg_btn.pack(side="left", padx=5)
        
        self.play_bg_btn = ctk.CTkButton(
            bg_btn_frame,
            text="Play",
            command=lambda: self._play_audio(self.negative_files[-1] if self.negative_files else None),
            state="disabled" if not self.negative_files else "normal",
            fg_color="#555555",
            hover_color="#777777"
        )
        self.play_bg_btn.pack(side="left", padx=5)
        
        self.delete_bg_btn = ctk.CTkButton(
            bg_btn_frame,
            text="Delete Last",
            command=lambda: self._delete_last("background"),
            state="disabled" if not self.negative_files else "normal",
            fg_color="#555555",
            hover_color="#777777"
        )
        self.delete_bg_btn.pack(side="left", padx=5)
        
        # Training section
        train_frame = ctk.CTkFrame(content)
        train_frame.grid(row=3, column=0, padx=10, pady=10, sticky="ew")
        train_frame.grid_columnconfigure(1, weight=1)
        
        ctk.CTkLabel(
            train_frame, 
            text="Step 3: Train Model",
            font=ctk.CTkFont(size=16, weight="bold")
        ).grid(row=0, column=0, columnspan=2, padx=10, pady=(10, 5), sticky="w")
        
        ctk.CTkLabel(
            train_frame,
            text="Model Name:"
        ).grid(row=1, column=0, padx=10, pady=(10, 5), sticky="w")
        
        self.model_name = ctk.CTkEntry(train_frame)
        self.model_name.grid(row=1, column=1, padx=10, pady=(10, 5), sticky="ew")
        self.model_name.insert(0, "my_wake_word.pth")
        
        ctk.CTkLabel(
            train_frame,
            text="Required: At least 50 wake word samples and 10 background recordings"
        ).grid(row=2, column=0, columnspan=2, padx=10, pady=(10, 5), sticky="w")
        
        self.train_btn = ctk.CTkButton(
            train_frame,
            text="Start Training",
            command=self._start_training,
            state="disabled" if len(self.wake_word_files) < 50 or len(self.negative_files) < 10 else "normal",
            fg_color="#00AAAA",
            hover_color="#008888"
        )
        self.train_btn.grid(row=3, column=0, columnspan=2, padx=10, pady=(10, 20))
        
        # Progress frame
        progress_frame = ctk.CTkFrame(self)
        progress_frame.grid(row=3, column=0, padx=20, pady=(0, 20), sticky="ew")
        progress_frame.grid_columnconfigure(0, weight=1)
        
        self.progress_bar = ctk.CTkProgressBar(progress_frame)
        self.progress_bar.grid(row=0, column=0, padx=10, pady=(10, 5), sticky="ew")
        self.progress_bar.set(0)
        
        self.status_text = ctk.CTkLabel(
            progress_frame,
            text="Ready to begin recording samples"
        )
        self.status_text.grid(row=1, column=0, padx=10, pady=(0, 10), sticky="w")
        
        # Level meter for recording
        self.level_meter = ctk.CTkProgressBar(self)
        self.level_meter.grid(row=4, column=0, padx=20, pady=(0, 20), sticky="ew")
        self.level_meter.set(0)
        
        # Update recording level periodically
        self._update_recording_level()
    
    def _open_directory(self, path: Path):
        """Open directory in file explorer"""
        import subprocess
        import platform
        
        try:
            path_str = str(path.absolute())
            if platform.system() == "Windows":
                os.startfile(path_str)
            elif platform.system() == "Darwin":  # macOS
                subprocess.run(["open", path_str], check=False)
            else:  # Linux
                subprocess.run(["xdg-open", path_str], check=False)
        except Exception as e:
            logger.error(f"Error opening directory: {e}")
            self.status_text.configure(text=f"Error opening directory: {e}")
    
    def _import_files(self, mode: str):
        """Import WAV files from a directory"""
        from tkinter import filedialog
        import shutil
        
        # Select files to import
        filetypes = [("WAV Files", "*.wav")]
        files = filedialog.askopenfilenames(
            title=f"Select {mode} WAV files",
            filetypes=filetypes
        )
        
        if not files:
            return
        
        # Determine target directory
        if mode == "wake":
            target_dir = self.wake_word_dir
        else:
            target_dir = self.negative_dir
        
        # Copy files to target directory
        copied_count = 0
        for file_path in files:
            try:
                # Generate a unique filename
                base_name = f"{mode}_{len(os.listdir(target_dir)) + copied_count + 1}.wav"
                target_path = target_dir / base_name
                
                # Copy the file
                shutil.copy2(file_path, target_path)
                copied_count += 1
                
            except Exception as e:
                logger.error(f"Error importing file {file_path}: {e}")
        
        # Refresh file lists
        self._refresh_files()
        self._update_file_ui()
        
        # Update status
        self.status_text.configure(text=f"Imported {copied_count} files to {mode} directory")
    
    def _update_file_ui(self):
        """Update UI based on current file lists"""
        # Refresh file lists first
        self._refresh_files()
        
        # Update counters
        self.wake_count.configure(text=f"Current samples: {len(self.wake_word_files)}")
        self.bg_count.configure(text=f"Current samples: {len(self.negative_files)}")
        
        # Update button states
        self.play_wake_btn.configure(state="normal" if self.wake_word_files else "disabled")
        self.delete_wake_btn.configure(state="normal" if self.wake_word_files else "disabled")
        self.play_bg_btn.configure(state="normal" if self.negative_files else "disabled")
        self.delete_bg_btn.configure(state="normal" if self.negative_files else "disabled")
        
        # Update training button state
        self.train_btn.configure(
            state="normal" if len(self.wake_word_files) >= 50 and len(self.negative_files) >= 10 else "disabled"
        )
        
        # Update status
        self.status_text.configure(text="File lists refreshed")
    
    def _start_recording(self, mode: str):
        """Start recording audio sample"""
        if self.recording_thread and self.recording_thread.is_alive():
            self._stop_recording()
        
        try:
            if mode == "wake":
                filename = self.wake_word_dir / f"wake_{len(self.wake_word_files) + 1}.wav"
                duration = 2.0
                self.record_wake_btn.configure(state="disabled")
                self.stop_wake_btn.configure(state="normal")
                self.record_bg_btn.configure(state="disabled")
                self.status_text.configure(text="Recording wake word...")
            else:
                filename = self.negative_dir / f"background_{len(self.negative_files) + 1}.wav"
                duration = 5.0
                self.record_bg_btn.configure(state="disabled")
                self.stop_bg_btn.configure(state="normal")
                self.record_wake_btn.configure(state="disabled")
                self.status_text.configure(text="Recording background sounds...")
            
            logger.info(f"Starting recording thread for {mode} to {filename}")
            self.recording_thread = RecordingThread(filename, duration)
            self.recording_thread.start()
            
            # Start progress updater
            self._update_recording_progress(mode, duration)
            
            return True
        except Exception as e:
            logger.error(f"Error starting recording: {e}")
            self.status_text.configure(text=f"Error: {e}")
            return False
    
    def _update_recording_progress(self, mode: str, duration: float):
        """Update progress bar during recording"""
        if not self.recording_thread:
            # Thread doesn't exist
            self._recording_finished(mode)
            return
            
        if not self.recording_thread.is_alive():
            # Check for errors before finishing
            if hasattr(self.recording_thread, 'error') and self.recording_thread.error:
                self.status_text.configure(text=f"Error: {self.recording_thread.error}")
                logger.error(f"Recording error: {self.recording_thread.error}")
                
            # Small delay to ensure file is saved
            self.after(200, lambda: self._recording_finished(mode))
            return
        
        # Update the progress bar
        if hasattr(self.recording_thread, '_start_time'):
            elapsed = time.time() - self.recording_thread._start_time
            progress = min(1.0, elapsed / duration)
            self.progress_bar.set(progress)
        else:
            # No start time yet, just show indeterminate progress
            self.progress_bar.set(0.1)
        
        # Schedule next update
        self.after(100, lambda: self._update_recording_progress(mode, duration))
    
    def _recording_finished(self, mode: str):
        """Handle recording completion"""
        # Enable/disable buttons
        self.record_wake_btn.configure(state="normal")
        self.record_bg_btn.configure(state="normal")
        self.stop_wake_btn.configure(state="disabled")
        self.stop_bg_btn.configure(state="disabled")
        
        # Reset progress bar
        self.progress_bar.set(0)
        
        # Update file lists
        self._refresh_files()
        self._update_file_ui()
        
        # Update status message
        if mode == "wake":
            self.status_text.configure(text="Wake word sample recorded")
        else:
            self.status_text.configure(text="Background sample recorded")
    
    def _stop_recording(self):
        """Stop recording"""
        if self.recording_thread and self.recording_thread.is_alive():
            logger.info("Manually stopping recording")
            self.recording_thread.stop()
            # Let the thread finish
            self.recording_thread.join(timeout=1.0)
            self.status_text.configure(text="Recording stopped manually")
    
    def _update_recording_level(self):
        """Update recording level meter"""
        level = 0.0
        if self.recording_thread and self.recording_thread.is_alive():
            level = self.recording_thread.audio_level
        
        self.level_meter.set(level)
        
        # Schedule next update
        self.after(50, self._update_recording_level)
    
    def _play_audio(self, file_path: Optional[Path]):
        """Play an audio file"""
        if not file_path:
            return
            
        try:
            import platform
            system = platform.system()
            file_path_str = str(file_path)
            
            if system == "Windows":
                os.startfile(file_path_str)
            elif system == "Darwin":  # macOS
                import subprocess
                subprocess.run(["afplay", file_path_str], check=False)
            elif system == "Linux":
                import subprocess
                subprocess.run(["aplay", file_path_str], check=False)
            else:
                logger.warning(f"Audio playback not implemented for {system}")
                self.status_text.configure(text=f"Audio playback not supported on {system}")
                
        except Exception as e:
            logger.error(f"Error playing audio: {e}")
            self.status_text.configure(text=f"Error playing audio: {e}")
    
    def _delete_last(self, mode: str):
        """Delete the last recorded sample"""
        try:
            if mode == "wake" and self.wake_word_files:
                file_to_delete = self.wake_word_files[-1]
                logger.info(f"Deleting wake word sample: {file_to_delete}")
                os.remove(str(file_to_delete))
                self.wake_word_files = self.wake_word_files[:-1]
                self.status_text.configure(text="Wake word sample deleted")
            elif mode == "background" and self.negative_files:
                file_to_delete = self.negative_files[-1]
                logger.info(f"Deleting background sample: {file_to_delete}")
                os.remove(str(file_to_delete))
                self.negative_files = self.negative_files[:-1]
                self.status_text.configure(text="Background sample deleted")
                
            # Update UI
            self._update_file_ui()
            
        except Exception as e:
            logger.error(f"Error deleting file: {e}")
            self.status_text.configure(text=f"Error: {e}")
    
    def _start_training(self):
        """Start training the wake word model"""
        model_name = self.model_name.get()
        if not model_name:
            self.status_text.configure(text="Please enter a model name")
            return
            
        if not model_name.endswith(".pth"):
            model_name += ".pth"
        
        # Disable all buttons during training
        self._set_training_state(True)
        
        # Start training in a separate thread
        logger.info(f"Starting training with {len(self.wake_word_files)} wake word files and {len(self.negative_files)} background files")
        self.training_thread = TrainingThread(
            self.wake_word_files, 
            self.negative_files,
            model_name,
            self._update_training_progress
        )
        self.training_thread.start()
        
        # Check for training completion periodically
        self._check_training_completion()
    
    def _set_training_state(self, training: bool):
        """Set UI state during training"""
        state = "disabled" if training else "normal"
        
        self.record_wake_btn.configure(state=state)
        self.stop_wake_btn.configure(state="disabled")
        self.play_wake_btn.configure(state=state if self.wake_word_files else "disabled")
        self.delete_wake_btn.configure(state=state if self.wake_word_files else "disabled")
        
        self.record_bg_btn.configure(state=state)
        self.stop_bg_btn.configure(state="disabled")
        self.play_bg_btn.configure(state=state if self.negative_files else "disabled")
        self.delete_bg_btn.configure(state=state if self.negative_files else "disabled")
        
        self.train_btn.configure(state=state if len(self.wake_word_files) >= 50 and len(self.negative_files) >= 10 else "disabled")
        self.model_name.configure(state=state)
    
    def _update_training_progress(self, status_text: str, progress_value: int):
        """Update training progress in UI"""
        self.status_text.configure(text=status_text)
        if progress_value >= 0:
            self.progress_bar.set(progress_value / 100)
    
    def _check_training_completion(self):
        """Check if training has completed"""
        if not self.training_thread or not self.training_thread.is_alive():
            if hasattr(self.training_thread, "result"):
                if self.training_thread.result.get("success", False):
                    self._on_training_success(self.training_thread.result)
                else:
                    self._on_training_error(self.training_thread.result.get("error", "Unknown error"))
                
                self.training_thread = None
            return
        
        # Check again later
        self.after(500, self._check_training_completion)
    
    def _on_training_success(self, result: Dict[str, Any]):
        """Handle successful training completion"""
        self._set_training_state(False)
        self.status_text.configure(text="Training completed successfully!")
        self.progress_bar.set(1.0)
        
        # Show success dialog
        dialog = ctk.CTkToplevel(self)
        dialog.title("Training Complete")
        dialog.geometry("400x200")
        dialog.transient(self)
        dialog.grab_set()
        
        # Center the dialog
        dialog.update_idletasks()
        width = dialog.winfo_width()
        height = dialog.winfo_height()
        x = (dialog.winfo_screenwidth() // 2) - (width // 2)
        y = (dialog.winfo_screenheight() // 2) - (height // 2)
        dialog.geometry(f"{width}x{height}+{x}+{y}")
        
        ctk.CTkLabel(
            dialog, 
            text="Training Complete",
            font=ctk.CTkFont(size=18, weight="bold"),
            text_color="#00FFFF"
        ).pack(pady=(20, 10))
        
        ctk.CTkLabel(
            dialog, 
            text="The wake word model has been trained successfully!\n"
                 f"Model saved as: {Path(result['model_path']).name}"
        ).pack(pady=10, padx=20)
        
        ctk.CTkButton(
            dialog, 
            text="OK",
            command=dialog.destroy,
            fg_color="#00AAAA",
            hover_color="#008888"
        ).pack(pady=(10, 20))
        
        # Call completion callback
        if self.on_model_trained:
            self.on_model_trained(result)
    
    def _on_training_error(self, error: str):
        """Handle training error"""
        self._set_training_state(False)
        self.status_text.configure(text=f"Error during training: {error}")
        self.progress_bar.set(0)
        
        # Show error dialog
        dialog = ctk.CTkToplevel(self)
        dialog.title("Training Error")
        dialog.geometry("400x200")
        dialog.transient(self)
        dialog.grab_set()
        
        # Center the dialog
        dialog.update_idletasks()
        width = dialog.winfo_width()
        height = dialog.winfo_height()
        x = (dialog.winfo_screenwidth() // 2) - (width // 2)
        y = (dialog.winfo_screenheight() // 2) - (height // 2)
        dialog.geometry(f"{width}x{height}+{x}+{y}")
        
        ctk.CTkLabel(
            dialog, 
            text="Training Error",
            font=ctk.CTkFont(size=18, weight="bold"),
            text_color="#FF5555"
        ).pack(pady=(20, 10))
        
        ctk.CTkLabel(
            dialog, 
            text=f"An error occurred during training:\n{error}",
            wraplength=350
        ).pack(pady=10, padx=20)
        
        ctk.CTkButton(
            dialog, 
            text="OK",
            command=dialog.destroy,
            fg_color="#00AAAA",
            hover_color="#008888"
        ).pack(pady=(10, 20))
