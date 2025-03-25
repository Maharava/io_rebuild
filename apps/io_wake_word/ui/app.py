"""
Main application window for Io wake word engine using CustomTkinter
"""
import logging
import threading
from pathlib import Path
from typing import Optional

try:
    import customtkinter as ctk
    from PIL import Image
    import pystray
except ImportError:
    raise ImportError("UI dependencies not installed. Run: pip install io-wake-word[ui]")

from io_wake_word import AudioCapture, FeatureExtractor, WakeWordDetector
from io_wake_word.utils.actions import ActionHandler
from io_wake_word.utils.config import Config
from io_wake_word.utils.paths import get_base_dir, get_models_dir

from io_app.ui.config_panel import ConfigPanel
from io_app.ui.training_panel import TrainingPanel

logger = logging.getLogger("io_app.ui")

# Set CustomTkinter appearance
ctk.set_appearance_mode("dark")
ctk.set_default_color_theme("blue")

class AudioProcessor:
    """Audio processing for the UI application"""
    
    def __init__(self, config):
        """Initialize audio processor with configuration
        
        Args:
            config: Configuration dictionary
        """
        self.config = config
        self.running = False
        
        # Initialize components
        self.feature_extractor = FeatureExtractor(
            sample_rate=config["sample_rate"],
            frame_size=config["frame_size"],
            n_mfcc=config["feature_extraction"]["n_mfcc"],
            n_fft=config["feature_extraction"]["n_fft"],
            hop_length=config["feature_extraction"]["hop_length"],
        )
        
        self.detector = WakeWordDetector(
            model_path=config["model_path"],
            threshold=config["threshold"],
            debounce_time=config["debounce_time"],
        )
        
        self.action_handler = ActionHandler(
            action_config=config["action"],
            debounce_time=config["debounce_time"],
        )
        
        # Audio capture with callback
        self.audio_capture = AudioCapture(
            device_index=config["audio_device"],
            sample_rate=config["sample_rate"],
            frame_size=config["frame_size"],
            callback=self._process_audio,
        )
    
    def _process_audio(self, audio_frame):
        """Process audio frame"""
        if not self.running:
            return
        
        try:
            # Extract features
            features = self.feature_extractor.extract(audio_frame)
            
            # Detect wake word
            if features is not None:
                is_detected, confidence = self.detector.detect(features)
                
                # Trigger action if wake word detected
                if is_detected:
                    self.action_handler.trigger()
        except Exception as e:
            logger.error(f"Error processing audio: {e}")
    
    def set_ui_callback(self, callback):
        """Set UI callback for detections
        
        Args:
            callback: Function to call on detection
        """
        self.detector.register_callback(callback)
    
    def start(self):
        """Start audio processing"""
        if self.running:
            return
        
        self.running = True
        self.audio_capture.start()
        logger.info("Audio processor started")
    
    def stop(self):
        """Stop audio processing"""
        if not self.running:
            return
        
        self.running = False
        self.audio_capture.stop()
        logger.info("Audio processor stopped")
    
    def update_config(self, config):
        """Update configuration
        
        Args:
            config: New configuration dictionary
            
        Returns:
            True if successful, False otherwise
        """
        restart_needed = False
        load_success = True
        
        # Check if audio device changed
        if config["audio_device"] != self.config["audio_device"]:
            restart_needed = True
        
        # Check if model path changed
        model_path_changed = config.get("model_path") != self.config.get("model_path")
        
        # Update configuration
        self.config = config
        
        # Update detector threshold
        threshold = config.get("threshold", 0.85)
        self.detector.set_threshold(threshold)
        
        # Update model if path changed
        if model_path_changed and config.get("model_path"):
            load_success = self.detector.load_model(config["model_path"])
        
        # Update action handler
        self.action_handler.update_config(config["action"], config["debounce_time"])
        
        # Restart audio capture if needed
        if restart_needed and self.running:
            self.stop()
            
            # Update audio capture
            self.audio_capture = AudioCapture(
                device_index=config["audio_device"],
                sample_rate=config["sample_rate"],
                frame_size=config["frame_size"],
                callback=self._process_audio,
            )
            
            # Update feature extractor
            self.feature_extractor = FeatureExtractor(
                sample_rate=config["sample_rate"],
                frame_size=config["frame_size"],
                n_mfcc=config["feature_extraction"]["n_mfcc"],
                n_fft=config["feature_extraction"]["n_fft"],
                hop_length=config["feature_extraction"]["hop_length"],
            )
            
            self.start()
        
        return load_success


class IoApp:
    """Main application window with system tray integration"""
    
    def __init__(self):
        """Initialize the application"""
        self.config = Config.load()
        self.window = None
        self.tray_icon = None
        self.is_running = True
        
        # Get asset paths
        self.assets_dir = Path(__file__).parent / "assets"
        self.icon_path = self.assets_dir / "io.png"
        self.ico_path = self.assets_dir / "io_icon.ico"
        
        # Create audio processor
        self.audio_processor = AudioProcessor(self.config)
        
        # Create the main window or system tray based on preferences
        if self.config.get("start_minimized", False):
            self._create_system_tray()
        else:
            self._create_main_window()
    
    def _create_main_window(self):
        """Create the main application window"""
        self.window = ctk.CTk()
        self.window.title("Io Wake Word Engine")
        self.window.geometry("800x600")
        self.window.protocol("WM_DELETE_WINDOW", self._on_close)
        
        # Configure grid
        self.window.grid_columnconfigure(0, weight=1)
        self.window.grid_rowconfigure(0, weight=1)
        
        # Create tabview
        tabview = ctk.CTkTabview(self.window, fg_color=("#DCE4EE", "#2B2B2B"))
        tabview.grid(row=0, column=0, padx=20, pady=20, sticky="nsew")
        
        # Create tabs
        dashboard_tab = tabview.add("Dashboard")
        config_tab = tabview.add("Configuration")
        training_tab = tabview.add("Training")
        
        # Configure tabs
        dashboard_tab.grid_columnconfigure(0, weight=1)
        dashboard_tab.grid_rowconfigure(0, weight=1)
        config_tab.grid_columnconfigure(0, weight=1)
        config_tab.grid_rowconfigure(0, weight=1)
        training_tab.grid_columnconfigure(0, weight=1)
        training_tab.grid_rowconfigure(0, weight=1)
        
        # Dashboard content
        self._create_dashboard(dashboard_tab)
        
        # Configuration panel
        config_panel = ConfigPanel(
            config_tab, 
            self.config, 
            self.audio_processor.audio_capture,
            self.audio_processor.detector,
            self._on_config_save
        )
        config_panel.grid(row=0, column=0, padx=20, pady=20, sticky="nsew")
        
        # Training panel
        training_panel = TrainingPanel(
            training_tab,
            self.config,
            self._on_model_trained
        )
        training_panel.grid(row=0, column=0, padx=20, pady=20, sticky="nsew")
        
        # Bottom status bar
        self.status_bar = ctk.CTkLabel(
            self.window, 
            text="Io Wake Word Engine - Ready",
            anchor="w"
        )
        self.status_bar.grid(row=1, column=0, padx=10, pady=(0, 10), sticky="ew")
        
        # Start the audio processor if configured
        if self.config.get("autostart", False):
            self._toggle_detection(True)
    
    def _create_dashboard(self, parent):
        """Create dashboard with status and controls"""
        frame = ctk.CTkFrame(parent)
        frame.grid(row=0, column=0, padx=20, pady=20, sticky="nsew")
        
        # Configure grid
        frame.grid_columnconfigure(0, weight=1)
        frame.grid_rowconfigure(3, weight=1)
        
        # Title
        title = ctk.CTkLabel(
            frame, 
            text="Io Wake Word Engine",
            font=ctk.CTkFont(size=24, weight="bold"),
            text_color="#00FFFF"
        )
        title.grid(row=0, column=0, padx=20, pady=(20, 10))
        
        # Status indicators
        status_frame = ctk.CTkFrame(frame)
        status_frame.grid(row=1, column=0, padx=20, pady=10, sticky="ew")
        status_frame.grid_columnconfigure((0, 1), weight=1)
        
        # Detection status
        self.detection_status = ctk.CTkLabel(
            status_frame, 
            text="Detection: Inactive", 
            font=ctk.CTkFont(size=16)
        )
        self.detection_status.grid(row=0, column=0, padx=10, pady=10, sticky="w")
        
        # Model status
        model_name = "No model loaded"
        if self.config.get("model_path"):
            model_name = Path(self.config["model_path"]).name
        
        self.model_status = ctk.CTkLabel(
            status_frame, 
            text=f"Model: {model_name}", 
            font=ctk.CTkFont(size=16)
        )
        self.model_status.grid(row=0, column=1, padx=10, pady=10, sticky="w")
        
        # Add a detection indicator (visual feedback)
        self.detection_indicator = ctk.CTkLabel(
            status_frame,
            text="●",
            font=ctk.CTkFont(size=24),
            text_color="gray"
        )
        self.detection_indicator.grid(row=0, column=2, padx=10, pady=10, sticky="e")
        
        # Controls
        control_frame = ctk.CTkFrame(frame)
        control_frame.grid(row=2, column=0, padx=20, pady=10, sticky="ew")
        
        # Toggle button
        self.toggle_btn = ctk.CTkButton(
            control_frame,
            text="Start Detection",
            command=lambda: self._toggle_detection(not self.audio_processor.running),
            fg_color="#00AAAA",
            hover_color="#008888"
        )
        self.toggle_btn.pack(side="left", padx=10, pady=10)
        
        # Minimize to tray button
        minimize_btn = ctk.CTkButton(
            control_frame,
            text="Minimize to Tray",
            command=self._minimize_to_tray,
            fg_color="#444444",
            hover_color="#666666"
        )
        minimize_btn.pack(side="right", padx=10, pady=10)
        
        # Activity log
        log_frame = ctk.CTkFrame(frame)
        log_frame.grid(row=3, column=0, padx=20, pady=(10, 20), sticky="nsew")
        log_frame.grid_columnconfigure(0, weight=1)
        log_frame.grid_rowconfigure(1, weight=1)
        
        log_label = ctk.CTkLabel(log_frame, text="Activity Log", anchor="w")
        log_label.grid(row=0, column=0, padx=10, pady=(10, 0), sticky="w")
        
        self.log_text = ctk.CTkTextbox(log_frame, height=200, state="disabled")
        self.log_text.grid(row=1, column=0, padx=10, pady=10, sticky="nsew")
        
        # Set up detection callback
        self.audio_processor.set_ui_callback(self._on_detection)
    
    def _on_detection(self, confidence):
        """Handle wake word detection for UI updates"""
        # Update the log
        self._log(f"Wake word detected with confidence: {confidence:.4f}")
        
        # Flash the indicator
        self.detection_indicator.configure(text_color="#00FFFF")
        
        # Reset the indicator after 1 second
        if self.window:
            self.window.after(1000, lambda: self.detection_indicator.configure(text_color="gray"))
    
    def _create_system_tray(self):
        """Create system tray icon"""
        # Get icon image
        if self.icon_path.exists():
            icon_image = Image.open(self.icon_path)
        else:
            # Create a fallback icon if file doesn't exist
            icon_image = self._create_fallback_icon()
        
        # Create the tray icon
        self.tray_icon = pystray.Icon(
            "IoWakeWordDetector",
            icon_image,
            "Io Wake Word Engine",
            menu=pystray.Menu(
                pystray.MenuItem("Open", self._on_tray_open),
                pystray.MenuItem("Start Detection", self._on_tray_start),
                pystray.MenuItem("Stop Detection", self._on_tray_stop),
                pystray.MenuItem("Exit", self._on_tray_exit)
            )
        )
        
        # Start the tray icon in a separate thread
        threading.Thread(
            target=self.tray_icon.run,
            daemon=True
        ).start()
    
    def _create_fallback_icon(self, size=(64, 64)):
        """Create a simple icon image for the system tray as fallback"""
        from PIL import Image, ImageDraw
        
        image = Image.new('RGB', size, color=(0, 0, 0))
        dc = ImageDraw.Draw(image)
        
        # Draw a simple "Io" symbol (circle with line)
        dc.ellipse([(16, 16), (48, 48)], outline="#00FFFF", width=2)
        dc.line([(32, 12), (32, 52)], fill="#00FFFF", width=2)
        
        return image
    
    def _toggle_detection(self, enable):
        """Toggle wake word detection"""
        if enable:
            self.audio_processor.start()
            if self.toggle_btn:
                self.toggle_btn.configure(text="Stop Detection")
                self.detection_status.configure(text="Detection: Active", text_color="#00FFFF")
            self._log("Wake word detection started")
        else:
            self.audio_processor.stop()
            if self.toggle_btn:
                self.toggle_btn.configure(text="Start Detection")
                self.detection_status.configure(text="Detection: Inactive", text_color="white")
            self._log("Wake word detection stopped")
    
    def _log(self, message):
        """Add a message to the log"""
        if not hasattr(self, 'log_text') or self.log_text is None:
            return
            
        self.log_text.configure(state="normal")
        self.log_text.insert("end", f"{message}\n")
        self.log_text.see("end")
        self.log_text.configure(state="disabled")
    
    def _on_config_save(self, new_config):
        """Handle configuration updates"""
        self.config = new_config
        Config.save(new_config)
        
        # Update the audio processor
        self.audio_processor.update_config(new_config)
        
        # Update UI elements
        if self.model_status:
            model_name = "No model loaded"
            if new_config.get("model_path"):
                model_name = Path(new_config["model_path"]).name
            self.model_status.configure(text=f"Model: {model_name}")
        
        self._log("Configuration updated")
    
    def _on_model_trained(self, result):
        """Handle model training completion"""
        if result and result.get("success") and "model_path" in result:
            self.config["model_path"] = result["model_path"]
            Config.save(self.config)
            
            # Update the audio processor
            success = self.audio_processor.update_config(self.config)
            
            # Update UI elements
            if self.model_status:
                model_name = Path(result["model_path"]).name
                self.model_status.configure(text=f"Model: {model_name}")
            
            if success:
                self._log(f"New model trained and loaded: {model_name}")
            else:
                self._log(f"Model trained but could not be loaded. There may be a compatibility issue.")
    
    def _minimize_to_tray(self):
        """Minimize the application to system tray"""
        self.window.withdraw()
        self._create_system_tray()
        self._log("Application minimized to system tray")
    
    def _on_close(self):
        """Handle window close event"""
        if self.config.get("minimize_on_close", True):
            self._minimize_to_tray()
        else:
            self._exit_app()
    
    def _exit_app(self):
        """Exit the application"""
        # Stop detection
        self.audio_processor.stop()
        
        # Stop the tray icon if it exists
        if self.tray_icon:
            self.tray_icon.stop()
        
        # Mark as not running
        self.is_running = False
        
        # Close the window if it exists
        if self.window:
            self.window.quit()
    
    def _on_tray_open(self, icon, item):
        """Handle tray menu open action"""
        # Stop the tray icon
        icon.stop()
        self.tray_icon = None
        
        # Show the window
        if not self.window:
            self._create_main_window()
        else:
            self.window.deiconify()
    
    def _on_tray_start(self, icon, item):
        """Handle tray menu start action"""
        self._toggle_detection(True)
    
    def _on_tray_stop(self, icon, item):
        """Handle tray menu stop action"""
        self._toggle_detection(False)
    
    def _on_tray_exit(self, icon, item):
        """Handle tray menu exit action"""
        icon.stop()
        self.tray_icon = None
        self._exit_app()
    
    def run(self):
        """Run the application"""
        if self.window:
            self.window.mainloop()
            
        # Keep the main thread alive if only the tray is running
        while self.is_running and not self.window:
            import time
            time.sleep(0.1)