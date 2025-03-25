"""
Configuration panel for Io wake word engine UI
"""
import logging
import os
import threading
import time
import tkinter as tk
from pathlib import Path
from tkinter import filedialog
from typing import Any, Callable, Dict, List, Optional

import customtkinter as ctk

from io_wake_word.audio.capture import AudioCapture
from io_wake_word.models.detector import WakeWordDetector
from io_wake_word.utils.paths import get_models_dir

logger = logging.getLogger("io_app.ui")

class ConfigPanel(ctk.CTkFrame):
    """Configuration panel using CustomTkinter"""
    
    def __init__(
        self,
        parent,
        config: Dict[str, Any],
        audio_capture: AudioCapture,
        detector: WakeWordDetector,
        on_save_callback: Optional[Callable] = None,
    ):
        """Initialize configuration panel
        
        Args:
            parent: Parent widget
            config: Configuration dictionary
            audio_capture: AudioCapture instance
            detector: WakeWordDetector instance
            on_save_callback: Callback for configuration save
        """
        super().__init__(parent)
        
        self.config = config.copy()
        self.audio_capture = audio_capture
        self.detector = detector
        self.on_save_callback = on_save_callback
        
        # Create UI elements
        self._create_ui()
    
    def _create_ui(self):
        """Create configuration UI elements"""
        # Configure grid
        self.grid_columnconfigure(0, weight=1)
        
        # Title
        title = ctk.CTkLabel(
            self, 
            text="Io Configuration",
            font=ctk.CTkFont(size=20, weight="bold"),
            text_color="#00FFFF"
        )
        title.grid(row=0, column=0, padx=20, pady=(20, 10))
        
        # Main content frame with scrolling
        content_frame = ctk.CTkScrollableFrame(self)
        content_frame.grid(row=1, column=0, padx=20, pady=10, sticky="nsew")
        content_frame.grid_columnconfigure(1, weight=1)
        
        # Audio device selection
        row = 0
        ctk.CTkLabel(content_frame, text="Audio Device:", anchor="w").grid(
            row=row, column=0, padx=10, pady=10, sticky="w"
        )
        
        # Get audio devices
        devices = self.audio_capture.list_devices()
        device_options = [f"{d['index']}: {d['name']}" for d in devices]
        
        # Find current device index
        current_device = 0
        if self.config.get("audio_device") is not None:
            for i, d in enumerate(devices):
                if d["index"] == self.config["audio_device"]:
                    current_device = i
                    break
        
        self.device_var = tk.StringVar(value=device_options[current_device] if device_options else "")
        self.device_menu = ctk.CTkOptionMenu(
            content_frame,
            values=device_options,
            variable=self.device_var
        )
        self.device_menu.grid(row=row, column=1, padx=10, pady=10, sticky="ew")
        
        # Test audio button
        test_audio_btn = ctk.CTkButton(
            content_frame,
            text="Test",
            command=self._test_audio,
            width=60,
            fg_color="#00AAAA",
            hover_color="#008888"
        )
        test_audio_btn.grid(row=row, column=2, padx=10, pady=10)
        
        # Wake word model
        row += 1
        ctk.CTkLabel(content_frame, text="Wake Word Model:", anchor="w").grid(
            row=row, column=0, padx=10, pady=10, sticky="w"
        )
        
        # Get models
        models_dir = get_models_dir()
        model_files = list(models_dir.glob("*.pth"))
        model_names = [model.name for model in model_files]
        
        # Default value
        current_model = ""
        if self.config.get("model_path"):
            model_name = Path(self.config["model_path"]).name
            if model_name in model_names:
                current_model = model_name
        
        self.model_var = tk.StringVar(value=current_model)
        model_frame = ctk.CTkFrame(content_frame)
        model_frame.grid(row=row, column=1, padx=10, pady=10, sticky="ew")
        model_frame.grid_columnconfigure(0, weight=1)
        
        self.model_menu = ctk.CTkOptionMenu(
            model_frame,
            values=model_names if model_names else ["No models found"],
            variable=self.model_var
        )
        self.model_menu.pack(side="left", fill="x", expand=True)
        
        browse_btn = ctk.CTkButton(
            content_frame,
            text="Browse",
            command=self._browse_model,
            width=60,
            fg_color="#444444",
            hover_color="#666666"
        )
        browse_btn.grid(row=row, column=2, padx=10, pady=10)
        
        # Detection threshold
        row += 1
        ctk.CTkLabel(content_frame, text="Detection Threshold:", anchor="w").grid(
            row=row, column=0, padx=10, pady=10, sticky="w"
        )
        
        threshold_frame = ctk.CTkFrame(content_frame)
        threshold_frame.grid(row=row, column=1, columnspan=2, padx=10, pady=10, sticky="ew")
        
        self.threshold_var = tk.DoubleVar(value=self.config.get("threshold", 0.85))
        self.threshold_slider = ctk.CTkSlider(
            threshold_frame,
            from_=0.5,
            to=0.99,
            variable=self.threshold_var,
            number_of_steps=49
        )
        self.threshold_slider.pack(side="left", fill="x", expand=True, padx=10)
        
        threshold_label = ctk.CTkLabel(threshold_frame, textvariable=self.threshold_var, width=40)
        threshold_label.pack(side="right", padx=10)
        
        # Debounce time
        row += 1
        ctk.CTkLabel(content_frame, text="Debounce Time (s):", anchor="w").grid(
            row=row, column=0, padx=10, pady=10, sticky="w"
        )
        
        debounce_frame = ctk.CTkFrame(content_frame)
        debounce_frame.grid(row=row, column=1, columnspan=2, padx=10, pady=10, sticky="ew")
        
        self.debounce_var = tk.DoubleVar(value=self.config.get("debounce_time", 3.0))
        self.debounce_slider = ctk.CTkSlider(
            debounce_frame,
            from_=0.5,
            to=10.0,
            variable=self.debounce_var,
            number_of_steps=19
        )
        self.debounce_slider.pack(side="left", fill="x", expand=True, padx=10)
        
        debounce_label = ctk.CTkLabel(debounce_frame, textvariable=self.debounce_var, width=40)
        debounce_label.pack(side="right", padx=10)
        
        # Action type
        row += 1
        ctk.CTkLabel(content_frame, text="Action on Detection:", anchor="w").grid(
            row=row, column=0, padx=10, pady=10, sticky="w")
        
        action_frame = ctk.CTkFrame(content_frame)
        action_frame.grid(row=row, column=1, columnspan=2, padx=10, pady=10, sticky="ew")
        
        self.action_type = tk.StringVar(value=self.config.get("action", {}).get("type", "notification"))
        
        action_option1 = ctk.CTkRadioButton(
            action_frame, text="Notification", 
            variable=self.action_type, value="notification",
            command=self._on_action_type_change
        )
        action_option1.pack(anchor="w", padx=10, pady=5)
        
        action_option2 = ctk.CTkRadioButton(
            action_frame, text="Run Command", 
            variable=self.action_type, value="command",
            command=self._on_action_type_change
        )
        action_option2.pack(anchor="w", padx=10, pady=5)
        
        action_option3 = ctk.CTkRadioButton(
            action_frame, text="Custom Script", 
            variable=self.action_type, value="custom_script",
            command=self._on_action_type_change
        )
        action_option3.pack(anchor="w", padx=10, pady=5)
        
        # Action parameters
        row += 1
        self.param_label = ctk.CTkLabel(content_frame, text="Parameter:", anchor="w")
        self.param_label.grid(row=row, column=0, padx=10, pady=10, sticky="w")
        
        self.param_entry = ctk.CTkEntry(content_frame)
        self.param_entry.grid(row=row, column=1, columnspan=2, padx=10, pady=10, sticky="ew")
        
        # Update parameter label based on action type
        self._on_action_type_change()
        self._set_param_value()
        
        # Startup options
        row += 1
        options_frame = ctk.CTkFrame(content_frame)
        options_frame.grid(row=row, column=0, columnspan=3, padx=10, pady=10, sticky="ew")
        
        self.autostart_var = tk.BooleanVar(value=self.config.get("autostart", False))
        autostart_check = ctk.CTkCheckBox(
            options_frame, 
            text="Start detection automatically on launch",
            variable=self.autostart_var
        )
        autostart_check.pack(anchor="w", padx=10, pady=5)
        
        self.minimize_var = tk.BooleanVar(value=self.config.get("minimize_on_close", True))
        minimize_check = ctk.CTkCheckBox(
            options_frame, 
            text="Minimize to system tray when closing window",
            variable=self.minimize_var
        )
        minimize_check.pack(anchor="w", padx=10, pady=5)
        
        # Buttons
        row += 1
        buttons_frame = ctk.CTkFrame(self)
        buttons_frame.grid(row=2, column=0, padx=20, pady=(10, 20), sticky="ew")
        
        # Test configuration button
        test_btn = ctk.CTkButton(
            buttons_frame,
            text="Test Configuration",
            command=self._test_config,
            fg_color="#00AAAA",
            hover_color="#008888"
        )
        test_btn.pack(side="left", padx=10, pady=10)
        
        # Save configuration button
        save_btn = ctk.CTkButton(
            buttons_frame,
            text="Save Configuration",
            command=self._save_config,
            fg_color="#00AAAA",
            hover_color="#008888"
        )
        save_btn.pack(side="right", padx=10, pady=10)
    
    def _browse_model(self):
        """Open file browser to select model"""
        models_dir = get_models_dir()
        
        model_path = filedialog.askopenfilename(
            title="Select Model File",
            initialdir=str(models_dir),
            filetypes=[("PyTorch Models", "*.pth")]
        )
        
        if model_path:
            # Update model menu
            model_name = Path(model_path).name
            
            # Update combobox values
            current_values = self.model_menu.cget("values")
            if "No models found" in current_values:
                current_values = []
                
            if model_name not in current_values:
                self.model_menu.configure(values=[*current_values, model_name])
            
            self.model_var.set(model_name)
    
    def _test_audio(self):
        """Test audio input device"""
        # Get selected device index
        device_str = self.device_var.get()
        if not device_str:
            return
            
        device_index = int(device_str.split(":")[0])
        
        # Create popup dialog
        dialog = ctk.CTkToplevel(self)
        dialog.title("Audio Test")
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
        
        # Create labels
        ctk.CTkLabel(
            dialog, 
            text="Audio Device Test",
            font=ctk.CTkFont(size=16, weight="bold")
        ).pack(pady=(20, 10))
        
        level_frame = ctk.CTkFrame(dialog)
        level_frame.pack(fill="x", padx=20, pady=10)
        
        level_label = ctk.CTkLabel(level_frame, text="Audio Level:")
        level_label.pack(side="left", padx=10)
        
        # Audio level bar
        audio_level = ctk.CTkProgressBar(level_frame, width=200)
        audio_level.pack(side="left", padx=10, fill="x", expand=True)
        audio_level.set(0)
        
        info_text = ctk.CTkLabel(
            dialog, 
            text="Speak into your microphone to test.\nThe bar should move when you speak."
        )
        info_text.pack(pady=10)
        
        # Close button
        close_btn = ctk.CTkButton(
            dialog, 
            text="Close",
            command=dialog.destroy,
            fg_color="#00AAAA",
            hover_color="#008888"
        )
        close_btn.pack(pady=(10, 20))
        
        # Create temporary audio capture for testing
        test_capture = AudioCapture(device_index=device_index)
        test_capture.start()
        
        # Update audio level periodically
        def update_level():
            if not dialog.winfo_exists():
                test_capture.stop()
                return
                
            try:
                level = test_capture.get_audio_level()
                audio_level.set(min(1.0, level * 5))  # Amplify for better visualization
            except Exception as e:
                logger.error(f"Error getting audio level: {e}")
            
            # Schedule next update
            dialog.after(100, update_level)
        
        # Start updating level
        update_level()
    
    def _on_action_type_change(self):
        """Update UI based on selected action type"""
        action_type = self.action_type.get()
        
        if action_type == "notification":
            self.param_label.configure(text="Message:")
        elif action_type == "command":
            self.param_label.configure(text="Command:")
        elif action_type == "custom_script":
            self.param_label.configure(text="Script Path:")
        
        self._set_param_value()
    
    def _set_param_value(self):
        """Set parameter value based on action type"""
        action_type = self.action_type.get()
        action_params = self.config.get("action", {}).get("params", {})
        
        param_value = ""
        if action_type == "notification":
            param_value = action_params.get("message", "Wake word detected!")
        elif action_type == "command":
            param_value = action_params.get("command", "")
        elif action_type == "custom_script":
            param_value = action_params.get("script_path", "")
        
        self.param_entry.delete(0, tk.END)
        self.param_entry.insert(0, param_value)
    
    def _test_config(self):
        """Test current configuration"""
        # Get selected threshold
        threshold = self.threshold_var.get()
        
        # Temporarily set detector threshold
        original_threshold = self.detector.threshold
        self.detector.set_threshold(threshold)
        
        # Create dialog
        dialog = ctk.CTkToplevel(self)
        dialog.title("Configuration Test")
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
        
        # Testing message
        ctk.CTkLabel(
            dialog, 
            text="Configuration Test",
            font=ctk.CTkFont(size=16, weight="bold")
        ).pack(pady=(20, 10))
        
        ctk.CTkLabel(
            dialog, 
            text="Speak your wake word now to test the detection.\n"
                 "The detector is listening with the current threshold setting."
        ).pack(pady=10, padx=20)
        
        # Status indicator
        status_var = tk.StringVar(value="Listening...")
        status = ctk.CTkLabel(
            dialog,
            textvariable=status_var,
            font=ctk.CTkFont(size=14)
        )
        status.pack(pady=10)
        
        # Close button
        close_btn = ctk.CTkButton(
            dialog, 
            text="Close",
            command=lambda: self._close_test_dialog(dialog, original_threshold),
            fg_color="#00AAAA",
            hover_color="#008888"
        )
        close_btn.pack(pady=(10, 20))
        
        # Set up detection callback
        def on_detection(confidence):
            status_var.set(f"Detected! Confidence: {confidence:.2f}")
            status.configure(text_color="#00FFFF")
            dialog.after(2000, lambda: status_var.set("Listening..."))
            dialog.after(2000, lambda: status.configure(text_color="white"))
        
        # Register and store original callback
        original_callbacks = self.detector.callbacks.copy()
        self.detector.callbacks = [on_detection]
        
        # Restore original callbacks when dialog is closed
        def on_dialog_closed():
            self.detector.callbacks = original_callbacks
            self.detector.set_threshold(original_threshold)
            if dialog.winfo_exists():
                dialog.destroy()
            
        dialog.protocol("WM_DELETE_WINDOW", on_dialog_closed)
    
    def _close_test_dialog(self, dialog, original_threshold):
        """Close test dialog and restore original threshold"""
        # Restore original threshold
        self.detector.set_threshold(original_threshold)
        
        # Close dialog
        dialog.destroy()
    
    def _save_config(self):
        """Save configuration from UI values"""
        try:
            # Parse audio device
            device_str = self.device_var.get()
            device_index = None
            if device_str:
                device_index = int(device_str.split(":")[0])
            
            # Parse model path
            model_name = self.model_var.get()
            model_path = None
            if model_name and model_name != "No models found":
                model_path = str(get_models_dir() / model_name)
            
            # Parse threshold and debounce
            threshold = self.threshold_var.get()
            debounce_time = self.debounce_var.get()
            
            # Parse action type and parameters
            action_type = self.action_type.get()
            param_value = self.param_entry.get()
            
            action_params = {}
            if action_type == "notification":
                action_params = {"message": param_value or "Wake word detected!"}
            elif action_type == "command":
                action_params = {"command": param_value}
            elif action_type == "custom_script":
                action_params = {"script_path": param_value}
            
            # Update configuration
            self.config["audio_device"] = device_index
            self.config["model_path"] = model_path
            self.config["threshold"] = threshold
            self.config["debounce_time"] = debounce_time
            self.config["action"] = {
                "type": action_type,
                "params": action_params
            }
            
            # Additional options
            self.config["autostart"] = self.autostart_var.get()
            self.config["minimize_on_close"] = self.minimize_var.get()
            
            # Call save callback
            if self.on_save_callback:
                self.on_save_callback(self.config)
            
            # Show success message
            dialog = ctk.CTkToplevel(self)
            dialog.title("Configuration Saved")
            dialog.geometry("300x150")
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
                text="Configuration Saved",
                font=ctk.CTkFont(size=16, weight="bold")
            ).pack(pady=(20, 10))
            
            ctk.CTkLabel(
                dialog, 
                text="Your configuration has been saved successfully."
            ).pack(pady=10)
            
            ctk.CTkButton(
                dialog, 
                text="OK",
                command=dialog.destroy,
                fg_color="#00AAAA",
                hover_color="#008888"
            ).pack(pady=(10, 20))
            
            return True
            
        except Exception as e:
            logger.error(f"Error saving configuration: {e}")
            
            # Show error message
            dialog = ctk.CTkToplevel(self)
            dialog.title("Error")
            dialog.geometry("300x150")
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
                text="Error",
                font=ctk.CTkFont(size=16, weight="bold"),
                text_color="#FF5555"
            ).pack(pady=(20, 10))
            
            ctk.CTkLabel(
                dialog, 
                text=f"Error saving configuration:\n{str(e)}"
            ).pack(pady=10)
            
            ctk.CTkButton(
                dialog, 
                text="OK",
                command=dialog.destroy,
                fg_color="#00AAAA",
                hover_color="#008888"
            ).pack(pady=(10, 20))
            
            return False