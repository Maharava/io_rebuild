"""
Io Wake Word Detection Engine Package
A fully offline wake word detection system using a lightweight CNN architecture
"""

__version__ = "0.2.0"

# Import core components for easier access
from io_wake_word.audio import AudioCapture, FeatureExtractor, VoiceActivityDetector
from io_wake_word.models import WakeWordDetector, WakeWordModel, SimpleWakeWordModel, WakeWordTrainer
from io_wake_word.utils import Config, ActionHandler
from io_wake_word.utils.paths import get_base_dir, get_models_dir, get_data_dir, ensure_app_directories