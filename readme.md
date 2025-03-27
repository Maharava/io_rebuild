# Io Wake Word Detection Engine

**WORK IN PROGESS!!!** If you have somehow stumbled upon this pre-actual release, know that ```pip install``` doesn't work from the global repos, you need to clone this and install it manually. The code needs considerably more testing before it's ready.

A lightweight, privacy-focused wake word detection system that runs entirely offline. This Python package provides tools for training custom wake words, detecting them in real-time, and executing actions when wake words are detected.

## Features

- **Privacy-First Design**: All processing happens on your device - no data sent to the cloud
- **Lightweight Architecture**: Uses a compact CNN model (~100K parameters) that runs efficiently on CPUs
- **Customisable**: Train your own wake words with simple audio recording tools
- **Flexible Integration**: Use as a Python library, CLI tool, or standalone application
- **Modern Package Structure**: Clean, typed codebase with proper documentation

## Installation

### Basic Installation

```bash
# Install the base package
pip install io-wake-word
```

### With UI Components

```bash
# Install with UI support (for the graphical application)
pip install io-wake-word[ui]
```

### Development Installation

```bash
# Clone the repository
git clone https://github.com/yourusername/io-wake-word.git
cd io-wake-word

# Install in development mode with dev dependencies
pip install -e ".[ui,dev]"
```

## Standalone Usage

### Command Line Interface

The package provides a comprehensive command-line interface for all operations:

#### Initialize and Test Audio

```bash
# Initialize and view available audio devices
io-wake-word init
```

#### Record Training Data

```bash
# Record wake word samples
io-wake-word record --output-dir ~/wake_word_data/wake_word --count 10 --prefix wake

# Record background noise samples
io-wake-word record --output-dir ~/wake_word_data/negative --count 10 --duration 5 --prefix background
```

#### Train a Model

```bash
# Train a wake word model
io-wake-word train --wake-word-dir ~/wake_word_data/wake_word --negative-dir ~/wake_word_data/negative --output ~/my_wake_word_model.pth
```

#### Run Detection

```bash
# Run detection with notification when wake word is detected
io-wake-word detect --model ~/my_wake_word_model.pth --action notification --message "Wake word detected!"

# Run detection that executes a command
io-wake-word detect --model ~/my_wake_word_model.pth --action command --command "python my_script.py"
```

#### Analyze Audio or Models

```bash
# Analyze audio files
io-wake-word analyze --audio ~/wake_word_data/wake_word

# Analyze a trained model
io-wake-word analyze --model ~/my_wake_word_model.pth
```

### Graphical Application

The graphical application provides a user-friendly interface for recording samples, training models, and running detection:

```bash
# First ensure you've installed with UI components
pip install io-wake-word[ui]

# Launch the GUI application
python -m apps.io_app.main
```

## Integration into Larger Projects

The `io_wake_word` package is designed to be easily integrated into larger Python projects. Here are common integration patterns:

### Basic Detection Integration

```python
import time
from io_wake_word.audio import AudioCapture, FeatureExtractor
from io_wake_word.models import WakeWordDetector

# Create components
capture = AudioCapture()
extractor = FeatureExtractor()
detector = WakeWordDetector(model_path="path/to/model.pth")

# Register a detection callback
def on_wake_word(confidence):
    print(f"Wake word detected with confidence {confidence:.4f}")
    # Trigger your application's functionality here
    
detector.register_callback(on_wake_word)

# Start audio capture
capture.start()

# Process in a loop
try:
    while True:
        # Get audio buffer
        audio = capture.get_buffer()
        
        # Extract features
        features = extractor.extract(audio)
        
        # Detect wake word
        if features is not None:
            detector.detect(features)
            
        # Sleep to avoid high CPU usage
        time.sleep(0.01)
        
except KeyboardInterrupt:
    # Clean up on exit
    capture.stop()
```

### Custom Action Handler Integration

Use the ActionHandler to trigger custom functionality when a wake word is detected:

```python
from io_wake_word.utils.actions import ActionHandler

# Create a custom action configuration
action_config = {
    "type": "command",
    "params": {
        "command": "python your_application_script.py --with-args"
    }
}

# Create action handler
action_handler = ActionHandler(action_config)

# In your wake word callback
def on_wake_word(confidence):
    print(f"Wake word detected with confidence {confidence:.4f}")
    # Trigger your action
    action_handler.trigger()
```

### Integration with Threading

For applications that need to run detection in a background thread:

```python
import threading
import queue
import time

class WakeWordThread(threading.Thread):
    def __init__(self, model_path, detection_queue):
        super().__init__()
        self.daemon = True
        self.model_path = model_path
        self.detection_queue = detection_queue
        self.running = True
        
    def run(self):
        from io_wake_word.audio import AudioCapture, FeatureExtractor
        from io_wake_word.models import WakeWordDetector
        
        # Create components
        capture = AudioCapture()
        extractor = FeatureExtractor()
        detector = WakeWordDetector(model_path=self.model_path)
        
        # Define callback that puts events in the queue
        def on_detection(confidence):
            self.detection_queue.put(("wake_word", confidence))
            
        detector.register_callback(on_detection)
        
        # Start capture
        capture.start()
        
        # Process while running
        while self.running:
            audio = capture.get_buffer()
            features = extractor.extract(audio)
            if features is not None:
                detector.detect(features)
            time.sleep(0.01)
            
        # Clean up
        capture.stop()
    
    def stop(self):
        self.running = False

# In your main application:
detection_queue = queue.Queue()
wake_word_thread = WakeWordThread("model.pth", detection_queue)
wake_word_thread.start()

# Process detection events
try:
    while True:
        try:
            event_type, confidence = detection_queue.get(timeout=0.1)
            if event_type == "wake_word":
                print(f"Wake word detected: {confidence}")
                # Your application logic here
        except queue.Empty:
            # No detection events, continue
            pass
except KeyboardInterrupt:
    wake_word_thread.stop()
    wake_word_thread.join()
```

### Integration with Asynchronous Applications

For asynchronous applications (e.g., with asyncio):

```python
import asyncio
from io_wake_word.audio import AudioCapture, FeatureExtractor
from io_wake_word.models import WakeWordDetector

class AsyncWakeWordDetector:
    def __init__(self, model_path):
        self.model_path = model_path
        self.capture = AudioCapture()
        self.extractor = FeatureExtractor()
        self.detector = WakeWordDetector(model_path=model_path)
        self.callbacks = []
        
        # Register internal callback
        self.detector.register_callback(self._on_detection)
        
    def register_callback(self, callback):
        self.callbacks.append(callback)
        
    def _on_detection(self, confidence):
        # Convert regular callback to async tasks
        for callback in self.callbacks:
            asyncio.create_task(callback(confidence))
    
    async def start(self):
        self.capture.start()
        
    async def stop(self):
        self.capture.stop()
    
    async def run_detection(self):
        while True:
            audio = self.capture.get_buffer()
            features = self.extractor.extract(audio)
            if features is not None:
                self.detector.detect(features)
            await asyncio.sleep(0.01)

# In your async application:
async def on_wake_word(confidence):
    print(f"Wake word detected: {confidence}")
    # Your async application logic here

async def main():
    detector = AsyncWakeWordDetector("model.pth")
    detector.register_callback(on_wake_word)
    
    await detector.start()
    
    try:
        detection_task = asyncio.create_task(detector.run_detection())
        # Your other async tasks here
        await asyncio.gather(detection_task)
    except asyncio.CancelledError:
        await detector.stop()

# Run the async application
asyncio.run(main())
```

## Training Custom Wake Words

For best results when training your own wake words:

1. Record at least 50 samples of your wake word
   - Vary your tone, speed, and distance from the microphone
   - Include samples from different speakers if possible
   
2. Record at least 10 minutes of background noise
   - Include typical ambient sounds from your environment
   - Add samples of speech that isn't your wake word
   - Include music or other audio that might be playing
   
3. Use a consistent microphone for both training and detection
   - Ideally use the same microphone you'll use in production
   
4. Choose wake words with 2-4 syllables for better accuracy
   - Single-syllable wake words often have higher false positive rates
   - Very long wake words may be harder to detect reliably

```python
from io_wake_word.models.trainer import WakeWordTrainer

# Create a trainer
trainer = WakeWordTrainer()

# Train from directories
trainer.train_from_directories(
    wake_word_dir="path/to/wake_word_samples",
    negative_dir="path/to/background_samples",
    output_path="path/to/output/model.pth",
    progress_callback=lambda msg, progress: print(f"{msg} - {progress}%")
)
```

## Configuration

The package uses a configuration file stored in `~/.io_wake_word/config/config.json`. You can programmatically access and modify the configuration:

```python
from io_wake_word.utils.config import Config

# Load current configuration
config = Config.load()

# Modify configuration
config["threshold"] = 0.80
config["action"] = {
    "type": "command",
    "params": {"command": "your_command_here"}
}

# Save configuration
Config.save(config)
```

## API Reference

### Audio Module

- `AudioCapture`: Handles microphone input and buffering
- `FeatureExtractor`: Converts audio frames to MFCC features
- `VoiceActivityDetector`: Filters out silent audio frames

### Models Module

- `WakeWordModel`: Standard CNN model for wake word detection
- `SimpleWakeWordModel`: Simplified CNN model (smaller)
- `WakeWordDetector`: Provides detection interface with callbacks
- `WakeWordTrainer`: Handles model training

### Utils Module

- `ActionHandler`: Executes actions when wake words are detected
- `Config`: Manages configuration loading/saving
- `paths`: Utility functions for file/directory management

## Requirements

- Python 3.9+
- PyAudio 0.2.11+
- NumPy 1.20+
- PyTorch 1.9+
- Librosa 0.8+
- TKinter (for UI components)

## Project Structure

- `src/io_wake_word/`: Core package
  - `audio/`: Audio processing modules (capture, feature extraction)
  - `models/`: Neural network models and detection logic
  - `utils/`: Configuration and utility functions
  - `diagnostics/`: Analysis and debugging tools
  - `cli/`: Command-line interface
- `apps/`: Standalone applications
  - `io_app/`: GUI application

## License

This project is licensed under the MIT License - see the LICENSE file for details.
