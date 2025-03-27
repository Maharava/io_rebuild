# Io Wake Word Detection

A lightweight, privacy-focused wake word detection system that runs entirely offline. This Python package provides tools for training custom wake word models, detecting wake words in real-time, and acting on detections.

## Features

- **Privacy-First Design**: All processing happens on your device - no data sent to the cloud
- **Lightweight Architecture**: Uses a compact CNN model that runs efficiently on CPUs
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

## Quick Start

### Command Line Interface

Initialize the system and list available audio devices:

```bash
# Initialize and view available audio devices
io-wake-word init
```

Record wake word samples:

```bash
# Record 10 samples of your wake word
io-wake-word record --output-dir ~/wake_word_data/wake_word --count 10 --prefix wake
```

Record background samples:

```bash
# Record background noise samples
io-wake-word record --output-dir ~/wake_word_data/negative --count 10 --duration 5 --prefix background
```

Train a model:

```bash
# Train a wake word model
io-wake-word train --wake-word-dir ~/wake_word_data/wake_word --negative-dir ~/wake_word_data/negative --output ~/my_wake_word_model.pth
```

Detect wake words:

```bash
# Run detection with notification when wake word is detected
io-wake-word detect --model ~/my_wake_word_model.pth --action notification --message "Wake word detected!"
```

### Python API

```python
import numpy as np
from io_wake_word.audio import AudioCapture, FeatureExtractor
from io_wake_word.models import WakeWordDetector

# Create components
capture = AudioCapture()
extractor = FeatureExtractor()
detector = WakeWordDetector(model_path="path/to/model.pth")

# Register a detection callback
def on_detection(confidence):
    print(f"Wake word detected with confidence {confidence:.4f}")
    
detector.register_callback(on_detection)

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
        import time
        time.sleep(0.01)
        
except KeyboardInterrupt:
    # Clean up on exit
    capture.stop()
```

### Graphical Application

```bash
# Launch the GUI application
io-wake-word-app
```

## Training Your Own Wake Word

For best results:

1. Record 50+ samples of your wake word in various tones and speeds
2. Record 10+ minutes of background noise from your typical environment
3. Use a consistent microphone for both training and detection
4. Aim for wake words with 2-4 syllables for better accuracy

## Project Structure

- `src/io_wake_word/`: Core package
  - `audio/`: Audio processing modules (capture, feature extraction)
  - `models/`: Neural network models and detection logic
  - `utils/`: Configuration and utility functions
  - `diagnostics/`: Analysis and debugging tools
  - `cli/`: Command-line interface
- `apps/`: Standalone applications
  - `io_app/`: GUI application

## Command Reference

| Command | Description |
|---------|-------------|
| `init` | Initialize the system and show available devices |
| `record` | Record audio samples for training |
| `train` | Train a wake word model |
| `detect` | Run wake word detection |
| `analyze` | Analyze audio files or models |
| `recover` | Recover models from checkpoints |

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

This project is licensed under the MIT License - see the LICENSE file for details.
