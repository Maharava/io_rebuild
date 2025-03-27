"""
Commands implementation for Io wake word CLI
"""
import logging
import os
import sys
import time
from pathlib import Path
from typing import List, Optional

import numpy as np

from io_wake_word.audio.capture import AudioCapture
from io_wake_word.audio.features import FeatureExtractor
from io_wake_word.diagnostics.analyzer import (analyze_audio_files,
                                             analyze_feature_extraction,
                                             analyze_model)
from io_wake_word.diagnostics.recovery import recover_model
from io_wake_word.models.detector import WakeWordDetector
from io_wake_word.models.trainer import WakeWordTrainer
from io_wake_word.utils.actions import ActionHandler
from io_wake_word.utils.config import Config
from io_wake_word.utils.paths import (ensure_app_directories, get_data_dir,
                                    get_models_dir, resolve_model_path)

logger = logging.getLogger("io_wake_word.cli")

def setup_logging(verbose: bool = False) -> None:
    """Set up logging for CLI
    
    Args:
        verbose: Whether to enable verbose logging
    """
    log_level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=log_level,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        handlers=[
            logging.StreamHandler(),
        ]
    )

def init_command(args) -> int:
    """Initialize Io wake word detection
    
    Args:
        args: Command-line arguments
        
    Returns:
        Exit code
    """
    # Create directories
    ensure_app_directories()
    
    print("Io wake word detection engine initialized successfully.")
    print(f"Data directory: {get_data_dir()}")
    print(f"Models directory: {get_models_dir()}")
    
    # List audio devices
    try:
        capture = AudioCapture()
        devices = capture.list_devices()
        
        if devices:
            print("\nAvailable audio devices:")
            for device in devices:
                print(f"  {device['index']}: {device['name']}")
        else:
            print("\nNo audio devices found.")
    except Exception as e:
        logger.error(f"Error listing audio devices: {e}")
        print("\nError listing audio devices.")
    
    return 0

def train_command(args) -> int:
    """Train a wake word model
    
    Args:
        args: Command-line arguments
        
    Returns:
        Exit code
    """
    # Validate arguments
    wake_word_dir = Path(args.wake_word_dir)
    negative_dir = Path(args.negative_dir)
    output_path = Path(args.output)
    
    if not wake_word_dir.exists():
        print(f"Error: Wake word directory not found: {wake_word_dir}")
        return 1
    
    if not negative_dir.exists():
        print(f"Error: Negative directory not found: {negative_dir}")
        return 1
    
    # Get wake word files
    wake_word_files = list(wake_word_dir.glob("*.wav"))
    if not wake_word_files:
        print(f"Error: No WAV files found in wake word directory")
        return 1
    
    # Get negative files
    negative_files = list(negative_dir.glob("*.wav"))
    if not negative_files:
        print(f"Error: No WAV files found in negative directory")
        return 1
    
    print(f"Found {len(wake_word_files)} wake word files and {len(negative_files)} negative files")
    
    if len(wake_word_files) < 10:
        print(f"Warning: Less than 10 wake word files may lead to poor model performance")
    
    if len(negative_files) < 10:
        print(f"Warning: Less than 10 negative files may lead to poor model performance")
    
    # Create trainer
    trainer = WakeWordTrainer(
        use_simple_model=args.simple_model,
    )
    
    # Progress callback
    def progress_callback(message, progress):
        print(f"{message} ({progress}%)")
    
    # Train model
    print("Starting model training...")
    success = trainer.train_from_directories(
        wake_word_dir=wake_word_dir,
        negative_dir=negative_dir,
        output_path=output_path,
        progress_callback=progress_callback,
    )
    
    if success:
        print(f"Model trained successfully and saved to {output_path}")
        return 0
    else:
        print("Error training model")
        return 1

def detect_command(args) -> int:
    """Run wake word detection
    
    Args:
        args: Command-line arguments
        
    Returns:
        Exit code
    """
    # Validate arguments
    model_path_str = args.model
    model_path = resolve_model_path(model_path_str)
    
    if not model_path.exists():
        print(f"Error: Model file not found: {model_path_str}")
        print(f"Looked in default models directory: {get_models_dir()}")
        return 1
    
    # Create detector
    detector = WakeWordDetector(
        model_path=model_path,
        threshold=args.threshold,
    )
    
    # Create AudioCapture
    capture = AudioCapture(
        device_index=args.device,
    )
    
    # Create FeatureExtractor
    feature_extractor = FeatureExtractor()
    
    # Create ActionHandler if action is specified
    action_handler = None
    if args.action:
        action_config = {
            "type": args.action,
            "params": {},
        }
        
        if args.action == "notification":
            action_config["params"]["message"] = args.message or "Wake word detected!"
        elif args.action == "command":
            if not args.command:
                print("Error: --command is required when action is 'command'")
                return 1
            action_config["params"]["command"] = args.command
        elif args.action == "custom_script":
            if not args.script:
                print("Error: --script is required when action is 'custom_script'")
                return 1
            action_config["params"]["script_path"] = args.script
        
        action_handler = ActionHandler(action_config)
    
    # Register callback
    def on_detection(confidence):
        timestamp = time.strftime("%H:%M:%S")
        print(f"[{timestamp}] Wake word detected! Confidence: {confidence:.4f}")
        
        if action_handler:
            action_handler.trigger()
    
    detector.register_callback(on_detection)
    
    # Start detection
    print(f"Starting wake word detection with model: {model_path}")
    print(f"Threshold: {args.threshold}")
    print("Listening... (Press Ctrl+C to stop)")
    
    try:
        # Start capture
        capture.start()
        
        # Process audio until interrupted
        while True:
            # Get audio buffer
            audio = capture.get_buffer()
            
            if len(audio) > 0:
                # Extract features
                features = feature_extractor.extract(audio)
                
                # Detect wake word
                if features is not None:
                    detector.detect(features)
            
            # Sleep to avoid busy waiting
            time.sleep(0.01)
            
    except KeyboardInterrupt:
        print("\nStopping detection...")
    finally:
        # Clean up
        capture.stop()
        if action_handler:
            action_handler.shutdown()
    
    return 0

def analyze_command(args) -> int:
    """Analyze audio files or model
    
    Args:
        args: Command-line arguments
        
    Returns:
        Exit code
    """
    if args.audio:
        # Analyze audio directory
        audio_dir = Path(args.audio)
        if not audio_dir.exists():
            print(f"Error: Audio directory not found: {audio_dir}")
            return 1
        
        print(f"Analyzing audio files in {audio_dir}...")
        results = analyze_audio_files(audio_dir, limit=args.limit)
        
        if results["files_analyzed"] == 0:
            print("No audio files were analyzed.")
            return 1
        
        print(f"Analyzed {results['files_analyzed']} files:")
        print(f"  Duration (s): {results['mean_duration']:.2f} mean, {results['min_duration']:.2f} min, {results['max_duration']:.2f} max")
        print(f"  Energy: {results['mean_energy']:.6f} mean, {results['min_energy']:.6f} min, {results['max_energy']:.6f} max")
        print(f"  Sample rates: {', '.join(map(str, results['sample_rates']))}")
        if results["has_multichannel"]:
            print("  Warning: Some files have multiple channels, which may cause issues")
        
    elif args.model:
        # Analyze model
        model_path_str = args.model
        model_path = resolve_model_path(model_path_str)
        
        if not model_path.exists():
            print(f"Error: Model file not found: {model_path_str}")
            print(f"Looked in default models directory: {get_models_dir()}")
            return 1
        
        print(f"Analyzing model: {model_path}...")
        results = analyze_model(model_path)
        
        if not results["success"]:
            print(f"Error analyzing model: {results.get('error', 'Unknown error')}")
            return 1
        
        print(f"Model type: {results['model_type']}")
        print(f"Total parameters: {results['total_params']}")
        print(f"Trainable parameters: {results['trainable_params']}")
        print("Layers:")
        for layer in results["layers"]:
            print(f"  {layer['name']}: {layer['type']} ({layer['params']} parameters)")
    
    return 0

def recover_command(args) -> int:
    """Recover a model from checkpoint
    
    Args:
        args: Command-line arguments
        
    Returns:
        Exit code
    """
    output_path = args.output
    if output_path:
        output_path = Path(output_path)
    
    print("Attempting to recover model from checkpoint...")
    success = recover_model(output_path)
    
    if success:
        print(f"Model recovered successfully to: {output_path or get_models_dir() / 'recovered_model.pth'}")
        return 0
    else:
        print("Failed to recover model")
        return 1

def record_command(args) -> int:
    """Record audio samples
    
    Args:
        args: Command-line arguments
        
    Returns:
        Exit code
    """
    # Validate arguments
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Create AudioCapture
    capture = AudioCapture(
        device_index=args.device,
    )
    
    # Record samples
    for i in range(args.count):
        filename = output_dir / f"{args.prefix or 'sample'}_{i+1}.wav"
        
        if args.countdown > 0:
            print(f"Recording sample {i+1}/{args.count} to {filename} in...")
            for j in range(args.countdown, 0, -1):
                print(f"{j}...", end=" ", flush=True)
                time.sleep(1)
            print("Recording...")
        else:
            print(f"Recording sample {i+1}/{args.count} to {filename}...")
        
        success = capture.save_sample(filename, args.duration)
        
        if success:
            print(f"Sample recorded to {filename}")
        else:
            print(f"Error recording sample to {filename}")
        
        if i < args.count - 1:
            if args.delay > 0:
                print(f"Waiting {args.delay} seconds before next recording...")
                time.sleep(args.delay)
    
    return 0

def add_init_parser(subparsers):
    """Add init subcommand parser
    
    Args:
        subparsers: Subparsers object
    """
    parser = subparsers.add_parser(
        "init",
        help="Initialize Io wake word detection",
    )
    parser.set_defaults(func=init_command)

def add_train_parser(subparsers):
    """Add train subcommand parser
    
    Args:
        subparsers: Subparsers object
    """
    parser = subparsers.add_parser(
        "train",
        help="Train a wake word model",
    )
    parser.add_argument(
        "--wake-word-dir",
        help="Directory containing wake word audio files",
        required=True,
    )
    parser.add_argument(
        "--negative-dir",
        help="Directory containing negative audio files",
        required=True,
    )
    parser.add_argument(
        "--output",
        help="Path to save the trained model",
        required=True,
    )
    parser.add_argument(
        "--simple-model",
        help="Use a simpler model architecture",
        action="store_true",
    )
    parser.set_defaults(func=train_command)

def add_detect_parser(subparsers):
    """Add detect subcommand parser
    
    Args:
        subparsers: Subparsers object
    """
    parser = subparsers.add_parser(
        "detect",
        help="Run wake word detection",
    )
    parser.add_argument(
        "--model",
        help="Path to the model file",
        required=True,
    )
    parser.add_argument(
        "--device",
        help="Audio device index",
        type=int,
    )
    parser.add_argument(
        "--threshold",
        help="Detection threshold (0.0-1.0)",
        type=float,
        default=0.85,
    )
    parser.add_argument(
        "--action",
        help="Action to take on detection",
        choices=["notification", "command", "custom_script"],
    )
    parser.add_argument(
        "--message",
        help="Notification message",
    )
    parser.add_argument(
        "--command",
        help="Command to run",
    )
    parser.add_argument(
        "--script",
        help="Path to custom script",
    )
    parser.set_defaults(func=detect_command)

def add_analyze_parser(subparsers):
    """Add analyze subcommand parser
    
    Args:
        subparsers: Subparsers object
    """
    parser = subparsers.add_parser(
        "analyze",
        help="Analyze audio files or model",
    )
    parser.add_argument(
        "--audio",
        help="Directory containing audio files to analyze",
    )
    parser.add_argument(
        "--model",
        help="Path to model file to analyze",
    )
    parser.add_argument(
        "--limit",
        help="Maximum number of files to analyze",
        type=int,
    )
    parser.set_defaults(func=analyze_command)

def add_recover_parser(subparsers):
    """Add recover subcommand parser
    
    Args:
        subparsers: Subparsers object
    """
    parser = subparsers.add_parser(
        "recover",
        help="Recover a model from checkpoint",
    )
    parser.add_argument(
        "--output",
        help="Path to save the recovered model",
    )
    parser.set_defaults(func=recover_command)

def add_record_parser(subparsers):
    """Add record subcommand parser
    
    Args:
        subparsers: Subparsers object
    """
    parser = subparsers.add_parser(
        "record",
        help="Record audio samples",
    )
    parser.add_argument(
        "--output-dir",
        help="Directory to save audio samples",
        required=True,
    )
    parser.add_argument(
        "--device",
        help="Audio device index",
        type=int,
    )
    parser.add_argument(
        "--count",
        help="Number of samples to record",
        type=int,
        default=1,
    )
    parser.add_argument(
        "--duration",
        help="Recording duration in seconds",
        type=float,
        default=3.0,
    )
    parser.add_argument(
        "--delay",
        help="Delay between recordings in seconds",
        type=float,
        default=1.0,
    )
    parser.add_argument(
        "--countdown",
        help="Countdown before recording in seconds",
        type=int,
        default=3,
    )
    parser.add_argument(
        "--prefix",
        help="Filename prefix for samples",
    )
    parser.set_defaults(func=record_command)