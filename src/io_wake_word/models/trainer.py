"""
Wake Word Trainer - A module for training wake word detection models
from audio samples.
"""
import logging
import os
import random
from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple, Union

import librosa
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import precision_recall_fscore_support
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

from io_wake_word.models.architecture import (WakeWordModel, create_model,
                                           save_model)

logger = logging.getLogger("io_wake_word.models")

class WakeWordDataset(Dataset):
    """Dataset for wake word training"""
    
    def __init__(
        self,
        wake_word_files: List[str],
        negative_files: List[str],
        n_mfcc: int = 13,
        n_fft: int = 2048,
        hop_length: int = 160,
        sample_rate: int = 16000,
    ):
        """Initialize dataset with file paths
        
        Args:
            wake_word_files: List of paths to wake word audio files
            negative_files: List of paths to negative audio files
            n_mfcc: Number of MFCC coefficients
            n_fft: FFT window size
            hop_length: Hop length for feature extraction
            sample_rate: Audio sample rate
        """
        self.wake_word_files = wake_word_files
        self.negative_files = negative_files
        self.n_mfcc = n_mfcc
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.sample_rate = sample_rate
        
        # All files
        self.files = wake_word_files + negative_files
        
        # Labels (1 for wake word, 0 for negative)
        self.labels = [1] * len(wake_word_files) + [0] * len(negative_files)
    
    def __len__(self) -> int:
        """Get dataset size"""
        return len(self.files)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """Get a sample from the dataset
        
        Args:
            idx: Index of the sample
            
        Returns:
            Tuple of (features, label)
        """
        file_path = self.files[idx]
        label = self.labels[idx]
        
        try:
            # Load audio file
            y, sr = librosa.load(file_path, sr=self.sample_rate)
            
            # Pad or truncate to 1 second
            if len(y) < self.sample_rate:
                y = np.pad(y, (0, self.sample_rate - len(y)))
            elif len(y) > self.sample_rate:
                y = y[:self.sample_rate]
            
            # Extract MFCCs
            mfccs = librosa.feature.mfcc(
                y=y, 
                sr=sr, 
                n_mfcc=self.n_mfcc,
                n_fft=self.n_fft, 
                hop_length=self.hop_length
            )
            
            # Ensure consistent dimensions
            if mfccs.shape[1] > 101:
                mfccs = mfccs[:, :101]
            elif mfccs.shape[1] < 101:
                pad_width = 101 - mfccs.shape[1]
                mfccs = np.pad(mfccs, ((0, 0), (0, pad_width)))
            
            # Normalize features
            mfccs = (mfccs - np.mean(mfccs)) / (np.std(mfccs) + 1e-8)
            
            return torch.from_numpy(mfccs).float(), torch.tensor([label]).float()
            
        except Exception as e:
            logger.error(f"Error loading {file_path}: {e}")
            # Return zeros for failed files
            mfccs = np.zeros((self.n_mfcc, 101))
            return torch.from_numpy(mfccs).float(), torch.tensor([0]).float()


class WakeWordTrainer:
    """Train wake word detection models"""
    
    def __init__(
        self,
        n_mfcc: int = 13,
        n_fft: int = 2048,
        hop_length: int = 160,
        num_frames: int = 101,
        use_simple_model: bool = False,
    ):
        """Initialize trainer with feature parameters
        
        Args:
            n_mfcc: Number of MFCC coefficients
            n_fft: FFT window size
            hop_length: Hop length for feature extraction
            num_frames: Number of time frames
            use_simple_model: Whether to use the simplified model
        """
        self.n_mfcc = n_mfcc
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.num_frames = num_frames
        self.use_simple_model = use_simple_model
        
        # Create diagnostics directory
        self.diagnostics_dir = Path.home() / ".io_wake_word" / "training_diagnostics"
        self.diagnostics_dir.mkdir(parents=True, exist_ok=True)
    
    def prepare_data(
        self,
        wake_word_files: List[Union[str, Path]],
        negative_files: List[Union[str, Path]],
        val_split: float = 0.2,
        batch_size: int = 32,
    ) -> Tuple[DataLoader, DataLoader]:
        """Prepare training and validation data loaders
        
        Args:
            wake_word_files: List of paths to wake word audio files
            negative_files: List of paths to negative audio files
            val_split: Fraction of data to use for validation
            batch_size: Batch size for training
            
        Returns:
            Tuple of (train_loader, val_loader)
        """
        if not wake_word_files or not negative_files:
            logger.error("No training files provided")
            return None, None
        
        # Convert Path objects to strings if needed
        wake_word_files = [str(f) for f in wake_word_files]
        negative_files = [str(f) for f in negative_files]
        
        # Shuffle files
        random.seed(42)  # For reproducibility
        random.shuffle(wake_word_files)
        random.shuffle(negative_files)
        
        # Split into training and validation
        n_wake_val = max(1, int(len(wake_word_files) * val_split))
        n_neg_val = max(1, int(len(negative_files) * val_split))
        
        train_wake = wake_word_files[:-n_wake_val]
        val_wake = wake_word_files[-n_wake_val:]
        
        train_neg = negative_files[:-n_neg_val]
        val_neg = negative_files[-n_neg_val:]
        
        # Create datasets
        train_dataset = WakeWordDataset(
            train_wake, train_neg, 
            n_mfcc=self.n_mfcc, 
            n_fft=self.n_fft, 
            hop_length=self.hop_length
        )
        
        val_dataset = WakeWordDataset(
            val_wake, val_neg, 
            n_mfcc=self.n_mfcc, 
            n_fft=self.n_fft, 
            hop_length=self.hop_length
        )
        
        # Create data loaders
        train_loader = DataLoader(
            train_dataset, 
            batch_size=batch_size, 
            shuffle=True, 
            num_workers=0
        )
        
        val_loader = DataLoader(
            val_dataset, 
            batch_size=batch_size, 
            shuffle=False, 
            num_workers=0
        )
        
        logger.info(
            f"Data prepared: {len(train_wake)} wake word and {len(train_neg)} "
            f"negative samples for training, {len(val_wake)} wake word and "
            f"{len(val_neg)} negative samples for validation"
        )
        
        return train_loader, val_loader
    
    def train(
        self,
        train_loader: DataLoader,
        val_loader: DataLoader,
        num_epochs: int = 100,
        patience: int = 20,
        progress_callback: Optional[Callable[[str, int], None]] = None,
    ) -> Optional[nn.Module]:
        """Train the wake word model
        
        Args:
            train_loader: DataLoader for training data
            val_loader: DataLoader for validation data
            num_epochs: Maximum number of training epochs
            patience: Early stopping patience
            progress_callback: Callback for reporting progress
            
        Returns:
            Trained model or None if training failed
        """
        logger.info("Starting model training")
        
        # Create model
        model = create_model(
            n_mfcc=self.n_mfcc, 
            num_frames=self.num_frames, 
            use_simple_model=self.use_simple_model
        )
        
        # Loss and optimizer
        criterion = nn.BCELoss()
        optimizer = optim.Adam(model.parameters(), lr=0.001)
        
        # Training variables
        best_val_loss = float('inf')
        best_model = None
        patience_counter = 0
        
        # Training loop
        for epoch in range(num_epochs):
            model.train()
            running_loss = 0.0
            correct = 0
            total = 0
            
            # Update progress
            if progress_callback:
                progress_value = 30 + int(50 * epoch / num_epochs)
                progress_callback(f"Training epoch {epoch+1}/{num_epochs}...", progress_value)
            
            # Training loop
            for inputs, labels in train_loader:
                # Zero the parameter gradients
                optimizer.zero_grad()
                
                # Forward pass
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                
                # Backward pass and optimize
                loss.backward()
                optimizer.step()
                
                # Statistics
                running_loss += loss.item() * inputs.size(0)
                predicted = (outputs > 0.5).float()
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
            
            train_loss = running_loss / len(train_loader.dataset)
            train_acc = correct / total
            
            # Validation
            model.eval()
            val_loss = 0.0
            val_correct = 0
            val_total = 0
            all_preds = []
            all_labels = []
            
            with torch.no_grad():
                for inputs, labels in val_loader:
                    outputs = model(inputs)
                    loss = criterion(outputs, labels)
                    
                    val_loss += loss.item() * inputs.size(0)
                    predicted = (outputs > 0.5).float()
                    val_total += labels.size(0)
                    val_correct += (predicted == labels).sum().item()
                    
                    all_preds.extend(predicted.cpu().numpy())
                    all_labels.extend(labels.cpu().numpy())
            
            val_loss = val_loss / len(val_loader.dataset)
            val_acc = val_correct / val_total
            
            # Calculate precision, recall, f1
            precision, recall, f1, _ = precision_recall_fscore_support(
                np.array(all_labels).flatten(),
                np.array(all_preds).flatten(),
                average='binary',
                zero_division=0
            )
            
            # Log metrics
            logger.info(
                f"Epoch {epoch+1}/{num_epochs} - "
                f"Train Loss: {train_loss:.4f}, Acc: {train_acc:.4f} | "
                f"Val Loss: {val_loss:.4f}, Acc: {val_acc:.4f}, "
                f"Precision: {precision:.4f}, Recall: {recall:.4f}, F1: {f1:.4f}"
            )
            
            # Save checkpoint
            if epoch % 20 == 0:
                checkpoint_path = self.diagnostics_dir / f"model_epoch_{epoch}.pth"
                torch.save(model.state_dict(), checkpoint_path)
                logger.info(f"Saved checkpoint to {checkpoint_path}")
            
            # Check for improvement
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_model = model.state_dict().copy()
                patience_counter = 0
                
                # Save best model
                best_model_path = self.diagnostics_dir / "model_best.pth"
                torch.save(best_model, best_model_path)
                logger.info(f"New best model saved! Val Loss: {val_loss:.4f}")
            else:
                patience_counter += 1
                if patience_counter >= patience:
                    logger.info(f"Early stopping at epoch {epoch+1}")
                    break
        
        # Final model preparation
        if best_model is not None:
            model.load_state_dict(best_model)
            logger.info("Loaded best model from validation")
            
            # Final inference test
            logger.info("Testing inference on validation data")
            self.test_inference(model, val_loader)
            
            return model
        else:
            logger.error("Training failed to produce a valid model")
            return None
    
    def test_inference(
        self, 
        model: nn.Module, 
        val_loader: DataLoader
    ) -> Tuple[float, float, float]:
        """Test inference on validation data
        
        Args:
            model: Trained model
            val_loader: DataLoader for validation data
            
        Returns:
            Tuple of (precision, recall, f1)
        """
        model.eval()
        all_preds = []
        all_labels = []
        
        with torch.no_grad():
            for inputs, labels in val_loader:
                outputs = model(inputs)
                predicted = (outputs > 0.5).float()
                
                all_preds.extend(predicted.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
        
        # Calculate metrics
        precision, recall, f1, _ = precision_recall_fscore_support(
            np.array(all_labels).flatten(),
            np.array(all_preds).flatten(),
            average='binary',
            zero_division=0
        )
        
        logger.info(
            f"Final model performance: "
            f"Precision: {precision:.4f}, Recall: {recall:.4f}, F1: {f1:.4f}"
        )
        
        return precision, recall, f1
    
    def train_from_directories(
        self,
        wake_word_dir: Union[str, Path],
        negative_dir: Union[str, Path],
        output_path: Union[str, Path],
        progress_callback: Optional[Callable[[str, int], None]] = None,
    ) -> bool:
        """Train a model from directories of audio files
        
        This is a convenience method for training from directories
        
        Args:
            wake_word_dir: Directory containing wake word audio files
            negative_dir: Directory containing negative audio files
            output_path: Path to save the trained model
            progress_callback: Callback for reporting progress
            
        Returns:
            True if training was successful, False otherwise
        """
        try:
            # Convert to Path objects
            wake_word_dir = Path(wake_word_dir)
            negative_dir = Path(negative_dir)
            output_path = Path(output_path)
            
            # Get audio files
            wake_word_files = list(wake_word_dir.glob("*.wav"))
            negative_files = list(negative_dir.glob("*.wav"))
            
            if not wake_word_files:
                logger.error(f"No wake word files found in {wake_word_dir}")
                return False
                
            if not negative_files:
                logger.error(f"No negative files found in {negative_dir}")
                return False
            
            # Prepare data
            if progress_callback:
                progress_callback("Preparing training data...", 10)
            
            train_loader, val_loader = self.prepare_data(wake_word_files, negative_files)
            if train_loader is None or val_loader is None:
                return False
            
            # Train model
            if progress_callback:
                progress_callback("Training model...", 30)
            
            model = self.train(train_loader, val_loader, progress_callback=progress_callback)
            if model is None:
                return False
            
            # Save model
            if progress_callback:
                progress_callback("Saving model...", 90)
            
            success = save_model(model, output_path)
            
            if success and progress_callback:
                progress_callback("Training complete!", 100)
            
            return success
            
        except Exception as e:
            logger.error(f"Error in train_from_directories: {e}")
            return False