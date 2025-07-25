"""
Training script for news headline topic classification.

This module contains the training loop, evaluation functions, and
main training pipeline for the news headline classifier.
"""

import os
import json
import logging
from pathlib import Path
from typing import Dict, List, Tuple, Optional

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset, random_split
import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

from model import NewsHeadlineClassifier
from utils import (
    setup_logging, load_and_preprocess_data, create_vocabulary,
    texts_to_sequences, pad_sequences, save_model_artifacts
)


class Trainer:
    """
    Trainer class for the news headline classifier.
    
    Args:
        model (nn.Module): PyTorch model to train.
        device (torch.device): Device to run training on.
        model_save_dir (str): Directory to save model artifacts.
    """
    
    def __init__(self, model: nn.Module, device: torch.device, model_save_dir: str):
        self.model = model.to(device)
        self.device = device
        self.model_save_dir = Path(model_save_dir)
        self.model_save_dir.mkdir(parents=True, exist_ok=True)
        
        # Training history
        self.train_losses = []
        self.val_losses = []
        self.val_accuracies = []
        
        # Setup logging
        self.logger = setup_logging()
    
    def train_epoch(
        self, 
        train_loader: DataLoader, 
        optimizer: optim.Optimizer, 
        criterion: nn.Module
    ) -> float:
        """
        Train the model for one epoch.
        
        Args:
            train_loader (DataLoader): Training data loader.
            optimizer (optim.Optimizer): Optimizer for training.
            criterion (nn.Module): Loss function.
        
        Returns:
            float: Average training loss for the epoch.
        """
        self.model.train()
        total_loss = 0.0
        num_batches = 0
        
        for batch_idx, batch in enumerate(train_loader):
            input_ids, lengths, labels = [x.to(self.device) for x in batch]
            
            # Zero gradients
            optimizer.zero_grad()
            
            # Forward pass
            logits = self.model(input_ids, lengths)
            loss = criterion(logits, labels)
            
            # Backward pass
            loss.backward()
            
            # Gradient clipping to prevent exploding gradients
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            
            # Update weights
            optimizer.step()
            
            total_loss += loss.item()
            num_batches += 1
            
            if batch_idx % 50 == 0:
                self.logger.info(
                    f'Batch {batch_idx}/{len(train_loader)}, Loss: {loss.item():.4f}'
                )
        
        return total_loss / num_batches
    
    def evaluate(
        self, 
        val_loader: DataLoader, 
        criterion: nn.Module
    ) -> Tuple[float, float, np.ndarray, np.ndarray]:
        """
        Evaluate the model on validation data.
        
        Args:
            val_loader (DataLoader): Validation data loader.
            criterion (nn.Module): Loss function.
        
        Returns:
            Tuple[float, float, np.ndarray, np.ndarray]: 
                Average loss, accuracy, predictions, true labels.
        """
        self.model.eval()
        total_loss = 0.0
        num_batches = 0
        all_predictions = []
        all_labels = []
        
        with torch.no_grad():
            for batch in val_loader:
                input_ids, lengths, labels = [x.to(self.device) for x in batch]
                
                # Forward pass
                logits = self.model(input_ids, lengths)
                loss = criterion(logits, labels)
                
                # Get predictions
                predictions = torch.argmax(logits, dim=-1)
                
                total_loss += loss.item()
                num_batches += 1
                
                all_predictions.extend(predictions.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
        
        avg_loss = total_loss / num_batches
        accuracy = accuracy_score(all_labels, all_predictions)
        
        return avg_loss, accuracy, np.array(all_predictions), np.array(all_labels)
    
    def train(
        self,
        train_loader: DataLoader,
        val_loader: DataLoader,
        num_epochs: int,
        learning_rate: float = 1e-3,
        weight_decay: float = 1e-5,
        patience: int = 5
    ) -> Dict:
        """
        Full training loop with early stopping.
        
        Args:
            train_loader (DataLoader): Training data loader.
            val_loader (DataLoader): Validation data loader.
            num_epochs (int): Maximum number of epochs.
            learning_rate (float): Learning rate for optimizer.
            weight_decay (float): Weight decay for L2 regularization.
            patience (int): Early stopping patience.
        
        Returns:
            Dict: Training history and final metrics.
        """
        # Setup optimizer and loss function
        optimizer = optim.Adam(
            self.model.parameters(), 
            lr=learning_rate, 
            weight_decay=weight_decay
        )
        criterion = nn.CrossEntropyLoss()
        
        # Early stopping variables
        best_val_loss = float('inf')
        patience_counter = 0
        best_model_state = None
        
        self.logger.info(f"Starting training for {num_epochs} epochs...")
        
        for epoch in range(num_epochs):
            self.logger.info(f"\nEpoch {epoch + 1}/{num_epochs}")
            
            # Train
            train_loss = self.train_epoch(train_loader, optimizer, criterion)
            
            # Validate
            val_loss, val_accuracy, val_preds, val_labels = self.evaluate(val_loader, criterion)
            
            # Store history
            self.train_losses.append(train_loss)
            self.val_losses.append(val_loss)
            self.val_accuracies.append(val_accuracy)
            
            self.logger.info(
                f"Train Loss: {train_loss:.4f}, "
                f"Val Loss: {val_loss:.4f}, "
                f"Val Accuracy: {val_accuracy:.4f}"
            )
            
            # Early stopping check
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_counter = 0
                best_model_state = self.model.state_dict().copy()
                self.logger.info("New best model saved!")
            else:
                patience_counter += 1
                if patience_counter >= patience:
                    self.logger.info(f"Early stopping triggered after {epoch + 1} epochs")
                    break
        
        # Load best model
        if best_model_state is not None:
            self.model.load_state_dict(best_model_state)
        
        # Final evaluation
        final_val_loss, final_accuracy, final_preds, final_labels = self.evaluate(
            val_loader, criterion
        )
        
        return {
            'train_losses': self.train_losses,
            'val_losses': self.val_losses,
            'val_accuracies': self.val_accuracies,
            'final_accuracy': final_accuracy,
            'final_predictions': final_preds,
            'final_labels': final_labels
        }
    
    def plot_training_history(self, save_path: Optional[str] = None):
        """
        Plot training and validation metrics.
        
        Args:
            save_path (str, optional): Path to save the plot.
        """
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
        
        # Loss plot
        epochs = range(1, len(self.train_losses) + 1)
        ax1.plot(epochs, self.train_losses, 'b-', label='Training Loss')
        ax1.plot(epochs, self.val_losses, 'r-', label='Validation Loss')
        ax1.set_title('Training and Validation Loss')
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('Loss')
        ax1.legend()
        ax1.grid(True)
        
        # Accuracy plot
        ax2.plot(epochs, self.val_accuracies, 'g-', label='Validation Accuracy')
        ax2.set_title('Validation Accuracy')
        ax2.set_xlabel('Epoch')
        ax2.set_ylabel('Accuracy')
        ax2.legend()
        ax2.grid(True)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()


def plot_confusion_matrix(
    y_true: np.ndarray, 
    y_pred: np.ndarray, 
    class_names: List[str],
    save_path: Optional[str] = None
):
    """
    Plot confusion matrix.
    
    Args:
        y_true (np.ndarray): True labels.
        y_pred (np.ndarray): Predicted labels.
        class_names (List[str]): Names of classes.
        save_path (str, optional): Path to save the plot.
    """
    cm = confusion_matrix(y_true, y_pred)
    
    plt.figure(figsize=(8, 6))
    sns.heatmap(
        cm, 
        annot=True, 
        fmt='d', 
        cmap='Blues',
        xticklabels=class_names,
        yticklabels=class_names
    )
    plt.title('Confusion Matrix')
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()


def create_data_loaders(
    sequences: np.ndarray,
    lengths: np.ndarray,
    labels: np.ndarray,
    batch_size: int = 32,
    val_split: float = 0.2,
    random_seed: int = 42
) -> Tuple[DataLoader, DataLoader]:
    """
    Create training and validation data loaders.
    
    Args:
        sequences (np.ndarray): Padded token sequences.
        lengths (np.ndarray): Actual sequence lengths.
        labels (np.ndarray): Target labels.
        batch_size (int): Batch size for data loaders.
        val_split (float): Fraction of data for validation.
        random_seed (int): Random seed for reproducibility.
    
    Returns:
        Tuple[DataLoader, DataLoader]: Training and validation data loaders.
    """
    # Convert to tensors
    sequences_tensor = torch.tensor(sequences, dtype=torch.long)
    lengths_tensor = torch.tensor(lengths, dtype=torch.long)
    labels_tensor = torch.tensor(labels, dtype=torch.long)
    
    # Create dataset
    dataset = TensorDataset(sequences_tensor, lengths_tensor, labels_tensor)
    
    # Split into train and validation
    total_size = len(dataset)
    val_size = int(val_split * total_size)
    train_size = total_size - val_size
    
    train_dataset, val_dataset = random_split(
        dataset, 
        [train_size, val_size],
        generator=torch.Generator().manual_seed(random_seed)
    )
    
    # Create data loaders
    train_loader = DataLoader(
        train_dataset, 
        batch_size=batch_size, 
        shuffle=True,
        num_workers=2,
        pin_memory=True
    )
    
    val_loader = DataLoader(
        val_dataset, 
        batch_size=batch_size, 
        shuffle=False,
        num_workers=2,
        pin_memory=True
    )
    
    return train_loader, val_loader


def main():
    """Main training function."""
    # Configuration
    config = {
        'data_path': 'data/processed/headlines.csv',
        'output_dir': 'outputs',
        'batch_size': 32,
        'embedding_dim': 128,
        'hidden_dim': 128,
        'max_len': 50,
        'min_freq': 2,
        'learning_rate': 1e-3,
        'weight_decay': 1e-5,
        'num_epochs': 50,
        'patience': 7,
        'use_gru': True,  # Set to False for mean pooling
        'random_seed': 42
    }
    
    # Set random seeds for reproducibility
    torch.manual_seed(config['random_seed'])
    np.random.seed(config['random_seed'])
    
    # Setup device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger = setup_logging()
    logger.info(f"Using device: {device}")
    
    # Load and preprocess data
    logger.info("Loading and preprocessing data...")
    df = load_and_preprocess_data(config['data_path'])
    
    # Create vocabulary
    vocab, word_to_idx, idx_to_word = create_vocabulary(
        df['processed_headline'].tolist(), 
        min_freq=config['min_freq']
    )
    
    # Convert texts to sequences
    sequences = texts_to_sequences(df['processed_headline'].tolist(), word_to_idx)
    padded_sequences, lengths = pad_sequences(sequences, max_len=config['max_len'])
    
    # Encode labels
    unique_topics = sorted(df['topic'].unique())
    topic_to_idx = {topic: idx for idx, topic in enumerate(unique_topics)}
    labels = df['topic'].map(topic_to_idx).values
    
    logger.info(f"Dataset size: {len(df)}")
    logger.info(f"Vocabulary size: {len(vocab)}")
    logger.info(f"Number of classes: {len(unique_topics)}")
    logger.info(f"Classes: {unique_topics}")
    
    # Create data loaders
    train_loader, val_loader = create_data_loaders(
        padded_sequences, lengths, labels,
        batch_size=config['batch_size'],
        random_seed=config['random_seed']
    )
    
    # Initialize model
    model = NewsHeadlineClassifier(
        vocab_size=len(vocab),
        embedding_dim=config['embedding_dim'],
        hidden_dim=config['hidden_dim'],
        num_classes=len(unique_topics),
        use_gru=config['use_gru']
    )
    
    logger.info(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # Initialize trainer
    trainer = Trainer(model, device, config['output_dir'])
    
    # Train model
    history = trainer.train(
        train_loader=train_loader,
        val_loader=val_loader,
        num_epochs=config['num_epochs'],
        learning_rate=config['learning_rate'],
        weight_decay=config['weight_decay'],
        patience=config['patience']
    )
    
    # Save model and artifacts
    artifacts = {
        'vocab': vocab,
        'word_to_idx': word_to_idx,
        'idx_to_word': idx_to_word,
        'topic_to_idx': topic_to_idx,
        'unique_topics': unique_topics,
        'config': config,
        'history': history
    }
    
    save_model_artifacts(model, artifacts, config['output_dir'])
    
    # Generate evaluation report
    logger.info("\n" + "="*50)
    logger.info("FINAL EVALUATION RESULTS")
    logger.info("="*50)
    logger.info(f"Final Validation Accuracy: {history['final_accuracy']:.4f}")
    
    # Classification report
    report = classification_report(
        history['final_labels'], 
        history['final_predictions'],
        target_names=unique_topics,
        digits=4
    )
    logger.info(f"\nClassification Report:\n{report}")
    
    # Plot training history
    trainer.plot_training_history(
        save_path=os.path.join(config['output_dir'], 'training_history.png')
    )
    
    # Plot confusion matrix
    plot_confusion_matrix(
        history['final_labels'],
        history['final_predictions'],
        unique_topics,
        save_path=os.path.join(config['output_dir'], 'confusion_matrix.png')
    )
    
    logger.info(f"Training completed! Model and artifacts saved to {config['output_dir']}")


if __name__ == "__main__":
    main()