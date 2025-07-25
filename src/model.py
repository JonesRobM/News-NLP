"""
News headline topic classification model.

This module contains the PyTorch model architecture for multiclass
text classification of news headlines.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class NewsHeadlineClassifier(nn.Module):
    """
    A neural network model for classifying news headlines into topics.
    
    The model uses an embedding layer followed by either mean pooling
    or GRU-based feature extraction, then a linear classifier.
    
    Args:
        vocab_size (int): Size of the vocabulary.
        embedding_dim (int): Dimension of word embeddings.
        hidden_dim (int): Hidden dimension for GRU (if use_gru=True).
        num_classes (int): Number of output classes.
        use_gru (bool): Whether to use GRU or mean pooling. Defaults to False.
        dropout_rate (float): Dropout rate for regularization. Defaults to 0.3.
        padding_idx (int): Index of padding token in vocabulary. Defaults to 0.
    """
    
    def __init__(
        self, 
        vocab_size: int,
        embedding_dim: int,
        hidden_dim: int,
        num_classes: int,
        use_gru: bool = False,
        dropout_rate: float = 0.3,
        padding_idx: int = 0
    ):
        super(NewsHeadlineClassifier, self).__init__()
        
        self.use_gru = use_gru
        
        # Embedding layer
        self.embedding = nn.Embedding(
            vocab_size, 
            embedding_dim, 
            padding_idx=padding_idx
        )
        
        # Feature extraction layers
        if use_gru:
            self.gru = nn.GRU(
                embedding_dim,
                hidden_dim,
                batch_first=True,
                bidirectional=True
            )
            classifier_input_dim = hidden_dim * 2  # Bidirectional
        else:
            classifier_input_dim = embedding_dim
        
        # Dropout for regularization
        self.dropout = nn.Dropout(dropout_rate)
        
        # Classification head
        self.classifier = nn.Linear(classifier_input_dim, num_classes)
        
        # Initialize weights
        self._init_weights()
    
    def _init_weights(self):
        """Initialize model weights using Xavier uniform distribution."""
        for name, param in self.named_parameters():
            if 'weight' in name and param.dim() > 1:
                nn.init.xavier_uniform_(param)
            elif 'bias' in name:
                nn.init.constant_(param, 0)
    
    def forward(self, input_ids: torch.Tensor, lengths: torch.Tensor = None) -> torch.Tensor:
        """
        Forward pass of the model.
        
        Args:
            input_ids (torch.Tensor): Input token IDs of shape (batch_size, seq_len).
            lengths (torch.Tensor, optional): Actual lengths of sequences for masking.
                Required for mean pooling to handle padding correctly.
        
        Returns:
            torch.Tensor: Logits of shape (batch_size, num_classes).
        """
        # Get embeddings
        embedded = self.embedding(input_ids)  # (batch_size, seq_len, embedding_dim)
        
        if self.use_gru:
            # GRU-based feature extraction
            if lengths is not None:
                # Pack sequences for efficient processing of variable lengths
                packed = nn.utils.rnn.pack_padded_sequence(
                    embedded, lengths.cpu(), batch_first=True, enforce_sorted=False
                )
                gru_out, _ = self.gru(packed)
                gru_out, _ = nn.utils.rnn.pad_packed_sequence(gru_out, batch_first=True)
            else:
                gru_out, _ = self.gru(embedded)
            
            # Use the last hidden state (mean of forward and backward)
            if lengths is not None:
                # Get the actual last output for each sequence
                batch_size = gru_out.size(0)
                last_outputs = []
                for i, length in enumerate(lengths):
                    last_outputs.append(gru_out[i, length-1, :])
                features = torch.stack(last_outputs)
            else:
                features = gru_out[:, -1, :]  # (batch_size, hidden_dim * 2)
        else:
            # Mean pooling
            if lengths is not None:
                # Create mask to ignore padding tokens
                mask = torch.arange(embedded.size(1), device=embedded.device)[None, :] < lengths[:, None]
                mask = mask.unsqueeze(-1).float()  # (batch_size, seq_len, 1)
                
                # Apply mask and compute mean
                masked_embedded = embedded * mask
                features = masked_embedded.sum(dim=1) / lengths.unsqueeze(-1).float()
            else:
                # Simple mean without considering padding
                features = embedded.mean(dim=1)  # (batch_size, embedding_dim)
        
        # Apply dropout
        features = self.dropout(features)
        
        # Classification
        logits = self.classifier(features)  # (batch_size, num_classes)
        
        return logits
    
    def predict(self, input_ids: torch.Tensor, lengths: torch.Tensor = None) -> torch.Tensor:
        """
        Make predictions with the model (applies softmax).
        
        Args:
            input_ids (torch.Tensor): Input token IDs.
            lengths (torch.Tensor, optional): Actual lengths of sequences.
        
        Returns:
            torch.Tensor: Predicted class probabilities.
        """
        self.eval()
        with torch.no_grad():
            logits = self.forward(input_ids, lengths)
            probabilities = F.softmax(logits, dim=-1)
        return probabilities


class SimpleMLP(nn.Module):
    """
    A simple MLP baseline for text classification using bag-of-words features.
    
    Args:
        vocab_size (int): Size of the vocabulary.
        hidden_dim (int): Hidden layer dimension.
        num_classes (int): Number of output classes.
        dropout_rate (float): Dropout rate. Defaults to 0.3.
    """
    
    def __init__(
        self,
        vocab_size: int,
        hidden_dim: int,
        num_classes: int,
        dropout_rate: float = 0.3
    ):
        super(SimpleMLP, self).__init__()
        
        self.hidden = nn.Linear(vocab_size, hidden_dim)
        self.dropout = nn.Dropout(dropout_rate)
        self.output = nn.Linear(hidden_dim, num_classes)
        
        # Initialize weights
        nn.init.xavier_uniform_(self.hidden.weight)
        nn.init.xavier_uniform_(self.output.weight)
        nn.init.constant_(self.hidden.bias, 0)
        nn.init.constant_(self.output.bias, 0)
    
    def forward(self, bow_features: torch.Tensor) -> torch.Tensor:
        """
        Forward pass for bag-of-words input.
        
        Args:
            bow_features (torch.Tensor): Bag-of-words features of shape (batch_size, vocab_size).
        
        Returns:
            torch.Tensor: Logits of shape (batch_size, num_classes).
        """
        hidden = F.relu(self.hidden(bow_features))
        hidden = self.dropout(hidden)
        logits = self.output(hidden)
        return logits