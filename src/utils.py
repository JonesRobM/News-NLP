"""
Utility functions for news headline topic classification.

This module contains helper functions for data preprocessing, vocabulary
creation, text processing, and model management.
"""

import os
import re
import json
import pickle
import logging
from pathlib import Path
from typing import Dict, List, Tuple, Set, Any
from collections import Counter

import torch
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder


def setup_logging(log_level: int = logging.INFO) -> logging.Logger:
    """
    Setup logging configuration.
    
    Args:
        log_level (int): Logging level. Defaults to logging.INFO.
    
    Returns:
        logging.Logger: Configured logger instance.
    """
    logging.basicConfig(
        level=log_level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    return logging.getLogger(__name__)


def clean_text(text: str) -> str:
    """
    Clean and preprocess text.
    
    Args:
        text (str): Raw text to clean.
    
    Returns:
        str: Cleaned text.
    """
    if not isinstance(text, str):
        return ""
    
    # Convert to lowercase
    text = text.lower()
    
    # Remove special characters but keep spaces and basic punctuation
    text = re.sub(r'[^a-zA-Z0-9\s\.\,\!\?\-]', '', text)
    
    # Remove extra whitespace
    text = re.sub(r'\s+', ' ', text)
    
    # Strip leading/trailing whitespace
    text = text.strip()
    
    return text


def tokenize_text(text: str) -> List[str]:
    """
    Tokenize text into words.
    
    Args:
        text (str): Text to tokenize.
    
    Returns:
        List[str]: List of tokens.
    """
    # Simple whitespace tokenization
    tokens = text.split()
    
    # Remove empty tokens
    tokens = [token for token in tokens if token.strip()]
    
    return tokens


def load_and_preprocess_data(file_path: str) -> pd.DataFrame:
    """
    Load and preprocess the headline dataset.
    
    Args:
        file_path (str): Path to the CSV file.
    
    Returns:
        pd.DataFrame: Preprocessed dataframe with cleaned text.
    
    Raises:
        FileNotFoundError: If the data file doesn't exist.
        ValueError: If required columns are missing.
    """
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Data file not found: {file_path}")
    
    # Load data
    df = pd.read_csv(file_path)
    
    # Check required columns
    required_columns = ['headline', 'topic']
    missing_columns = [col for col in required_columns if col not in df.columns]
    if missing_columns:
        raise ValueError(f"Missing required columns: {missing_columns}")
    
    # Remove rows with missing values
    df = df.dropna(subset=required_columns)
    
    # Clean headlines
    df['processed_headline'] = df['headline'].apply(clean_text)
    
    # Remove empty headlines after processing
    df = df[df['processed_headline'].str.len() > 0]
    
    # Reset index
    df = df.reset_index(drop=True)
    
    logger = setup_logging()
    logger.info(f"Loaded {len(df)} samples from {file_path}")
    logger.info(f"Topics: {sorted(df['topic'].unique())}")
    
    return df


def create_vocabulary(
    texts: List[str], 
    min_freq: int = 2,
    max_vocab_size: int = None
) -> Tuple[Set[str], Dict[str, int], Dict[int, str]]:
    """
    Create vocabulary from a list of texts.
    
    Args:
        texts (List[str]): List of text strings.
        min_freq (int): Minimum frequency for a word to be included.
        max_vocab_size (int, optional): Maximum vocabulary size.
    
    Returns:
        Tuple[Set[str], Dict[str, int], Dict[int, str]]: 
            Vocabulary set, word-to-index mapping, index-to-word mapping.
    """
    # Special tokens
    PAD_TOKEN = '<pad>'
    UNK_TOKEN = '<unk>'
    
    # Count word frequencies
    word_counts = Counter()
    for text in texts:
        tokens = tokenize_text(text)
        word_counts.update(tokens)
    
    # Filter by minimum frequency
    filtered_words = [word for word, count in word_counts.items() if count >= min_freq]
    
    # Sort by frequency (most frequent first)
    filtered_words.sort(key=lambda x: word_counts[x], reverse=True)
    
    # Limit vocabulary size if specified
    if max_vocab_size:
        filtered_words = filtered_words[:max_vocab_size - 2]  # -2 for special tokens
    
    # Create vocabulary with special tokens
    vocab = {PAD_TOKEN, UNK_TOKEN} | set(filtered_words)
    
    # Create mappings
    word_to_idx = {PAD_TOKEN: 0, UNK_TOKEN: 1}
    for i, word in enumerate(filtered_words, start=2):
        word_to_idx[word] = i
    
    idx_to_word = {idx: word for word, idx in word_to_idx.items()}
    
    logger = setup_logging()
    logger.info(f"Created vocabulary with {len(vocab)} words")
    logger.info(f"Filtered {len(word_counts) - len(filtered_words)} low-frequency words")
    
    return vocab, word_to_idx, idx_to_word


def texts_to_sequences(texts: List[str], word_to_idx: Dict[str, int]) -> List[List[int]]:
    """
    Convert texts to sequences of token indices.
    
    Args:
        texts (List[str]): List of text strings.
        word_to_idx (Dict[str, int]): Word to index mapping.
    
    Returns:
        List[List[int]]: List of token sequences.
    """
    sequences = []
    unk_idx = word_to_idx.get('<unk>', 1)
    
    for text in texts:
        tokens = tokenize_text(text)
        sequence = [word_to_idx.get(token, unk_idx) for token in tokens]
        sequences.append(sequence)
    
    return sequences


def pad_sequences(
    sequences: List[List[int]], 
    max_len: int = None, 
    pad_value: int = 0
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Pad sequences to a fixed length.
    
    Args:
        sequences (List[List[int]]): List of token sequences.
        max_len (int, optional): Maximum sequence length. If None, uses the longest sequence.
        pad_value (int): Value to use for padding.
    
    Returns:
        Tuple[np.ndarray, np.ndarray]: Padded sequences and original lengths.
    """
    if max_len is None:
        max_len = max(len(seq) for seq in sequences)
    
    # Store original lengths
    lengths = [min(len(seq), max_len) for seq in sequences]
    
    # Pad sequences
    padded_sequences = []
    for seq in sequences:
        if len(seq) >= max_len:
            # Truncate if too long
            padded_seq = seq[:max_len]
        else:
            # Pad if too short
            padded_seq = seq + [pad_value] * (max_len - len(seq))
        
        padded_sequences.append(padded_seq)
    
    return np.array(padded_sequences), np.array(lengths)


def encode_labels(labels: List[str]) -> Tuple[np.ndarray, Dict[str, int], List[str]]:
    """
    Encode string labels to numerical values.
    
    Args:
        labels (List[str]): List of string labels.
    
    Returns:
        Tuple[np.ndarray, Dict[str, int], List[str]]: 
            Encoded labels, label-to-index mapping, unique labels.
    """
    label_encoder = LabelEncoder()
    encoded_labels = label_encoder.fit_transform(labels)
    
    # Create mappings
    unique_labels = label_encoder.classes_.tolist()
    label_to_idx = {label: idx for idx, label in enumerate(unique_labels)}
    
    return encoded_labels, label_to_idx, unique_labels


def save_model_artifacts(
    model: torch.nn.Module, 
    artifacts: Dict[str, Any], 
    save_dir: str
):
    """
    Save model and associated artifacts.
    
    Args:
        model (torch.nn.Module): Trained PyTorch model.
        artifacts (Dict[str, Any]): Dictionary of artifacts to save.
        save_dir (str): Directory to save artifacts.
    """
    save_path = Path(save_dir)
    save_path.mkdir(parents=True, exist_ok=True)
    
    # Save model state dict
    model_path = save_path / 'model.pth'
    torch.save(model.state_dict(), model_path)
    
    # Save artifacts
    artifacts_path = save_path / 'artifacts.pkl'
    with open(artifacts_path, 'wb') as f:
        pickle.dump(artifacts, f)
    
    # Save config as JSON for easy reading
    config_path = save_path / 'config.json'
    with open(config_path, 'w') as f:
        # Convert numpy arrays to lists for JSON serialization
        config_copy = artifacts['config'].copy()
        json.dump(config_copy, f, indent=2)
    
    # Save vocabulary
    vocab_path = save_path / 'vocabulary.json'
    vocab_data = {
        'word_to_idx': artifacts['word_to_idx'],
        'idx_to_word': artifacts['idx_to_word'],
        'topic_to_idx': artifacts['topic_to_idx'],
        'unique_topics': artifacts['unique_topics']
    }
    with open(vocab_path, 'w') as f:
        json.dump(vocab_data, f, indent=2)
    
    logger = setup_logging()
    logger.info(f"Model and artifacts saved to {save_dir}")


def load_model_artifacts(load_dir: str) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    """
    Load model artifacts from directory.
    
    Args:
        load_dir (str): Directory containing saved artifacts.
    
    Returns:
        Tuple[Dict[str, Any], Dict[str, Any]]: Artifacts and vocabulary data.
    
    Raises:
        FileNotFoundError: If required files are not found.
    """
    load_path = Path(load_dir)
    
    # Check if files exist
    required_files = ['artifacts.pkl', 'vocabulary.json']
    for file_name in required_files:
        if not (load_path / file_name).exists():
            raise FileNotFoundError(f"Required file not found: {file_name}")
    
    # Load artifacts
    artifacts_path = load_path / 'artifacts.pkl'
    with open(artifacts_path, 'rb') as f:
        artifacts = pickle.load(f)
    
    # Load vocabulary
    vocab_path = load_path / 'vocabulary.json'
    with open(vocab_path, 'r') as f:
        vocab_data = json.load(f)
    
    return artifacts, vocab_data


def predict_single_headline(
    headline: str,
    model: torch.nn.Module,
    word_to_idx: Dict[str, int],
    unique_topics: List[str],
    max_len: int = 50,
    device: torch.device = None
) -> Tuple[str, float, Dict[str, float]]:
    """
    Predict topic for a single headline.
    
    Args:
        headline (str): Input headline text.
        model (torch.nn.Module): Trained model.
        word_to_idx (Dict[str, int]): Word to index mapping.
        unique_topics (List[str]): List of topic names.
        max_len (int): Maximum sequence length.
        device (torch.device, optional): Device for inference.
    
    Returns:
        Tuple[str, float, Dict[str, float]]: 
            Predicted topic, confidence, all topic probabilities.
    """
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    model.eval()
    
    # Preprocess headline
    cleaned_headline = clean_text(headline)
    tokens = tokenize_text(cleaned_headline)
    
    # Convert to sequence
    unk_idx = word_to_idx.get('<unk>', 1)
    sequence = [word_to_idx.get(token, unk_idx) for token in tokens]
    
    # Pad sequence
    if len(sequence) >= max_len:
        padded_sequence = sequence[:max_len]
        length = max_len
    else:
        padded_sequence = sequence + [0] * (max_len - len(sequence))
        length = len(sequence)
    
    # Convert to tensors
    input_ids = torch.tensor([padded_sequence], dtype=torch.long).to(device)
    lengths = torch.tensor([length], dtype=torch.long).to(device)
    
    # Make prediction
    with torch.no_grad():
        probabilities = model.predict(input_ids, lengths)
        probabilities = probabilities.cpu().numpy()[0]
    
    # Get prediction
    predicted_idx = np.argmax(probabilities)
    predicted_topic = unique_topics[predicted_idx]
    confidence = probabilities[predicted_idx]
    
    # Create probability dictionary
    topic_probs = {topic: float(prob) for topic, prob in zip(unique_topics, probabilities)}
    
    return predicted_topic, float(confidence), topic_probs


def calculate_dataset_statistics(df: pd.DataFrame) -> Dict[str, Any]:
    """
    Calculate statistics about the dataset.
    
    Args:
        df (pd.DataFrame): Dataset dataframe.
    
    Returns:
        Dict[str, Any]: Dictionary containing dataset statistics.
    """
    stats = {}
    
    # Basic statistics
    stats['total_samples'] = len(df)
    stats['num_topics'] = df['topic'].nunique()
    stats['topic_distribution'] = df['topic'].value_counts().to_dict()
    
    # Text length statistics
    headline_lengths = df['processed_headline'].str.split().str.len()
    stats['avg_headline_length'] = float(headline_lengths.mean())
    stats['min_headline_length'] = int(headline_lengths.min())
    stats['max_headline_length'] = int(headline_lengths.max())
    stats['median_headline_length'] = float(headline_lengths.median())
    
    # Vocabulary statistics
    all_words = []
    for headline in df['processed_headline']:
        all_words.extend(tokenize_text(headline))
    
    word_counts = Counter(all_words)
    stats['total_words'] = len(all_words)
    stats['unique_words'] = len(word_counts)
    stats['avg_word_frequency'] = float(np.mean(list(word_counts.values())))
    
    return stats


def create_sample_data(output_path: str, num_samples_per_topic: int = 100):
    """
    Create sample data for testing purposes.
    
    Args:
        output_path (str): Path to save the sample data.
        num_samples_per_topic (int): Number of samples per topic.
    """
    # Sample headlines for different topics
    sample_data = {
        'politics': [
            "Government announces new economic policy changes",
            "Parliament debates voting reform legislation",
            "Prime Minister addresses nation on budget concerns",
            "Opposition party criticizes current tax proposals",
            "Election campaign begins with candidate announcements"
        ],
        'technology': [
            "New artificial intelligence breakthrough announced",
            "Tech company releases innovative smartphone model",
            "Cybersecurity experts warn of data breach risks",
            "Software giant updates privacy protection measures",
            "Researchers develop quantum computing advancement"
        ],
        'sport': [
            "Football team wins championship final match",
            "Olympic athlete breaks world record performance",
            "Basketball season ends with surprising tournament results",
            "Tennis player advances to semifinals competition",
            "Cricket match postponed due to weather conditions"
        ]
    }
    
    # Generate full dataset
    headlines = []
    topics = []
    
    for topic, sample_headlines in sample_data.items():
        for i in range(num_samples_per_topic):
            # Use sample headlines cyclically and add variation
            base_headline = sample_headlines[i % len(sample_headlines)]
            headlines.append(base_headline)
            topics.append(topic)
    
    # Create DataFrame
    df = pd.DataFrame({
        'headline': headlines,
        'topic': topics
    })
    
    # Save to CSV
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    df.to_csv(output_path, index=False)
    
    print(f"Sample data created with {len(df)} samples saved to {output_path}")