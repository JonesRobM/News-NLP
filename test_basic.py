"""
Basic tests for the news headline topic classifier.

This module contains simple tests to verify that the core functionality
of the classifier works correctly.
"""

import os
import sys
import unittest
import tempfile
import shutil
import torch
import pandas as pd
import numpy as np

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from model import NewsHeadlineClassifier, SimpleMLP
from utils import (
    clean_text, tokenize_text, create_vocabulary, texts_to_sequences,
    pad_sequences, encode_labels, create_sample_data, calculate_dataset_statistics
)


class TestTextProcessing(unittest.TestCase):
    """Test text processing utilities."""
    
    def test_clean_text(self):
        """Test text cleaning function."""
        # Test basic cleaning
        text = "This is a TEST with Numbers123 and symbols!@#"
        cleaned = clean_text(text)
        expected = "this is a test with numbers123 and symbols!"
        self.assertEqual(cleaned, expected)
        
        # Test empty string
        self.assertEqual(clean_text(""), "")
        
        # Test None input
        self.assertEqual(clean_text(None), "")
        
        # Test whitespace normalization
        text = "Multiple   spaces    here"
        cleaned = clean_text(text)
        expected = "multiple spaces here"
        self.assertEqual(cleaned, expected)
    
    def test_tokenize_text(self):
        """Test text tokenization function."""
        # Test basic tokenization
        text = "This is a test sentence"
        tokens = tokenize_text(text)
        expected = ["This", "is", "a", "test", "sentence"]
        self.assertEqual(tokens, expected)
        
        # Test empty string
        self.assertEqual(tokenize_text(""), [])
        
        # Test single word
        self.assertEqual(tokenize_text("word"), ["word"])
    
    def test_create_vocabulary(self):
        """Test vocabulary creation."""
        texts = [
            "this is a test",
            "this is another test",
            "vocabulary test here",
            "test vocabulary creation"
        ]
        
        vocab, word_to_idx, idx_to_word = create_vocabulary(texts, min_freq=1)
        
        # Check that vocabulary contains expected words
        self.assertIn("test", vocab)
        self.assertIn("this", vocab)
        self.assertIn("<pad>", vocab)
        self.assertIn("<unk>", vocab)
        
        # Check mappings
        self.assertEqual(word_to_idx["<pad>"], 0)
        self.assertEqual(word_to_idx["<unk>"], 1)
        
        # Check bidirectional mapping
        for word, idx in word_to_idx.items():
            self.assertEqual(idx_to_word[idx], word)
    
    def test_texts_to_sequences(self):
        """Test text to sequence conversion."""
        texts = ["hello world", "world test"]
        word_to_idx = {"<pad>": 0, "<unk>": 1, "hello": 2, "world": 3, "test": 4}
        
        sequences = texts_to_sequences(texts, word_to_idx)
        
        expected = [[2, 3], [3, 4]]
        self.assertEqual(sequences, expected)
    
    def test_pad_sequences(self):
        """Test sequence padding."""
        sequences = [[1, 2, 3], [4, 5], [6, 7, 8, 9]]
        
        padded, lengths = pad_sequences(sequences, max_len=4, pad_value=0)
        
        expected_padded = np.array([
            [1, 2, 3, 0],
            [4, 5, 0, 0],
            [6, 7, 8, 9]
        ])
        expected_lengths = np.array([3, 2, 4])
        
        np.testing.assert_array_equal(padded, expected_padded)
        np.testing.assert_array_equal(lengths, expected_lengths)
    
    def test_encode_labels(self):
        """Test label encoding."""
        labels = ["politics", "sport", "politics", "tech", "sport"]
        encoded, label_to_idx, unique_labels = encode_labels(labels)
        
        # Check that all labels are encoded
        self.assertEqual(len(set(encoded)), 3)  # 3 unique labels
        self.assertEqual(len(unique_labels), 3)
        
        # Check consistency
        for label in labels:
            self.assertIn(label, unique_labels)


class TestModels(unittest.TestCase):
    """Test model architectures."""
    
    def setUp(self):
        """Set up test parameters."""
        self.vocab_size = 1000
        self.embedding_dim = 64
        self.hidden_dim = 32
        self.num_classes = 4
        self.batch_size = 8
        self.seq_len = 20
    
    def test_news_headline_classifier_mean_pooling(self):
        """Test NewsHeadlineClassifier with mean pooling."""
        model = NewsHeadlineClassifier(
            vocab_size=self.vocab_size,
            embedding_dim=self.embedding_dim,
            hidden_dim=self.hidden_dim,
            num_classes=self.num_classes,
            use_gru=False
        )
        
        # Test forward pass
        input_ids = torch.randint(0, self.vocab_size, (self.batch_size, self.seq_len))
        lengths = torch.randint(5, self.seq_len, (self.batch_size,))
        
        output = model(input_ids, lengths)
        
        # Check output shape
        expected_shape = (self.batch_size, self.num_classes)
        self.assertEqual(output.shape, expected_shape)
    
    def test_news_headline_classifier_gru(self):
        """Test NewsHeadlineClassifier with GRU."""
        model = NewsHeadlineClassifier(
            vocab_size=self.vocab_size,
            embedding_dim=self.embedding_dim,
            hidden_dim=self.hidden_dim,
            num_classes=self.num_classes,
            use_gru=True
        )
        
        # Test forward pass
        input_ids = torch.randint(0, self.vocab_size, (self.batch_size, self.seq_len))
        lengths = torch.randint(5, self.seq_len, (self.batch_size,))
        
        output = model(input_ids, lengths)
        
        # Check output shape
        expected_shape = (self.batch_size, self.num_classes)
        self.assertEqual(output.shape, expected_shape)
    
    def test_simple_mlp(self):
        """Test SimpleMLP model."""
        model = SimpleMLP(
            vocab_size=self.vocab_size,
            hidden_dim=self.hidden_dim,
            num_classes=self.num_classes
        )
        
        # Test forward pass
        bow_features = torch.randn(self.batch_size, self.vocab_size)
        output = model(bow_features)
        
        # Check output shape
        expected_shape = (self.batch_size, self.num_classes)
        self.assertEqual(output.shape, expected_shape)
    
    def test_model_prediction_mode(self):
        """Test model prediction mode."""
        model = NewsHeadlineClassifier(
            vocab_size=self.vocab_size,
            embedding_dim=self.embedding_dim,
            hidden_dim=self.hidden_dim,
            num_classes=self.num_classes,
            use_gru=False
        )
        
        input_ids = torch.randint(0, self.vocab_size, (self.batch_size, self.seq_len))
        lengths = torch.randint(5, self.seq_len, (self.batch_size,))
        
        # Test prediction method
        probabilities = model.predict(input_ids, lengths)
        
        # Check that probabilities sum to 1
        prob_sums = probabilities.sum(dim=1)
        np.testing.assert_allclose(prob_sums.numpy(), 1.0, rtol=1e-5)
        
        # Check that all probabilities are non-negative
        self.assertTrue(torch.all(probabilities >= 0))


class TestDataUtilities(unittest.TestCase):
    """Test data utility functions."""
    
    def setUp(self):
        """Set up temporary directory for tests."""
        self.temp_dir = tempfile.mkdtemp()
    
    def tearDown(self):
        """Clean up temporary directory."""
        shutil.rmtree(self.temp_dir)
    
    def test_create_sample_data(self):
        """Test sample data creation."""
        sample_file = os.path.join(self.temp_dir, "sample_data.csv")
        
        create_sample_data(sample_file, num_samples_per_topic=10)
        
        # Check that file was created
        self.assertTrue(os.path.exists(sample_file))
        
        # Load and check data
        df = pd.read_csv(sample_file)
        
        # Check structure
        self.assertIn('headline', df.columns)
        self.assertIn('topic', df.columns)
        
        # Check content
        self.assertEqual(len(df), 30)  # 10 samples * 3 topics
        self.assertEqual(df['topic'].nunique(), 3)
    
    def test_calculate_dataset_statistics(self):
        """Test dataset statistics calculation."""
        # Create sample dataframe
        data = {
            'processed_headline': [
                'government announces new policy',
                'tech company releases product',
                'sports team wins championship',
                'business reports quarterly earnings'
            ],
            'topic': ['politics', 'technology', 'sport', 'business']
        }
        df = pd.DataFrame(data)
        
        stats = calculate_dataset_statistics(df)
        
        # Check required statistics
        self.assertEqual(stats['total_samples'], 4)
        self.assertEqual(stats['num_topics'], 4)
        self.assertIn('avg_headline_length', stats)
        self.assertIn('unique_words', stats)


class TestIntegration(unittest.TestCase):
    """Integration tests for the complete pipeline."""
    
    def setUp(self):
        """Set up test data."""
        self.sample_headlines = [
            "Government announces new economic policy",
            "Tech startup raises million in funding",
            "Football team wins championship final",
            "Stock market reaches record high"
        ]
        self.sample_topics = ["politics", "technology", "sport", "business"]
    
    def test_end_to_end_preprocessing(self):
        """Test complete preprocessing pipeline."""
        # Clean texts
        cleaned_headlines = [clean_text(h) for h in self.sample_headlines]
        
        # Create vocabulary
        vocab, word_to_idx, idx_to_word = create_vocabulary(cleaned_headlines, min_freq=1)
        
        # Convert to sequences
        sequences = texts_to_sequences(cleaned_headlines, word_to_idx)
        
        # Pad sequences
        padded_sequences, lengths = pad_sequences(sequences, max_len=10)
        
        # Encode labels
        encoded_labels, label_to_idx, unique_labels = encode_labels(self.sample_topics)
        
        # Check final shapes
        self.assertEqual(padded_sequences.shape[0], len(self.sample_headlines))
        self.assertEqual(len(encoded_labels), len(self.sample_topics))
        self.assertEqual(len(unique_labels), len(set(self.sample_topics)))
    
    def test_model_with_preprocessed_data(self):
        """Test model with preprocessed data."""
        # Preprocess data
        cleaned_headlines = [clean_text(h) for h in self.sample_headlines]
        vocab, word_to_idx, idx_to_word = create_vocabulary(cleaned_headlines, min_freq=1)
        sequences = texts_to_sequences(cleaned_headlines, word_to_idx)
        padded_sequences, lengths = pad_sequences(sequences, max_len=10)
        encoded_labels, label_to_idx, unique_labels = encode_labels(self.sample_topics)
        
        # Create model
        model = NewsHeadlineClassifier(
            vocab_size=len(vocab),
            embedding_dim=32,
            hidden_dim=16,
            num_classes=len(unique_labels),
            use_gru=False
        )
        
        # Test forward pass
        input_ids = torch.tensor(padded_sequences, dtype=torch.long)
        lengths_tensor = torch.tensor(lengths, dtype=torch.long)
        
        output = model(input_ids, lengths_tensor)
        
        # Check output shape
        expected_shape = (len(self.sample_headlines), len(unique_labels))
        self.assertEqual(output.shape, expected_shape)


def run_tests():
    """Run all tests."""
    # Create test suite
    test_suite = unittest.TestSuite()
    
    # Add test classes
    test_classes = [
        TestTextProcessing,
        TestModels,
        TestDataUtilities,
        TestIntegration
    ]
    
    for test_class in test_classes:
        tests = unittest.TestLoader().loadTestsFromTestCase(test_class)
        test_suite.addTests(tests)
    
    # Run tests
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(test_suite)
    
    return result.wasSuccessful()


if __name__ == "__main__":
    print("Running basic tests for News Headlines Topic Classifier...")
    print("=" * 60)
    
    success = run_tests()
    
    if success:
        print("\n" + "=" * 60)
        print("✅ All tests passed successfully!")
        print("The basic functionality is working correctly.")
    else:
        print("\n" + "=" * 60)
        print("❌ Some tests failed!")
        print("Please check the error messages above.")
        sys.exit(1)