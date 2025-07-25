"""
Evaluation script for the news headline topic classifier.

This script provides comprehensive evaluation of trained models including
detailed metrics, error analysis, and performance visualization.
"""

import os
import sys
import argparse
import json
from pathlib import Path
from typing import Dict, List, Tuple, Optional

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import pandas as pd
import numpy as np
from sklearn.metrics import (
    accuracy_score, precision_recall_fscore_support,
    classification_report, confusion_matrix
)
import matplotlib.pyplot as plt
import seaborn as sns

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from model import NewsHeadlineClassifier
from utils import (
    load_model_artifacts, load_and_preprocess_data, texts_to_sequences,
    pad_sequences, setup_logging, predict_single_headline
)


class ModelEvaluator:
    """
    Comprehensive model evaluation class.
    
    Args:
        model (NewsHeadlineClassifier): Trained model.
        artifacts (Dict): Model artifacts.
        vocab_data (Dict): Vocabulary data.
        device (torch.device): Device for inference.
    """
    
    def __init__(
        self,
        model: NewsHeadlineClassifier,
        artifacts: Dict,
        vocab_data: Dict,
        device: torch.device = None
    ):
        self.model = model.to(device if device else torch.device('cpu'))
        self.artifacts = artifacts
        self.vocab_data = vocab_data
        self.device = device if device else torch.device('cpu')
        self.logger = setup_logging()
        
        self.model.eval()
    
    def evaluate_dataset(
        self,
        test_data: pd.DataFrame,
        batch_size: int = 32
    ) -> Dict:
        """
        Evaluate model on a test dataset.
        
        Args:
            test_data (pd.DataFrame): Test dataset with 'headline' and 'topic' columns.
            batch_size (int): Batch size for evaluation.
        
        Returns:
            Dict: Comprehensive evaluation results.
        """
        self.logger.info("Starting dataset evaluation...")
        
        # Preprocess test data
        headlines = test_data['headline'].tolist()
        true_topics = test_data['topic'].tolist()
        
        # Convert to sequences
        cleaned_headlines = [headline.lower().strip() for headline in headlines]
        sequences = texts_to_sequences(cleaned_headlines, self.vocab_data['word_to_idx'])
        padded_sequences, lengths = pad_sequences(
            sequences, 
            max_len=self.artifacts['config']['max_len']
        )
        
        # Encode true labels
        topic_to_idx = self.artifacts['topic_to_idx']
        true_labels = [topic_to_idx.get(topic, -1) for topic in true_topics]
        
        # Filter out unknown labels
        valid_indices = [i for i, label in enumerate(true_labels) if label != -1]
        if len(valid_indices) != len(true_labels):
            self.logger.warning(f"Filtered out {len(true_labels) - len(valid_indices)} samples with unknown topics")
        
        padded_sequences = padded_sequences[valid_indices]
        lengths = lengths[valid_indices]
        true_labels = [true_labels[i] for i in valid_indices]
        headlines = [headlines[i] for i in valid_indices]
        
        # Create data loader
        dataset = TensorDataset(
            torch.tensor(padded_sequences, dtype=torch.long),
            torch.tensor(lengths, dtype=torch.long),
            torch.tensor(true_labels, dtype=torch.long)
        )
        data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
        
        # Evaluate
        all_predictions = []
        all_probabilities = []
        all_true_labels = []
        
        with torch.no_grad():
            for batch in data_loader:
                input_ids, batch_lengths, labels = [x.to(self.device) for x in batch]
                
                # Get predictions
                logits = self.model(input_ids, batch_lengths)
                probabilities = torch.softmax(logits, dim=-1)
                predictions = torch.argmax(logits, dim=-1)
                
                all_predictions.extend(predictions.cpu().numpy())
                all_probabilities.extend(probabilities.cpu().numpy())
                all_true_labels.extend(labels.cpu().numpy())
        
        # Calculate metrics
        results = self._calculate_metrics(
            all_true_labels, all_predictions, all_probabilities, headlines
        )
        
        self.logger.info(f"Evaluation completed on {len(all_true_labels)} samples")
        return results
    
    def _calculate_metrics(
        self,
        true_labels: List[int],
        predictions: List[int],
        probabilities: List[List[float]],
        headlines: List[str]
    ) -> Dict:
        """
        Calculate comprehensive evaluation metrics.
        
        Args:
            true_labels (List[int]): True label indices.
            predictions (List[int]): Predicted label indices.
            probabilities (List[List[float]]): Prediction probabilities.
            headlines (List[str]): Original headlines.
        
        Returns:
            Dict: Evaluation metrics and analysis.
        """
        unique_topics = self.artifacts['unique_topics']
        
        # Basic metrics
        accuracy = accuracy_score(true_labels, predictions)
        precision, recall, f1, support = precision_recall_fscore_support(
            true_labels, predictions, average=None, zero_division=0
        )
        
        # Macro and weighted averages
        macro_precision, macro_recall, macro_f1, _ = precision_recall_fscore_support(
            true_labels, predictions, average='macro', zero_division=0
        )
        weighted_precision, weighted_recall, weighted_f1, _ = precision_recall_fscore_support(
            true_labels, predictions, average='weighted', zero_division=0
        )
        
        # Confusion matrix
        cm = confusion_matrix(true_labels, predictions)
        
        # Per-class metrics
        per_class_metrics = {}
        for i, topic in enumerate(unique_topics):
            per_class_metrics[topic] = {
                'precision': float(precision[i]) if i < len(precision) else 0.0,
                'recall': float(recall[i]) if i < len(recall) else 0.0,
                'f1_score': float(f1[i]) if i < len(f1) else 0.0,
                'support': int(support[i]) if i < len(support) else 0
            }
        
        # Error analysis
        error_analysis = self._analyze_errors(
            true_labels, predictions, probabilities, headlines, unique_topics
        )
        
        # Confidence analysis
        confidence_analysis = self._analyze_confidence(probabilities, true_labels, predictions)
        
        return {
            'overall_metrics': {
                'accuracy': float(accuracy),
                'macro_precision': float(macro_precision),
                'macro_recall': float(macro_recall),
                'macro_f1': float(macro_f1),
                'weighted_precision': float(weighted_precision),
                'weighted_recall': float(weighted_recall),
                'weighted_f1': float(weighted_f1),
                'total_samples': len(true_labels)
            },
            'per_class_metrics': per_class_metrics,
            'confusion_matrix': cm.tolist(),
            'error_analysis': error_analysis,
            'confidence_analysis': confidence_analysis
        }
    
    def _analyze_errors(
        self,
        true_labels: List[int],
        predictions: List[int],
        probabilities: List[List[float]],
        headlines: List[str],
        unique_topics: List[str]
    ) -> Dict:
        """
        Analyze prediction errors in detail.
        
        Args:
            true_labels (List[int]): True labels.
            predictions (List[int]): Predictions.
            probabilities (List[List[float]]): Probabilities.
            headlines (List[str]): Headlines.
            unique_topics (List[str]): Topic names.
        
        Returns:
            Dict: Error analysis results.
        """
        errors = []
        correct = []
        
        for i, (true_idx, pred_idx, probs, headline) in enumerate(
            zip(true_labels, predictions, probabilities, headlines)
        ):
            true_topic = unique_topics[true_idx]
            pred_topic = unique_topics[pred_idx]
            confidence = float(probs[pred_idx])
            
            if true_idx != pred_idx:
                errors.append({
                    'headline': headline,
                    'true_topic': true_topic,
                    'predicted_topic': pred_topic,
                    'confidence': confidence,
                    'true_probability': float(probs[true_idx])
                })
            else:
                correct.append({
                    'headline': headline,
                    'topic': true_topic,
                    'confidence': confidence
                })
        
        # Most common error types
        error_pairs = {}
        for error in errors:
            pair = (error['true_topic'], error['predicted_topic'])
            if pair not in error_pairs:
                error_pairs[pair] = []
            error_pairs[pair].append(error)
        
        # Sort by frequency
        common_errors = sorted(
            error_pairs.items(),
            key=lambda x: len(x[1]),
            reverse=True
        )[:10]  # Top 10 error types
        
        return {
            'total_errors': len(errors),
            'error_rate': len(errors) / len(true_labels),
            'most_common_errors': [
                {
                    'error_type': f"{pair[0]} -> {pair[1]}",
                    'count': len(examples),
                    'examples': examples[:3]  # Top 3 examples
                }
                for pair, examples in common_errors
            ],
            'low_confidence_errors': sorted(
                errors, key=lambda x: x['confidence']
            )[:10],  # 10 lowest confidence errors
            'high_confidence_errors': sorted(
                errors, key=lambda x: x['confidence'], reverse=True
            )[:5]  # 5 highest confidence errors (surprising mistakes)
        }
    
    def _analyze_confidence(
        self,
        probabilities: List[List[float]],
        true_labels: List[int],
        predictions: List[int]
    ) -> Dict:
        """
        Analyze prediction confidence distribution.
        
        Args:
            probabilities (List[List[float]]): Prediction probabilities.
            true_labels (List[int]): True labels.
            predictions (List[int]): Predictions.
        
        Returns:
            Dict: Confidence analysis results.
        """
        confidences = [float(probs[pred_idx]) for probs, pred_idx in zip(probabilities, predictions)]
        correct_mask = [true == pred for true, pred in zip(true_labels, predictions)]
        
        correct_confidences = [conf for conf, correct in zip(confidences, correct_mask) if correct]
        incorrect_confidences = [conf for conf, correct in zip(confidences, correct_mask) if not correct]
        
        return {
            'overall_confidence': {
                'mean': float(np.mean(confidences)),
                'std': float(np.std(confidences)),
                'min': float(np.min(confidences)),
                'max': float(np.max(confidences)),
                'median': float(np.median(confidences))
            },
            'correct_predictions_confidence': {
                'mean': float(np.mean(correct_confidences)) if correct_confidences else 0.0,
                'std': float(np.std(correct_confidences)) if correct_confidences else 0.0
            },
            'incorrect_predictions_confidence': {
                'mean': float(np.mean(incorrect_confidences)) if incorrect_confidences else 0.0,
                'std': float(np.std(incorrect_confidences)) if incorrect_confidences else 0.0
            },
            'calibration_bins': self._calculate_calibration(confidences, correct_mask)
        }
    
    def _calculate_calibration(self, confidences: List[float], correct_mask: List[bool]) -> Dict:
        """Calculate calibration statistics."""
        bins = np.linspace(0, 1, 11)  # 10 bins
        bin_accuracies = []
        bin_confidences = []
        bin_counts = []
        
        for i in range(len(bins) - 1):
            mask = np.logical_and(
                np.array(confidences) >= bins[i],
                np.array(confidences) < bins[i+1]
            )
            
            if mask.sum() > 0:
                bin_accuracy = np.array(correct_mask)[mask].mean()
                bin_confidence = np.array(confidences)[mask].mean()
                bin_count = mask.sum()
            else:
                bin_accuracy = 0.0
                bin_confidence = 0.0
                bin_count = 0
            
            bin_accuracies.append(float(bin_accuracy))
            bin_confidences.append(float(bin_confidence))
            bin_counts.append(int(bin_count))
        
        return {
            'bin_boundaries': bins.tolist(),
            'bin_accuracies': bin_accuracies,
            'bin_confidences': bin_confidences,
            'bin_counts': bin_counts
        }
    
    def save_evaluation_report(self, results: Dict, output_path: str):
        """
        Save evaluation results to JSON file.
        
        Args:
            results (Dict): Evaluation results.
            output_path (str): Output file path.
        """
        with open(output_path, 'w') as f:
            json.dump(results, f, indent=2)
        
        self.logger.info(f"Evaluation report saved to: {output_path}")
    
    def plot_evaluation_results(self, results: Dict, save_dir: str = None):
        """
        Create visualization plots from evaluation results.
        
        Args:
            results (Dict): Evaluation results.
            save_dir (str, optional): Directory to save plots.
        """
        # Confusion Matrix
        self._plot_confusion_matrix(results, save_dir)
        
        # Per-class Performance
        self._plot_per_class_metrics(results, save_dir)
        
        # Confidence Distribution
        self._plot_confidence_distribution(results, save_dir)
        
        # Calibration Plot
        self._plot_calibration(results, save_dir)
    
    def _plot_confusion_matrix(self, results: Dict, save_dir: str = None):
        """Plot confusion matrix."""
        cm = np.array(results['confusion_matrix'])
        unique_topics = self.artifacts['unique_topics']
        
        plt.figure(figsize=(10, 8))
        sns.heatmap(
            cm, 
            annot=True, 
            fmt='d', 
            cmap='Blues',
            xticklabels=unique_topics,
            yticklabels=unique_topics
        )
        plt.title('Confusion Matrix')
        plt.xlabel('Predicted Label')
        plt.ylabel('True Label')
        plt.tight_layout()
        
        if save_dir:
            plt.savefig(os.path.join(save_dir, 'confusion_matrix.png'), dpi=300, bbox_inches='tight')
        plt.show()
    
    def _plot_per_class_metrics(self, results: Dict, save_dir: str = None):
        """Plot per-class performance metrics."""
        per_class = results['per_class_metrics']
        topics = list(per_class.keys())
        
        metrics = ['precision', 'recall', 'f1_score']
        metric_values = {metric: [per_class[topic][metric] for topic in topics] for metric in metrics}
        
        x = np.arange(len(topics))
        width = 0.25
        
        fig, ax = plt.subplots(figsize=(12, 6))
        
        for i, metric in enumerate(metrics):
            ax.bar(x + i*width, metric_values[metric], width, label=metric.capitalize())
        
        ax.set_xlabel('Topics')
        ax.set_ylabel('Score')
        ax.set_title('Per-Class Performance Metrics')
        ax.set_xticks(x + width)
        ax.set_xticklabels(topics, rotation=45, ha='right')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_dir:
            plt.savefig(os.path.join(save_dir, 'per_class_metrics.png'), dpi=300, bbox_inches='tight')
        plt.show()
    
    def _plot_confidence_distribution(self, results: Dict, save_dir: str = None):
        """Plot confidence distribution."""
        # This would require the raw confidence data, which we don't store in results
        # For now, just create a placeholder or skip
        pass
    
    def _plot_calibration(self, results: Dict, save_dir: str = None):
        """Plot calibration diagram."""
        calibration = results['confidence_analysis']['calibration_bins']
        
        bin_confidences = calibration['bin_confidences']
        bin_accuracies = calibration['bin_accuracies']
        
        plt.figure(figsize=(8, 6))
        plt.plot([0, 1], [0, 1], 'k--', label='Perfect calibration')
        plt.plot(bin_confidences, bin_accuracies, 'ro-', label='Model calibration')
        
        plt.xlabel('Mean Predicted Probability')
        plt.ylabel('Fraction of Positives')
        plt.title('Calibration Plot')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        if save_dir:
            plt.savefig(os.path.join(save_dir, 'calibration_plot.png'), dpi=300, bbox_inches='tight')
        plt.show()


def main():
    """Main evaluation function."""
    parser = argparse.ArgumentParser(
        description="Evaluate trained news headline classifier"
    )
    
    parser.add_argument(
        '--model_dir',
        type=str,
        default='outputs',
        help='Directory containing trained model'
    )
    
    parser.add_argument(
        '--test_data',
        type=str,
        help='Path to test data CSV file'
    )
    
    parser.add_argument(
        '--output_dir',
        type=str,
        default='evaluation_results',
        help='Directory to save evaluation results'
    )
    
    parser.add_argument(
        '--batch_size',
        type=int,
        default=32,
        help='Batch size for evaluation'
    )
    
    parser.add_argument(
        '--device',
        type=str,
        default='auto',
        help='Device to use (cpu, cuda, or auto)'
    )
    
    args = parser.parse_args()
    
    # Setup device
    if args.device == 'auto':
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    else:
        device = torch.device(args.device)
    
    print(f"Using device: {device}")
    
    # Load model and artifacts
    try:
        artifacts, vocab_data = load_model_artifacts(args.model_dir)
        
        # Create model
        model = NewsHeadlineClassifier(
            vocab_size=len(artifacts['vocab']),
            embedding_dim=artifacts['config']['embedding_dim'],
            hidden_dim=artifacts['config']['hidden_dim'],
            num_classes=len(artifacts['unique_topics']),
            use_gru=artifacts['config']['use_gru']
        )
        
        # Load weights
        model_path = os.path.join(args.model_dir, 'model.pth')
        model.load_state_dict(torch.load(model_path, map_location=device))
        
        print("✅ Model loaded successfully")
        
    except Exception as e:
        print(f"❌ Error loading model: {e}")
        return
    
    # Load test data
    if args.test_data:
        if not os.path.exists(args.test_data):
            print(f"❌ Test data file not found: {args.test_data}")
            return
        test_df = pd.read_csv(args.test_data)
    else:
        # Use a portion of the training data as test data
        training_data_path = os.path.join('data/processed', 'headlines.csv')
        if os.path.exists(training_data_path):
            full_df = pd.read_csv(training_data_path)
            # Use 20% as test data
            test_df = full_df.sample(frac=0.2, random_state=42)
            print(f"Using 20% of training data ({len(test_df)} samples) for evaluation")
        else:
            print("❌ No test data provided and no training data found")
            return
    
    print(f"Test dataset: {len(test_df)} samples")
    
    # Create evaluator
    evaluator = ModelEvaluator(model, artifacts, vocab_data, device)
    
    # Run evaluation
    print("🔄 Running evaluation...")
    results = evaluator.evaluate_dataset(test_df, args.batch_size)
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Save results
    evaluator.save_evaluation_report(
        results, 
        os.path.join(args.output_dir, 'evaluation_report.json')
    )
    
    # Create visualizations
    evaluator.plot_evaluation_results(results, args.output_dir)
    
    # Print summary
    print("\n" + "="*60)
    print("📊 EVALUATION SUMMARY")
    print("="*60)
    
    overall = results['overall_metrics']
    print(f"Overall Accuracy: {overall['accuracy']:.4f}")
    print(f"Macro F1-Score: {overall['macro_f1']:.4f}")
    print(f"Weighted F1-Score: {overall['weighted_f1']:.4f}")
    print(f"Total Samples: {overall['total_samples']}")
    
    print(f"\nPer-Class Performance:")
    for topic, metrics in results['per_class_metrics'].items():
        print(f"  {topic}: F1={metrics['f1_score']:.3f}, Support={metrics['support']}")
    
    print(f"\nError Analysis:")
    error_analysis = results['error_analysis']
    print(f"  Total Errors: {error_analysis['total_errors']}")
    print(f"  Error Rate: {error_analysis['error_rate']:.4f}")
    
    print(f"\n📁 Results saved to: {args.output_dir}")


if __name__ == "__main__":
    main()