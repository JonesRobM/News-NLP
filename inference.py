"""
Inference script for news headline topic classification.

This script demonstrates how to load a trained model and make predictions
on new headlines.
"""

import os
import sys
import argparse
import torch
from typing import List, Dict, Tuple

# Add src directory to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from model import NewsHeadlineClassifier
from utils import load_model_artifacts, predict_single_headline, setup_logging


def load_trained_model(model_dir: str) -> Tuple[NewsHeadlineClassifier, Dict, Dict]:
    """
    Load a trained model and its artifacts.
    
    Args:
        model_dir (str): Directory containing model artifacts.
    
    Returns:
        Tuple[NewsHeadlineClassifier, Dict, Dict]: Model, artifacts, and vocabulary.
    """
    logger = setup_logging()
    
    # Load artifacts
    artifacts, vocab_data = load_model_artifacts(model_dir)
    
    # Create model instance
    model = NewsHeadlineClassifier(
        vocab_size=len(artifacts['vocab']),
        embedding_dim=artifacts['config']['embedding_dim'],
        hidden_dim=artifacts['config']['hidden_dim'],
        num_classes=len(artifacts['unique_topics']),
        use_gru=artifacts['config']['use_gru'],
        dropout_rate=artifacts['config'].get('dropout_rate', 0.3)
    )
    
    # Load trained weights
    model_path = os.path.join(model_dir, 'model.pth')
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file not found: {model_path}")
    
    model.load_state_dict(torch.load(model_path, map_location='cpu'))
    model.eval()
    
    logger.info(f"Model loaded successfully from {model_dir}")
    logger.info(f"Model supports topics: {artifacts['unique_topics']}")
    
    return model, artifacts, vocab_data


def predict_headlines(
    headlines: List[str],
    model: NewsHeadlineClassifier,
    artifacts: Dict,
    vocab_data: Dict,
    show_probabilities: bool = False
) -> List[Dict]:
    """
    Predict topics for multiple headlines.
    
    Args:
        headlines (List[str]): List of headlines to classify.
        model (NewsHeadlineClassifier): Trained model.
        artifacts (Dict): Model artifacts.
        vocab_data (Dict): Vocabulary data.
        show_probabilities (bool): Whether to include all topic probabilities.
    
    Returns:
        List[Dict]: List of prediction results.
    """
    results = []
    
    for headline in headlines:
        topic, confidence, probabilities = predict_single_headline(
            headline=headline,
            model=model,
            word_to_idx=vocab_data['word_to_idx'],
            unique_topics=artifacts['unique_topics'],
            max_len=artifacts['config']['max_len']
        )
        
        result = {
            'headline': headline,
            'predicted_topic': topic,
            'confidence': confidence
        }
        
        if show_probabilities:
            result['all_probabilities'] = probabilities
        
        results.append(result)
    
    return results


def interactive_mode(model: NewsHeadlineClassifier, artifacts: Dict, vocab_data: Dict):
    """
    Run interactive prediction mode.
    
    Args:
        model (NewsHeadlineClassifier): Trained model.
        artifacts (Dict): Model artifacts.
        vocab_data (Dict): Vocabulary data.
    """
    print("\n" + "="*60)
    print("Interactive News Headline Topic Classifier")
    print("="*60)
    print(f"Available topics: {', '.join(artifacts['unique_topics'])}")
    print("Enter headlines to classify (type 'quit' to exit):")
    print("-"*60)
    
    while True:
        try:
            headline = input("\nEnter headline: ").strip()
            
            if headline.lower() in ['quit', 'exit', 'q']:
                print("Goodbye!")
                break
            
            if not headline:
                continue
            
            # Make prediction
            topic, confidence, probabilities = predict_single_headline(
                headline=headline,
                model=model,
                word_to_idx=vocab_data['word_to_idx'],
                unique_topics=artifacts['unique_topics'],
                max_len=artifacts['config']['max_len']
            )
            
            # Display results
            print(f"\nPrediction Results:")
            print(f"  Headline: {headline}")
            print(f"  Predicted Topic: {topic}")
            print(f"  Confidence: {confidence:.3f}")
            
            # Show top 3 probabilities
            sorted_probs = sorted(probabilities.items(), key=lambda x: x[1], reverse=True)
            print(f"  Top 3 probabilities:")
            for i, (topic_name, prob) in enumerate(sorted_probs[:3], 1):
                print(f"    {i}. {topic_name}: {prob:.3f}")
        
        except KeyboardInterrupt:
            print("\nGoodbye!")
            break
        except Exception as e:
            print(f"Error processing headline: {str(e)}")


def main():
    """Main function for inference script."""
    parser = argparse.ArgumentParser(
        description="News Headline Topic Classification Inference"
    )
    
    parser.add_argument(
        '--model_dir',
        type=str,
        default='outputs',
        help='Directory containing trained model artifacts'
    )
    
    parser.add_argument(
        '--headlines',
        nargs='+',
        help='Headlines to classify (space-separated)'
    )
    
    parser.add_argument(
        '--interactive',
        action='store_true',
        help='Run in interactive mode'
    )
    
    parser.add_argument(
        '--show_probabilities',
        action='store_true',
        help='Show probabilities for all topics'
    )
    
    parser.add_argument(
        '--examples',
        action='store_true',
        help='Run with example headlines'
    )
    
    args = parser.parse_args()
    
    # Load model
    try:
        model, artifacts, vocab_data = load_trained_model(args.model_dir)
    except Exception as e:
        print(f"Error loading model: {str(e)}")
        return
    
    # Run examples
    if args.examples:
        example_headlines = [
            "Government announces new tax policy changes",
            "Tech giant releases revolutionary smartphone",
            "Stock market reaches record high levels",
            "Football team wins championship final",
            "Scientists discover new cancer treatment"
        ]
        
        print("Example Headlines Classification:")
        print("="*50)
        
        results = predict_headlines(
            example_headlines, model, artifacts, vocab_data, args.show_probabilities
        )
        
        for result in results:
            print(f"\nHeadline: {result['headline']}")
            print(f"Topic: {result['predicted_topic']}")
            print(f"Confidence: {result['confidence']:.3f}")
            
            if args.show_probabilities:
                print("All probabilities:")
                for topic, prob in result['all_probabilities'].items():
                    print(f"  {topic}: {prob:.3f}")
        
        return
    
    # Interactive mode
    if args.interactive:
        interactive_mode(model, artifacts, vocab_data)
        return
    
    # Classify provided headlines
    if args.headlines:
        results = predict_headlines(
            args.headlines, model, artifacts, vocab_data, args.show_probabilities
        )
        
        print("Classification Results:")
        print("="*40)
        
        for result in results:
            print(f"\nHeadline: {result['headline']}")
            print(f"Topic: {result['predicted_topic']}")
            print(f"Confidence: {result['confidence']:.3f}")
            
            if args.show_probabilities:
                print("All probabilities:")
                for topic, prob in result['all_probabilities'].items():
                    print(f"  {topic}: {prob:.3f}")
    else:
        # Default to interactive mode if no headlines provided
        interactive_mode(model, artifacts, vocab_data)


if __name__ == "__main__":
    main()