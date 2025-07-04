#!/usr/bin/env python3
"""
Standalone Visualization App for Subset Scanning Adversarial Detection
=====================================================================

This app provides an interactive way to explore the qualitative visualizations
from the subset scanning adversarial detection demo.

Run this script to see the visualizations:
    python visualization_app.py

The app will:
1. Load sample data (or use dummy data if no real data is available)
2. Display interactive visualizations one by one
3. Save the visualizations to files
4. Provide explanations for each visualization

Each visualization opens in a separate window - close it to proceed to the next one.
"""

import numpy as np
import matplotlib.pyplot as plt
import os
import sys

# Add src to path for imports
sys.path.append('src')

from visualization.qualitative_analysis import create_interactive_visualization_app


def load_sample_data():
    """
    Try to load real data from the demo, fall back to dummy data if not available.
    """
    try:
        # Try to import and load real data
        from data.dataset import load_mnist_data
        from models.cnn_model import MNISTCNN, load_model
        from attacks.fgsm_attack import generate_fgsm_attack
        from utils.data_utils import prepare_mixed_dataset_for_scanning
        from utils.art_utils import create_art_classifier
        from art.defences.detector.evasion import SubsetScanningDetector
        
        print("Loading real data from the demo...")
        
        # Load MNIST test data
        _, test_loader, _, _ = load_mnist_data(batch_size=100, test_size=1000)
        test_data, test_labels = next(iter(test_loader))
        
        # Load model
        model = MNISTCNN()
        model_path = 'models/mnist_cnn_model.pth'
        
        if os.path.exists(model_path):
            model = load_model(model, model_path)
            device = 'cuda' if torch.cuda.is_available() else 'cpu'
            
            # Generate adversarial examples
            eps = 0.3
            adv_data, adv_preds, _ = generate_fgsm_attack(
                model, test_data, test_labels, eps=eps, device=device
            )
            
            # Create mixed dataset
            contamination_rate = 0.1
            x_combined, y_true_anomaly, _ = prepare_mixed_dataset_for_scanning(
                test_data.numpy(), adv_data, contamination_rate=contamination_rate
            )
            
            # Run subset scanning detection
            classifier = create_art_classifier(model, device)
            x_combined_flat = x_combined.reshape(x_combined.shape[0], -1)
            
            detector = SubsetScanningDetector(
                classifier=classifier,
                window_size=x_combined_flat.shape[1],
                verbose=False
            )
            
            scores, _, _ = detector.detect(x_combined_flat)
            
            print("✓ Real data loaded successfully!")
            return {
                'clean_data': test_data.numpy(),
                'adversarial_data': adv_data,
                'true_labels': test_labels.numpy(),
                'adv_predictions': adv_preds,
                'scores': scores,
                'y_true': y_true_anomaly
            }
        else:
            print("⚠️  No trained model found, using dummy data...")
            return None
            
    except Exception as e:
        print(f"⚠️  Could not load real data: {e}")
        print("Using dummy data for demonstration...")
        return None


def create_dummy_data():
    """
    Create dummy data for demonstration purposes.
    """
    print("Creating dummy data for visualization demonstration...")
    
    # Generate dummy data
    np.random.seed(42)
    n_samples = 1000
    
    # Dummy images (simulate MNIST-like data)
    clean_data = np.random.rand(n_samples, 1, 28, 28)
    # Add some structure to make it look more like digits
    clean_data = clean_data * 0.3 + 0.1  # Reduce contrast
    
    # Create adversarial data with subtle perturbations
    adversarial_data = clean_data + np.random.normal(0, 0.05, clean_data.shape)
    adversarial_data = np.clip(adversarial_data, 0, 1)  # Clip to valid range
    
    # Dummy labels and predictions
    true_labels = np.random.randint(0, 10, n_samples)
    adv_predictions = np.random.randint(0, 10, n_samples)
    
    # Create realistic detection scores (adversarial should have higher scores)
    clean_scores = np.random.normal(0, 0.5, n_samples // 2)
    adv_scores = np.random.normal(2, 0.8, n_samples // 2)
    scores = np.concatenate([clean_scores, adv_scores])
    
    # Create true labels for detection (0=clean, 1=adversarial)
    y_true = np.concatenate([np.zeros(n_samples // 2), np.ones(n_samples // 2)])
    
    print("✓ Dummy data created successfully!")
    return {
        'clean_data': clean_data,
        'adversarial_data': adversarial_data,
        'true_labels': true_labels,
        'adv_predictions': adv_predictions,
        'scores': scores,
        'y_true': y_true
    }


def main():
    """
    Main function to run the visualization app.
    """
    print("=" * 80)
    print("🎨 SUBSET SCANNING VISUALIZATION APP")
    print("=" * 80)
    print("This app provides interactive visualizations for adversarial detection.")
    print("Each visualization will open in a separate window.")
    print("Close each window to proceed to the next visualization.")
    print()
    
    # Try to load real data, fall back to dummy data
    data = load_sample_data()
    if data is None:
        data = create_dummy_data()
    
    print("\n" + "=" * 80)
    print("📊 DATA SUMMARY")
    print("=" * 80)
    print(f"   • Clean images: {data['clean_data'].shape}")
    print(f"   • Adversarial images: {data['adversarial_data'].shape}")
    print(f"   • Number of samples: {len(data['clean_data'])}")
    print(f"   • Clean samples: {np.sum(data['y_true'] == 0)}")
    print(f"   • Adversarial samples: {np.sum(data['y_true'] == 1)}")
    print(f"   • Detection scores range: [{data['scores'].min():.3f}, {data['scores'].max():.3f}]")
    print()
    
    # Run the interactive visualization app
    create_interactive_visualization_app(
        clean_data=data['clean_data'],
        adversarial_data=data['adversarial_data'],
        true_labels=data['true_labels'],
        adv_predictions=data['adv_predictions'],
        scores=data['scores'],
        y_true=data['y_true']
    )
    
    print("\n" + "=" * 80)
    print("🎉 VISUALIZATION APP COMPLETED!")
    print("=" * 80)
    print("Thank you for exploring the subset scanning adversarial detection visualizations!")
    print()
    print("To run the full demo with real data:")
    print("   1. Train the model: python src/models/cnn_model.py")
    print("   2. Run the showcase: python showcase_subset_scanning_demo.py")
    print("=" * 80)


if __name__ == "__main__":
    main() 