"""
Simple demo for instantiating and running ART's SubsetScanningDetector.
"""

import numpy as np
import torch
import os
import sys
from art.defences.detector.evasion import SubsetScanningDetector

# Add src to path for imports
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from data.dataset import load_mnist_data
from models.cnn_model import MNISTCNN, load_model
from attacks.fgsm_attack import generate_fgsm_attack
from utils.art_utils import create_art_classifier, extract_features_from_layer
from utils.data_utils import prepare_mixed_dataset_for_scanning


def main():
    """
    Main function demonstrating SubsetScanningDetector usage.
    """
    print("ART SubsetScanningDetector Demo")
    print("=" * 40)
    # 1. Load data and model
    print("1. Loading data and model...")
    _, test_loader, _, _ = load_mnist_data(batch_size=100, test_size=200)
    test_data, test_labels = next(iter(test_loader))
    model = MNISTCNN()
    model_path = 'models/mnist_cnn_model.pth'
    if not os.path.exists(model_path):
        print(f"Error: No trained model found at {model_path}")
        print("Please run train_model.py first to train the model.")
        return
    model = load_model(model, model_path)
    # 2. Generate adversarial examples
    print("2. Generating FGSM adversarial examples...")
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    adv_data, adv_labels, _ = generate_fgsm_attack(
        model, test_data, test_labels, eps=0.3, device=device
    )
    # 3. Prepare mixed dataset
    print("3. Preparing mixed dataset...")
    contamination_rate = 0.1  # 10% adversarial samples
    x_combined, y_true_anomaly, sample_info = prepare_mixed_dataset_for_scanning(
        test_data.numpy(), adv_data, contamination_rate=contamination_rate
    )
    print(f"Mixed dataset prepared:")
    print(f"  Total samples: {len(x_combined)}")
    print(f"  Clean samples: {np.sum(y_true_anomaly == 0)}")
    print(f"  Adversarial samples: {np.sum(y_true_anomaly == 1)}")
    # 4. Create ART classifier
    print("4. Creating ART classifier...")
    classifier = create_art_classifier(model, device)
    # 5. Method 1: Detection on raw flattened pixels
    print("\n5. Method 1: Detection on raw flattened pixels")
    print("-" * 50)
    x_combined_flat = x_combined.reshape(x_combined.shape[0], -1)
    print(f"Flattened data shape: {x_combined_flat.shape}")
    print(f"Feature dimension: {x_combined_flat.shape[1]} (28*28 = 784)")
    detector_raw = SubsetScanningDetector(
        classifier=classifier, 
        window_size=x_combined_flat.shape[1]  # 784 for MNIST
    )
    print("Running subset scanning detection on raw pixels...")
    scores_raw, p_values_raw, scan_stats_raw = detector_raw.detect(x_combined_flat)
    print(f"Detection completed!")
    print(f"  Scores shape: {scores_raw.shape}")
    print(f"  P-values shape: {p_values_raw.shape}")
    print(f"  Scan stats: {scan_stats_raw}")
    # 6. Method 2: Detection on extracted features
    print("\n6. Method 2: Detection on extracted features")
    print("-" * 50)
    print("Extracting features from fc2 layer...")
    clean_features = extract_features_from_layer(model, test_data, 'fc2')
    adv_features = extract_features_from_layer(model, torch.from_numpy(adv_data), 'fc2')
    x_features_combined, y_features_true, _ = prepare_mixed_dataset_for_scanning(
        clean_features, adv_features, contamination_rate=contamination_rate
    )
    print(f"Feature extraction completed:")
    print(f"  Feature dimension: {x_features_combined.shape[1]}")
    print(f"  Total samples: {len(x_features_combined)}")
    detector_features = SubsetScanningDetector(
        classifier=classifier,
        window_size=x_features_combined.shape[1]  # Feature dimension
    )
    print("Running subset scanning detection on features...")
    scores_features, p_values_features, scan_stats_features = detector_features.detect(x_features_combined)
    print(f"Detection completed!")
    print(f"  Scores shape: {scores_features.shape}")
    print(f"  P-values shape: {p_values_features.shape}")
    print(f"  Scan stats: {scan_stats_features}")
    # 7. Compare results
    print("\n7. Results Comparison")
    print("=" * 50)
    print("Raw Pixel Detection:")
    print(f"  Score mean: {np.mean(scores_raw):.4f}")
    print(f"  Score std: {np.std(scores_raw):.4f}")
    print(f"  Min score: {np.min(scores_raw):.4f}")
    print(f"  Max score: {np.max(scores_raw):.4f}")
    print("\nFeature-based Detection:")
    print(f"  Score mean: {np.mean(scores_features):.4f}")
    print(f"  Score std: {np.std(scores_features):.4f}")
    print(f"  Min score: {np.min(scores_features):.4f}")
    print(f"  Max score: {np.max(scores_features):.4f}")
    threshold_raw = np.percentile(scores_raw, 95)
    threshold_features = np.percentile(scores_features, 95)
    raw_detected = np.sum(scores_raw > threshold_raw)
    features_detected = np.sum(scores_features > threshold_features)
    print(f"\nDetection Rates (95th percentile threshold):")
    print(f"  Raw pixels: {raw_detected}/{len(scores_raw)} ({raw_detected/len(scores_raw):.1%})")
    print(f"  Features: {features_detected}/{len(scores_features)} ({features_detected/len(scores_features):.1%})")
    print("\nDemo completed successfully!")

if __name__ == "__main__":
    main() 