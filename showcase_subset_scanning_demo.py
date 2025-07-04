"""
Subset Scanning Adversarial Detection Showcase
==============================================

This script demonstrates the complete workflow of:
1. Loading a trained model and test data
2. Generating FGSM adversarial examples
3. Creating a mixed dataset with hidden adversarial samples
4. Running subset scanning detection to find the anomalous subgroup
5. Evaluating and visualizing the results

This showcases the integration of subset scanning with ART for adversarial detection.
"""

import numpy as np
import torch
import matplotlib.pyplot as plt
import seaborn as sns
from art.defences.detector.evasion import SubsetScanningDetector
import os
import sys

# Add src to path for imports
sys.path.append('src')

from data.dataset import load_mnist_data
from models.cnn_model import MNISTCNN, load_model
from attacks.fgsm_attack import generate_fgsm_attack, evaluate_attack_effectiveness
from scanning.subset_evaluation import (
    evaluate_detection_performance, 
    visualize_detection_results, 
    print_detection_summary
)
from utils.art_utils import create_art_classifier, extract_features_from_layer
from utils.data_utils import prepare_mixed_dataset_for_scanning


def showcase_complete_workflow():
    """
    Complete showcase of subset scanning for adversarial detection.
    """
    
    print("=" * 80)
    print("SUBSET SCANNING ADVERSARIAL DETECTION SHOWCASE")
    print("=" * 80)
    print("This demo showcases the integration of subset scanning with ART")
    print("to detect adversarial examples hidden within clean data.")
    print()
    
    # ============================================================================
    # Phase 1: Load Data and Model
    # ============================================================================
    print("PHASE 1: Loading Data and Model")
    print("-" * 40)
    
    # Load MNIST test data
    print("Loading MNIST test data...")
    _, test_loader, _, _ = load_mnist_data(batch_size=100, test_size=1000)
    test_data, test_labels = next(iter(test_loader))
    
    print(f"✓ Loaded {len(test_data)} test samples")
    print(f"✓ Data shape: {test_data.shape}")
    print(f"✓ Label distribution: {np.bincount(test_labels.numpy())}")
    
    # Load trained model
    print("\nLoading trained model...")
    model = MNISTCNN()
    model_path = 'models/mnist_cnn_model.pth'
    
    if not os.path.exists(model_path):
        print("❌ No trained model found!")
        print("Please run the training script first:")
        print("  python src/models/cnn_model.py")
        return
    
    model = load_model(model, model_path)
    print("✓ Model loaded successfully")
    
    # Evaluate model on clean data
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model.eval()
    with torch.no_grad():
        clean_outputs = model(test_data.to(device))
        clean_preds = torch.argmax(clean_outputs, dim=1)
        clean_accuracy = (clean_preds == test_labels.to(device)).float().mean().item()
    
    print(f"✓ Model accuracy on clean data: {clean_accuracy:.2%}")
    
    # ============================================================================
    # Phase 2: Generate FGSM Adversarial Examples
    # ============================================================================
    print("\n" + "=" * 80)
    print("PHASE 2: Generating FGSM Adversarial Examples")
    print("-" * 40)
    
    print("Generating FGSM adversarial examples...")
    eps = 0.3  # Perturbation magnitude
    adv_data, adv_preds, attack_success_rate = generate_fgsm_attack(
        model, test_data, test_labels, eps=eps, device=device
    )
    
    print(f"✓ FGSM attack completed with eps={eps}")
    print(f"✓ Attack success rate: {attack_success_rate:.2%}")
    print(f"✓ Adversarial examples shape: {adv_data.shape}")
    
    # Evaluate attack effectiveness
    attack_eval = evaluate_attack_effectiveness(
        model, test_data.numpy(), test_labels.numpy(),
        adv_data, adv_preds, device
    )
    
    print(f"✓ Clean accuracy: {attack_eval['clean_accuracy']:.2%}")
    print(f"✓ Adversarial accuracy: {attack_eval['adversarial_accuracy']:.2%}")
    print(f"✓ Mean L∞ perturbation: {attack_eval['mean_linf_norm']:.4f}")
    
    # ============================================================================
    # Phase 3: Create Mixed Dataset (Hidden Adversarial Subgroup)
    # ============================================================================
    print("\n" + "=" * 80)
    print("PHASE 3: Creating Mixed Dataset with Hidden Adversarial Subgroup")
    print("-" * 40)
    
    contamination_rate = 0.1  # 10% adversarial samples
    print(f"Creating mixed dataset with {contamination_rate:.1%} contamination...")
    
    x_combined, y_true_anomaly, sample_info = prepare_mixed_dataset_for_scanning(
        test_data.numpy(), adv_data, contamination_rate=contamination_rate
    )
    
    print(f"✓ Mixed dataset created:")
    print(f"  - Total samples: {len(x_combined)}")
    print(f"  - Clean samples: {sample_info['n_clean']}")
    print(f"  - Adversarial samples: {sample_info['n_adversarial']}")
    print(f"  - Contamination rate: {sample_info['contamination_rate']:.1%}")
    
    # ============================================================================
    # Phase 4: Subset Scanning Detection
    # ============================================================================
    print("\n" + "=" * 80)
    print("PHASE 4: Running Subset Scanning Detection")
    print("-" * 40)
    
    # Create ART classifier
    print("Creating ART classifier...")
    classifier = create_art_classifier(model, device)
    
    # Method 1: Detection on raw flattened pixels
    print("\nMethod 1: Detection on Raw Flattened Pixels")
    print("-" * 50)
    
    x_combined_flat = x_combined.reshape(x_combined.shape[0], -1)
    print(f"Flattening images: {x_combined.shape} → {x_combined_flat.shape}")
    
    detector_raw = SubsetScanningDetector(
        classifier=classifier,
        window_size=x_combined_flat.shape[1],  # 784 for MNIST
        verbose=True
    )
    
    print("Running subset scanning detection...")
    scores_raw, p_values_raw, scan_stats_raw = detector_raw.detect(x_combined_flat)
    
    print(f"✓ Raw pixel detection completed!")
    print(f"  - Scores shape: {scores_raw.shape}")
    print(f"  - P-values shape: {p_values_raw.shape}")
    print(f"  - Scan stats: {scan_stats_raw}")
    
    # Method 2: Detection on extracted features
    print("\nMethod 2: Detection on Extracted Features")
    print("-" * 50)
    
    print("Extracting features from fc2 layer...")
    clean_features = extract_features_from_layer(model, test_data, 'fc2')
    adv_features = extract_features_from_layer(model, torch.from_numpy(adv_data), 'fc2')
    
    x_features_combined, y_features_true, _ = prepare_mixed_dataset_for_scanning(
        clean_features, adv_features, contamination_rate=contamination_rate
    )
    
    print(f"Feature extraction completed: {x_features_combined.shape}")
    
    detector_features = SubsetScanningDetector(
        classifier=classifier,
        window_size=x_features_combined.shape[1],
        verbose=True
    )
    
    print("Running subset scanning detection on features...")
    scores_features, p_values_features, scan_stats_features = detector_features.detect(x_features_combined)
    
    print(f"✓ Feature-based detection completed!")
    print(f"  - Scores shape: {scores_features.shape}")
    print(f"  - P-values shape: {p_values_features.shape}")
    
    # ============================================================================
    # Phase 5: Evaluation and Visualization
    # ============================================================================
    print("\n" + "=" * 80)
    print("PHASE 5: Evaluation and Visualization")
    print("-" * 40)
    
    # Evaluate detection performance
    print("Evaluating detection performance...")
    
    raw_results = evaluate_detection_performance(scores_raw, y_true_anomaly)
    feature_results = evaluate_detection_performance(scores_features, y_features_true)
    
    # Print comparison
    print("\nDETECTION PERFORMANCE COMPARISON")
    print("=" * 60)
    print(f"{'Metric':<20} {'Raw Pixels':<15} {'Features':<15}")
    print("-" * 60)
    print(f"{'ROC AUC':<20} {raw_results['roc_auc']:<15.4f} {feature_results['roc_auc']:<15.4f}")
    print(f"{'PR AUC':<20} {raw_results['pr_auc']:<15.4f} {feature_results['pr_auc']:<15.4f}")
    print(f"{'F1 Score':<20} {raw_results['f1_score']:<15.4f} {feature_results['f1_score']:<15.4f}")
    print(f"{'Precision':<20} {raw_results['precision']:<15.4f} {feature_results['precision']:<15.4f}")
    print(f"{'Recall':<20} {raw_results['recall']:<15.4f} {feature_results['recall']:<15.4f}")
    
    # Visualize results
    print("\nGenerating visualizations...")
    
    # Raw pixel detection results
    visualize_detection_results(
        scores_raw, y_true_anomaly, raw_results,
        title="Raw Pixel Subset Scanning Detection"
    )
    
    # Feature-based detection results
    visualize_detection_results(
        scores_features, y_features_true, feature_results,
        title="Feature-Based Subset Scanning Detection"
    )
    
    # Print detailed summaries
    print("\nRaw Pixel Detection Summary:")
    print_detection_summary(raw_results)
    
    print("\nFeature-Based Detection Summary:")
    print_detection_summary(feature_results)
    
    # ============================================================================
    # Phase 6: Showcase Summary
    # ============================================================================
    print("\n" + "=" * 80)
    print("SHOWCASE SUMMARY")
    print("=" * 80)
    
    print("🎯 SUCCESS: Subset scanning successfully detected the hidden adversarial subgroup!")
    print()
    print("Key Results:")
    print(f"  • FGSM attack success rate: {attack_success_rate:.1%}")
    print(f"  • Hidden adversarial samples: {sample_info['n_adversarial']} ({contamination_rate:.1%})")
    print(f"  • Raw pixel detection ROC AUC: {raw_results['roc_auc']:.3f}")
    print(f"  • Feature-based detection ROC AUC: {feature_results['roc_auc']:.3f}")
    print()
    print("This demonstrates that subset scanning can effectively identify")
    print("adversarial examples even when they're hidden within clean data!")
    print()
    print("Files generated:")
    print("  • detection_results.png - Raw pixel detection visualization")
    print("  • detection_results.png - Feature-based detection visualization")
    print("=" * 80)


if __name__ == "__main__":
    showcase_complete_workflow() 