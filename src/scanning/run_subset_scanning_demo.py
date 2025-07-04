"""
Comprehensive demo for running ART's SubsetScanningDetector.
"""

import numpy as np
import torch
import torch.nn as nn
from art.defences.detector.evasion import SubsetScanningDetector
from art.estimators.classification import PyTorchClassifier
import os
import sys

# Add src to path for imports
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from data.dataset import load_mnist_data
from models.cnn_model import MNISTCNN, load_model
from attacks.fgsm_attack import generate_fgsm_attack
from scanning.subset_evaluation import (
    evaluate_detection_performance,
    visualize_detection_results,
    print_detection_summary
)
from utils.art_utils import create_art_classifier, extract_features_from_layer
from utils.data_utils import prepare_mixed_dataset_for_scanning


def create_art_classifier(model, device='cpu'):
    """
    Create an ART classifier from a PyTorch model.
    
    Args:
        model: PyTorch model
        device (str): Device to run on
        
    Returns:
        PyTorchClassifier: ART classifier
    """
    criterion = nn.CrossEntropyLoss()
    
    art_classifier = PyTorchClassifier(
        model=model,
        loss=criterion,
        input_shape=(1, 28, 28),  # MNIST shape
        nb_classes=10,
        device_type=device
    )
    
    return art_classifier


def extract_features_from_model(model, data, layer_name='fc2'):
    """
    Extract features from a specific layer of the model.
    
    Args:
        model: PyTorch model
        data: Input data
        layer_name (str): Name of the layer to extract features from
        
    Returns:
        np.ndarray: Extracted features
    """
    model.eval()
    features = []
    
    with torch.no_grad():
        for batch in data:
            # Forward pass until the specified layer
            x = batch[0] if isinstance(batch, (list, tuple)) else batch
            
            if layer_name == 'conv1':
                x = model.conv1(x)
                x = torch.relu(x)
                x = torch.max_pool2d(x, 2, 2)
            elif layer_name == 'conv2':
                x = model.conv1(x)
                x = torch.relu(x)
                x = torch.max_pool2d(x, 2, 2)
                x = model.conv2(x)
                x = torch.relu(x)
                x = torch.max_pool2d(x, 2, 2)
            elif layer_name == 'conv3':
                x = model.conv1(x)
                x = torch.relu(x)
                x = torch.max_pool2d(x, 2, 2)
                x = model.conv2(x)
                x = torch.relu(x)
                x = torch.max_pool2d(x, 2, 2)
                x = model.conv3(x)
                x = torch.relu(x)
                x = torch.max_pool2d(x, 2, 2)
            elif layer_name == 'fc1':
                x = model.conv1(x)
                x = torch.relu(x)
                x = torch.max_pool2d(x, 2, 2)
                x = model.conv2(x)
                x = torch.relu(x)
                x = torch.max_pool2d(x, 2, 2)
                x = model.conv3(x)
                x = torch.relu(x)
                x = torch.max_pool2d(x, 2, 2)
                x = x.view(x.size(0), -1)
                x = model.fc1(x)
                x = torch.relu(x)
            elif layer_name == 'fc2':
                x = model.conv1(x)
                x = torch.relu(x)
                x = torch.max_pool2d(x, 2, 2)
                x = model.conv2(x)
                x = torch.relu(x)
                x = torch.max_pool2d(x, 2, 2)
                x = model.conv3(x)
                x = torch.relu(x)
                x = torch.max_pool2d(x, 2, 2)
                x = x.view(x.size(0), -1)
                x = model.fc1(x)
                x = torch.relu(x)
                x = model.fc2(x)
            
            # Flatten features
            x = x.view(x.size(0), -1)
            features.append(x.cpu().numpy())
    
    return np.concatenate(features, axis=0)


def run_subset_scanning_on_features(model, clean_data, adversarial_data, 
                                   layer='fc2', scoring_function='BerkJones', 
                                   device='cpu', contamination_rate=0.1):
    """
    Run subset scanning detection on extracted features.
    
    Args:
        model: PyTorch model
        clean_data: Clean test data
        adversarial_data: Adversarial test data
        layer (str): Layer to extract features from
        scoring_function (str): Scoring function for subset scanning
        device (str): Device to run on
        contamination_rate (float): Rate of adversarial samples in mixed dataset
        
    Returns:
        dict: Detection results
    """
    
    print(f"Extracting features from layer: {layer}")
    
    # Extract features from clean and adversarial data
    clean_features = extract_features_from_model(model, [clean_data], layer)
    adv_features = extract_features_from_model(model, [adversarial_data], layer)
    
    print(f"Feature extraction completed:")
    print(f"  Clean features shape: {clean_features.shape}")
    print(f"  Adversarial features shape: {adv_features.shape}")
    print(f"  Feature dimension: {clean_features.shape[1]}")
    
    # Prepare mixed dataset
    x_combined, y_true_anomaly, sample_info = prepare_mixed_dataset_for_scanning(
        clean_features, adv_features, contamination_rate=contamination_rate
    )
    
    # Create ART classifier
    art_classifier = create_art_classifier(model, device)
    
    print(f"\nInitializing SubsetScanningDetector...")
    print(f"  Window size: {x_combined.shape[1]} (feature dimension)")
    print(f"  Scoring function: {scoring_function}")
    print(f"  Total samples: {len(x_combined)}")
    
    # Initialize detector with feature dimension as window size
    detector = SubsetScanningDetector(
        classifier=art_classifier,
        window_size=x_combined.shape[1],  # Use feature dimension
        scoring_function=scoring_function,
        verbose=True
    )
    
    # Run detection
    print(f"\nRunning subset scanning detection...")
    print("This may take a while depending on the number of samples and features...")
    
    # For unsupervised detection, we don't need labels in detect method
    # The detector will analyze the data and return anomaly scores
    scores, p_values, scan_stats = detector.detect(x_combined)
    
    print(f"Detection completed!")
    print(f"  Scores shape: {scores.shape}")
    print(f"  P-values shape: {p_values.shape}")
    print(f"  Scan stats keys: {list(scan_stats.keys()) if scan_stats else 'None'}")
    
    # Evaluate detection performance
    results = evaluate_detection_performance(scores, y_true_anomaly)
    
    # Add additional information
    results.update({
        'layer': layer,
        'scoring_function': scoring_function,
        'feature_dimension': x_combined.shape[1],
        'sample_info': sample_info,
        'p_values': p_values,
        'scan_stats': scan_stats
    })
    
    return results


def run_subset_scanning_on_raw_pixels(model, clean_data, adversarial_data,
                                     scoring_function='BerkJones', device='cpu',
                                     contamination_rate=0.1):
    """
    Run subset scanning detection on raw flattened pixels.
    
    Args:
        model: PyTorch model
        clean_data: Clean test data
        adversarial_data: Adversarial test data
        scoring_function (str): Scoring function for subset scanning
        device (str): Device to run on
        contamination_rate (float): Rate of adversarial samples in mixed dataset
        
    Returns:
        dict: Detection results
    """
    
    print(f"Preparing raw pixel data for subset scanning...")
    
    # Flatten the images: (batch_size, channels, height, width) -> (batch_size, features)
    clean_flat = clean_data.reshape(clean_data.shape[0], -1)
    adv_flat = adversarial_data.reshape(adversarial_data.shape[0], -1)
    
    print(f"Data flattening completed:")
    print(f"  Clean data shape: {clean_data.shape} -> {clean_flat.shape}")
    print(f"  Adversarial data shape: {adversarial_data.shape} -> {adv_flat.shape}")
    print(f"  Feature dimension: {clean_flat.shape[1]} (28*28 = 784)")
    
    # Prepare mixed dataset
    x_combined, y_true_anomaly, sample_info = prepare_mixed_dataset_for_scanning(
        clean_flat, adv_flat, contamination_rate=contamination_rate
    )
    
    # Create ART classifier
    art_classifier = create_art_classifier(model, device)
    
    print(f"\nInitializing SubsetScanningDetector for raw pixels...")
    print(f"  Window size: {x_combined.shape[1]} (784 pixels)")
    print(f"  Scoring function: {scoring_function}")
    print(f"  Total samples: {len(x_combined)}")
    
    # Initialize detector
    detector = SubsetScanningDetector(
        classifier=art_classifier,
        window_size=x_combined.shape[1],  # 784 for MNIST
        scoring_function=scoring_function,
        verbose=True
    )
    
    # Run detection
    print(f"\nRunning subset scanning detection on raw pixels...")
    print("This may take a while for 784-dimensional data...")
    
    scores, p_values, scan_stats = detector.detect(x_combined)
    
    print(f"Detection completed!")
    print(f"  Scores shape: {scores.shape}")
    print(f"  P-values shape: {p_values.shape}")
    
    # Evaluate detection performance
    results = evaluate_detection_performance(scores, y_true_anomaly)
    
    # Add additional information
    results.update({
        'layer': 'raw_pixels',
        'scoring_function': scoring_function,
        'feature_dimension': x_combined.shape[1],
        'sample_info': sample_info,
        'p_values': p_values,
        'scan_stats': scan_stats
    })
    
    return results


def compare_feature_vs_raw_detection(model, clean_data, adversarial_data, device='cpu'):
    """
    Compare detection performance between feature-based and raw pixel detection.
    
    Args:
        model: PyTorch model
        clean_data: Clean test data
        adversarial_data: Adversarial test data
        device (str): Device to run on
        
    Returns:
        dict: Comparison results
    """
    
    print("Comparing feature-based vs raw pixel detection...")
    
    # Test on features
    print(f"\n{'='*60}")
    print("FEATURE-BASED DETECTION")
    print(f"{'='*60}")
    
    feature_results = run_subset_scanning_on_features(
        model, clean_data, adversarial_data,
        layer='fc2', scoring_function='BerkJones',
        device=device, contamination_rate=0.1
    )
    
    # Test on raw pixels
    print(f"\n{'='*60}")
    print("RAW PIXEL DETECTION")
    print(f"{'='*60}")
    
    raw_results = run_subset_scanning_on_raw_pixels(
        model, clean_data, adversarial_data,
        scoring_function='BerkJones', device=device,
        contamination_rate=0.1
    )
    
    # Compare results
    comparison = {
        'feature_based': feature_results,
        'raw_pixels': raw_results
    }
    
    print(f"\n{'='*80}")
    print("DETECTION METHOD COMPARISON")
    print(f"{'='*80}")
    
    print(f"{'Method':<20} {'ROC AUC':<10} {'PR AUC':<10} {'F1 Score':<10} {'Feature Dim':<12}")
    print(f"{'-'*80}")
    
    for method, results in comparison.items():
        print(f"{method:<20} {results['roc_auc']:<10.4f} {results['pr_auc']:<10.4f} "
              f"{results['f1_score']:<10.4f} {results['feature_dimension']:<12}")
    
    return comparison


def main():
    """
    Main function to run the subset scanning demo.
    """
    
    print("Subset Scanning Adversarial Detection Demo")
    print("=" * 50)
    
    # Load data
    print("Loading MNIST data...")
    _, test_loader, _, _ = load_mnist_data(batch_size=100, test_size=500)
    test_data, test_labels = next(iter(test_loader))
    
    # Load model
    print("Loading trained model...")
    model = MNISTCNN()
    model_path = 'models/mnist_cnn_model.pth'
    
    if not os.path.exists(model_path):
        print(f"Error: No trained model found at {model_path}")
        print("Please run train_model.py first to train the model.")
        return
    
    model = load_model(model, model_path)
    
    # Generate adversarial examples
    print("Generating FGSM adversarial examples...")
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    adv_data, adv_labels, _ = generate_fgsm_attack(
        model, test_data, test_labels, eps=0.3, device=device
    )
    
    print(f"Data preparation completed:")
    print(f"  Clean samples: {len(test_data)}")
    print(f"  Adversarial samples: {len(adv_data)}")
    print(f"  Device: {device}")
    
    # Run comparison
    comparison_results = compare_feature_vs_raw_detection(
        model, test_data.numpy(), adv_data, device=device
    )
    
    # Visualize results
    print("\nGenerating visualizations...")
    
    # Feature-based detection results
    feature_results = comparison_results['feature_based']
    visualize_detection_results(
        feature_results['scores'], 
        feature_results['sample_info']['y_true_anomaly'],
        feature_results,
        title="Feature-Based Subset Scanning Detection"
    )
    
    # Raw pixel detection results
    raw_results = comparison_results['raw_pixels']
    visualize_detection_results(
        raw_results['scores'],
        raw_results['sample_info']['y_true_anomaly'],
        raw_results,
        title="Raw Pixel Subset Scanning Detection"
    )
    
    # Print detailed summaries
    print("\nFeature-Based Detection Summary:")
    print_detection_summary(feature_results)
    
    print("\nRaw Pixel Detection Summary:")
    print_detection_summary(raw_results)
    
    print("\nDemo completed successfully!")
    print("Check the generated plots for detailed results.")


if __name__ == "__main__":
    main() 