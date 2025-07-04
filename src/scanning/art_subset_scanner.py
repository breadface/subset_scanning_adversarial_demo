"""
ART Subset Scanning Detector for adversarial detection.
"""

import numpy as np
import torch
from art.defences.detector.evasion import SubsetScanningDetector
from art.estimators.classification import PyTorchClassifier
import torch.nn as nn
import os


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


def run_subset_scanning_detection(model, clean_data, adversarial_data, 
                                 background_data=None, layer='fc2', 
                                 scoring_function='BerkJones', device='cpu'):
    """
    Run subset scanning detection using ART's SubsetScanningDetector.
    
    Args:
        model: PyTorch model
        clean_data: Clean test data
        adversarial_data: Adversarial test data
        background_data: Background data for comparison (if None, uses clean_data)
        layer (str): Layer name or index to extract activations from
        scoring_function (str): 'BerkJones', 'HigherCriticism', or 'KolmarovSmirnov'
        device (str): Device to run on
        
    Returns:
        dict: Detection results
    """
    
    # Create ART classifier
    art_classifier = create_art_classifier(model, device)
    
    # Use clean data as background if not provided
    if background_data is None:
        background_data = clean_data
    
    print(f"Initializing SubsetScanningDetector...")
    print(f"  Layer: {layer}")
    print(f"  Scoring function: {scoring_function}")
    print(f"  Background data size: {len(background_data)}")
    print(f"  Clean data size: {len(clean_data)}")
    print(f"  Adversarial data size: {len(adversarial_data)}")
    
    # Initialize detector
    detector = SubsetScanningDetector(
        classifier=art_classifier,
        bgd_data=background_data,
        layer=layer,
        scoring_function=scoring_function,
        verbose=True
    )
    
    # Run the scan
    print(f"\nRunning subset scanning detection...")
    clean_scores, adv_scores, detection_power = detector.scan(
        clean_x=clean_data,
        adv_x=adversarial_data,
        clean_size=None,  # Use all clean data
        adv_size=None,    # Use all adversarial data
        run=10            # Number of runs for statistical significance
    )
    
    # Calculate detection metrics
    clean_mean = np.mean(clean_scores)
    adv_mean = np.mean(adv_scores)
    clean_std = np.std(clean_scores)
    adv_std = np.std(adv_scores)
    
    # Determine threshold (e.g., 95th percentile of clean scores)
    threshold = np.percentile(clean_scores, 95)
    
    # Calculate detection rates
    clean_detected = np.sum(clean_scores > threshold)
    adv_detected = np.sum(adv_scores > threshold)
    
    clean_detection_rate = clean_detected / len(clean_scores)
    adv_detection_rate = adv_detected / len(adv_scores)
    
    results = {
        'clean_scores': clean_scores,
        'adversarial_scores': adv_scores,
        'detection_power': detection_power,
        'clean_mean': clean_mean,
        'adversarial_mean': adv_mean,
        'clean_std': clean_std,
        'adversarial_std': adv_std,
        'threshold': threshold,
        'clean_detection_rate': clean_detection_rate,
        'adversarial_detection_rate': adv_detection_rate,
        'true_positive_rate': adv_detection_rate,
        'false_positive_rate': clean_detection_rate,
        'layer': layer,
        'scoring_function': scoring_function
    }
    
    print(f"\nDetection Results:")
    print(f"  Detection Power: {detection_power:.4f}")
    print(f"  Clean Score Mean: {clean_mean:.4f} ± {clean_std:.4f}")
    print(f"  Adversarial Score Mean: {adv_mean:.4f} ± {adv_std:.4f}")
    print(f"  Threshold (95th percentile): {threshold:.4f}")
    print(f"  True Positive Rate: {adv_detection_rate:.2%}")
    print(f"  False Positive Rate: {clean_detection_rate:.2%}")
    
    return results


def experiment_with_different_layers(model, clean_data, adversarial_data, 
                                   layers=['conv1', 'conv2', 'conv3', 'fc1', 'fc2'],
                                   scoring_functions=['BerkJones', 'HigherCriticism'],
                                   device='cpu'):
    """
    Experiment with subset scanning on different layers and scoring functions.
    
    Args:
        model: PyTorch model
        clean_data: Clean test data
        adversarial_data: Adversarial test data
        layers (list): List of layer names to test
        scoring_functions (list): List of scoring functions to test
        device (str): Device to run on
        
    Returns:
        dict: Results for each combination
    """
    
    results = {}
    
    for layer in layers:
        for scoring_function in scoring_functions:
            print(f"\n{'='*50}")
            print(f"Testing Layer: {layer}, Scoring Function: {scoring_function}")
            print(f"{'='*50}")
            
            try:
                layer_results = run_subset_scanning_detection(
                    model, clean_data, adversarial_data,
                    layer=layer, scoring_function=scoring_function, device=device
                )
                
                results[f"{layer}_{scoring_function}"] = layer_results
                
            except Exception as e:
                print(f"Error with layer {layer} and scoring function {scoring_function}: {e}")
                results[f"{layer}_{scoring_function}"] = {'error': str(e)}
    
    return results


def compare_detection_methods(model, clean_data, adversarial_data, device='cpu'):
    """
    Compare different detection methods and configurations.
    
    Args:
        model: PyTorch model
        clean_data: Clean test data
        adversarial_data: Adversarial test data
        device (str): Device to run on
        
    Returns:
        dict: Comparison results
    """
    
    print("Comparing subset scanning detection methods...")
    
    # Test different layers
    layers = ['fc2', 'fc1', 'conv3']  # Focus on later layers for better detection
    
    # Test different scoring functions
    scoring_functions = ['BerkJones', 'HigherCriticism', 'KolmarovSmirnov']
    
    results = experiment_with_different_layers(
        model, clean_data, adversarial_data,
        layers=layers, scoring_functions=scoring_functions, device=device
    )
    
    # Print summary
    print(f"\n{'='*60}")
    print("DETECTION METHOD COMPARISON SUMMARY")
    print(f"{'='*60}")
    
    for key, result in results.items():
        if 'error' not in result:
            print(f"\n{key}:")
            print(f"  Detection Power: {result['detection_power']:.4f}")
            print(f"  True Positive Rate: {result['true_positive_rate']:.2%}")
            print(f"  False Positive Rate: {result['false_positive_rate']:.2%}")
            print(f"  Score Separation: {result['adversarial_mean'] - result['clean_mean']:.4f}")
    
    return results


if __name__ == "__main__":
    # Test the subset scanning detector
    from src.data.dataset import load_mnist_data
    from src.models.cnn_model import MNISTCNN, load_model
    from src.attacks.fgsm_attack import generate_fgsm_attack
    
    # Load data
    _, test_loader, _, _ = load_mnist_data(batch_size=100, test_size=200)
    test_data, test_labels = next(iter(test_loader))
    
    # Load model
    model = MNISTCNN()
    model_path = 'models/mnist_cnn_model.pth'
    
    if os.path.exists(model_path):
        model = load_model(model, model_path)
    else:
        print("No trained model found. Please run train_model.py first.")
        exit(1)
    
    # Generate adversarial examples
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    adv_data, adv_labels, _ = generate_fgsm_attack(
        model, test_data, test_labels, eps=0.3, device=device
    )
    
    # Run subset scanning detection
    results = run_subset_scanning_detection(
        model, test_data.numpy(), adv_data,
        layer='fc2', scoring_function='BerkJones', device=device
    )
    
    print("Subset scanning detection test completed!") 