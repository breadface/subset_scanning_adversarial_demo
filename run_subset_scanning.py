#!/usr/bin/env python3
"""
Run ART's subset scanning detection on adversarial examples.
"""

import sys
import os
sys.path.append('src')

from src.data.dataset import load_mnist_data
from src.models.cnn_model import MNISTCNN, load_model
from src.attacks.fgsm_attack import generate_fgsm_attack
from src.scanning.art_subset_scanner import run_subset_scanning_detection, compare_detection_methods
import torch


def main():
    print("Running ART Subset Scanning Detection...")
    
    # Load test data
    print("Loading test data...")
    _, test_loader, _, _ = load_mnist_data(batch_size=100, test_size=200)
    test_data, test_labels = next(iter(test_loader))
    
    # Load trained model
    model = MNISTCNN()
    model_path = 'models/mnist_cnn_model.pth'
    
    if not os.path.exists(model_path):
        print("No trained model found. Run train_model.py first.")
        return
    
    model = load_model(model, model_path)
    
    # Generate adversarial examples
    print("Generating adversarial examples...")
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    adv_data, adv_labels, success_rate = generate_fgsm_attack(
        model, test_data, test_labels, eps=0.3, device=device
    )
    
    print(f"Attack success rate: {success_rate:.1%}")
    
    # Run subset scanning detection
    print("\nRunning subset scanning detection...")
    results = run_subset_scanning_detection(
        model, test_data.numpy(), adv_data,
        layer='fc2', scoring_function='BerkJones', device=device
    )
    
    # Compare different methods
    print("\nComparing different detection methods...")
    comparison_results = compare_detection_methods(
        model, test_data.numpy(), adv_data, device=device
    )
    
    print("\nDetection completed!")


if __name__ == "__main__":
    main() 