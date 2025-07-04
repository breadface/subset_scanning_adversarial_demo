#!/usr/bin/env python3
"""
Run FGSM attack on trained MNIST model.
"""

import sys
import os
sys.path.append('src')

from src.data.dataset import load_mnist_data
from src.models.cnn_model import MNISTCNN, load_model
from src.attacks.fgsm_attack import generate_fgsm_attack, experiment_with_eps_values
import torch


def main():
    print("Running FGSM attack demo...")
    
    # Load test data
    print("Loading test data...")
    _, test_loader, _, _ = load_mnist_data(batch_size=100, test_size=100)
    test_data, test_labels = next(iter(test_loader))
    
    # Load trained model
    model = MNISTCNN()
    model_path = 'models/mnist_cnn_model.pth'
    
    if not os.path.exists(model_path):
        print("No trained model found. Run train_model.py first.")
        return
    
    model = load_model(model, model_path)
    
    # Run attack with different eps values
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    eps_values = [0.1, 0.2, 0.3, 0.4, 0.5]
    
    print(f"Testing FGSM attack with eps values: {eps_values}")
    results = experiment_with_eps_values(model, test_data, test_labels, eps_values, device)
    
    # Show results
    print("\nAttack Results:")
    for eps, result in results.items():
        print(f"eps={eps}: {result['success_rate']:.1%} success rate")


if __name__ == "__main__":
    main() 