#!/usr/bin/env python3
"""
Prepare data for subset scanning by mixing clean and adversarial samples.
"""

import sys
import os
sys.path.append('src')

from src.data.dataset import load_mnist_data
from src.models.cnn_model import MNISTCNN, load_model
from src.attacks.fgsm_attack import generate_fgsm_attack
from src.data.subset_data import prepare_subset_scanning_data, save_scanning_data
import torch


def main():
    print("Preparing data for subset scanning...")
    
    # Load test data
    print("Loading test data...")
    _, test_loader, _, _ = load_mnist_data(batch_size=100, test_size=500)
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
    
    # Prepare subset scanning data with different contamination rates
    contamination_rates = [0.05, 0.1, 0.15, 0.2]
    
    for rate in contamination_rates:
        print(f"\nPreparing data with {rate:.1%} contamination...")
        
        scanning_data = prepare_subset_scanning_data(
            test_data.numpy(), test_labels.numpy(),
            adv_data, adv_labels,
            model, contamination_rate=rate, device=device
        )
        
        # Save data
        filename = f"scanning_data_contamination_{int(rate*100)}.npz"
        save_scanning_data(scanning_data, f"data/{filename}")
    
    print("\nData preparation completed!")


if __name__ == "__main__":
    main() 