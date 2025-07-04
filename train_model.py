#!/usr/bin/env python3
"""
Main script to train the MNIST CNN model for adversarial demo.
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from src.data.dataset import load_mnist_data, get_data_statistics
from src.models.cnn_model import MNISTCNN, train_model, evaluate_model, save_model
import torch


def main():
    """Main training function."""
    
    print("=" * 50)
    print("MNIST CNN Training for Adversarial Demo")
    print("=" * 50)
    
    # Configuration
    BATCH_SIZE = 64
    TRAIN_SIZE = 5000  # Use subset for quick demo
    TEST_SIZE = 1000
    EPOCHS = 10
    LEARNING_RATE = 0.001
    
    # Check for GPU
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")
    
    # Load data
    print("\n1. Loading MNIST dataset...")
    train_loader, test_loader, train_dataset, test_dataset = load_mnist_data(
        batch_size=BATCH_SIZE,
        train_size=TRAIN_SIZE,
        test_size=TEST_SIZE
    )
    
    print(f"Training samples: {len(train_dataset)}")
    print(f"Test samples: {len(test_dataset)}")
    
    # Get data statistics
    print("\n2. Dataset statistics:")
    stats = get_data_statistics(test_loader)
    print(f"   Data shape: {stats['shape']}")
    print(f"   Mean: {stats['mean']:.4f}")
    print(f"   Std: {stats['std']:.4f}")
    print(f"   Min: {stats['min']:.4f}")
    print(f"   Max: {stats['max']:.4f}")
    print(f"   Number of classes: {stats['num_classes']}")
    
    # Create model
    print("\n3. Creating CNN model...")
    model = MNISTCNN(num_classes=10)
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    print(f"   Total parameters: {total_params:,}")
    print(f"   Trainable parameters: {trainable_params:,}")
    
    # Train model
    print(f"\n4. Training model for {EPOCHS} epochs...")
    train_losses, test_accuracies = train_model(
        model=model,
        train_loader=train_loader,
        test_loader=test_loader,
        epochs=EPOCHS,
        lr=LEARNING_RATE,
        device=device
    )
    
    # Evaluate final model
    print("\n5. Evaluating final model...")
    results = evaluate_model(model, test_loader, device)
    
    print(f"   Final test accuracy: {results['overall_accuracy']:.2f}%")
    print("\n   Per-class accuracies:")
    for i, acc in enumerate(results['class_accuracies']):
        print(f"     Class {i}: {acc:.2f}%")
    
    # Save model
    print("\n6. Saving trained model...")
    model_path = 'models/mnist_cnn_model.pth'
    save_model(model, model_path)
    
    print("\n" + "=" * 50)
    print("Training completed successfully!")
    print(f"Model saved to: {model_path}")
    print("Ready for adversarial attacks and subset scanning!")
    print("=" * 50)


if __name__ == "__main__":
    main() 