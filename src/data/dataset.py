"""
Dataset loading and preprocessing utilities for MNIST.
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import numpy as np


def load_mnist_data(batch_size=64, train_size=None, test_size=None):
    """
    Load and preprocess MNIST dataset.
    
    Args:
        batch_size (int): Batch size for data loaders
        train_size (int, optional): Number of training samples to use
        test_size (int, optional): Number of test samples to use
    
    Returns:
        tuple: (train_loader, test_loader, train_dataset, test_dataset)
    """
    
    # Define transformations
    transform = transforms.Compose([
        transforms.ToTensor(),  # Convert to tensor and normalize to [0, 1]
        transforms.Normalize((0.1307,), (0.3081,))  # MNIST mean and std
    ])
    
    # Load datasets
    train_dataset = datasets.MNIST(
        root='./data', 
        train=True, 
        download=True, 
        transform=transform
    )
    
    test_dataset = datasets.MNIST(
        root='./data', 
        train=False, 
        download=True, 
        transform=transform
    )
    
    # Limit dataset size if specified (useful for quick demos)
    if train_size is not None:
        train_dataset = torch.utils.data.Subset(train_dataset, range(train_size))
    if test_size is not None:
        test_dataset = torch.utils.data.Subset(test_dataset, range(test_size))
    
    # Create data loaders
    train_loader = DataLoader(
        train_dataset, 
        batch_size=batch_size, 
        shuffle=True,
        num_workers=0  # Set to 0 for compatibility
    )
    
    test_loader = DataLoader(
        test_dataset, 
        batch_size=batch_size, 
        shuffle=False,
        num_workers=0
    )
    
    return train_loader, test_loader, train_dataset, test_dataset


def get_mnist_sample_data(n_samples=1000):
    """
    Get a subset of MNIST data as numpy arrays for subset scanning.
    
    Args:
        n_samples (int): Number of samples to return
    
    Returns:
        tuple: (X, y) where X is data and y is labels
    """
    # Load data without normalization for subset scanning
    transform = transforms.Compose([
        transforms.ToTensor(),
    ])
    
    dataset = datasets.MNIST(
        root='./data', 
        train=False, 
        download=True, 
        transform=transform
    )
    
    # Get subset
    subset = torch.utils.data.Subset(dataset, range(n_samples))
    loader = DataLoader(subset, batch_size=n_samples, shuffle=False)
    
    # Convert to numpy
    data, labels = next(iter(loader))
    X = data.numpy().reshape(n_samples, -1)  # Flatten to 2D
    y = labels.numpy()
    
    return X, y


def preprocess_for_attack(data, normalize=True):
    """
    Preprocess data for adversarial attacks.
    
    Args:
        data (torch.Tensor): Input data
        normalize (bool): Whether to normalize data
    
    Returns:
        torch.Tensor: Preprocessed data
    """
    if normalize:
        # Normalize to [0, 1] range for attacks
        data = (data - data.min()) / (data.max() - data.min())
    
    return data


def get_data_statistics(loader):
    """
    Get basic statistics about the dataset.
    
    Args:
        loader (DataLoader): Data loader
    
    Returns:
        dict: Statistics including mean, std, min, max
    """
    all_data = []
    all_labels = []
    
    for data, labels in loader:
        all_data.append(data)
        all_labels.append(labels)
    
    all_data = torch.cat(all_data, dim=0)
    all_labels = torch.cat(all_labels, dim=0)
    
    stats = {
        'mean': all_data.mean().item(),
        'std': all_data.std().item(),
        'min': all_data.min().item(),
        'max': all_data.max().item(),
        'shape': all_data.shape,
        'num_classes': len(torch.unique(all_labels)),
        'class_distribution': torch.bincount(all_labels).tolist()
    }
    
    return stats


if __name__ == "__main__":
    # Test the data loading
    print("Loading MNIST data...")
    train_loader, test_loader, train_dataset, test_dataset = load_mnist_data(
        batch_size=32, train_size=1000, test_size=500
    )
    
    print(f"Training samples: {len(train_dataset)}")
    print(f"Test samples: {len(test_dataset)}")
    
    # Get statistics
    stats = get_data_statistics(test_loader)
    print(f"Data statistics: {stats}")
    
    # Test sample data for subset scanning
    X, y = get_mnist_sample_data(n_samples=100)
    print(f"Sample data shape: {X.shape}")
    print(f"Sample labels shape: {y.shape}")
    print(f"Unique labels: {np.unique(y)}") 