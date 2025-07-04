"""
Data preparation for subset scanning with mixed clean/adversarial samples.
"""

import numpy as np
import torch
from torch.utils.data import DataLoader
import os


def create_mixed_dataset(clean_data, clean_labels, adversarial_data, adversarial_labels, 
                        contamination_rate=0.1, shuffle=True, seed=42):
    """
    Create a mixed dataset with clean and adversarial samples.
    
    Args:
        clean_data: Clean test data
        clean_labels: Labels for clean data
        adversarial_data: Adversarial test data  
        adversarial_labels: Labels for adversarial data
        contamination_rate (float): Fraction of adversarial samples to include
        shuffle (bool): Whether to shuffle the final dataset
        seed (int): Random seed for reproducibility
        
    Returns:
        tuple: (mixed_data, mixed_labels, ground_truth, sample_info)
    """
    
    np.random.seed(seed)
    
    # Determine number of adversarial samples to include
    n_clean = len(clean_data)
    n_adversarial = len(adversarial_data)
    n_adversarial_to_include = int(n_clean * contamination_rate)
    
    print(f"Creating mixed dataset:")
    print(f"  Clean samples: {n_clean}")
    print(f"  Available adversarial samples: {n_adversarial}")
    print(f"  Adversarial samples to include: {n_adversarial_to_include}")
    print(f"  Contamination rate: {contamination_rate:.1%}")
    
    # Randomly select adversarial samples
    if n_adversarial_to_include > n_adversarial:
        print(f"Warning: Requested {n_adversarial_to_include} adversarial samples, "
              f"but only {n_adversarial} available. Using all available.")
        n_adversarial_to_include = n_adversarial
    
    adv_indices = np.random.choice(n_adversarial, n_adversarial_to_include, replace=False)
    selected_adversarial_data = adversarial_data[adv_indices]
    selected_adversarial_labels = adversarial_labels[adv_indices]
    
    # Combine datasets
    mixed_data = np.concatenate([clean_data, selected_adversarial_data], axis=0)
    mixed_labels = np.concatenate([clean_labels, selected_adversarial_labels], axis=0)
    
    # Create ground truth labels (0 for clean, 1 for adversarial)
    ground_truth = np.concatenate([
        np.zeros(n_clean, dtype=int),  # Clean samples
        np.ones(n_adversarial_to_include, dtype=int)  # Adversarial samples
    ])
    
    # Create sample info for tracking
    sample_info = {
        'clean_indices': np.arange(n_clean),
        'adversarial_indices': np.arange(n_clean, n_clean + n_adversarial_to_include),
        'n_clean': n_clean,
        'n_adversarial': n_adversarial_to_include,
        'contamination_rate': contamination_rate
    }
    
    # Shuffle if requested
    if shuffle:
        shuffle_indices = np.random.permutation(len(mixed_data))
        mixed_data = mixed_data[shuffle_indices]
        mixed_labels = mixed_labels[shuffle_indices]
        ground_truth = ground_truth[shuffle_indices]
        
        # Update sample info with new indices
        sample_info['clean_indices'] = np.where(ground_truth == 0)[0]
        sample_info['adversarial_indices'] = np.where(ground_truth == 1)[0]
    
    print(f"Final mixed dataset: {len(mixed_data)} samples")
    print(f"  Clean: {np.sum(ground_truth == 0)}")
    print(f"  Adversarial: {np.sum(ground_truth == 1)}")
    
    return mixed_data, mixed_labels, ground_truth, sample_info


def create_multiple_contamination_levels(clean_data, clean_labels, adversarial_data, 
                                       adversarial_labels, contamination_rates=[0.05, 0.1, 0.15, 0.2]):
    """
    Create multiple datasets with different contamination levels.
    
    Args:
        clean_data: Clean test data
        clean_labels: Labels for clean data
        adversarial_data: Adversarial test data
        adversarial_labels: Labels for adversarial data
        contamination_rates (list): List of contamination rates to test
        
    Returns:
        dict: Dictionary with datasets for each contamination rate
    """
    
    datasets = {}
    
    for rate in contamination_rates:
        print(f"\nCreating dataset with {rate:.1%} contamination...")
        
        mixed_data, mixed_labels, ground_truth, sample_info = create_mixed_dataset(
            clean_data, clean_labels, adversarial_data, adversarial_labels,
            contamination_rate=rate, shuffle=True
        )
        
        datasets[rate] = {
            'data': mixed_data,
            'labels': mixed_labels,
            'ground_truth': ground_truth,
            'sample_info': sample_info
        }
    
    return datasets


def extract_features_for_scanning(data, model, device='cpu'):
    """
    Extract features from data for subset scanning.
    
    Args:
        data: Input data (numpy array)
        model: Trained PyTorch model
        device (str): Device to run on
        
    Returns:
        np.ndarray: Extracted features
    """
    
    model.eval()
    model = model.to(device)
    
    features = []
    batch_size = 32
    
    with torch.no_grad():
        for i in range(0, len(data), batch_size):
            batch = data[i:i+batch_size]
            batch_tensor = torch.FloatTensor(batch).to(device)
            
            # Get model outputs (logits)
            outputs = model(batch_tensor)
            batch_features = outputs.cpu().numpy()
            
            features.append(batch_features)
    
    features = np.concatenate(features, axis=0)
    
    print(f"Extracted features shape: {features.shape}")
    return features


def prepare_subset_scanning_data(clean_data, clean_labels, adversarial_data, adversarial_labels,
                                model, contamination_rate=0.1, device='cpu'):
    """
    Prepare complete dataset for subset scanning.
    
    Args:
        clean_data: Clean test data
        clean_labels: Labels for clean data
        adversarial_data: Adversarial test data
        adversarial_labels: Labels for adversarial data
        model: Trained model for feature extraction
        contamination_rate (float): Fraction of adversarial samples
        device (str): Device to run on
        
    Returns:
        dict: Complete dataset for subset scanning
    """
    
    print("Preparing data for subset scanning...")
    
    # Create mixed dataset
    mixed_data, mixed_labels, ground_truth, sample_info = create_mixed_dataset(
        clean_data, clean_labels, adversarial_data, adversarial_labels,
        contamination_rate=contamination_rate
    )
    
    # Extract features
    features = extract_features_for_scanning(mixed_data, model, device)
    
    # Create final dataset
    scanning_data = {
        'raw_data': mixed_data,
        'features': features,
        'labels': mixed_labels,
        'ground_truth': ground_truth,
        'sample_info': sample_info,
        'contamination_rate': contamination_rate,
        'n_samples': len(mixed_data),
        'n_features': features.shape[1]
    }
    
    print(f"Subset scanning data prepared:")
    print(f"  Samples: {scanning_data['n_samples']}")
    print(f"  Features: {scanning_data['n_features']}")
    print(f"  Contamination: {contamination_rate:.1%}")
    
    return scanning_data


def save_scanning_data(data, filepath):
    """
    Save subset scanning data to file.
    
    Args:
        data (dict): Subset scanning data
        filepath (str): Path to save data
    """
    
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    np.savez_compressed(filepath, **data)
    print(f"Scanning data saved to {filepath}")


def load_scanning_data(filepath):
    """
    Load subset scanning data from file.
    
    Args:
        filepath (str): Path to load data from
        
    Returns:
        dict: Subset scanning data
    """
    
    data = np.load(filepath, allow_pickle=True)
    scanning_data = {}
    
    for key in data.keys():
        if key == 'sample_info':
            scanning_data[key] = data[key].item()
        else:
            scanning_data[key] = data[key]
    
    print(f"Scanning data loaded from {filepath}")
    return scanning_data


if __name__ == "__main__":
    # Test the data preparation
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
    
    # Prepare subset scanning data
    scanning_data = prepare_subset_scanning_data(
        test_data.numpy(), test_labels.numpy(),
        adv_data, adv_labels,
        model, contamination_rate=0.1, device=device
    )
    
    print("Data preparation test completed!") 