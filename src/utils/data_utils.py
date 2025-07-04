"""
Data utility functions for mixing clean and adversarial samples for scanning.
"""

import numpy as np

def prepare_mixed_dataset_for_scanning(clean_data, adversarial_data, contamination_rate=0.1, shuffle=True, seed=42):
    """
    Prepare mixed dataset for subset scanning evaluation.
    Args:
        clean_data: Clean test data
        adversarial_data: Adversarial test data
        contamination_rate (float): Fraction of adversarial samples (e.g., 0.1 for 10%)
        shuffle (bool): Whether to shuffle the final dataset
        seed (int): Random seed for reproducibility
    Returns:
        tuple: (x_combined, y_true_anomaly, sample_info)
    """
    np.random.seed(seed)
    num_clean = len(clean_data)
    num_adv = int(num_clean * contamination_rate)
    if num_adv > len(adversarial_data):
        num_adv = len(adversarial_data)
    x_combined = np.concatenate((clean_data[:num_clean], adversarial_data[:num_adv]))
    y_true_anomaly = np.array([0] * num_clean + [1] * num_adv)
    sample_info = {
        'clean_indices': np.arange(num_clean),
        'adversarial_indices': np.arange(num_clean, num_clean + num_adv),
        'n_clean': num_clean,
        'n_adversarial': num_adv,
        'contamination_rate': contamination_rate,
        'total_samples': num_clean + num_adv
    }
    if shuffle:
        shuffle_indices = np.random.permutation(len(x_combined))
        x_combined = x_combined[shuffle_indices]
        y_true_anomaly = y_true_anomaly[shuffle_indices]
        sample_info['clean_indices'] = np.where(y_true_anomaly == 0)[0]
        sample_info['adversarial_indices'] = np.where(y_true_anomaly == 1)[0]
    return x_combined, y_true_anomaly, sample_info 