"""
FGSM (Fast Gradient Sign Method) attack implementation using ART.
"""

import torch
import torch.nn as nn
import numpy as np
from art.attacks.evasion import FastGradientMethod
from art.estimators.classification import PyTorchClassifier
import matplotlib.pyplot as plt
import os


class ARTModelWrapper:
    """
    Wrapper class to make PyTorch models compatible with ART.
    """
    
    def __init__(self, model, device='cpu'):
        """
        Initialize the wrapper.
        
        Args:
            model: PyTorch model
            device (str): Device to run on
        """
        self.model = model
        self.device = device
        self.model.to(device)
        self.model.eval()
    
    def predict(self, x):
        """
        Predict function for ART.
        
        Args:
            x (np.ndarray): Input data
            
        Returns:
            np.ndarray: Predictions
        """
        self.model.eval()
        with torch.no_grad():
            x_tensor = torch.FloatTensor(x).to(self.device)
            outputs = self.model(x_tensor)
            return outputs.cpu().numpy()


def create_art_classifier(model, device='cpu'):
    """
    Create an ART classifier from a PyTorch model.
    
    Args:
        model: PyTorch model
        device (str): Device to run on
        
    Returns:
        PyTorchClassifier: ART classifier
    """
    # Define loss function
    criterion = nn.CrossEntropyLoss()
    
    # Create ART classifier
    art_classifier = PyTorchClassifier(
        model=model,
        loss=criterion,
        input_shape=(1, 28, 28),  # MNIST shape
        nb_classes=10,
        device_type=device
    )
    
    return art_classifier


def generate_fgsm_attack(model, data, labels, eps=0.3, device='cpu'):
    """
    Generate FGSM adversarial examples.
    
    Args:
        model: PyTorch model
        data (torch.Tensor): Clean input data
        labels (torch.Tensor): True labels
        eps (float): Perturbation magnitude
        device (str): Device to run on
        
    Returns:
        tuple: (adversarial_examples, predictions, success_rate)
    """
    
    # Create ART classifier
    art_classifier = create_art_classifier(model, device)
    
    # Convert data to numpy
    if isinstance(data, torch.Tensor):
        data_np = data.cpu().numpy()
    else:
        data_np = data
    
    if isinstance(labels, torch.Tensor):
        labels_np = labels.cpu().numpy()
    else:
        labels_np = labels
    
    # Initialize FGSM attack
    fgsm_attack = FastGradientMethod(
        estimator=art_classifier,
        eps=eps,
        eps_step=eps/10,
        targeted=False,
        num_random_init=0,
        batch_size=32,
        minimal=False
    )
    
    # Generate adversarial examples
    print(f"Generating FGSM adversarial examples with eps={eps}...")
    adversarial_examples = fgsm_attack.generate(x=data_np, y=labels_np)
    
    # Get predictions on adversarial examples
    predictions = art_classifier.predict(adversarial_examples)
    predicted_labels = np.argmax(predictions, axis=1)
    
    # Calculate success rate (misclassification rate)
    success_rate = np.mean(predicted_labels != labels_np)
    
    print(f"FGSM attack completed:")
    print(f"  - Perturbation magnitude (eps): {eps}")
    print(f"  - Success rate: {success_rate:.2%}")
    print(f"  - Number of samples: {len(data_np)}")
    
    return adversarial_examples, predicted_labels, success_rate


def evaluate_attack_effectiveness(model, clean_data, clean_labels, 
                                adversarial_data, adversarial_labels, device='cpu'):
    """
    Evaluate the effectiveness of the attack.
    
    Args:
        model: PyTorch model
        clean_data: Clean input data
        clean_labels: True labels
        adversarial_data: Adversarial examples
        adversarial_labels: Predictions on adversarial examples
        device (str): Device to run on
        
    Returns:
        dict: Attack evaluation metrics
    """
    
    model.eval()
    model = model.to(device)
    
    # Evaluate on clean data
    with torch.no_grad():
        clean_tensor = torch.FloatTensor(clean_data).to(device)
        clean_outputs = model(clean_tensor)
        clean_pred = torch.argmax(clean_outputs, dim=1).cpu().numpy()
        clean_accuracy = np.mean(clean_pred == clean_labels)
    
    # Evaluate on adversarial data
    with torch.no_grad():
        adv_tensor = torch.FloatTensor(adversarial_data).to(device)
        adv_outputs = model(adv_tensor)
        adv_pred = torch.argmax(adv_outputs, dim=1).cpu().numpy()
        adv_accuracy = np.mean(adv_pred == clean_labels)  # Compare with true labels
    
    # Calculate perturbation statistics
    perturbations = adversarial_data - clean_data
    l2_norm = np.linalg.norm(perturbations.reshape(len(perturbations), -1), axis=1)
    linf_norm = np.max(np.abs(perturbations), axis=(1, 2, 3))
    
    results = {
        'clean_accuracy': clean_accuracy,
        'adversarial_accuracy': adv_accuracy,
        'attack_success_rate': 1 - adv_accuracy,
        'mean_l2_norm': np.mean(l2_norm),
        'mean_linf_norm': np.mean(linf_norm),
        'max_linf_norm': np.max(linf_norm),
        'perturbation_stats': {
            'l2_norm': l2_norm,
            'linf_norm': linf_norm
        }
    }
    
    return results


def visualize_attack_results(clean_data, adversarial_data, clean_labels, 
                           adversarial_labels, true_labels, num_samples=5):
    """
    Visualize the attack results.
    
    Args:
        clean_data: Clean input data
        adversarial_data: Adversarial examples
        clean_labels: Predictions on clean data
        adversarial_labels: Predictions on adversarial data
        true_labels: True labels
        num_samples (int): Number of samples to visualize
    """
    
    fig, axes = plt.subplots(3, num_samples, figsize=(15, 9))
    
    for i in range(num_samples):
        # Clean image
        axes[0, i].imshow(clean_data[i].squeeze(), cmap='gray')
        axes[0, i].set_title(f'Clean\nTrue: {true_labels[i]}\nPred: {clean_labels[i]}')
        axes[0, i].axis('off')
        
        # Adversarial image
        axes[1, i].imshow(adversarial_data[i].squeeze(), cmap='gray')
        axes[1, i].set_title(f'Adversarial\nTrue: {true_labels[i]}\nPred: {adversarial_labels[i]}')
        axes[1, i].axis('off')
        
        # Perturbation
        perturbation = adversarial_data[i] - clean_data[i]
        axes[2, i].imshow(perturbation.squeeze(), cmap='RdBu', vmin=-0.5, vmax=0.5)
        axes[2, i].set_title(f'Perturbation\nL∞: {np.max(np.abs(perturbation)):.3f}')
        axes[2, i].axis('off')
    
    plt.tight_layout()
    plt.savefig('attack_visualization.png', dpi=150, bbox_inches='tight')
    plt.show()


def experiment_with_eps_values(model, data, labels, eps_values, device='cpu'):
    """
    Experiment with different epsilon values for FGSM attack.
    
    Args:
        model: PyTorch model
        data: Input data
        labels: True labels
        eps_values (list): List of epsilon values to try
        device (str): Device to run on
        
    Returns:
        dict: Results for each epsilon value
    """
    
    results = {}
    
    for eps in eps_values:
        print(f"\nTesting eps = {eps}")
        
        # Generate adversarial examples
        adv_examples, adv_preds, success_rate = generate_fgsm_attack(
            model, data, labels, eps=eps, device=device
        )
        
        # Evaluate effectiveness
        eval_results = evaluate_attack_effectiveness(
            model, data, labels, adv_examples, adv_preds, device
        )
        
        results[eps] = {
            'success_rate': success_rate,
            'adversarial_accuracy': eval_results['adversarial_accuracy'],
            'mean_l2_norm': eval_results['mean_l2_norm'],
            'mean_linf_norm': eval_results['mean_linf_norm'],
            'adversarial_examples': adv_examples,
            'adversarial_predictions': adv_preds
        }
        
        print(f"  Success rate: {success_rate:.2%}")
        print(f"  Adversarial accuracy: {eval_results['adversarial_accuracy']:.2%}")
        print(f"  Mean L∞ norm: {eval_results['mean_linf_norm']:.4f}")
    
    return results


if __name__ == "__main__":
    # Test the FGSM attack
    from src.data.dataset import load_mnist_data
    from src.models.cnn_model import MNISTCNN, load_model
    
    # Load data
    _, test_loader, _, _ = load_mnist_data(
        batch_size=100, test_size=100
    )
    
    # Get a batch of test data
    test_data, test_labels = next(iter(test_loader))
    
    # Load or create model
    model = MNISTCNN()
    model_path = 'models/mnist_cnn_model.pth'
    
    if os.path.exists(model_path):
        model = load_model(model, model_path)
    else:
        print("No trained model found. Please run train_model.py first.")
        exit(1)
    
    # Test FGSM attack
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    # Experiment with different epsilon values
    eps_values = [0.1, 0.2, 0.3, 0.4, 0.5]
    results = experiment_with_eps_values(
        model, test_data, test_labels, eps_values, device
    )
    
    # Print summary
    print("\n" + "="*50)
    print("FGSM Attack Results Summary")
    print("="*50)
    for eps, result in results.items():
        print(f"eps={eps}: Success rate={result['success_rate']:.2%}, "
              f"L∞ norm={result['mean_linf_norm']:.4f}") 