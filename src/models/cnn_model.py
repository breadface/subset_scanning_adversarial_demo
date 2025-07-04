"""
Simple CNN model for MNIST classification.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
import numpy as np
import os


class MNISTCNN(nn.Module):
    """
    Simple CNN for MNIST digit classification.
    """
    
    def __init__(self, num_classes=10):
        super(MNISTCNN, self).__init__()
        
        # Convolutional layers
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        
        # Pooling layer
        self.pool = nn.MaxPool2d(2, 2)
        
        # Dropout for regularization
        self.dropout = nn.Dropout(0.25)
        
        # Fully connected layers
        self.fc1 = nn.Linear(128 * 3 * 3, 512)
        self.fc2 = nn.Linear(512, num_classes)
        
    def forward(self, x):
        # First conv block
        x = self.pool(F.relu(self.conv1(x)))  # 28x28 -> 14x14
        
        # Second conv block
        x = self.pool(F.relu(self.conv2(x)))  # 14x14 -> 7x7
        
        # Third conv block
        x = self.pool(F.relu(self.conv3(x)))  # 7x7 -> 3x3
        
        # Flatten
        x = x.view(-1, 128 * 3 * 3)
        
        # Fully connected layers
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        
        return x


def train_model(model, train_loader, test_loader, epochs=10, lr=0.001, device='cpu'):
    """
    Train the CNN model.
    
    Args:
        model: The CNN model
        train_loader: Training data loader
        test_loader: Test data loader
        epochs (int): Number of training epochs
        lr (float): Learning rate
        device (str): Device to train on ('cpu' or 'cuda')
    
    Returns:
        tuple: (train_losses, test_accuracies)
    """
    
    model = model.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    
    train_losses = []
    test_accuracies = []
    
    print(f"Training on {device}")
    print(f"Epochs: {epochs}, Learning rate: {lr}")
    
    for epoch in range(epochs):
        # Training phase
        model.train()
        running_loss = 0.0
        
        for batch_idx, (data, target) in enumerate(train_loader):
            data, target = data.to(device), target.to(device)
            
            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
            
            if batch_idx % 100 == 0:
                print(f'Epoch {epoch+1}/{epochs}, Batch {batch_idx}/{len(train_loader)}, '
                      f'Loss: {loss.item():.4f}')
        
        # Calculate average training loss
        avg_train_loss = running_loss / len(train_loader)
        train_losses.append(avg_train_loss)
        
        # Evaluation phase
        model.eval()
        correct = 0
        total = 0
        
        with torch.no_grad():
            for data, target in test_loader:
                data, target = data.to(device), target.to(device)
                output = model(data)
                _, predicted = torch.max(output.data, 1)
                total += target.size(0)
                correct += (predicted == target).sum().item()
        
        accuracy = 100 * correct / total
        test_accuracies.append(accuracy)
        
        print(f'Epoch {epoch+1}/{epochs}: '
              f'Train Loss: {avg_train_loss:.4f}, '
              f'Test Accuracy: {accuracy:.2f}%')
    
    return train_losses, test_accuracies


def evaluate_model(model, test_loader, device='cpu'):
    """
    Evaluate the trained model.
    
    Args:
        model: The trained model
        test_loader: Test data loader
        device (str): Device to evaluate on
    
    Returns:
        dict: Evaluation metrics
    """
    model.eval()
    model = model.to(device)
    
    correct = 0
    total = 0
    all_predictions = []
    all_targets = []
    
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            _, predicted = torch.max(output.data, 1)
            
            total += target.size(0)
            correct += (predicted == target).sum().item()
            
            all_predictions.extend(predicted.cpu().numpy())
            all_targets.extend(target.cpu().numpy())
    
    accuracy = 100 * correct / total
    
    # Calculate per-class accuracy
    class_correct = [0] * 10
    class_total = [0] * 10
    
    for i in range(len(all_targets)):
        class_total[all_targets[i]] += 1
        if all_predictions[i] == all_targets[i]:
            class_correct[all_targets[i]] += 1
    
    class_accuracies = [100 * class_correct[i] / class_total[i] 
                       if class_total[i] > 0 else 0 
                       for i in range(10)]
    
    return {
        'overall_accuracy': accuracy,
        'class_accuracies': class_accuracies,
        'predictions': all_predictions,
        'targets': all_targets
    }


def save_model(model, filepath):
    """
    Save the trained model.
    
    Args:
        model: The model to save
        filepath (str): Path to save the model
    """
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    torch.save(model.state_dict(), filepath)
    print(f"Model saved to {filepath}")


def load_model(model, filepath, device='cpu'):
    """
    Load a trained model.
    
    Args:
        model: The model architecture
        filepath (str): Path to the saved model
        device (str): Device to load the model on
    
    Returns:
        The loaded model
    """
    model.load_state_dict(torch.load(filepath, map_location=device))
    model = model.to(device)
    print(f"Model loaded from {filepath}")
    return model


if __name__ == "__main__":
    # Test the model
    from src.data.dataset import load_mnist_data
    
    # Load data
    train_loader, test_loader, _, _ = load_mnist_data(
        batch_size=64, train_size=2000, test_size=1000
    )
    
    # Create model
    model = MNISTCNN()
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # Train model
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    train_losses, test_accuracies = train_model(
        model, train_loader, test_loader, epochs=5, device=device
    )
    
    # Evaluate model
    results = evaluate_model(model, test_loader, device)
    print(f"Final test accuracy: {results['overall_accuracy']:.2f}%")
    
    # Save model
    save_model(model, 'models/mnist_cnn_model.pth') 