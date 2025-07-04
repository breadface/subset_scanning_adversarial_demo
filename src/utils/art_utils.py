"""
ART utility functions for classifier creation and feature extraction.
"""

import torch
import torch.nn as nn
from art.estimators.classification import PyTorchClassifier


def create_art_classifier(model, device='cpu', input_shape=(1, 28, 28), nb_classes=10):
    """
    Create an ART PyTorchClassifier from a PyTorch model.
    """
    criterion = nn.CrossEntropyLoss()
    art_classifier = PyTorchClassifier(
        model=model,
        loss=criterion,
        input_shape=input_shape,
        nb_classes=nb_classes,
        device_type=device
    )
    return art_classifier


def extract_features_from_layer(model, data, layer_name='fc2'):
    """
    Extract features from a specific layer of the model.
    Args:
        model: PyTorch model
        data: Input data (torch.Tensor)
        layer_name (str): Name of the layer to extract features from
    Returns:
        np.ndarray: Extracted features
    """
    model.eval()
    with torch.no_grad():
        x = data
        if layer_name == 'fc2':
            x = model.conv1(x)
            x = torch.relu(x)
            x = torch.max_pool2d(x, 2, 2)
            x = model.conv2(x)
            x = torch.relu(x)
            x = torch.max_pool2d(x, 2, 2)
            x = model.conv3(x)
            x = torch.relu(x)
            x = torch.max_pool2d(x, 2, 2)
            x = x.view(x.size(0), -1)
            x = model.fc1(x)
            x = torch.relu(x)
            x = model.fc2(x)
        # Add more layers as needed
        x = x.view(x.size(0), -1)
        features = x.cpu().numpy()
    return features 