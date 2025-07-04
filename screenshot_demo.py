#!/usr/bin/env python3
"""
Screenshot Demo for Subset Scanning Visualizations
==================================================

This script demonstrates how to programmatically capture screenshots
of the interactive visualizations for documentation or presentation purposes.

Usage:
    python screenshot_demo.py

This will:
1. Create the visualizations
2. Save them as high-resolution PNG files
3. Demonstrate how to capture screenshots programmatically
"""

import numpy as np
import matplotlib.pyplot as plt
import os
import sys
from PIL import Image
import io

# Add src to path for imports
sys.path.append('src')

from visualization.qualitative_analysis import (
    visualize_original_vs_adversarial,
    visualize_detection_scores_distribution,
    visualize_detected_anomalies,
    visualize_perturbation_analysis
)


def create_dummy_data():
    """
    Create dummy data for demonstration purposes.
    """
    print("Creating dummy data for screenshot demonstration...")
    
    # Generate dummy data
    np.random.seed(42)
    n_samples = 1000
    
    # Dummy images (simulate MNIST-like data)
    clean_data = np.random.rand(n_samples, 1, 28, 28)
    clean_data = clean_data * 0.3 + 0.1  # Reduce contrast
    
    # Create adversarial data with subtle perturbations
    adversarial_data = clean_data + np.random.normal(0, 0.05, clean_data.shape)
    adversarial_data = np.clip(adversarial_data, 0, 1)  # Clip to valid range
    
    # Dummy labels and predictions
    true_labels = np.random.randint(0, 10, n_samples)
    adv_predictions = np.random.randint(0, 10, n_samples)
    
    # Create realistic detection scores (adversarial should have higher scores)
    clean_scores = np.random.normal(0, 0.5, n_samples // 2)
    adv_scores = np.random.normal(2, 0.8, n_samples // 2)
    scores = np.concatenate([clean_scores, adv_scores])
    
    # Create true labels for detection (0=clean, 1=adversarial)
    y_true = np.concatenate([np.zeros(n_samples // 2), np.ones(n_samples // 2)])
    
    return {
        'clean_data': clean_data,
        'adversarial_data': adversarial_data,
        'true_labels': true_labels,
        'adv_predictions': adv_predictions,
        'scores': scores,
        'y_true': y_true
    }


def capture_plot_as_image(fig, filename, dpi=300):
    """
    Capture a matplotlib figure as a high-resolution image.
    
    Args:
        fig: Matplotlib figure object
        filename: Output filename
        dpi: Resolution (dots per inch)
    """
    # Save the figure to a bytes buffer
    buf = io.BytesIO()
    fig.savefig(buf, format='png', dpi=dpi, bbox_inches='tight')
    buf.seek(0)
    
    # Open with PIL and save
    img = Image.open(buf)
    img.save(filename, 'PNG', dpi=(dpi, dpi))
    buf.close()
    
    print(f"✓ Screenshot saved: {filename}")


def create_screenshots():
    """
    Create high-resolution screenshots of all visualizations.
    """
    print("=" * 80)
    print("📸 SCREENSHOT DEMONSTRATION")
    print("=" * 80)
    print("Creating high-resolution screenshots of all visualizations...")
    print()
    
    # Create output directory
    output_dir = 'screenshots'
    os.makedirs(output_dir, exist_ok=True)
    
    # Load data
    data = create_dummy_data()
    
    # 1. Original vs Adversarial
    print("1️⃣  Creating Original vs Adversarial screenshot...")
    fig, axes = plt.subplots(4, 8, figsize=(20, 12))
    
    # Select random samples
    indices = np.random.choice(len(data['clean_data']), 8, replace=False)
    
    for i, idx in enumerate(indices):
        # Original image
        axes[0, i].imshow(data['clean_data'][idx].squeeze(), cmap='gray')
        axes[0, i].set_title(f'Original\nTrue: {data["true_labels"][idx]}', fontsize=10)
        axes[0, i].axis('off')
        
        # Adversarial image
        axes[1, i].imshow(data['adversarial_data'][idx].squeeze(), cmap='gray')
        axes[1, i].set_title(f'Adversarial\nPred: {data["adv_predictions"][idx]}', fontsize=10)
        axes[1, i].axis('off')
        
        # Perturbation
        perturbation = data['adversarial_data'][idx] - data['clean_data'][idx]
        im = axes[2, i].imshow(perturbation.squeeze(), cmap='RdBu', vmin=-0.5, vmax=0.5)
        axes[2, i].set_title(f'Perturbation\nL∞: {np.max(np.abs(perturbation)):.3f}', fontsize=10)
        axes[2, i].axis('off')
        
        # Magnified perturbation
        magnified = perturbation * 5
        axes[3, i].imshow(magnified.squeeze(), cmap='RdBu', vmin=-0.5, vmax=0.5)
        axes[3, i].set_title(f'Magnified (×5)\nL2: {np.linalg.norm(perturbation):.3f}', fontsize=10)
        axes[3, i].axis('off')
    
    plt.tight_layout()
    plt.suptitle('Original vs Adversarial Images: Subtle Perturbations', y=0.98, fontsize=16)
    
    # Add colorbar
    cbar_ax = fig.add_axes([0.92, 0.15, 0.02, 0.3])
    cbar = plt.colorbar(im, cax=cbar_ax)
    cbar.set_label('Perturbation Magnitude', rotation=270, labelpad=15)
    
    # Capture screenshot
    capture_plot_as_image(fig, os.path.join(output_dir, 'original_vs_adversarial_screenshot.png'))
    plt.close()
    
    # 2. Detection Scores Distribution
    print("\n2️⃣  Creating Detection Scores Distribution screenshot...")
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    
    # Separate scores
    clean_scores = data['scores'][data['y_true'] == 0]
    adv_scores = data['scores'][data['y_true'] == 1]
    
    # Histogram comparison
    axes[0, 0].hist(clean_scores, bins=30, alpha=0.7, label='Clean', density=True, color='blue')
    axes[0, 0].hist(adv_scores, bins=30, alpha=0.7, label='Adversarial', density=True, color='red')
    axes[0, 0].set_xlabel('Detection Score')
    axes[0, 0].set_ylabel('Density')
    axes[0, 0].set_title('Score Distribution: Clean vs Adversarial')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    
    # Box plot
    data_to_plot = [clean_scores, adv_scores]
    axes[0, 1].boxplot(data_to_plot, labels=['Clean', 'Adversarial'])
    axes[0, 1].set_ylabel('Detection Score')
    axes[0, 1].set_title('Score Distribution: Box Plot')
    axes[0, 1].grid(True, alpha=0.3)
    
    # Violin plot
    axes[1, 0].violinplot(data_to_plot, positions=[1, 2])
    axes[1, 0].set_xticks([1, 2])
    axes[1, 0].set_xticklabels(['Clean', 'Adversarial'])
    axes[1, 0].set_ylabel('Detection Score')
    axes[1, 0].set_title('Score Distribution: Violin Plot')
    axes[1, 0].grid(True, alpha=0.3)
    
    # ROC curve
    from sklearn.metrics import roc_curve
    fpr, tpr, thresholds = roc_curve(data['y_true'], data['scores'])
    auc_score = np.trapz(tpr, fpr)
    axes[1, 1].plot(fpr, tpr, label=f'ROC Curve (AUC = {auc_score:.3f})')
    axes[1, 1].plot([0, 1], [0, 1], 'k--', alpha=0.5)
    axes[1, 1].set_xlabel('False Positive Rate')
    axes[1, 1].set_ylabel('True Positive Rate')
    axes[1, 1].set_title('ROC Curve')
    axes[1, 1].legend()
    axes[1, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.suptitle('Detection Score Analysis', y=0.98, fontsize=16)
    
    # Capture screenshot
    capture_plot_as_image(fig, os.path.join(output_dir, 'detection_scores_screenshot.png'))
    plt.close()
    
    # 3. Perturbation Analysis
    print("\n3️⃣  Creating Perturbation Analysis screenshot...")
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    
    # Calculate perturbations
    perturbations = data['adversarial_data'] - data['clean_data']
    l2_norms = np.linalg.norm(perturbations.reshape(len(perturbations), -1), axis=1)
    linf_norms = np.max(np.abs(perturbations), axis=(1, 2, 3))
    
    # L2 norm distribution
    axes[0, 0].hist(l2_norms, bins=30, alpha=0.7, color='purple')
    axes[0, 0].set_xlabel('L2 Norm')
    axes[0, 0].set_ylabel('Count')
    axes[0, 0].set_title('Perturbation L2 Norm Distribution')
    axes[0, 0].grid(True, alpha=0.3)
    
    # L∞ norm distribution
    axes[0, 1].hist(linf_norms, bins=30, alpha=0.7, color='orange')
    axes[0, 1].set_xlabel('L∞ Norm')
    axes[0, 1].set_ylabel('Count')
    axes[0, 1].set_title('Perturbation L∞ Norm Distribution')
    axes[0, 1].grid(True, alpha=0.3)
    
    # L2 vs L∞ scatter
    axes[0, 2].scatter(l2_norms, linf_norms, alpha=0.6, s=20)
    axes[0, 2].set_xlabel('L2 Norm')
    axes[0, 2].set_ylabel('L∞ Norm')
    axes[0, 2].set_title('L2 vs L∞ Norm Relationship')
    axes[0, 2].grid(True, alpha=0.3)
    
    # Average perturbation pattern
    avg_perturbation = np.mean(perturbations, axis=0)
    im1 = axes[1, 0].imshow(avg_perturbation.squeeze(), cmap='RdBu', vmin=-0.1, vmax=0.1)
    axes[1, 0].set_title('Average Perturbation Pattern')
    axes[1, 0].axis('off')
    
    # Perturbation variance
    var_perturbation = np.var(perturbations, axis=0)
    im2 = axes[1, 1].imshow(var_perturbation.squeeze(), cmap='viridis')
    axes[1, 1].set_title('Perturbation Variance')
    axes[1, 1].axis('off')
    
    # Perturbation magnitude by pixel position
    pixel_perturbations = perturbations.reshape(len(perturbations), -1)
    pixel_means = np.mean(np.abs(pixel_perturbations), axis=0)
    pixel_means_2d = pixel_means.reshape(28, 28)
    im3 = axes[1, 2].imshow(pixel_means_2d, cmap='hot')
    axes[1, 2].set_title('Average Perturbation Magnitude by Pixel')
    axes[1, 2].axis('off')
    
    # Add colorbars
    plt.colorbar(im1, ax=axes[1, 0], fraction=0.046, pad=0.04)
    plt.colorbar(im2, ax=axes[1, 1], fraction=0.046, pad=0.04)
    plt.colorbar(im3, ax=axes[1, 2], fraction=0.046, pad=0.04)
    
    plt.tight_layout()
    plt.suptitle('Perturbation Analysis', y=0.98, fontsize=16)
    
    # Capture screenshot
    capture_plot_as_image(fig, os.path.join(output_dir, 'perturbation_analysis_screenshot.png'))
    plt.close()
    
    print("\n" + "=" * 80)
    print("✅ SCREENSHOT DEMONSTRATION COMPLETED!")
    print("=" * 80)
    print("High-resolution screenshots saved to:")
    print(f"   📁 {output_dir}/")
    print("   • original_vs_adversarial_screenshot.png")
    print("   • detection_scores_screenshot.png")
    print("   • perturbation_analysis_screenshot.png")
    print()
    print("These screenshots can be used for:")
    print("   • Documentation and reports")
    print("   • Presentations and slides")
    print("   • Publications and papers")
    print("   • Social media and outreach")
    print("=" * 80)


if __name__ == "__main__":
    create_screenshots() 