"""
Qualitative visualization for subset scanning adversarial detection.
Shows original vs adversarial images, perturbation patterns, and detected anomalies.
"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import torch
from sklearn.metrics import roc_curve
import os
import time


def visualize_original_vs_adversarial(clean_data, adversarial_data, true_labels, 
                                    adv_predictions, num_samples=8, save_path='original_vs_adversarial.png',
                                    show_interactive=True):
    """
    Visualize pairs of original and adversarial images to show subtle differences.
    
    Args:
        clean_data: Original clean images
        adversarial_data: FGSM perturbed images
        true_labels: True labels
        adv_predictions: Model predictions on adversarial examples
        num_samples: Number of image pairs to show
        save_path: Path to save the visualization
        show_interactive: Whether to display interactively
    """
    
    fig, axes = plt.subplots(4, num_samples, figsize=(20, 12))
    
    # Select random samples
    indices = np.random.choice(len(clean_data), num_samples, replace=False)
    
    for i, idx in enumerate(indices):
        # Original image
        axes[0, i].imshow(clean_data[idx].squeeze(), cmap='gray')
        axes[0, i].set_title(f'Original\nTrue: {true_labels[idx]}', fontsize=10)
        axes[0, i].axis('off')
        
        # Adversarial image
        axes[1, i].imshow(adversarial_data[idx].squeeze(), cmap='gray')
        axes[1, i].set_title(f'Adversarial\nPred: {adv_predictions[idx]}', fontsize=10)
        axes[1, i].axis('off')
        
        # Perturbation (difference)
        perturbation = adversarial_data[idx] - clean_data[idx]
        im = axes[2, i].imshow(perturbation.squeeze(), cmap='RdBu', vmin=-0.5, vmax=0.5)
        axes[2, i].set_title(f'Perturbation\nL∞: {np.max(np.abs(perturbation)):.3f}', fontsize=10)
        axes[2, i].axis('off')
        
        # Magnified perturbation (for subtle changes)
        magnified = perturbation * 5  # Amplify for visibility
        axes[3, i].imshow(magnified.squeeze(), cmap='RdBu', vmin=-0.5, vmax=0.5)
        axes[3, i].set_title(f'Magnified (×5)\nL2: {np.linalg.norm(perturbation):.3f}', fontsize=10)
        axes[3, i].axis('off')
    
    plt.tight_layout()
    plt.suptitle('Original vs Adversarial Images: Subtle Perturbations', y=0.98, fontsize=16)
    
    # Add colorbar for perturbation
    cbar_ax = fig.add_axes([0.92, 0.15, 0.02, 0.3])
    cbar = plt.colorbar(im, cax=cbar_ax)
    cbar.set_label('Perturbation Magnitude', rotation=270, labelpad=15)
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"✓ Original vs adversarial visualization saved to {save_path}")
    
    if show_interactive:
        print("\n🖼️  Displaying Original vs Adversarial Images...")
        print("   - Row 1: Original images with true labels")
        print("   - Row 2: Adversarial images with model predictions")
        print("   - Row 3: Perturbation patterns (red=positive, blue=negative)")
        print("   - Row 4: Magnified perturbations (×5) for visibility")
        print("   - Close the window to continue...")
        plt.show()
        print("✓ Visualization displayed successfully!")


def visualize_detection_scores_distribution(scores, y_true, save_path='detection_scores_distribution.png',
                                          show_interactive=True):
    """
    Visualize the distribution of detection scores for clean vs adversarial samples.
    
    Args:
        scores: Detection scores from subset scanning
        y_true: True labels (0 for clean, 1 for adversarial)
        save_path: Path to save the visualization
        show_interactive: Whether to display interactively
    """
    
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    
    # Separate scores
    clean_scores = scores[y_true == 0]
    adv_scores = scores[y_true == 1]
    
    # 1. Histogram comparison
    axes[0, 0].hist(clean_scores, bins=30, alpha=0.7, label='Clean', density=True, color='blue')
    axes[0, 0].hist(adv_scores, bins=30, alpha=0.7, label='Adversarial', density=True, color='red')
    axes[0, 0].set_xlabel('Detection Score')
    axes[0, 0].set_ylabel('Density')
    axes[0, 0].set_title('Score Distribution: Clean vs Adversarial')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    
    # 2. Box plot
    data_to_plot = [clean_scores, adv_scores]
    axes[0, 1].boxplot(data_to_plot, labels=['Clean', 'Adversarial'])
    axes[0, 1].set_ylabel('Detection Score')
    axes[0, 1].set_title('Score Distribution: Box Plot')
    axes[0, 1].grid(True, alpha=0.3)
    
    # 3. Violin plot
    axes[1, 0].violinplot(data_to_plot, positions=[1, 2])
    axes[1, 0].set_xticks([1, 2])
    axes[1, 0].set_xticklabels(['Clean', 'Adversarial'])
    axes[1, 0].set_ylabel('Detection Score')
    axes[1, 0].set_title('Score Distribution: Violin Plot')
    axes[1, 0].grid(True, alpha=0.3)
    
    # 4. ROC curve
    fpr, tpr, thresholds = roc_curve(y_true, scores)
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
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"✓ Detection scores distribution saved to {save_path}")
    
    if show_interactive:
        print("\n📊 Displaying Detection Score Analysis...")
        print("   - Top Left: Histogram comparison of score distributions")
        print("   - Top Right: Box plot showing score ranges and outliers")
        print("   - Bottom Left: Violin plot showing density distributions")
        print("   - Bottom Right: ROC curve with AUC score")
        print("   - Close the window to continue...")
        plt.show()
        print("✓ Score analysis displayed successfully!")


def visualize_detected_anomalies(clean_data, adversarial_data, scores, y_true, 
                               threshold_percentile=95, num_samples=12, save_path='detected_anomalies.png',
                               show_interactive=True):
    """
    Visualize images that were detected as anomalous by the subset scanning detector.
    
    Args:
        clean_data: Original clean images
        adversarial_data: Adversarial images
        scores: Detection scores
        y_true: True labels
        threshold_percentile: Percentile to use as detection threshold
        num_samples: Number of samples to show
        save_path: Path to save the visualization
        show_interactive: Whether to display interactively
    """
    
    # Determine threshold
    threshold = np.percentile(scores, threshold_percentile)
    
    # Find detected anomalies (high scores)
    detected_indices = np.where(scores > threshold)[0]
    
    # Separate true positives (adversarial detected as anomalous) and false positives (clean detected as anomalous)
    true_positives = detected_indices[y_true[detected_indices] == 1]
    false_positives = detected_indices[y_true[detected_indices] == 0]
    
    print(f"Detection threshold (95th percentile): {threshold:.4f}")
    print(f"True positives (adversarial detected): {len(true_positives)}")
    print(f"False positives (clean detected): {len(false_positives)}")
    
    # Create visualization
    fig, axes = plt.subplots(4, num_samples, figsize=(20, 16))
    
    # Show true positives (correctly detected adversarial)
    num_tp = min(len(true_positives), num_samples // 2)
    if num_tp > 0:
        tp_indices = np.random.choice(true_positives, num_tp, replace=False)
        for i, idx in enumerate(tp_indices):
            # Original
            axes[0, i].imshow(clean_data[idx].squeeze(), cmap='gray')
            axes[0, i].set_title(f'Original\nScore: {scores[idx]:.3f}', fontsize=9)
            axes[0, i].axis('off')
            
            # Adversarial
            axes[1, i].imshow(adversarial_data[idx].squeeze(), cmap='gray')
            axes[1, i].set_title(f'Adversarial\nScore: {scores[idx]:.3f}', fontsize=9)
            axes[1, i].axis('off')
    
    # Show false positives (clean images detected as anomalous)
    num_fp = min(len(false_positives), num_samples // 2)
    if num_fp > 0:
        fp_indices = np.random.choice(false_positives, num_fp, replace=False)
        for i, idx in enumerate(fp_indices):
            col_idx = i + num_tp
            if col_idx < num_samples:
                # Original
                axes[0, col_idx].imshow(clean_data[idx].squeeze(), cmap='gray')
                axes[0, col_idx].set_title(f'Original (FP)\nScore: {scores[idx]:.3f}', fontsize=9, color='red')
                axes[0, col_idx].axis('off')
                
                # No adversarial for false positives
                axes[1, col_idx].text(0.5, 0.5, 'N/A', ha='center', va='center', transform=axes[1, col_idx].transAxes)
                axes[1, col_idx].set_title('N/A', fontsize=9)
                axes[1, col_idx].axis('off')
    
    # Show score distribution for detected samples
    detected_scores = scores[detected_indices]
    axes[2, 0].hist(detected_scores, bins=20, alpha=0.7, color='orange')
    axes[2, 0].axvline(threshold, color='red', linestyle='--', label=f'Threshold: {threshold:.3f}')
    axes[2, 0].set_xlabel('Detection Score')
    axes[2, 0].set_ylabel('Count')
    axes[2, 0].set_title('Scores of Detected Anomalies')
    axes[2, 0].legend()
    axes[2, 0].grid(True, alpha=0.3)
    
    # Show score vs true label scatter
    axes[2, 1].scatter(scores[y_true == 0], np.zeros_like(scores[y_true == 0]), 
                      alpha=0.6, label='Clean', color='blue', s=20)
    axes[2, 1].scatter(scores[y_true == 1], np.ones_like(scores[y_true == 1]), 
                      alpha=0.6, label='Adversarial', color='red', s=20)
    axes[2, 1].axhline(y=threshold, color='red', linestyle='--', alpha=0.7)
    axes[2, 1].set_xlabel('Detection Score')
    axes[2, 1].set_ylabel('True Label')
    axes[2, 1].set_title('Score vs True Label')
    axes[2, 1].legend()
    axes[2, 1].grid(True, alpha=0.3)
    
    # Show top detected anomalies (highest scores)
    top_indices = np.argsort(scores)[-num_samples:]
    top_scores = scores[top_indices]
    top_labels = ['Adv' if y_true[i] == 1 else 'Clean' for i in top_indices]
    
    axes[3, 0].barh(range(len(top_scores)), top_scores, color=['red' if label == 'Adv' else 'blue' for label in top_labels])
    axes[3, 0].set_yticks(range(len(top_scores)))
    axes[3, 0].set_yticklabels([f'{label}\n{i}' for i, label in zip(top_indices, top_labels)])
    axes[3, 0].set_xlabel('Detection Score')
    axes[3, 0].set_title('Top Detected Anomalies')
    axes[3, 0].grid(True, alpha=0.3)
    
    # Hide unused subplots
    for i in range(2, num_samples):
        axes[2, i].axis('off')
        axes[3, i].axis('off')
    
    plt.tight_layout()
    plt.suptitle('Detected Anomalies Analysis', y=0.98, fontsize=16)
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"✓ Detected anomalies visualization saved to {save_path}")
    
    if show_interactive:
        print("\n🔍 Displaying Detected Anomalies Analysis...")
        print("   - Rows 1-2: Images detected as anomalous (TP=green, FP=red)")
        print("   - Bottom Left: Histogram of detection scores")
        print("   - Bottom Right: Score vs true label scatter plot")
        print("   - Bottom Center: Top detected anomalies ranking")
        print("   - Close the window to continue...")
        plt.show()
        print("✓ Anomaly analysis displayed successfully!")


def visualize_perturbation_analysis(clean_data, adversarial_data, save_path='perturbation_analysis.png',
                                  show_interactive=True):
    """
    Analyze and visualize perturbation patterns across the dataset.
    
    Args:
        clean_data: Original clean images
        adversarial_data: Adversarial images
        save_path: Path to save the visualization
        show_interactive: Whether to display interactively
    """
    
    # Calculate perturbations
    perturbations = adversarial_data - clean_data
    
    # Calculate statistics
    l2_norms = np.linalg.norm(perturbations.reshape(len(perturbations), -1), axis=1)
    linf_norms = np.max(np.abs(perturbations), axis=(1, 2, 3))
    
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    
    # 1. L2 norm distribution
    axes[0, 0].hist(l2_norms, bins=30, alpha=0.7, color='purple')
    axes[0, 0].set_xlabel('L2 Norm')
    axes[0, 0].set_ylabel('Count')
    axes[0, 0].set_title('Perturbation L2 Norm Distribution')
    axes[0, 0].grid(True, alpha=0.3)
    
    # 2. L∞ norm distribution
    axes[0, 1].hist(linf_norms, bins=30, alpha=0.7, color='orange')
    axes[0, 1].set_xlabel('L∞ Norm')
    axes[0, 1].set_ylabel('Count')
    axes[0, 1].set_title('Perturbation L∞ Norm Distribution')
    axes[0, 1].grid(True, alpha=0.3)
    
    # 3. L2 vs L∞ scatter
    axes[0, 2].scatter(l2_norms, linf_norms, alpha=0.6, s=20)
    axes[0, 2].set_xlabel('L2 Norm')
    axes[0, 2].set_ylabel('L∞ Norm')
    axes[0, 2].set_title('L2 vs L∞ Norm Relationship')
    axes[0, 2].grid(True, alpha=0.3)
    
    # 4. Average perturbation pattern
    avg_perturbation = np.mean(perturbations, axis=0)
    im1 = axes[1, 0].imshow(avg_perturbation.squeeze(), cmap='RdBu', vmin=-0.1, vmax=0.1)
    axes[1, 0].set_title('Average Perturbation Pattern')
    axes[1, 0].axis('off')
    
    # 5. Perturbation variance
    var_perturbation = np.var(perturbations, axis=0)
    im2 = axes[1, 1].imshow(var_perturbation.squeeze(), cmap='viridis')
    axes[1, 1].set_title('Perturbation Variance')
    axes[1, 1].axis('off')
    
    # 6. Perturbation magnitude by pixel position
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
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"✓ Perturbation analysis saved to {save_path}")
    
    if show_interactive:
        print("\n📈 Displaying Perturbation Analysis...")
        print("   - Top Left: Distribution of L2 perturbation norms")
        print("   - Top Center: Distribution of L∞ perturbation norms")
        print("   - Top Right: Relationship between L2 and L∞ norms")
        print("   - Bottom Left: Average perturbation pattern across dataset")
        print("   - Bottom Center: Variance in perturbation patterns")
        print("   - Bottom Right: Average perturbation magnitude by pixel position")
        print("   - Close the window to continue...")
        plt.show()
        print("✓ Perturbation analysis displayed successfully!")


def create_interactive_visualization_app(clean_data, adversarial_data, true_labels, 
                                       adv_predictions, scores, y_true):
    """
    Create an interactive visualization app that allows users to explore the results.
    
    Args:
        clean_data: Original clean images
        adversarial_data: Adversarial images
        true_labels: True labels
        adv_predictions: Model predictions on adversarial examples
        scores: Detection scores
        y_true: True labels for detection evaluation
    """
    
    print("\n" + "=" * 80)
    print("🎨 INTERACTIVE VISUALIZATION APP")
    print("=" * 80)
    print("This app allows you to explore the adversarial detection results interactively.")
    print("Each visualization will be displayed in a separate window.")
    print("Close each window to proceed to the next visualization.")
    print()
    
    # Create output directory for saved images
    output_dir = 'interactive_visualizations'
    os.makedirs(output_dir, exist_ok=True)
    
    # 1. Original vs Adversarial
    print("1️⃣  Original vs Adversarial Images")
    print("   Showing subtle perturbations and their effects...")
    visualize_original_vs_adversarial(
        clean_data, adversarial_data, true_labels, adv_predictions,
        save_path=os.path.join(output_dir, 'original_vs_adversarial.png'),
        show_interactive=True
    )
    
    # 2. Detection Scores
    print("\n2️⃣  Detection Score Analysis")
    print("   Analyzing how well the detector separates clean from adversarial...")
    visualize_detection_scores_distribution(
        scores, y_true,
        save_path=os.path.join(output_dir, 'detection_scores_distribution.png'),
        show_interactive=True
    )
    
    # 3. Detected Anomalies
    print("\n3️⃣  Detected Anomalies")
    print("   Exploring what the detector identified as suspicious...")
    visualize_detected_anomalies(
        clean_data, adversarial_data, scores, y_true,
        save_path=os.path.join(output_dir, 'detected_anomalies.png'),
        show_interactive=True
    )
    
    # 4. Perturbation Analysis
    print("\n4️⃣  Perturbation Analysis")
    print("   Understanding the patterns in adversarial perturbations...")
    visualize_perturbation_analysis(
        clean_data, adversarial_data,
        save_path=os.path.join(output_dir, 'perturbation_analysis.png'),
        show_interactive=True
    )
    
    print("\n" + "=" * 80)
    print("✅ INTERACTIVE VISUALIZATION COMPLETED!")
    print("=" * 80)
    print("All visualizations have been displayed and saved to:")
    print(f"   📁 {output_dir}/")
    print()
    print("Key insights from the visualizations:")
    print("   • Original vs Adversarial: Shows how subtle the perturbations are")
    print("   • Score Analysis: Demonstrates detection performance")
    print("   • Detected Anomalies: Shows what the detector flags as suspicious")
    print("   • Perturbation Analysis: Reveals patterns in adversarial attacks")
    print("=" * 80)


def create_comprehensive_qualitative_analysis(clean_data, adversarial_data, true_labels, 
                                            adv_predictions, scores, y_true, output_dir='visualizations',
                                            interactive_mode=True):
    """
    Create all qualitative visualizations in one comprehensive analysis.
    
    Args:
        clean_data: Original clean images
        adversarial_data: Adversarial images
        true_labels: True labels
        adv_predictions: Model predictions on adversarial examples
        scores: Detection scores
        y_true: True labels for detection evaluation
        output_dir: Directory to save visualizations
        interactive_mode: Whether to show interactive visualizations
    """
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    if interactive_mode:
        # Use interactive app
        create_interactive_visualization_app(
            clean_data, adversarial_data, true_labels, adv_predictions, scores, y_true
        )
    else:
        # Use batch mode (original behavior)
        print("Creating comprehensive qualitative analysis...")
        print("=" * 60)
        
        # 1. Original vs adversarial comparison
        print("1. Creating original vs adversarial visualization...")
        visualize_original_vs_adversarial(
            clean_data, adversarial_data, true_labels, adv_predictions,
            save_path=os.path.join(output_dir, 'original_vs_adversarial.png'),
            show_interactive=False
        )
        
        # 2. Detection scores distribution
        print("\n2. Creating detection scores distribution...")
        visualize_detection_scores_distribution(
            scores, y_true,
            save_path=os.path.join(output_dir, 'detection_scores_distribution.png'),
            show_interactive=False
        )
        
        # 3. Detected anomalies
        print("\n3. Creating detected anomalies visualization...")
        visualize_detected_anomalies(
            clean_data, adversarial_data, scores, y_true,
            save_path=os.path.join(output_dir, 'detected_anomalies.png'),
            show_interactive=False
        )
        
        # 4. Perturbation analysis
        print("\n4. Creating perturbation analysis...")
        visualize_perturbation_analysis(
            clean_data, adversarial_data,
            save_path=os.path.join(output_dir, 'perturbation_analysis.png'),
            show_interactive=False
        )
        
        print("\n" + "=" * 60)
        print("✓ Comprehensive qualitative analysis completed!")
        print(f"✓ All visualizations saved to '{output_dir}' directory")
        print("=" * 60)


if __name__ == "__main__":
    # Test the visualization functions with dummy data
    print("Testing qualitative visualization functions...")
    
    # Generate dummy data
    np.random.seed(42)
    n_samples = 1000
    
    # Dummy images
    clean_data = np.random.rand(n_samples, 1, 28, 28)
    adversarial_data = clean_data + np.random.normal(0, 0.1, clean_data.shape)
    
    # Dummy labels and predictions
    true_labels = np.random.randint(0, 10, n_samples)
    adv_predictions = np.random.randint(0, 10, n_samples)
    
    # Dummy detection scores
    scores = np.random.normal(0, 1, n_samples)
    y_true = np.random.randint(0, 2, n_samples)
    
    # Create interactive visualizations
    create_interactive_visualization_app(
        clean_data, adversarial_data, true_labels, adv_predictions, scores, y_true
    ) 