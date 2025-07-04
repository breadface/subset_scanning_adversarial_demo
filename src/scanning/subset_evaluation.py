"""
Subset scanning evaluation with mixed clean/adversarial data.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score, precision_recall_curve, auc
from sklearn.metrics import confusion_matrix, classification_report
import seaborn as sns
from utils.data_utils import prepare_mixed_dataset_for_scanning


def evaluate_detection_performance(scores, y_true, threshold=None):
    """
    Evaluate detection performance using various metrics.
    
    Args:
        scores: Detection scores (higher = more likely to be adversarial)
        y_true: True labels (0 for clean, 1 for adversarial)
        threshold: Detection threshold (if None, will be optimized)
        
    Returns:
        dict: Performance metrics
    """
    
    # Calculate ROC curve and AUC
    roc_auc = roc_auc_score(y_true, scores)
    
    # Calculate precision-recall curve
    precision, recall, pr_thresholds = precision_recall_curve(y_true, scores)
    pr_auc = auc(recall, precision)
    
    # Determine optimal threshold (Youden's J statistic)
    from sklearn.metrics import roc_curve
    fpr, tpr, roc_thresholds = roc_curve(y_true, scores)
    j_scores = tpr - fpr
    optimal_idx = np.argmax(j_scores)
    optimal_threshold = roc_thresholds[optimal_idx]
    
    # Use provided threshold or optimal threshold
    if threshold is None:
        threshold = optimal_threshold
    
    # Calculate predictions
    y_pred = (scores > threshold).astype(int)
    
    # Calculate confusion matrix
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    
    # Calculate metrics
    accuracy = (tp + tn) / (tp + tn + fp + fn)
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
    
    results = {
        'roc_auc': roc_auc,
        'pr_auc': pr_auc,
        'threshold': threshold,
        'optimal_threshold': optimal_threshold,
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1_score': f1_score,
        'specificity': specificity,
        'true_positives': tp,
        'false_positives': fp,
        'true_negatives': tn,
        'false_negatives': fn,
        'scores': scores,
        'predictions': y_pred
    }
    
    return results


def visualize_detection_results(scores, y_true, results, title="Detection Results"):
    """
    Visualize detection results with multiple plots.
    
    Args:
        scores: Detection scores
        y_true: True labels
        results: Performance metrics dictionary
        title: Plot title
    """
    
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    
    # 1. Score distributions
    clean_scores = scores[y_true == 0]
    adv_scores = scores[y_true == 1]
    
    axes[0, 0].hist(clean_scores, bins=30, alpha=0.7, label='Clean', density=True)
    axes[0, 0].hist(adv_scores, bins=30, alpha=0.7, label='Adversarial', density=True)
    axes[0, 0].axvline(results['threshold'], color='red', linestyle='--', label=f'Threshold: {results["threshold"]:.3f}')
    axes[0, 0].set_xlabel('Detection Score')
    axes[0, 0].set_ylabel('Density')
    axes[0, 0].set_title('Score Distributions')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    
    # 2. ROC curve
    from sklearn.metrics import roc_curve
    fpr, tpr, _ = roc_curve(y_true, scores)
    axes[0, 1].plot(fpr, tpr, label=f'ROC (AUC = {results["roc_auc"]:.3f})')
    axes[0, 1].plot([0, 1], [0, 1], 'k--', alpha=0.5)
    axes[0, 1].set_xlabel('False Positive Rate')
    axes[0, 1].set_ylabel('True Positive Rate')
    axes[0, 1].set_title('ROC Curve')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)
    
    # 3. Precision-Recall curve
    from sklearn.metrics import precision_recall_curve
    precision, recall, _ = precision_recall_curve(y_true, scores)
    axes[1, 0].plot(recall, precision, label=f'PR (AUC = {results["pr_auc"]:.3f})')
    axes[1, 0].set_xlabel('Recall')
    axes[1, 0].set_ylabel('Precision')
    axes[1, 0].set_title('Precision-Recall Curve')
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)
    
    # 4. Confusion matrix
    cm = confusion_matrix(y_true, results['predictions'])
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=axes[1, 1])
    axes[1, 1].set_xlabel('Predicted')
    axes[1, 1].set_ylabel('Actual')
    axes[1, 1].set_title('Confusion Matrix')
    
    plt.tight_layout()
    plt.suptitle(title, y=1.02, fontsize=16)
    plt.savefig('detection_results.png', dpi=150, bbox_inches='tight')
    plt.show()


def print_detection_summary(results):
    """
    Print a comprehensive summary of detection results.
    
    Args:
        results: Performance metrics dictionary
    """
    
    print(f"\n{'='*60}")
    print("DETECTION PERFORMANCE SUMMARY")
    print(f"{'='*60}")
    
    print(f"ROC AUC: {results['roc_auc']:.4f}")
    print(f"PR AUC: {results['pr_auc']:.4f}")
    print(f"Optimal Threshold: {results['optimal_threshold']:.4f}")
    print(f"Used Threshold: {results['threshold']:.4f}")
    
    print(f"\nClassification Metrics:")
    print(f"  Accuracy: {results['accuracy']:.4f}")
    print(f"  Precision: {results['precision']:.4f}")
    print(f"  Recall (TPR): {results['recall']:.4f}")
    print(f"  Specificity (TNR): {results['specificity']:.4f}")
    print(f"  F1-Score: {results['f1_score']:.4f}")
    
    print(f"\nConfusion Matrix:")
    print(f"  True Positives: {results['true_positives']}")
    print(f"  False Positives: {results['false_positives']}")
    print(f"  True Negatives: {results['true_negatives']}")
    print(f"  False Negatives: {results['false_negatives']}")


def compare_detection_methods(method_results, title="Method Comparison"):
    """
    Compare multiple detection methods.
    
    Args:
        method_results: Dictionary with method names as keys and results as values
        title: Plot title
    """
    
    # Extract metrics for comparison
    methods = list(method_results.keys())
    roc_aucs = [method_results[m]['roc_auc'] for m in methods]
    pr_aucs = [method_results[m]['pr_auc'] for m in methods]
    f1_scores = [method_results[m]['f1_score'] for m in methods]
    
    # Create comparison plot
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    # ROC AUC comparison
    axes[0].bar(methods, roc_aucs, color='skyblue')
    axes[0].set_ylabel('ROC AUC')
    axes[0].set_title('ROC AUC Comparison')
    axes[0].tick_params(axis='x', rotation=45)
    
    # PR AUC comparison
    axes[1].bar(methods, pr_aucs, color='lightgreen')
    axes[1].set_ylabel('PR AUC')
    axes[1].set_title('PR AUC Comparison')
    axes[1].tick_params(axis='x', rotation=45)
    
    # F1 Score comparison
    axes[2].bar(methods, f1_scores, color='salmon')
    axes[2].set_ylabel('F1 Score')
    axes[2].set_title('F1 Score Comparison')
    axes[2].tick_params(axis='x', rotation=45)
    
    plt.tight_layout()
    plt.suptitle(title, y=1.02, fontsize=16)
    plt.savefig('method_comparison.png', dpi=150, bbox_inches='tight')
    plt.show()
    
    # Print comparison table
    print(f"\n{'='*80}")
    print("METHOD COMPARISON TABLE")
    print(f"{'='*80}")
    print(f"{'Method':<20} {'ROC AUC':<10} {'PR AUC':<10} {'F1 Score':<10} {'Precision':<10} {'Recall':<10}")
    print(f"{'-'*80}")
    
    for method in methods:
        results = method_results[method]
        print(f"{method:<20} {results['roc_auc']:<10.4f} {results['pr_auc']:<10.4f} "
              f"{results['f1_score']:<10.4f} {results['precision']:<10.4f} {results['recall']:<10.4f}")


if __name__ == "__main__":
    # Test the evaluation functions
    print("Testing subset scanning evaluation functions...")
    
    # Generate dummy data for testing
    np.random.seed(42)
    n_samples = 1000
    n_features = 10
    
    # Generate clean and adversarial scores
    clean_scores = np.random.normal(0, 1, n_samples)
    adv_scores = np.random.normal(2, 1, n_samples)
    
    # Combine scores
    all_scores = np.concatenate([clean_scores, adv_scores])
    y_true = np.concatenate([np.zeros(n_samples), np.ones(n_samples)])
    
    # Evaluate performance
    results = evaluate_detection_performance(all_scores, y_true)
    
    # Print summary
    print_detection_summary(results)
    
    print("Evaluation test completed!") 