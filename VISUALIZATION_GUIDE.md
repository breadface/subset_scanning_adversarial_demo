# Visualization Guide for Subset Scanning Adversarial Detection

This guide explains how to use the interactive visualization features of the subset scanning adversarial detection demo.

## 🎨 Overview

The project now includes comprehensive interactive visualizations that allow you to:

1. **Explore visualizations interactively** - Each plot opens in a separate window
2. **Take screenshots** - High-resolution images for documentation
3. **Understand the data** - Detailed explanations for each visualization
4. **Save results** - All visualizations are automatically saved to files

## 🚀 Quick Start

### Interactive Visualization App

```bash
# Run the interactive visualization app
python visualization_app.py
```

This will:
- Load real data if available, or create realistic dummy data
- Display 4 different visualizations one by one
- Save all visualizations to the `interactive_visualizations/` directory
- Provide detailed explanations for each visualization

### Screenshot Demo

```bash
# Create high-resolution screenshots
python screenshot_demo.py
```

This will:
- Generate publication-ready screenshots
- Save them to the `screenshots/` directory
- Use 300 DPI resolution for high quality

## 📊 Visualization Types

### 1. Original vs Adversarial Images

**Purpose**: Shows the subtle differences between clean and adversarial images

**What you'll see**:
- **Row 1**: Original images with true labels
- **Row 2**: Adversarial images with model predictions
- **Row 3**: Perturbation patterns (red=positive, blue=negative)
- **Row 4**: Magnified perturbations (×5) for visibility

**Key insights**:
- Demonstrates how imperceptible adversarial perturbations can be
- Shows the systematic patterns in FGSM attacks
- Reveals which pixels are most affected

### 2. Detection Score Analysis

**Purpose**: Analyzes how well the subset scanning detector separates clean from adversarial samples

**What you'll see**:
- **Top Left**: Histogram comparison of score distributions
- **Top Right**: Box plot showing score ranges and outliers
- **Bottom Left**: Violin plot showing density distributions
- **Bottom Right**: ROC curve with AUC score

**Key insights**:
- Shows the separation between clean and adversarial score distributions
- Demonstrates detection performance through ROC analysis
- Reveals the effectiveness of the subset scanning approach

### 3. Detected Anomalies

**Purpose**: Shows what the detector identified as suspicious

**What you'll see**:
- **Rows 1-2**: Images detected as anomalous (TP=green, FP=red)
- **Bottom Left**: Histogram of detection scores
- **Bottom Right**: Score vs true label scatter plot
- **Bottom Center**: Top detected anomalies ranking

**Key insights**:
- Shows true positives (correctly detected adversarial)
- Shows false positives (clean images incorrectly flagged)
- Demonstrates the detection threshold and its effectiveness

### 4. Perturbation Analysis

**Purpose**: Understands the patterns in adversarial perturbations across the dataset

**What you'll see**:
- **Top Left**: Distribution of L2 perturbation norms
- **Top Center**: Distribution of L∞ perturbation norms
- **Top Right**: Relationship between L2 and L∞ norms
- **Bottom Left**: Average perturbation pattern across dataset
- **Bottom Center**: Variance in perturbation patterns
- **Bottom Right**: Average perturbation magnitude by pixel position

**Key insights**:
- Reveals systematic patterns in adversarial attacks
- Shows which image regions are most vulnerable
- Demonstrates the statistical properties of perturbations

## 🖥️ Interactive Features

### Window Controls

Each visualization window provides:
- **Zoom**: Click and drag to zoom in/out
- **Pan**: Right-click and drag to pan around
- **Save**: Use the save button in the toolbar
- **Home**: Reset to original view
- **Forward/Back**: Navigate through zoom history

### Navigation

- **Close window**: Click the X button or press Ctrl+W
- **Next visualization**: Close current window to proceed
- **Pause**: Keep window open to examine details

## 📁 Output Files

### Interactive Visualizations

Files saved to `interactive_visualizations/`:
- `original_vs_adversarial.png` - Side-by-side comparison
- `detection_scores_distribution.png` - Score analysis
- `detected_anomalies.png` - Anomaly detection results
- `perturbation_analysis.png` - Perturbation patterns

### High-Resolution Screenshots

Files saved to `screenshots/`:
- `original_vs_adversarial_screenshot.png` - 300 DPI
- `detection_scores_screenshot.png` - 300 DPI
- `perturbation_analysis_screenshot.png` - 300 DPI

## 🛠️ Customization

### Modify Visualization Parameters

You can customize the visualizations by modifying the functions in `src/visualization/qualitative_analysis.py`:

```python
# Change number of samples displayed
visualize_original_vs_adversarial(..., num_samples=12)

# Change detection threshold
visualize_detected_anomalies(..., threshold_percentile=90)

# Disable interactive display
visualize_original_vs_adversarial(..., show_interactive=False)
```

### Create Custom Visualizations

You can create your own visualizations by using the utility functions:

```python
from src.visualization.qualitative_analysis import create_interactive_visualization_app

# Use your own data
create_interactive_visualization_app(
    clean_data=your_clean_data,
    adversarial_data=your_adversarial_data,
    true_labels=your_labels,
    adv_predictions=your_predictions,
    scores=your_scores,
    y_true=your_y_true
)
```

## 📈 Use Cases

### For Research

- **Paper figures**: Use high-resolution screenshots
- **Presentations**: Use interactive visualizations for live demos
- **Analysis**: Explore data interactively to understand patterns

### For Education

- **Teaching**: Show students the subtle nature of adversarial attacks
- **Demonstrations**: Interactive exploration of detection results
- **Documentation**: Clear visual explanations of concepts

### For Development

- **Debugging**: Visualize detection results to understand issues
- **Validation**: Verify that detection is working correctly
- **Comparison**: Compare different detection methods visually

## 🔧 Troubleshooting

### Common Issues

1. **No display window appears**:
   - Check if you're running in a headless environment
   - Try setting `show_interactive=False` for file-only output

2. **Matplotlib backend issues**:
   - Try: `export MPLBACKEND=TkAgg` (Linux/Mac)
   - Or: `export MPLBACKEND=Qt5Agg` (if available)

3. **Memory issues with large datasets**:
   - Reduce `num_samples` parameter
   - Use smaller batch sizes

### Performance Tips

- **Large datasets**: Use sampling to reduce visualization time
- **High resolution**: Increase DPI only when needed for final output
- **Interactive mode**: Disable for batch processing

## 📚 Further Reading

- [Matplotlib Documentation](https://matplotlib.org/)
- [Seaborn Visualization Guide](https://seaborn.pydata.org/)
- [ART Subset Scanning Documentation](https://adversarial-robustness-toolbox.readthedocs.io/)

## 🤝 Contributing

To add new visualizations:

1. Create new functions in `src/visualization/qualitative_analysis.py`
2. Add them to the `create_interactive_visualization_app` function
3. Update this guide with documentation
4. Test with both real and dummy data

---

**Note**: The visualizations are designed to work with both real data from the demo and dummy data for demonstration purposes. The dummy data provides realistic-looking examples when real data is not available. 