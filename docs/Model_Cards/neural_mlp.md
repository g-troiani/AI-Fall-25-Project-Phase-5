# Model Card: Multi-Layer Perceptron (MLP)

## Model Details

**Model Type**: Feed-forward Neural Network (MLP)
**Framework**: PyTorch 2.0+
**Version**: 1.0
**Date**: December 2024

### Architecture

- **Input Layer**: 12 features
- **Hidden Layers**: [128, 64] neurons with ReLU activation
- **Dropout**: 0.3 after each hidden layer (training only)
- **Output Layer**: 2 classes (softmax)
- **Optimizer**: Adam (lr=0.001)
- **Loss**: Cross-entropy

### MC Dropout for Uncertainty

- **Inference Mode**: Dropout enabled during prediction
- **Monte Carlo Samples**: 20 forward passes
- **Uncertainty Metric**: Prediction variance across samples

## Intended Use

- Binary classification with uncertainty quantification
- Identifying low-confidence predictions for human review
- Comparing neural vs. classical ML performance

## Training Data

- NFL Week 18 tracking (70/15/15 split)
- 12 numerical features (normalized)
- Batch size: 32, Epochs: 50
- Early stopping on validation loss

## Performance Metrics

| Metric | Value |
|--------|-------|
| Accuracy | 84.7% |
| F1-Score | 0.77 |
| ROC-AUC | 0.88 |
| ECE (Calibration) | 0.12 (needs calibration) |
| Inference Latency | 45 ms (with MC Dropout, 20 samples) |

**MC Dropout Benefits**:
- Prediction uncertainty: σ² ∈ [0.01, 0.25]
- High uncertainty (σ² > 0.15) flags 8% of predictions
- Manual review of high-uncertainty predictions improves F1 to 0.81

## Limitations

1. **Calibration**: ECE=0.12 indicates overconfidence (needs temperature scaling)
2. **MC Dropout Overhead**: 20x slower inference than single forward pass
3. **Hyperparameter Sensitivity**: Learning rate, dropout rate critical
4. **No Temporal Modeling**: Treats each frame independently (use LSTM for sequences)

## Ethical Considerations

### Bias
- Same data biases as classical models
- Dropout can introduce additional variance in predictions

### Transparency
- Less interpretable than tree-based models
- MC Dropout provides uncertainty, but not feature importance
- Use gradient-based attribution (e.g., Integrated Gradients) for explanations

### Safety
- **Confidence Thresholds**: Flag predictions with σ² > 0.15 for review
- **Calibration Required**: Apply temperature scaling before deployment

---

**Last Updated**: December 6, 2024
