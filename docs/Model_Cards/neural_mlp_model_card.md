# Model Card: Neural Network (MLP)

## Model Details
- **Model Type**: Multi-Layer Perceptron (MLP)
- **Framework**: PyTorch
- **Version**: 1.0
- **Date**: December 2025

## Model Description
Multi-Layer Perceptron for binary classification with dropout regularization and MC Dropout for uncertainty quantification.

## Architecture
- **Input**: 12 features
- **Hidden Layers**: [128, 64]
- **Activation**: ReLU
- **Dropout**: 0.3
- **Output**: Binary classification (softmax)

## Intended Use
- Binary classification from tabular features
- Alternative to classical ML with non-linear decision boundaries
- Uncertainty estimation via MC Dropout
- Ensemble component for model comparison

## Training Data
- **Samples**: 293 training, 68 validation
- **Features**: 12 tabular features
- **Class Balance**: 19.1% positive (minority class)
- **Preprocessing**: StandardScaler normalization

## Performance (Validation Set)
| Metric | Value |
|--------|-------|
| Accuracy | 0.7647 |

## Training Configuration
- **Optimizer**: Adam
- **Learning Rate**: 0.001
- **Epochs**: 50
- **Batch Size**: 32
- **Loss**: Cross-Entropy

## Limitations
1. No temporal information (first-frame features only)
2. Sensitive to feature scaling (requires normalization)
3. May overfit on small datasets
4. Black-box model requires post-hoc explanations

## Ethical Considerations

### Bias Sources
- **Sampling Bias**: Model learns patterns from available training data only
- **Feature Bias**: Feature engineering decisions may embed assumptions
- **Class Imbalance**: Minority class may be under-predicted

### Privacy
- No personally identifiable information (PII) in features
- Model weights do not encode individual data points
- Input features are normalized and anonymized

### Transparency
- MC Dropout provides uncertainty estimates
- SHAP/LIME can be applied for feature importance
- Training logs available in run_log.csv

### Safety
- **Confidence Threshold**: Use MC Dropout std < 0.15 for high-confidence predictions
- **Human-in-the-Loop**: High uncertainty predictions should be flagged for review
- **Fallback**: Use classical ML baseline for comparison

## Reproducibility
- **Seed**: 42
- **Config**: configs/default.yaml
- **Data**: data/base/input.csv (immutable)
- **Dependencies**: See requirements.txt

---
*Generated: 2025-12-07T09:11:35.020509*
