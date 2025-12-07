# Model Card: Classical ML (DecisionTree)

## Model Details
- **Model Type**: DecisionTree Classifier
- **Framework**: scikit-learn
- **Version**: 1.0
- **Date**: December 2025

## Model Description
DecisionTree classifier for binary classification with SMOTE resampling and class weighting.

## Intended Use
- Baseline classification for trajectory prediction
- Fast inference for real-time applications
- Interpretable feature importance (for tree-based models)

## Training Data
- **Samples**: 293 training, 68 validation
- **Features**: 12 tabular features
- **Class Balance**: 19.1% positive (minority class)
- **Resampling**: SMOTE applied to training data

## Performance (Validation Set)
| Metric | Value |
|--------|-------|
| Accuracy | 0.7647 |
| Precision | 0.4000 |
| Recall | 0.4615 |
| F1 Score | 0.4286 |
| F1-macro | 0.6402 |
| ROC-AUC | 0.6490 |
| ECE (Calibration) | 0.1324 |
| Latency (batch=1) | 0.05 ms |
| Latency (batch=32) | 0.06 ms |

## Limitations
1. Uses only first-frame features (no temporal context)
2. Class imbalance may affect minority class predictions
3. May not generalize to different game scenarios

## Ethical Considerations

### Bias Sources
- **Sampling Bias**: Training data may not represent all game scenarios equally
- **Label Leakage**: Temporal ordering preserved to prevent information leakage
- **Class Imbalance**: SMOTE used but may introduce synthetic artifacts

### Privacy
- No personally identifiable information (PII) in features
- Positions anonymized; no re-identification possible

### Transparency
- Feature importance available for tree-based models
- All hyperparameters logged in run_log.csv

### Safety
- **Confidence Threshold**: Use predicted probability > 0.7 for high-confidence predictions
- **Human-in-the-Loop**: Flag predictions with probability in [0.4, 0.6] for manual review

## Reproducibility
- **Seed**: 42
- **Config**: configs/default.yaml
- **Data**: data/base/input.csv (immutable)

---
*Generated: 2025-12-07T09:11:35.020397*
