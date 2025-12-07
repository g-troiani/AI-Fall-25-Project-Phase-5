# Model Card: Classical ML (DecisionTree)

## Model Description
DecisionTree classifier for binary classification.

## Intended Use
- Baseline classification for trajectory prediction
- Fast inference for real-time applications
- Interpretable feature importance

## Training Data
- Samples: 293 training, 68 validation
- Features: 12 features
- Class balance: 19.1% positive

## Performance (Validation Set)
- Accuracy: 0.7647058823529411

## Limitations
- Uses only first-frame features (no temporal context)
- Class imbalance may affect minority class predictions

## Ethical Considerations
- Predictions should be used as decision support, not autonomous decisions
- Performance may vary across different scenarios
