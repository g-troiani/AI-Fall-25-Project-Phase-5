# Model Card: Neural Network (MLP)

## Model Description
Multi-Layer Perceptron for binary classification with dropout regularization.

## Architecture
- Input: 12 features
- Hidden Layers: [128, 64]
- Activation: ReLU
- Dropout: 0.3
- Output: Binary classification (softmax)

## Intended Use
- Binary classification from tabular features
- Alternative to classical ML with non-linear decision boundaries
- Ensemble component for model comparison

## Training Data
- Samples: 293 training, 68 validation
- Features: 12 features
- Class balance: 19.1% positive

## Performance (Validation Set)
- Accuracy: 0.7647058823529411

## Training Configuration
- Optimizer: Adam
- Learning Rate: 0.001
- Epochs: 50
- Batch Size: 32

## Limitations
- No temporal information (first-frame features only)
- Sensitive to feature scaling (requires normalization)
- May overfit on small datasets

## Ethical Considerations
- Predictions should be validated before deployment
- Model explanations (e.g., SHAP) recommended for transparency
