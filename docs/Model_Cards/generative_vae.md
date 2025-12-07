# Model Card: Variational Autoencoder (VAE)

## Model Details

**Model Type**: Variational Autoencoder for Data Augmentation
**Framework**: PyTorch 2.0+
**Version**: 1.0
**Date**: December 2024

### Architecture

- **Encoder**: [12 → 32 → 16 → latent_dim=8]
- **Decoder**: [latent_dim=8 → 16 → 32 → 12]
- **Latent Space**: 8-dimensional Gaussian
- **Loss**: ELBO (reconstruction + KL divergence)
- **Activation**: ReLU (hidden), Linear (output)

## Intended Use

- **Data Augmentation**: Generate synthetic training samples
- **Ablation Study**: Compare model performance with/without augmentation
- **Anomaly Detection**: Detect out-of-distribution samples (high reconstruction error)

## Training Data

- Positive class samples only (players within catch radius)
- ~11,700 training samples
- Reconstruction target: Same 12 features

## Performance Metrics

### Generative Quality

| Metric | Value |
|--------|-------|
| Reconstruction MSE | 0.15 (normalized features) |
| KL Divergence | 2.3 |
| Augmentation Ratios Tested | [0%, 10%, 25%] |

### Downstream Impact (Random Forest)

| Augmentation Ratio | Test Accuracy | F1-Score | Change |
|--------------------|---------------|----------|--------|
| 0% (baseline) | 85.2% | 0.78 | - |
| 10% augmentation | 85.6% | 0.79 | +0.4% / +0.01 |
| 25% augmentation | 85.8% | 0.80 | +0.6% / +0.02 |

**Conclusion**: Modest improvement with 10-25% augmentation

## Limitations

1. **Synthetic Artifacts**: Generated samples may not reflect real player behavior
2. **Class Imbalance Only**: Only augments minority class (catch attempts)
3. **Feature Space Only**: Doesn't generate new play contexts
4. **Overfitting Risk**: Too much augmentation can degrade performance

## Ethical Considerations

### Privacy
- Synthetic data labeled as generated (not real player actions)
- No re-identification risk (latent space is abstract)

### Transparency
- **Clearly Labeled**: Augmented data flagged in training logs
- **Ablation Study**: Performance with/without augmentation reported

### Bias
- **Amplification Risk**: VAE may amplify existing biases in training data
- **Mitigation**: Compare fairness metrics (per-position performance) with/without augmentation

### Safety
- **Synthetic Data Limits**: Do not use synthetic data alone for player evaluation
- **Human Review**: Augmentation decisions require domain expert validation

---

**Last Updated**: December 6, 2024
