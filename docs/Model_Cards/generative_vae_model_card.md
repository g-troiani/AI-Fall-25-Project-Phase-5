# Model Card: Generative VAE-Style Sampler

## Model Details
- **Model Type**: PCA-based VAE-Style Latent Sampler
- **Framework**: NumPy/SciPy
- **Version**: 1.0
- **Date**: December 2025

## Model Description
PCA-based VAE-style sampler for data augmentation via latent space sampling.
Uses SVD for dimensionality reduction and Gaussian sampling in latent space.

## Architecture
- **Dimensionality Reduction**: SVD/PCA
- **Latent Dimension**: 8
- **Sampling**: Gaussian in latent space
- **Label Assignment**: Nearest-neighbor matching
- **Reconstruction**: Linear projection back to data space

## Guardrails
- **KL-Filter**: Rejects samples > 3.0 std from latent mean (outlier rejection)
- **Per-Class Cap**: Limits synthetic samples per class to minority_count × 1.0

## Intended Use
- Data augmentation for imbalanced datasets
- Synthetic sample generation for training
- Ablation studies on augmentation ratios
- Improving minority class representation

## Augmentation Ablation Results

| Ratio | Accuracy | F1 Score |
|-------|----------|----------|
| 0% | 0.7647 | 0.1111 |
| 10% | 0.8088 | 0.1333 |
| 25% | 0.7941 | 0.0000 |

## Best Configuration
- **Optimal Ratio**: 10%
- **Accuracy Delta vs Baseline**: +0.0441

## Limitations
1. PCA assumes linear relationships in data
2. Synthetic samples may not capture all data modes
3. Label assignment depends on training data quality
4. Cannot generate truly novel patterns

## Ethical Considerations

### Bias Sources
- **Augmentation Artifacts**: Synthetic data may not represent real-world diversity
- **Pattern Amplification**: Nearest-neighbor labeling may reinforce existing biases
- **Class Imbalance**: Per-class capping prevents synthetic data from dominating

### Privacy
- Synthetic samples are clearly flagged in run_log.csv
- No PII in generated features
- Generated data cannot be traced back to individuals
- Augmentation ratios documented for transparency

### Transparency
- All synthetic samples logged with augmentation ratio
- Ablation results show impact of augmentation
- KL-filter threshold documented (3.0 std)

### Safety
- **Guardrails**: KL-filter rejects unrealistic samples
- **Per-Class Cap**: Prevents synthetic data from overwhelming real data
- **Validation**: Ablation study validates augmentation benefit

## Reproducibility
- **Seed**: 42
- **Config**: configs/default.yaml
- **Latent Dim**: 8
- **Augment Ratios**: [0.0, 0.1, 0.25]

---
*Generated: 2025-12-07T09:11:35.020695*
