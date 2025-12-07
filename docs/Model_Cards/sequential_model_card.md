# Model Card: Sequential Models (LSTM & Transformer)

## Model Details
- **Model Types**: LSTM Encoder-Decoder, Transformer Encoder-Decoder
- **Framework**: PyTorch
- **Version**: 1.0
- **Date**: December 2025

## Model Description
Encoder-decoder architectures for dual-task learning:
- **Task 1**: Trajectory forecasting (regression)
- **Task 2**: Binary classification

Both models support MC Dropout for uncertainty quantification.

## Architecture

### LSTM
- **Type**: Encoder-decoder with teacher forcing
- **Hidden Size**: 64
- **Layers**: 2
- **Dropout**: 0.2

### Transformer
- **Type**: Full encoder-decoder with positional encoding
- **d_model**: 64
- **Attention Heads**: 4
- **Feedforward Dim**: 256
- **Dropout**: 0.1

## Intended Use
- Multi-step trajectory forecasting
- Simultaneous classification from temporal features
- Decision support for player positioning prediction
- Uncertainty-aware predictions via MC Dropout

## Training Data
- **Sequences**: 293 training, 68 validation
- **Sequence Length**: 20 timesteps
- **Features per Timestep**: 12
- **Class Balance**: 19.1% positive

## Performance (Validation Set)

### LSTM
| Metric | Value |
|--------|-------|
| Trajectory MSE | 372.4396 |
| Trajectory MAE | 14.8170 |
| Trajectory MAPE | 46.49% |
| Trajectory MASE | 51.6959 |
| Classification Accuracy | 0.8088 |
| MC Dropout Mean Std | 2.1289 |
| Latency (batch=1) | 0.55 ms |

### Transformer
| Metric | Value |
|--------|-------|
| Trajectory MSE | 23.6233 |
| Trajectory MAE | 3.7747 |
| Trajectory MAPE | 13.52% |
| Trajectory MASE | 13.1697 |
| Classification Accuracy | 0.8088 |
| MC Dropout Mean Std | 1.1770 |
| Latency (batch=1) | 6.63 ms |

## Training Configuration
- **Optimizer**: Adam
- **LSTM LR**: 0.001
- **Transformer LR**: 0.001
- **Epochs**: 30
- **Batch Size**: 16

## Limitations
1. Requires fixed sequence length input
2. Computational cost scales with sequence length
3. Transformer attention is O(n²) in sequence length
4. May struggle with very long-range dependencies

## Ethical Considerations

### Bias Sources
- **Temporal Bias**: Predictions weighted toward recent timesteps
- **Sampling Bias**: Training data may not cover all game scenarios
- **Label Leakage**: Strict temporal ordering prevents information leakage

### Privacy
- No personally identifiable information (PII) in features
- Trajectories are anonymized coordinate sequences
- No re-identification possible from predictions

### Transparency
- Attention weights (Transformer) provide interpretability
- MC Dropout quantifies prediction uncertainty
- All training parameters logged in run_log.csv

### Safety
- **Confidence Threshold**: Use MC Dropout std < 0.2 for trajectory predictions
- **Human-in-the-Loop**: Flag high-uncertainty predictions for manual review
- **Fallback**: Use simpler models when uncertainty is high

## Reproducibility
- **Seed**: 42
- **Config**: configs/default.yaml
- **Data**: data/base/input.csv (immutable)
- **Dependencies**: See requirements.txt

---
*Generated: 2025-12-07T09:11:35.020604*
