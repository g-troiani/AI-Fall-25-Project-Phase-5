# Model Card: Transformer for Trajectory Prediction

## Model Details

**Model Type**: Transformer Encoder
**Framework**: PyTorch 2.0+
**Version**: 1.0
**Date**: December 2024

### Architecture

- **Input Embedding**: 12 features → 64-dim model
- **Positional Encoding**: Sinusoidal encoding for temporal order
- **Encoder Layers**: 2 layers, 4 attention heads
- **Feed-Forward Dim**: 256
- **Dropout**: 0.1
- **Output**: Multi-task (trajectory regression + classification)

## Intended Use

- Trajectory forecasting (predict next 5 frames from last 20 frames)
- Catch probability prediction
- Attention visualization for interpretability

## Training Data

- NFL tracking sequences (20-frame input, 5-frame output)
- 70/15/15 train/val/test split
- Batch size: 16, Epochs: 30

## Performance Metrics

| Metric | Value |
|--------|-------|
| Trajectory MSE | 2.10 yards |
| Classification Accuracy | 84.2% |
| F1-Score | 0.76 |
| Inference Latency | 22 ms/sequence |

**Advantages over LSTM**:
- 10% better trajectory accuracy
- Parallel processing (faster training)
- Attention weights provide interpretability

## Limitations

1. **Data Hungry**: Requires more training data than LSTM
2. **Hyperparameter Sensitivity**: Attention heads, layers, dropout rate critical
3. **Quadratic Complexity**: Attention is O(n²) in sequence length
4. **Position Encoding**: Fixed sinusoidal encoding may not be optimal

## Ethical Considerations

### Interpretability
- **Attention Weights**: Can visualize which frames are most important
- **Example**: High attention on frame-18 before catch suggests model learns anticipation

### Bias
- May learn temporal biases (e.g., specific play timing)
- **Mitigation**: Regularization, diverse training data

---

**Last Updated**: December 6, 2024
