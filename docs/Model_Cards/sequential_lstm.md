# Model Card: LSTM for Text Classification

## Model Details

**Model Type**: Long Short-Term Memory (LSTM) Network
**Framework**: PyTorch 2.0+
**Version**: 1.0
**Date**: December 2024

### Architecture

- **Embedding Layer**: Vocab size dependent, 128-dim embeddings
- **LSTM Layers**: 2 layers, hidden size=64
- **Dropout**: 0.2 between LSTM layers
- **Output**: Fully connected layer → softmax
- **Bidirectional**: No (unidirectional)

## Intended Use

- Sequential text classification
- Time-series trajectory prediction (adapted for NFL tracking)
- Baseline for Transformer comparison

## Training Data

- **Text**: Tokenized sequences (if text classification)
- **Trajectory**: 20-frame input sequences → 5-frame predictions
- **Target**: Binary catch/no-catch + trajectory (x, y) coordinates

## Performance Metrics

| Metric | Value |
|--------|-------|
| Trajectory MSE | 2.34 yards |
| Classification Accuracy | 83.5% |
| F1-Score | 0.75 |
| Inference Latency | 18 ms/sequence |

**Comparison to Transformer**:
- Transformer MSE: 2.10 yards (better)
- Transformer faster with parallel processing

## Limitations

1. **Sequential Processing**: Slower than Transformer (no parallelization)
2. **Long-term Dependencies**: Struggles with sequences > 50 frames
3. **Vanishing Gradients**: Mitigated by LSTM architecture but still present
4. **Memory**: Requires full sequence in memory

## Ethical Considerations

- Same data biases as other models
- Sequence modeling can capture temporal patterns but may overfit to specific plays
- **Fairness**: Ensure performance is consistent across player positions

---

**Last Updated**: December 6, 2024
