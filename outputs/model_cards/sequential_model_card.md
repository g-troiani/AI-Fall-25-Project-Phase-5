# Model Card: Sequential Models (LSTM & Transformer)

## Model Description
Encoder-decoder architectures for dual-task learning:
- Task 1: Trajectory forecasting (regression)
- Task 2: Binary classification

## Architecture
### LSTM
- Encoder-decoder with teacher forcing
- Hidden size: 64
- Layers: 2

### Transformer
- Full encoder-decoder with positional encoding
- d_model: 64
- Heads: 4

## Training Data
- Sequences: 293 training
- Sequence length: 20
- Features per timestep: 12

## Performance (Validation Set)
### LSTM
- Trajectory MSE: 372.439581907559
- Classification Accuracy: 0.8088235294117647

### Transformer
- Trajectory MSE: 23.623291788774218
- Classification Accuracy: 0.8088235294117647

## Limitations
- Requires fixed sequence length input
- Computational cost scales with sequence length

## Ethical Considerations
- Uncertainty should be quantified for safety-critical applications
- Model confidence should be validated before deployment
