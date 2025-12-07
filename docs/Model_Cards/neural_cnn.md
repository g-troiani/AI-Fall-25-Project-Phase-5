# Model Card: Convolutional Neural Network (CNN)

## Model Details

**Model Type**: CNN for Image Classification
**Framework**: PyTorch 2.0+
**Version**: 1.0
**Date**: December 2024

### Architecture

- **Input**: 3-channel images (RGB), 224x224 pixels
- **Conv Layers**: [32, 64, 128] filters with 3x3 kernels
- **Pooling**: Max pooling (2x2) after each conv block
- **Batch Normalization**: After each conv layer
- **Fully Connected**: [256, 128] → num_classes
- **Dropout**: 0.5 (MC Dropout enabled)

## Intended Use

- Image classification tasks (e.g., field vision analysis, player identification)
- Transfer learning base for sports video analysis
- Benchmark for vision-based catch prediction

**Note**: Current project uses tabular tracking data, not images. CNN is included for completeness (P3 requirement) but not trained on project data.

## Training Data

- **Intended**: Sports video frames or field heatmaps
- **Current**: Placeholder (no image data in project dataset)

## Performance Metrics

*Not applicable - No image data available for training*

Placeholder metrics (for demonstration):
- Accuracy: N/A
- F1-Score: N/A
- Inference Latency: ~50 ms/image (estimated)

## Limitations

1. **No Image Data**: Project uses tracking data, not video/images
2. **Requires Large Dataset**: CNNs need 10k+ images for good performance
3. **Compute Intensive**: GPU required for efficient training
4. **Transfer Learning Recommended**: Use pretrained models (ResNet, EfficientNet) for small datasets

## Ethical Considerations

- Same privacy/bias considerations as other models
- Image data introduces additional concerns:
  - Player identification privacy
  - Demographic bias in image recognition
  - Video recording consent

---

**Last Updated**: December 6, 2024
**Status**: Placeholder (no training data available)
