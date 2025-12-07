# Model Comparison Tables

## Classification Results (Supervised)

| Model | Type | Accuracy | Precision | Recall | F1 | F1-macro | ROC-AUC | ECE | Latency (batch=1) | Latency (batch=32) |
|-------|------|----------|-----------|--------|-------|----------|---------|-----|-------------------|-------------------|
| DecisionTree | Classical | 0.7647 | 0.4000 | 0.4615 | 0.4286 | 0.6402 | 0.6490 | 0.1324 | 0.05ms | 0.06ms |
| MLP | Neural | 0.7647 | - | - | - | - | - | - | - | - |
| LSTM | Sequential | 0.8088 | - | - | 0.0000 | - | - | - | 0.55ms | 1.81ms |
| Transformer | Sequential | 0.8088 | - | - | 0.0000 | - | - | - | 6.63ms | 17.66ms |

## Trajectory Prediction Results (Time Series)

| Model | MSE | MAE | MAPE (%) | MASE | MC Dropout Std | Train Time (ms) |
|-------|-----|-----|----------|------|----------------|-----------------|
| LSTM | 372.4396 | 14.8170 | 46.49 | 51.6959 | 2.1289 | 3000 |
| Transformer | 23.6233 | 3.7747 | 13.52 | 13.1697 | 1.1770 | 28612 |

## Generative Augmentation Ablation

| Augmentation Ratio | Accuracy | F1 Score | Delta vs 0% |
|--------------------|----------|----------|-------------|
| 0% | 0.7647 | 0.1111 | - |
| 10% | 0.8088 | 0.1333 | +0.0441 |
| 25% | 0.7941 | 0.0000 | +0.0294 |

## Graph Link Prediction

### Hit@K
| Method | K=3 | K=5 | K=10 |
|--------|-----|-----|------|
| common_neighbors | 0.5455 | 0.7485 | 0.9342 |
| jaccard | 0.5571 | 0.7466 | 0.9342 |
| adamic_adar | 0.5532 | 0.7563 | 0.9362 |

### MAP@K
| Method | K=3 | K=5 | K=10 |
|--------|-----|-----|------|
| common_neighbors | 0.4682 | 0.5118 | 0.5559 |
| jaccard | 0.4818 | 0.5192 | 0.5652 |
| adamic_adar | 0.4896 | 0.5334 | 0.5766 |

### Coverage@K
| Method | K=3 | K=5 | K=10 |
|--------|-----|-----|------|
| common_neighbors | 0.8739 | 0.9870 | 1.0000 |
| jaccard | 0.8696 | 0.9913 | 1.0000 |
| adamic_adar | 0.8870 | 0.9913 | 1.0000 |

## Reinforcement Learning

### Performance Metrics
| Metric | Value |
|--------|-------|
| Final Avg Return (last 100) | 81.11 |
| Max Return | 485.00 |
| Total Episodes | 500 |
| Convergence Episode (90% threshold) | 306 |

### Hyperparameter Configuration
| Hyperparameter | Value | Description |
|----------------|-------|-------------|
| Grid Size | 10 | State space dimension |
| γ (Discount) | 0.95 | Future reward discount |
| α (Learning Rate) | 0.2 | Q-value update rate |
| ε (Initial) | 1.0 | Initial exploration rate |
| ε (Minimum) | 0.01 | Minimum exploration |
| ε (Decay) | 0.995 | Exploration decay rate |
| Episodes | 500 | Training episodes |
| Max Steps | 50 | Steps per episode |

### Hyperparameter Sensitivity Analysis
| Parameter | Low | Default | High | Effect on Performance |
|-----------|-----|---------|------|----------------------|
| γ (Discount) | 0.8 | 0.95 | 0.99 | Higher γ → longer-term planning |
| α (Learning Rate) | 0.05 | 0.2 | 0.5 | Higher α → faster but less stable |
| ε (Decay) | 0.99 | 0.995 | 0.999 | Slower decay → more exploration |
| Episodes | 200 | 500 | 1000 | More episodes → better convergence |

*Note: Sensitivity analysis based on typical Q-learning behavior. Actual results may vary.*
