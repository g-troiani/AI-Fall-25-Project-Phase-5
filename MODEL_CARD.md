# Model Card: NFL Player Trajectory Prediction System

## Model Details

### Basic Information
- **Model Name**: Integrated ML Pipeline for NFL Trajectory Prediction
- **Version**: 1.0
- **Date**: December 2025
- **Type**: Multi-model ensemble (Classical ML, Neural Networks, Sequential Models)
- **Framework**: PyTorch, scikit-learn
- **License**: MIT

### Developers
- **Organization**: Miami Dade College - CAI4510C Machine Learning Capstone
- **Contact**: gianmaria.troiani@mdc.edu

### Model Architecture
The system comprises multiple model families:

| Component | Architecture | Purpose |
|-----------|-------------|---------|
| Classical ML | DecisionTree, RandomForest, LogisticRegression | Binary classification baseline |
| Neural Network | MLP [128, 64] with MC Dropout | Classification with uncertainty |
| LSTM | Encoder-Decoder (hidden=64, layers=2) | Trajectory forecasting |
| Transformer | Encoder-Decoder (d_model=64, heads=4) | Trajectory forecasting |
| VAE Sampler | PCA-based latent space (dim=8) | Data augmentation |
| GAN | Generator-Discriminator (hidden=128) | Synthetic data generation |
| Q-Learning | Tabular Q-values (10x10 grid) | Dynamic positioning policy |

---

## Intended Use

### Primary Use Cases
1. **Catch Prediction**: Binary classification of whether a player will attempt to catch the ball
2. **Trajectory Forecasting**: Predict future player positions (5-step horizon)
3. **Play Recommendation**: Graph-based player-play link prediction
4. **Positioning Strategy**: RL-based optimal positioning suggestions

### Intended Users
- Sports analysts
- Coaching staff
- Broadcast teams (real-time predictions)
- Research teams studying player behavior

### Out-of-Scope Uses
- **NOT for**: Gambling predictions, player injury assessment, contract negotiations
- **NOT intended**: Real-time autonomous decision-making without human oversight

---

## Training Data

### Dataset
- **Source**: NFL Big Data Bowl 2023 (Kaggle)
- **Size**: 78,021 player-frame observations
- **Time Period**: Week 18, 2023 season
- **Features**: 12 engineered features from position, velocity, acceleration

### Data Splits
| Split | Samples | Class Balance |
|-------|---------|---------------|
| Train | 293 | 19.1% positive |
| Validation | 68 | 19.1% positive |
| Test | 91 | 19.8% positive |

### Preprocessing
- Sequence construction: 20 frames with 5-frame stride
- Normalization: StandardScaler (mean=0, std=1)
- Class balancing: SMOTE resampling applied

---

## Performance Metrics

### Classification (Validation Set)

| Model | Accuracy | Precision | Recall | F1 | ROC-AUC | ECE |
|-------|----------|-----------|--------|-----|---------|-----|
| DecisionTree | 0.765 | 0.400 | 0.462 | 0.429 | 0.649 | 0.132 |
| MLP | 0.765 | - | - | - | - | - |
| LSTM | 0.809 | - | - | - | - | - |
| Transformer | 0.809 | - | - | - | - | - |

### Trajectory Forecasting

| Model | MSE | MAE | MAPE | MC Dropout Std |
|-------|-----|-----|------|----------------|
| LSTM | 372.44 | 14.82 | 46.5% | 2.13 |
| Transformer | 23.62 | 3.77 | 13.5% | 1.18 |

### Link Prediction (Hit@K)

| Method | K=3 | K=5 | K=10 |
|--------|-----|-----|------|
| Common Neighbors | 0.545 | 0.749 | 0.934 |
| Jaccard | 0.557 | 0.747 | 0.934 |
| Adamic-Adar | 0.553 | 0.756 | 0.936 |

### Reinforcement Learning

| Metric | Value |
|--------|-------|
| Final Avg Return | 81.11 |
| Convergence Episode | 306 |
| Total Episodes | 500 |

---

## Limitations

### Technical Limitations
1. **Single Week Data**: Model trained on Week 18 only; may not generalize to other game contexts
2. **Fixed Sequence Length**: Requires exactly 20-frame input sequences
3. **Class Imbalance**: 19.1% minority class affects recall
4. **Computational Cost**: Transformer inference ~7ms (vs 0.05ms for DecisionTree)

### Known Failure Modes
- **Edge Cases**: Unusual formations not well represented in training
- **Fast Actions**: Very rapid direction changes may be underpredicted
- **Weather Effects**: No weather features; performance may vary in extreme conditions

### Assumptions
- Input features are normalized using training set statistics
- Players behave similarly across different game situations
- Ball trajectory features are available at inference time

---

## Ethical Considerations

### Bias Assessment
| Source | Risk | Mitigation |
|--------|------|------------|
| Sampling | Training data may not represent all scenarios | Stratified splits |
| Class Imbalance | Minority class underprediction | SMOTE, class weights |
| Position Bias | WR overrepresented | Per-position monitoring |

### Privacy
- **PII Status**: No personally identifiable information in features
- **Anonymization**: Player positions are coordinate-only; no names/IDs
- **Re-identification Risk**: LOW - trajectories cannot trace to individuals

### Fairness
- Model performance monitored across player positions
- No demographic attributes in feature set
- Equal treatment across teams

### Safety
- **Human-in-the-Loop**: Required for all high-stakes decisions
- **Confidence Thresholds**: Low-confidence predictions flagged
- **Uncertainty Quantification**: MC Dropout provides prediction intervals

---

## Explainability

### Global Interpretability
- **SHAP**: TreeExplainer for feature importance ranking
- **Feature Importance**: distance_to_ball, receiver_separation most important
- **Attention Weights**: Available for Transformer model

### Local Interpretability
- **LIME**: Per-prediction explanations with discretized rules
- **SHAP Values**: Per-instance feature contributions
- **MC Dropout**: Uncertainty estimates per prediction

### Top Features (SHAP Mean |Value|)
1. distance_to_ball (0.182)
2. receiver_separation (0.165)
3. speed (0.142)
4. time_to_ball (0.118)
5. angle_to_ball (0.095)

---

## Deployment

### Requirements
```
Python >= 3.9
torch >= 2.0.0
scikit-learn >= 1.3.0
numpy >= 1.24.0
```

### Inference
```python
from src.integration.orchestrate import load_model, predict

model = load_model('outputs/models/classical_model.pkl')
prediction = predict(model, features)
```

### Latency
| Model | Batch=1 | Batch=32 |
|-------|---------|----------|
| DecisionTree | 0.05ms | 0.06ms |
| MLP | ~1ms | ~2ms |
| Transformer | 6.63ms | 17.66ms |

---

## Maintenance

### Monitoring
- Track prediction distribution drift
- Monitor confidence score distribution
- Log all predictions for audit

### Update Procedures
- Retrain monthly with new game data
- Recalibrate confidence thresholds quarterly
- Full evaluation on holdout set before deployment

### Version History
| Version | Date | Changes |
|---------|------|---------|
| 1.0 | Dec 2025 | Initial release |

---

## References

1. Mitchell, M., et al. (2019). "Model Cards for Model Reporting." FAT* Conference.
2. Lundberg, S. M., & Lee, S. I. (2017). "A Unified Approach to Interpreting Model Predictions." NeurIPS.
3. Ribeiro, M. T., Singh, S., & Guestrin, C. (2016). "Why Should I Trust You?" KDD.

---

## Citation

```bibtex
@misc{nfl_trajectory_2025,
  title={Integrated ML Pipeline for NFL Player Trajectory Prediction},
  author={Troiani, Gianmaria and Aparicio, Jorge and Calvo, Ignacio},
  year={2025},
  institution={Miami Dade College}
}
```

---

*Last Updated: December 2025*
*Seed: 42 | Config: configs/default.yaml*
