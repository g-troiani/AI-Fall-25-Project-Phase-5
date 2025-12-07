# Model Card: XGBoost Classifier

## Model Details

**Model Type**: XGBoost (Extreme Gradient Boosting)
**Framework**: XGBoost 1.7+
**Version**: 1.0
**Date**: December 2024
**Developer**: Gianmaria Troiani

### Model Architecture

- **Algorithm**: Gradient boosted decision trees with regularization
- **Number of Estimators**: 100 boosting rounds
- **Max Depth**: 6 levels
- **Learning Rate**: 0.1
- **Regularization**: L2 (lambda=1), L1 (alpha=0)
- **Objective**: Binary logistic regression (log loss)

## Intended Use

- Binary classification for NFL catch prediction
- High-performance alternative to Random Forest
- Feature importance analysis with SHAP values
- Production deployment where speed + accuracy are critical

## Training Data

- Same as Random Forest: NFL Week 18 tracking data (~78k samples)
- 12 numerical features (position, velocity, acceleration)
- Target: within_catch_radius (binary)
- Class distribution: ~15% positive (balanced via scale_pos_weight)

## Performance Metrics

| Metric | Value |
|--------|-------|
| Accuracy | 86.1% |
| F1-Score | 0.80 |
| ROC-AUC | 0.91 |
| Inference Latency | 8.5 ms (batch=1) |

**Comparison to Random Forest**:
- +0.9% accuracy improvement
- +2% AUC improvement
- 30% faster inference

## Limitations

1. **Overfitting Risk**: More prone to overfitting than Random Forest without careful tuning
2. **Hyperparameter Sensitivity**: Performance depends heavily on learning rate, depth, regularization
3. **Class Imbalance**: Requires scale_pos_weight tuning
4. **Interpretability**: Less intuitive than Random Forest (use SHAP for explanations)

## Ethical Considerations

### Bias
- Same sampling, position, and labeling bias as Random Forest
- Mitigation: Balanced class weights (scale_pos_weight parameter)

### Privacy
- Uses publicly available NFL data
- No PII beyond player positions

### Transparency
- All hyperparameters tracked in configs/classic.yaml
- SHAP values provide local interpretability

### Safety
- Requires human review for high-stakes decisions
- Confidence thresholds recommended (>0.65 for high-confidence predictions)

## Deployment Recommendations

✅ Use when:
- Need better accuracy than Random Forest
- Fast inference is critical
- Willing to invest in hyperparameter tuning

🔀 Consider alternatives:
- Random Forest for better interpretability
- Neural networks for uncertainty quantification

---

**Last Updated**: December 6, 2024
