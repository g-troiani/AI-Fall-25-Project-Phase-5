# Ethics, Risk & Reproducibility Documentation

## Overview
This document addresses ethical considerations, risk mitigation, and reproducibility measures for the integrated ML pipeline. All components (Classical ML, Neural Networks, Sequential Models, Generative, Graph, RL) are covered.

---

## 1. Bias Sources

### 1.1 Sampling Bias
- **Issue**: Training data may not represent all game scenarios equally
- **Mitigation**: Stratified train/val/test splits preserve class ratios
- **Monitoring**: Class distribution logged in run_log.csv

### 1.2 Label Leakage
- **Issue**: Future information could leak into training features
- **Mitigation**: Strict temporal ordering in sequence construction
- **Validation**: Data pipeline ensures no lookahead in feature engineering

### 1.3 Augmentation Artifacts
- **Issue**: Synthetic data may not represent real-world diversity
- **Mitigation**:
  - KL-filter rejects latent space outliers (> 3.0 std from mean)
  - Per-class cap limits synthetic samples to minority_count × 1.0
- **Transparency**: Augmentation ratios logged with each run

### 1.4 Class Imbalance
- **Issue**: Minority class (19.1% positive) may be under-predicted
- **Mitigation**: SMOTE resampling, class weighting in classical models
- **Metrics**: F1-macro, balanced accuracy reported alongside accuracy

---

## 2. Privacy Considerations

### 2.1 Data Retention
- **PII Status**: No personally identifiable information (PII) in features
- **Anonymization**: Player positions are coordinate-only; no names or IDs exposed
- **Re-identification Risk**: Minimal; trajectories cannot be traced to individuals

### 2.2 Synthetic Data Handling
- **Labeling**: All synthetic samples clearly flagged in run_log.csv
- **Traceability**: Augmentation ratios and methods documented per run
- **Isolation**: Synthetic data stored separately from original data

### 2.3 Model Weights
- **Storage**: Model weights contain aggregated patterns, not individual data
- **Access Control**: Models saved to outputs/models/ with clear provenance

---

## 3. Transparency

### 3.1 Model Cards
Each major component has a comprehensive model card in `docs/Model_Cards/`:
- **classical_*.md**: Classical ML (LogReg, DecTree, RandomForest)
- **neural_mlp_model_card.md**: Multi-Layer Perceptron
- **sequential_model_card.md**: LSTM & Transformer
- **generative_vae_model_card.md**: VAE-style data augmentation
- **graph_link_prediction_model_card.md**: Link prediction algorithms
- **rl_qlearning_model_card.md**: Q-Learning agent

### 3.2 Model Card Contents
Each card documents:
- Intended use and limitations
- Training data characteristics
- Performance metrics (accuracy, F1, MSE, Hit@K, etc.)
- Ethical considerations specific to that component
- Reproducibility parameters

### 3.3 Logging
- **Run Log**: All experiments logged to `outputs/run_log.csv`
- **Contents**: Component, metric name, value, parameters, timestamp
- **Auditability**: Full experiment history preserved

---

## 4. Reproducibility

### 4.1 Seed Control
- **Global Seed**: 42
- **Libraries**: Applied to numpy, torch, sklearn, random
- **Determinism**: All random operations seeded for reproducibility

### 4.2 Configuration Management
- **Config File**: `configs/default.yaml`
- **Parameters Logged**: All hyperparameters recorded with each run
- **Version Control**: Configuration tracked in repository

### 4.3 Pinned Dependencies
- **Requirements**: `requirements.txt` lists all dependencies with versions
- **Key Libraries**:
  - numpy, pandas, scikit-learn
  - torch (PyTorch)
  - networkx (graph algorithms)
  - matplotlib (visualization)

### 4.4 Data Immutability
- **Original Data**: Preserved in `data/base/input.csv` (read-only)
- **Derived Data**: Processed data stored separately
- **Data Versioning**: Changes to data documented in data pipeline

---

## 5. Safety Considerations

### 5.1 Confidence Thresholds
| Component | Threshold | Action |
|-----------|-----------|--------|
| Classical ML | Probability > 0.7 | High confidence |
| Classical ML | Probability ∈ [0.4, 0.6] | Flag for review |
| Neural (MLP) | MC Dropout std < 0.15 | High confidence |
| Sequential | MC Dropout std < 0.2 | High confidence |
| Graph | Hit@K > 0.5 | Reliable recommendations |

### 5.2 Human-in-the-Loop (HITL)
- **Design Philosophy**: System provides decision support, not autonomous decisions
- **Flagging**: Low-confidence predictions automatically flagged
- **Override**: Human review required for ambiguous cases
- **Fallback**: Simpler baseline models available when uncertainty is high

### 5.3 Uncertainty Quantification
- **MC Dropout**: Available for MLP, LSTM, Transformer
- **Implementation**: Multiple forward passes with dropout enabled at inference
- **Metric**: Mean standard deviation across predictions

### 5.4 Deployment Safeguards
- **Simulation First**: RL agent tested in GridWorld before real deployment
- **Bounded Actions**: RL agent limited to 4 cardinal directions
- **Episode Limits**: Maximum steps prevent infinite loops
- **Validation Required**: All models should be validated on held-out test set

---

## 6. Component-Specific Ethical Notes

### 6.1 Classical ML
- Feature importance available for tree-based models
- SMOTE may introduce synthetic artifacts in minority class

### 6.2 Neural Networks
- Black-box nature requires post-hoc explanations (SHAP/LIME)
- Dropout provides regularization and uncertainty estimates

### 6.3 Sequential Models
- Attention weights (Transformer) provide some interpretability
- Temporal predictions may compound errors over horizon

### 6.4 Generative (VAE)
- Synthetic data clearly labeled and capped per class
- Ablation study validates augmentation benefit

### 6.5 Graph Link Prediction
- Coverage@K metric monitors recommendation diversity
- Cold-start problem acknowledged for new players/plays

### 6.6 Reinforcement Learning
- Q-values fully interpretable as state-action values
- Policy visualization available as grid heatmap

---

## 7. Run Summary

| Parameter | Value |
|-----------|-------|
| **Global Seed** | 42 |
| **Data Path** | data/base/input.csv |
| **Config File** | configs/default.yaml |
| **Output Directory** | outputs |
| **Training Samples** | 293 |
| **Validation Samples** | 68 |
| **Class Balance** | 19.1% positive |

### Components Run
- Classical ML: Baseline classifiers with comprehensive metrics
- Neural Network: MLP with MC Dropout
- Sequential: LSTM + Transformer encoder-decoder
- Generative: VAE-style augmentation with KL-filter guardrails
- Graph: Link prediction (Common Neighbors, Jaccard, Adamic-Adar)
- Reinforcement Learning: Q-Learning in dynamic GridWorld

---

## 8. Checklist for Deployment

- [ ] All model cards reviewed and approved
- [ ] Confidence thresholds configured for production
- [ ] HITL workflow implemented for low-confidence cases
- [ ] Monitoring dashboard set up for drift detection
- [ ] Fallback models identified and tested
- [ ] Privacy review completed (no PII exposure)
- [ ] Bias audit conducted on test set
- [ ] Reproducibility verified (same seed → same results)

---

*Generated: 2025-12-07T09:11:35.020941*
*Seed: 42*
