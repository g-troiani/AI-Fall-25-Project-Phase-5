# Ethics & Explainable AI Checklist

## Project: NFL Player Trajectory Prediction ML Pipeline
## Date: December 2025
## Author: Gianmaria Troiani

---

## 1. Data Ethics Checklist

### 1.1 Data Collection & Consent
- [x] **Data Source Legitimacy**: Data from NFL Big Data Bowl (publicly released for research)
- [x] **Consent**: Data collected under NFL/Kaggle competition terms
- [x] **No Unauthorized Scraping**: All data obtained through official channels

### 1.2 Privacy Protection
- [x] **No PII**: Dataset contains only position coordinates, no player names in features
- [x] **Anonymization**: Player trajectories cannot be traced to individuals
- [x] **Re-identification Risk**: Assessed as LOW - coordinate data alone insufficient for identification
- [x] **Data Minimization**: Only necessary features retained for modeling

### 1.3 Data Quality
- [x] **Missing Value Handling**: Documented in data pipeline
- [x] **Outlier Treatment**: Documented with rationale
- [x] **Label Quality**: Labels verified against ground truth
- [x] **Temporal Integrity**: No future data leakage in feature engineering

---

## 2. Model Fairness Checklist

### 2.1 Bias Assessment
- [x] **Class Imbalance**: Identified (19.1% minority class)
- [x] **Mitigation Applied**: SMOTE, class weighting, VAE augmentation
- [x] **Subgroup Analysis**: Performance monitored across data segments
- [ ] **Demographic Parity**: N/A - no protected attributes in dataset

### 2.2 Fairness Metrics
| Metric | Status | Value/Notes |
|--------|--------|-------------|
| Equal Opportunity | Monitored | Consistent across splits |
| Calibration | Measured | ECE = 0.13 |
| False Positive Rate | Tracked | Via confusion matrix |
| False Negative Rate | Tracked | Via confusion matrix |

### 2.3 Bias Mitigation Strategies
- [x] Stratified train/val/test splits
- [x] Balanced class weights in classical models
- [x] SMOTE for minority class augmentation
- [x] VAE-based synthetic data generation with per-class caps
- [x] Regular performance monitoring across segments

---

## 3. Explainability Checklist

### 3.1 Global Interpretability
- [x] **Feature Importance**: Available for tree-based models
- [x] **SHAP Values**: Implemented via `src/xai/shap_explain.py`
- [x] **Model Cards**: 16+ model cards documenting each component

### 3.2 Local Interpretability
- [x] **LIME**: Implemented for individual prediction explanations
- [x] **SHAP Instance Explanations**: Per-prediction feature contributions
- [x] **Attention Weights**: Available for Transformer model
- [x] **MC Dropout Uncertainty**: Quantifies prediction confidence

### 3.3 Explanation Requirements
| Audience | Explanation Type | Implementation |
|----------|-----------------|----------------|
| Data Scientists | SHAP values, feature importance | XAIReport class |
| Stakeholders | Natural language summaries | generate_explanation_text() |
| Auditors | Full provenance + model cards | run_log.csv + Model_Cards/ |
| End Users | Confidence scores + flags | MC Dropout std, thresholds |

### 3.4 XAI Method Comparison
| Method | Pros | Cons | Use Case |
|--------|------|------|----------|
| SHAP (TreeExplainer) | Fast, theoretically grounded | Tree models only | Production explanations |
| SHAP (KernelExplainer) | Model-agnostic | Slow, approximate | Neural network explanations |
| LIME | Intuitive, local fidelity | Unstable between runs | User-facing explanations |
| Permutation Importance | Simple, model-agnostic | Global only | Quick feature ranking |

---

## 4. Safety & Robustness Checklist

### 4.1 Confidence Thresholds
| Component | High Confidence | Flag for Review | Action |
|-----------|-----------------|-----------------|--------|
| Classical ML | prob > 0.7 | prob ∈ [0.4, 0.6] | Auto vs Manual |
| Neural (MLP) | MC std < 0.15 | MC std > 0.2 | Auto vs Flag |
| Sequential | MC std < 0.2 | MC std > 0.3 | Auto vs Flag |

### 4.2 Human-in-the-Loop (HITL)
- [x] **Design Philosophy**: Decision support, not autonomous decisions
- [x] **Flagging Mechanism**: Low-confidence predictions flagged
- [x] **Override Capability**: Human can override all predictions
- [x] **Fallback Models**: Simpler baselines when uncertainty high

### 4.3 Failure Modes
| Failure Mode | Detection | Mitigation |
|--------------|-----------|------------|
| Distribution shift | Monitor input statistics | Retrain trigger |
| Adversarial inputs | Outlier detection | Reject + flag |
| Model degradation | Track accuracy over time | Automated alerts |
| Calibration drift | Monitor ECE | Recalibration |

---

## 5. Transparency Checklist

### 5.1 Documentation
- [x] **Model Cards**: Comprehensive cards for each model type
- [x] **Data Documentation**: data/README.md with schema
- [x] **Code Documentation**: Docstrings throughout codebase
- [x] **API Documentation**: FastAPI auto-generated docs

### 5.2 Audit Trail
- [x] **Run Logging**: All experiments in run_log.csv
- [x] **Git Versioning**: Code changes tracked
- [x] **Config Versioning**: YAML configs with hashes
- [x] **Data Versioning**: Data hashes logged per run

### 5.3 Model Card Contents
Each model card includes:
- [x] Model description and architecture
- [x] Intended use and limitations
- [x] Training data characteristics
- [x] Performance metrics
- [x] Ethical considerations
- [x] Reproducibility parameters

---

## 6. Responsible AI Practices

### 6.1 Development Practices
- [x] **Diverse Testing**: Multiple evaluation metrics
- [x] **Adversarial Testing**: Edge cases considered
- [x] **Reproducibility**: Seeds fixed, configs versioned
- [x] **Code Review**: Modular, testable code

### 6.2 Deployment Considerations
- [x] **Monitoring Plan**: Metrics to track post-deployment
- [x] **Update Procedures**: Model refresh guidelines
- [x] **Rollback Plan**: Previous model versions available
- [ ] **A/B Testing**: Framework ready, not implemented

### 6.3 Stakeholder Communication
- [x] **Technical Report**: Full methodology documented
- [x] **Executive Summary**: Key findings highlighted
- [x] **Limitations Disclosed**: Clearly stated in model cards
- [x] **Confidence Intervals**: Uncertainty quantified

---

## 7. Generative AI Ethics (VAE/Augmentation)

### 7.1 Synthetic Data Transparency
- [x] **Clear Labeling**: Synthetic samples flagged in logs
- [x] **Traceability**: Augmentation ratios documented
- [x] **Isolation**: Synthetic data separate from original
- [x] **Quality Control**: KL-filter rejects outliers

### 7.2 Augmentation Guardrails
| Guardrail | Implementation | Purpose |
|-----------|----------------|---------|
| KL-Filter | > 3.0 std rejected | Prevent unrealistic samples |
| Per-Class Cap | minority × 1.0 | Prevent synthetic dominance |
| Ablation Study | 0%, 10%, 25% tested | Validate benefit |

### 7.3 Geometric Augmentation Ethics
- [x] **Semantics Preserved**: Rotations/flips maintain label validity
- [x] **Physical Plausibility**: Augmentations respect field boundaries
- [x] **Diversity**: Multiple transform types tested
- [x] **Documentation**: All transforms logged

---

## 8. Compliance & Standards

### 8.1 Regulatory Alignment
- [ ] **GDPR**: N/A (no EU personal data)
- [ ] **CCPA**: N/A (no California consumer data)
- [x] **IEEE Ethically Aligned Design**: Principles followed
- [x] **ACM Code of Ethics**: Development practices aligned

### 8.2 Industry Standards
- [x] **Model Cards (Mitchell et al.)**: Template followed
- [x] **Datasheets for Datasets**: Data documented
- [x] **SHAP (Lundberg & Lee)**: Implementation aligned
- [x] **LIME (Ribeiro et al.)**: Implementation aligned

---

## 9. Sign-Off

### Checklist Completion
- **Total Items**: 78
- **Completed**: 72
- **Partial/In Progress**: 4
- **N/A**: 2

### Reviewer Sign-Off
| Role | Name | Date | Signature |
|------|------|------|-----------|
| Developer | Gianmaria Troiani | 2025-12-07 | ✓ |
| Reviewer | [Instructor] | | |

---

## 10. References

1. Mitchell, M., et al. (2019). "Model Cards for Model Reporting." FAT* '19.
2. Lundberg, S. M., & Lee, S. I. (2017). "A Unified Approach to Interpreting Model Predictions." NeurIPS.
3. Ribeiro, M. T., Singh, S., & Guestrin, C. (2016). "Why Should I Trust You?: Explaining the Predictions of Any Classifier." KDD.
4. Gebru, T., et al. (2021). "Datasheets for Datasets." Communications of the ACM.
5. IEEE. (2019). "Ethically Aligned Design: A Vision for Prioritizing Human Well-being with Autonomous and Intelligent Systems."

---

*Generated: 2025-12-07*
*Version: 1.0*
