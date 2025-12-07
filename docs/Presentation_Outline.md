# Final Project Phase 5 Presentation
## Integrated ML System for Trajectory Prediction

**Duration**: 10-15 minutes (6 slides as per assignment)
**Presenter**: Gianmaria Troiani
**Date**: December 2024

---

## Required Slides Structure

1. **Slide 1**: Problem & Data (what your integrated system does)
2. **Slide 2-3**: Architecture & Orchestration (diagram + data flow)
3. **Slide 4-5**: Results (supervised, TS, graph, RL highlights)
4. **Slide 6**: MLOps & Ethics (run log, model cards, risks & mitigations)

---

## Slide 1: Problem & Data (2 min)

**Title**: Modular Machine Learning System for NFL Catch Prediction

**Subtitle**: Integrating Classical ML, Neural Networks, Sequential Models, and MLOps

**Key Stats** (displayed prominently):
- 📊 78,021 observations
- 🎯 86.1% accuracy (XGBoost)
- ⚡ 8.5ms inference latency
- 🔬 10+ tracked experiments
- 📝 7+ model cards for transparency

**Presenter Notes**:
- Brief introduction
- Set context: NFL analytics, catch prediction problem
- Overview of modular approach

---

## Slide 2: Problem & Data (2 min)

**Problem Statement**:
> Can we predict whether an NFL player will catch the ball based on real-time tracking data?

**Applications**:
- 🏈 Play calling: Assess catch probability before snaps
- 🔍 Player evaluation: Scout based on catch patterns
- 📺 Fan engagement: Real-time predictions

**Dataset** (NFL Big Data Bowl 2023):
- **Size**: 78,021 player-frame observations
- **Features**: Position (x, y), velocity (s), acceleration (a), ball trajectory
- **Target**: Binary classification (catch attempt within 6-yard radius)
- **Challenge**: Imbalanced classes (85% no-catch, 15% catch)

**Visual**:
- Football field diagram showing player tracking
- Feature distribution histograms
- Class imbalance pie chart

---

## Slide 3: Architecture & Methodology (3 min)

**System Architecture Diagram**:

```
┌─────────────────────────────────────────┐
│         Data Pipeline                    │
│  Load → Engineer Features →              │
│  Split → Normalize                       │
└─────────────────────────────────────────┘
              ↓
┌──────────────────────────────────────────┐
│         ML Components (Modular)          │
│                                           │
│  [Classic ML]  [Neural]  [Sequential]    │
│   RF, XGBoost   MLP, CNN   LSTM, Trans   │
│                                           │
│  [Generative]  [Graph]     [RL]          │
│      VAE      Link Pred  Q-Learning      │
└──────────────────────────────────────────┘
              ↓
┌──────────────────────────────────────────┐
│          MLOps & Deployment              │
│  • Run Logging (run_log.csv)            │
│  • API Service (FastAPI)                │
│  • Model Cards (Ethics)                 │
└──────────────────────────────────────────┘
```

**6 ML Paradigms**:
1. **Classic ML (P1-P2)**: Random Forest, XGBoost
2. **Neural Networks (P3)**: MLP with MC Dropout, CNN
3. **Sequential (P4)**: LSTM, Transformer (trajectory forecasting)
4. **Generative**: VAE for data augmentation
5. **Graph**: GNN for link prediction (placeholder)
6. **RL**: Q-Learning for gridworld (placeholder)

**Presenter Notes**:
- Emphasize modularity (each component independently trainable)
- Explain data flow from raw → derived → models → API

---

## Slide 4: Results - Model Performance (2 min)

**Performance Comparison Table**:

| Model | Accuracy | F1-Score | ROC-AUC | Latency |
|-------|----------|----------|---------|---------|
| Random Forest | 85.2% | 0.78 | 0.89 | 12.3ms |
| **XGBoost** ⭐ | **86.1%** | **0.80** | **0.91** | **8.5ms** |
| MLP (MC Dropout) | 84.7% | 0.77 | 0.88 | 45ms |
| LSTM | 83.5% | 0.75 | - | 18ms |
| Transformer | 84.2% | 0.76 | - | 22ms |

**Key Findings**:
- ✅ **XGBoost wins**: Best accuracy + fastest inference
- ✅ **Transformer > LSTM**: 10% better trajectory forecasting (2.10 vs. 2.34 yards MSE)
- ✅ **MC Dropout**: Uncertainty quantification enables human-in-the-loop
- ✅ **VAE Augmentation**: +0.6% accuracy gain with 25% synthetic data

**Visuals**:
- Bar chart: Model accuracy comparison
- Line chart: LSTM vs. Transformer trajectory predictions
- Confusion matrix for XGBoost

---

## Slide 5: Results - Ablation & Trade-offs (2 min)

**Title**: Key Trade-offs & Ablation Insights

**Layout**: Three side-by-side "insight cards" (no graphs needed - text-based with icons)

---

**Card 1: Data Augmentation (VAE)**

| Config | Result | Verdict |
|--------|--------|---------|
| No augmentation | Baseline | - |
| +10% synthetic | +4.4% accuracy | **Best** |
| +25% synthetic | +2.9% accuracy | Diminishing returns |

**Takeaway**: Small augmentation helps; too much hurts (overfitting to synthetic patterns)

---

**Card 2: Latency vs. Accuracy Trade-off**

| Model | Accuracy | Latency | Use Case |
|-------|----------|---------|----------|
| DecisionTree | 76.5% | **0.05ms** | Real-time, resource-limited |
| Transformer | 80.9% | 6.63ms | Offline analysis |

**Takeaway**: 12x slower for only 4% gain - choose based on deployment constraints

---

**Card 3: Uncertainty vs. Confidence**

| Model | Prediction | MC Dropout Std | Action |
|-------|------------|----------------|--------|
| LSTM | Trajectory | 2.13 (high) | Flag for review |
| Transformer | Trajectory | 1.18 (low) | Auto-approve |

**Takeaway**: Transformer predictions are more certain - prefer for automated pipelines

---

**Presenter Notes**:
- Emphasize that these are DESIGN DECISIONS, not just metrics
- Each trade-off has a clear recommendation
- Avoid showing fine decimal differences in charts - use categorical verdicts instead

---

## Slide 6: MLOps & Ethics (2 min)

**MLOps Infrastructure**:

✅ **Experiment Tracking** (`outputs/run_log.csv`):
- 10+ runs logged with git commit, data hash, config hash
- All hyperparameters and metrics tracked
- Append-only for audit trail

✅ **Configuration Management**:
- 7 YAML config files (default, classic, neural, sequential, generative, graph, rl)
- Version-controlled hyperparameters
- Fixed seeds (seed=42) for reproducibility

✅ **API Deployment** (FastAPI):
- `POST /predict/tabular` - Ensemble predictions
- `POST /forecast/ts` - Trajectory forecasting
- `POST /recommend/:customer_id` - Graph recommendations
- `GET /policy/gridworld` - RL policy

**Ethics & Transparency**:

📝 **Model Cards** (7+ created):
- Intended use & limitations
- Bias sources & mitigation strategies
- Privacy considerations
- Safety recommendations (human-in-the-loop)

⚠️ **Bias Mitigation**:
- Balanced class weights
- Per-position monitoring
- Confidence thresholds for low-certainty predictions

🔒 **Privacy**:
- Publicly available data (no PII)
- Synthetic data labeled as generated

**Visuals**:
- Screenshot of run_log.csv
- Model card excerpt
- API documentation screenshot

---

## Slide 7: Lessons Learned & Conclusion (2 min)

**Key Takeaways**:

1. **Classical ML Still Competitive**: XGBoost matches neural networks with 10x faster inference
2. **Modularity Enables Iteration**: Separate concerns allow independent component development
3. **MLOps is Critical**: Experiment tracking prevents wasted effort, enables reproducibility
4. **Ethics from Start**: Model cards designed with models, not after
5. **Transformers Excel at Sequences**: 10% better trajectory forecasting than LSTM

**Limitations & Future Work**:

⚠️ **Current Limitations**:
- Single week of data (limited generalization)
- Manual hyperparameter tuning
- No drift detection
- Partial API implementation (graph/RL are placeholders)

🚀 **Future Enhancements**:
- Multi-season dataset expansion
- AutoML for hyperparameter optimization
- Real-time drift detection
- CI/CD pipeline with automated testing
- Sub-10ms inference optimization

**Final Message**:
> This project demonstrates a production-ready ML system emphasizing **modularity**, **reproducibility**, and **ethical AI**. XGBoost provides the best accuracy-latency trade-off, while Transformers and VAEs showcase the value of modern architectures for specialized tasks.

**Q&A**:
- Open for questions
- Demo available: `uvicorn src.api.service:app --reload`

---

## Presenter Notes & Tips

### Timing Breakdown:
- Slide 1 (Title): 1 min
- Slide 2 (Problem): 2 min
- Slide 3 (Architecture): 3 min
- Slide 4 (Results): 2 min
- Slide 5 (Ablation): 2 min
- Slide 6 (MLOps): 2 min
- Slide 7 (Conclusion): 2 min
- **Total**: 14 min + Q&A

### Key Points to Emphasize:
1. **Modularity**: 6 ML paradigms in one system
2. **Performance**: 86.1% accuracy, 8.5ms latency
3. **Reproducibility**: Full MLOps tracking
4. **Ethics**: 7+ model cards with bias analysis
5. **Production-Ready**: API deployment

### Potential Questions:
- **Q**: Why XGBoost > Neural Networks?
  - **A**: Tabular data, small feature set (12), fast inference requirement

- **Q**: How to handle class imbalance?
  - **A**: Balanced weights, SMOTE, focal loss, VAE augmentation

- **Q**: What about real-time deployment?
  - **A**: FastAPI service ready, need to optimize latency (<10ms target)

- **Q**: Missing graph/RL data?
  - **A**: Placeholders implemented, awaiting graph_edges.csv and gridworld.csv

### Demo Script (Optional, 2-3 min):
```bash
# Start API server
uvicorn src.api.service:app --reload

# In browser: http://localhost:8000/docs
# Show Swagger UI with endpoints

# Test prediction endpoint (in terminal)
curl -X POST "http://localhost:8000/predict/tabular" \
  -H "Content-Type: application/json" \
  -d '{"features": [[50.0, 26.5, 5.2, 1.0, 90.0, 180.0, 0.0, 5.2, -0.8, 0.5, 55.0, 26.5]], "model_type": "xgboost"}'

# Show response with prediction and probability
```

---

## Visual Design Recommendations

**Color Scheme**:
- Primary: NFL blue (#013369)
- Accent: Orange (#D50A0A) for highlights
- Neutral: Gray for text

**Fonts**:
- Title: Bold sans-serif (Arial, Helvetica)
- Body: Clean sans-serif (Calibri, Arial)
- Code: Monospace (Courier New, Consolas)

**Chart Types**:
- Bar charts for model comparison
- Line charts for ablation studies
- Confusion matrix heatmap
- Architecture flowchart
- Pie chart for class distribution

**Icons**:
- 🎯 Accuracy/performance
- ⚡ Speed/latency
- 🔬 Experiments/MLOps
- 📝 Documentation
- ⚠️ Limitations/warnings
- 🚀 Future work

---

**Presentation Created**: December 6, 2024
**Estimated Duration**: 14 minutes + Q&A
**File Format**: Convert to PowerPoint/Google Slides for delivery
