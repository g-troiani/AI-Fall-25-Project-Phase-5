# System Architecture Diagram

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                         FINAL PROJECT PHASE 5                                │
│                      Integrated ML System Architecture                       │
└─────────────────────────────────────────────────────────────────────────────┘

                              ┌───────────────┐
                              │  Input Data   │
                              │  (input.csv)  │
                              └───────┬───────┘
                                      │
                              ┌───────▼───────┐
                              │ Data Pipeline │
                              │ - Load/Clean  │
                              │ - Features    │
                              │ - Sequences   │
                              └───────┬───────┘
                                      │
        ┌─────────────────────────────┼─────────────────────────────┐
        │                             │                             │
        ▼                             ▼                             ▼
┌───────────────┐           ┌───────────────┐           ┌───────────────┐
│   Classical   │           │    Neural     │           │  Sequential   │
│     (P1-P2)   │           │    (P3)       │           │    (P4)       │
│               │           │               │           │               │
│ - LogReg      │           │ - MLP         │           │ - LSTM        │
│ - DecTree     │           │ - MC Dropout  │           │ - Transformer │
│ - RandForest  │           │               │           │               │
└───────┬───────┘           └───────┬───────┘           └───────┬───────┘
        │                           │                           │
        └───────────────────────────┼───────────────────────────┘
                                    │
                    ┌───────────────┼───────────────┐
                    │               │               │
                    ▼               ▼               ▼
            ┌───────────┐   ┌───────────┐   ┌───────────┐
            │Generative │   │   Graph   │   │    RL     │
            │  (VAE)    │   │  (GNN)    │   │(Q-Learn)  │
            │           │   │           │   │           │
            │- Augment  │   │- Link Pred│   │- GridWorld│
            │- Ablation │   │- Hit@K    │   │- Policy   │
            └─────┬─────┘   └─────┬─────┘   └─────┬─────┘
                  │               │               │
                  └───────────────┼───────────────┘
                                  │
                          ┌───────▼───────┐
                          │    MLOps      │
                          │               │
                          │ - RunLogger   │
                          │ - run_log.csv │
                          │ - Model Cards │
                          │ - Ethics Docs │
                          └───────┬───────┘
                                  │
                          ┌───────▼───────┐
                          │   API Layer   │
                          │  (FastAPI)    │
                          │               │
                          │/predict/tabular
                          │/recommend/{id}│
                          │/forecast/ts   │
                          │/policy/grid   │
                          └───────────────┘
```

## Data Flow

1. **Input**: Raw CSV data loaded from `data/base/input.csv`
2. **Preprocessing**: Feature engineering, sequence construction, train/val/test split
3. **Training Stages**:
   - Classical ML: Baseline classifiers (LogReg, DecTree, RandomForest)
   - Neural: MLP with MC Dropout for uncertainty
   - Sequential: LSTM and Transformer for trajectory prediction
4. **Advanced Components**:
   - Generative: VAE-style augmentation for data synthesis
   - Graph: Link prediction for player-play relationships
   - RL: Q-learning agent for dynamic gridworld navigation
5. **MLOps**: All metrics logged to `run_log.csv` with model cards and ethics documentation
6. **API**: FastAPI endpoints for serving predictions

## Key Files

| Component | File |
|-----------|------|
| Orchestrator | `src/integration/orchestrate.py` |
| Classical ML | `src/classic/train_tabular.py` |
| Neural Network | `src/neural/train_mlp.py` |
| LSTM | `src/neural/train_text_lstm.py` |
| Transformer | `src/sequential/train_text_transformer.py` |
| Generative | `src/generative/vae_synth.py` |
| Graph | `src/graph/gnn_link_pred.py` |
| RL | `src/rl/q_learning.py` |
| API | `src/api/service.py` |
| MLOps | `src/mlops/utils.py`, `eval_pipeline.py` |
