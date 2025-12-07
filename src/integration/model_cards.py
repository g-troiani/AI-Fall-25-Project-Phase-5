"""
Model Cards and Ethics Documentation
=====================================
Generate model cards, fairness analysis, and ethics documentation.
Source: Project 5 (MLOps/Ethics)
"""

import os
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score
from collections import defaultdict


def generate_classical_model_card(model_name: str, metrics: dict, config: dict,
                                  n_train: int, n_test: int, n_features: int,
                                  class_balance: float, catch_radius: float) -> str:
    """Generate model card for classical ML model."""
    return f"""# Model Card: Classical ML ({model_name})

## Model Description
{model_name} classifier for binary classification (within {catch_radius} yards of ball landing).

## Intended Use
- Baseline classification for NFL player trajectory prediction
- Fast inference for real-time applications
- Interpretable feature importance

## Training Data
- Source: NFL Big Data Bowl 2026
- Samples: {n_train} training, {n_test} test
- Features: {n_features} numerical features
- Class balance: {class_balance:.1%} positive

## Performance (Test Set)
- Accuracy: {metrics.get('accuracy', 0):.3f}
- F1 Score: {metrics.get('f1', 0):.3f}
- ROC-AUC: {metrics.get('roc_auc', 0):.3f}
- Latency: {metrics.get('latency_ms', 0):.2f} ms

## Limitations
- Uses only first-frame features (no temporal context)
- Class imbalance may affect minority class predictions
- Limited to binary classification (no trajectory prediction)

## Ethical Considerations
- Predictions should be used as decision support, not autonomous decisions
- Performance may vary across player positions
- No personally identifiable information in features
"""


def generate_sequential_model_card(lstm_metrics: dict, transformer_metrics: dict,
                                   config: dict, n_train: int, n_test: int,
                                   input_seq_len: int, output_seq_len: int) -> str:
    """Generate model card for sequential models (LSTM & Transformer)."""
    return f"""# Model Card: Sequential Models (LSTM & Transformer)

## Model Description
Encoder-decoder architectures for dual-task learning:
- Task 1: Trajectory forecasting (regression)
- Task 2: Binary classification (catch proximity)

## Architecture
### LSTM
- Encoder: {config.get('lstm', {}).get('num_layers', 2)}-layer LSTM
- Hidden size: {config.get('lstm', {}).get('hidden_size', 64)}
- Teacher forcing ratio: {config.get('lstm', {}).get('teacher_forcing_ratio', 0.5)}

### Transformer
- d_model: {config.get('transformer', {}).get('d_model', 64)}
- Attention heads: {config.get('transformer', {}).get('nhead', 4)}
- Layers: {config.get('transformer', {}).get('num_layers', 2)}

## Training Data
- Samples: {n_train} training, {n_test} test
- Input sequence: {input_seq_len} frames
- Output sequence: {output_seq_len} frames

## Performance (Test Set)
### LSTM
- Mean L2 Error: {lstm_metrics.get('mean_l2', 0):.4f} yards
- Classification Accuracy: {lstm_metrics.get('class_accuracy', 0):.3f}
- Latency: {lstm_metrics.get('latency_ms', 0):.2f} ms

### Transformer
- Mean L2 Error: {transformer_metrics.get('mean_l2', 0):.4f} yards
- Classification Accuracy: {transformer_metrics.get('class_accuracy', 0):.3f}
- Latency: {transformer_metrics.get('latency_ms', 0):.2f} ms

## Limitations
- Requires full input sequence (no partial inputs)
- Classification head may compete with trajectory task
- Quadratic attention cost for Transformer

## Ethical Considerations
- Trajectory predictions should not be used for player evaluation
- Model uncertainty should inform confidence in predictions
"""


def generate_generative_model_card(ablation_results: dict, config: dict,
                                   latent_dim: int) -> str:
    """Generate model card for generative augmentation."""
    results_str = ""
    for ratio, metrics in ablation_results.items():
        results_str += f"- {ratio:.0%} augmentation: Accuracy = {metrics.get('accuracy', 0):.3f}\n"
    
    return f"""# Model Card: Generative Augmentation (VAE-style)

## Model Description
PCA-based latent space sampler for data augmentation.
- Learns latent distribution from training data
- Generates synthetic samples via latent space sampling
- Labels assigned via nearest-neighbor matching

## Architecture
- Latent dimension: {latent_dim}
- Encoding: PCA via SVD
- Sampling: Gaussian in latent space
- Decoding: Linear projection back to data space

## Augmentation Ablation Results
{results_str}

## Intended Use
- Augment minority class samples
- Increase training data diversity
- Regularization via synthetic examples

## Limitations
- Synthetic samples may not capture rare edge cases
- Nearest-neighbor labels may propagate existing biases
- Limited to linear latent space

## Ethical Considerations
- Synthetic data clearly flagged in all experiments
- No re-identification risk from synthetic trajectories
- Augmentation artifacts may affect model behavior
"""


def run_fairness_analysis(X_test: np.ndarray, Y_test: np.ndarray, 
                          metadata: list, models: dict, log_fn=None) -> pd.DataFrame:
    """
    Analyze model performance across player positions.
    
    Args:
        X_test: Test features
        Y_test: Test labels
        metadata: List of dicts with player_position
        models: Dict of {name: adapter} with predict method
        
    Returns:
        DataFrame with fairness metrics by position
    """
    if log_fn:
        log_fn("Running fairness analysis by player position...")
    
    # Get positions
    positions = [m.get('player_position', 'UNK') for m in metadata]
    unique_positions = list(set(positions))
    
    results = []
    
    for pos in unique_positions:
        mask = np.array([p == pos for p in positions])
        if mask.sum() < 2:
            continue
        
        X_pos = X_test[mask]
        Y_pos = Y_test[mask]
        
        row = {
            'position': pos,
            'n_samples': int(mask.sum()),
            'positive_rate': float(Y_pos.mean())
        }
        
        # Evaluate each model
        for name, adapter in models.items():
            pred = adapter.predict(X_pos)
            row[f'{name}_accuracy'] = accuracy_score(Y_pos, pred)
        
        results.append(row)
    
    return pd.DataFrame(results)


def generate_ethics_documentation(seed: int, config_hash: str, data_hash: str,
                                  catch_radius: float) -> str:
    """Generate ethics and risk assessment documentation."""
    return f"""# Ethics Documentation: NFL Trajectory Prediction System

## 1. Bias Sources

### Sampling Bias
- Data limited to specific weeks/seasons
- May not generalize to other weeks, seasons, or weather conditions
- Player performance varies throughout season

### Label Leakage
- Ball landing position known during training
- Real-world deployment would require prediction of ball landing

### Augmentation Artifacts
- Synthetic data may not capture rare edge cases
- Nearest-neighbor label assignment may propagate existing biases

## 2. Privacy Considerations

### Data Retention
- No PII stored in model features
- Player positions anonymized via position codes

### Synthetic Data
- Generated samples clearly flagged in run logs
- No re-identification risk from synthetic trajectories

## 3. Transparency

### Model Cards
- Model cards provided for each component (Classical, Sequential, Generative)
- Each documents intended use, limitations, and metrics

### Logging
- All experiments logged to run_log.csv
- Includes hyperparameters, metrics, and timestamps

## 4. Reproducibility

### Seed Control
- Global seed: {seed}
- Applied to numpy, torch, sklearn

### Configuration Management
- All hyperparameters stored in config files
- Config hash: {config_hash}

### Data Immutability
- Data hash: {data_hash}
- Original data preserved, derived data gitignored

## 5. Safety Considerations

### Uncertainty Quantification
- MC Dropout provides uncertainty estimates
- High uncertainty should trigger human review

### Human-in-the-Loop
- System designed for decision support, not autonomous decisions
- Coaches/analysts should validate predictions

### Confidence Thresholds
- Predictions below confidence threshold flagged for review
- Catch radius threshold: {catch_radius} yards

## 6. Intended Use

### Appropriate Uses
- Game strategy analysis
- Player development insights
- Post-game performance review

### Inappropriate Uses
- Real-time automated decisions without human oversight
- Player evaluation for contracts/trades
- Gambling or betting applications
"""


def save_model_cards(output_dir: str, cards: dict):
    """Save model cards to files."""
    cards_dir = os.path.join(output_dir, 'model_cards')
    os.makedirs(cards_dir, exist_ok=True)
    
    for name, content in cards.items():
        filepath = os.path.join(cards_dir, f'{name}_model_card.md')
        with open(filepath, 'w') as f:
            f.write(content)
    
    return cards_dir


def save_ethics_doc(output_dir: str, content: str):
    """Save ethics documentation to file."""
    filepath = os.path.join(output_dir, 'ethics_documentation.md')
    with open(filepath, 'w') as f:
        f.write(content)
    return filepath
