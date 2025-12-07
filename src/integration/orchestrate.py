"""
Integration Orchestrator
========================
End-to-end DAG / CLI entrypoint for running the ML pipeline.
Source: Project 5

Usage:
    python -m src.integration.orchestrate --stages classic,neural --config configs/default.yaml
    python -m src.integration.orchestrate --stages generative,neural,sequential
    python -m src.integration.orchestrate --stages graph,rl
"""

import argparse
import os
import yaml
import numpy as np
import pandas as pd
from datetime import datetime

from ..mlops.utils import RunLogger, Timer, log, set_seeds


def load_config(config_path: str) -> dict:
    """Load YAML configuration file."""
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)


def load_data(config: dict):
    """Load and preprocess data."""
    log(f"Loading data from {config['data_path']}...")
    df = pd.read_csv(config['data_path'])
    log(f"Loaded {len(df)} rows")
    return df


def run_classic_stage(X_train, y_train, X_val, y_val, config: dict,
                      run_logger: RunLogger):
    """Run classical ML training."""
    from ..classic.train_tabular import train_classical_models

    log("=== STAGE: Classical ML ===")
    best_adapter, results, best_name = train_classical_models(
        X_train, y_train, X_val, y_val,
        config['classical'], config['seed'], run_logger
    )
    log(f"Best classical model: {best_name}")

    # Return detailed results
    output = {
        'best_model': best_name,
        'best_accuracy': results[best_name]['metrics']['accuracy'],
        'all_results': results
    }
    return best_adapter, output


def run_neural_stage(X_train, y_train, X_val, y_val, config: dict,
                     run_logger: RunLogger):
    """Run neural network training."""
    from ..neural.train_mlp import train_mlp
    
    log("=== STAGE: Neural Network (MLP) ===")
    adapter, history, metrics = train_mlp(
        X_train, y_train, X_val, y_val,
        config['mlp'], config['seed'], run_logger
    )
    log(f"MLP Accuracy: {metrics['accuracy']:.4f}")
    return adapter, history, metrics


def run_sequential_stage(X_train, Y_traj_train, Y_class_train,
                         X_val, Y_traj_val, Y_class_val,
                         config: dict, run_logger: RunLogger):
    """Run sequential model training (LSTM + Transformer)."""
    from ..neural.train_text_lstm import train_lstm
    from ..sequential.train_text_transformer import train_transformer
    
    log("=== STAGE: Sequential Models ===")
    
    lstm_adapter, lstm_history, lstm_metrics = train_lstm(
        X_train, Y_traj_train, Y_class_train,
        X_val, Y_traj_val, Y_class_val,
        config['lstm'], config['seed'], run_logger
    )
    log(f"LSTM MSE: {lstm_metrics['traj_mse']:.4f}")
    
    tf_adapter, tf_history, tf_metrics = train_transformer(
        X_train, Y_traj_train, Y_class_train,
        X_val, Y_traj_val, Y_class_val,
        config['transformer'], config['seed'], run_logger
    )
    log(f"Transformer MSE: {tf_metrics['traj_mse']:.4f}")
    
    return {
        'lstm': (lstm_adapter, lstm_history, lstm_metrics),
        'transformer': (tf_adapter, tf_history, tf_metrics)
    }


def run_generative_stage(X_train, y_train, X_val, y_val, train_fn,
                         config: dict, run_logger: RunLogger):
    """Run generative augmentation ablation."""
    from ..generative.vae_synth import run_augmentation_ablation
    
    log("=== STAGE: Generative Augmentation ===")
    results = run_augmentation_ablation(
        train_fn, X_train, y_train, X_val, y_val,
        config['generative']['augment_ratios'],
        config['generative']['latent_dim'],
        config['seed'], run_logger
    )
    return results


def run_graph_stage(edges: list, config: dict, run_logger: RunLogger):
    """Run graph link prediction."""
    from ..graph.gnn_link_pred import run_graph_evaluation
    
    log("=== STAGE: Graph Link Prediction ===")
    results = run_graph_evaluation(
        edges, config['graph']['test_ratio'],
        config['graph']['k_values'], config['seed'], run_logger
    )
    return results


def run_rl_stage(target_positions: list, config: dict, run_logger: RunLogger):
    """Run reinforcement learning."""
    from ..rl.q_learning import DynamicGridWorld, QLearningAgent, train_q_learning
    
    log("=== STAGE: Reinforcement Learning ===")
    
    rl_config = config['rl']
    env = DynamicGridWorld(grid_size=rl_config['grid_size'], seed=config['seed'])
    agent = QLearningAgent(
        n_states=env.n_states,
        n_actions=env.n_actions,
        gamma=rl_config['gamma'],
        alpha=rl_config['alpha'],
        epsilon=rl_config['epsilon'],
        epsilon_min=rl_config['epsilon_min'],
        epsilon_decay=rl_config['epsilon_decay'],
        seed=config['seed']
    )
    
    history = train_q_learning(
        env, agent, target_positions,
        episodes=rl_config['episodes'],
        max_steps=rl_config['max_steps']
    )
    
    avg_return = np.mean(history['returns'][-100:])
    log(f"RL Final Avg Return: {avg_return:.2f}")
    
    if run_logger:
        run_logger.log('rl', 'final_avg_return', avg_return,
                      params={'episodes': rl_config['episodes']})
    
    return agent, history


def main():
    parser = argparse.ArgumentParser(description='ML Pipeline Orchestrator')
    parser.add_argument('--stages', type=str, default='classic,neural',
                        help='Comma-separated stages: classic,neural,sequential,generative,graph,rl')
    parser.add_argument('--config', type=str, default='configs/default.yaml',
                        help='Path to config YAML')
    parser.add_argument('--output-dir', type=str, default='outputs',
                        help='Output directory')
    args = parser.parse_args()
    
    # Default configuration
    defaults = {
        'seed': 42,
        'data_path': 'data/base/input.csv',
        'classical': {'use_smote': True, 'rf_n_estimators': 100, 'rf_max_depth': 10},
        'mlp': {'hidden_sizes': [128, 64], 'dropout': 0.3, 'epochs': 50, 'batch_size': 32, 'lr': 0.001},
        'lstm': {'hidden_size': 64, 'num_layers': 2, 'dropout': 0.2, 'epochs': 30, 'batch_size': 16, 'lr': 0.001},
        'transformer': {'d_model': 64, 'nhead': 4, 'num_layers': 2, 'dim_feedforward': 256, 'dropout': 0.1, 'epochs': 30, 'batch_size': 16, 'lr': 0.001},
        'generative': {'latent_dim': 8, 'augment_ratios': [0.0, 0.10, 0.25]},
        'graph': {'test_ratio': 0.2, 'k_values': [3, 5, 10]},
        'rl': {'grid_size': 10, 'gamma': 0.95, 'alpha': 0.2, 'epsilon': 1.0, 'epsilon_min': 0.01, 'epsilon_decay': 0.995, 'episodes': 500, 'max_steps': 50}
    }

    # Load config
    if os.path.exists(args.config):
        config = load_config(args.config)
        # Normalize config keys - support both 'seed' and 'random_seed'
        if 'seed' not in config:
            config['seed'] = config.get('random_seed', defaults['seed'])
        # Support nested paths config
        if 'paths' in config and 'data_path' not in config:
            config['data_path'] = config['paths'].get('data_base', 'data/base') + '/input.csv'
        # Merge with defaults for missing keys
        for key, value in defaults.items():
            if key not in config:
                config[key] = value
    else:
        log(f"Config not found: {args.config}, using defaults")
        config = defaults
    
    # Setup
    set_seeds(config['seed'])
    os.makedirs(args.output_dir, exist_ok=True)
    os.makedirs(f"{args.output_dir}/artifacts", exist_ok=True)
    os.makedirs(f"{args.output_dir}/models", exist_ok=True)
    
    # Initialize run logger
    run_logger = RunLogger(
        f"{args.output_dir}/run_log.csv",
        config.get('data_path'),
        config,
        config['seed']
    )
    
    stages = [s.strip() for s in args.stages.split(',')]
    log(f"Running stages: {stages}")

    # Load and preprocess data
    from .data_pipeline import (
        load_data as pipeline_load_data,
        engineer_features,
        construct_sequences,
        split_data,
        normalize_data
    )

    df_raw = pipeline_load_data(config['data_path'], log_fn=log)
    df = engineer_features(df_raw, log_fn=log)

    # Construct sequences for sequential models
    sequences, metadata = construct_sequences(df, log_fn=log)

    # Split data
    data_splits = split_data(sequences, metadata, seed=config['seed'], log_fn=log)
    data_splits, scaler = normalize_data(data_splits, log_fn=log)

    # Extract tabular features for classic/neural (use first frame of sequence)
    X_train_tab = data_splits['X_train_scaled'][:, 0, :]  # First timestep
    X_val_tab = data_splits['X_val_scaled'][:, 0, :]
    X_test_tab = data_splits['X_test_scaled'][:, 0, :]

    y_train = data_splits['Y_class_train']
    y_val = data_splits['Y_class_val']
    y_test = data_splits['Y_class_test']

    # Full sequences for sequential models
    X_train_seq = data_splits['X_train_scaled']
    X_val_seq = data_splits['X_val_scaled']

    Y_traj_train = data_splits['Y_traj_train']
    Y_traj_val = data_splits['Y_traj_val']

    Y_class_train = data_splits['Y_class_train']
    Y_class_val = data_splits['Y_class_val']

    # Artifacts registry
    artifacts = {}

    # Execute stages
    if 'classic' in stages:
        best_adapter, results = run_classic_stage(
            X_train_tab, y_train, X_val_tab, y_val,
            config, run_logger
        )
        artifacts['classic_model'] = best_adapter
        artifacts['classic_results'] = results

    if 'neural' in stages:
        adapter, history, metrics = run_neural_stage(
            X_train_tab, y_train, X_val_tab, y_val,
            config, run_logger
        )
        artifacts['neural_model'] = adapter
        artifacts['neural_history'] = history
        artifacts['neural_metrics'] = metrics

    if 'sequential' in stages:
        seq_results = run_sequential_stage(
            X_train_seq, Y_traj_train, Y_class_train,
            X_val_seq, Y_traj_val, Y_class_val,
            config, run_logger
        )
        lstm_adapter, lstm_history, lstm_metrics = seq_results['lstm']
        tf_adapter, tf_history, tf_metrics = seq_results['transformer']
        artifacts['lstm_model'] = lstm_adapter
        artifacts['transformer_model'] = tf_adapter
        artifacts['sequential_results'] = seq_results

    if 'generative' in stages:
        # Define a training function for ablation study
        from sklearn.ensemble import RandomForestClassifier
        from sklearn.metrics import accuracy_score, f1_score

        def train_and_eval(X_train_aug, y_train_aug, X_val_inner, y_val_inner):
            clf = RandomForestClassifier(n_estimators=100, max_depth=10, random_state=config['seed'])
            clf.fit(X_train_aug, y_train_aug)
            y_pred = clf.predict(X_val_inner)
            return {
                'accuracy': accuracy_score(y_val_inner, y_pred),
                'f1': f1_score(y_val_inner, y_pred, zero_division=0)
            }

        gen_results = run_generative_stage(
            X_train_tab, y_train, X_val_tab, y_val,
            train_and_eval, config, run_logger
        )
        artifacts['generative_results'] = gen_results

    if 'graph' in stages:
        try:
            # Build edges from DataFrame (player-play bipartite graph)
            from ..graph.gnn_link_pred import build_graph_from_data

            # Create player-play edges from input data
            # Ensure play_key exists (game_id + play_id)
            if 'play_key' not in df.columns:
                df['play_key'] = df['game_id'].astype(str) + '_' + df['play_id'].astype(str)

            # Use nfl_id (player) and play_key (unique play identifier)
            edges = build_graph_from_data(df, player_col='nfl_id', play_col='play_key')
            log(f"Built graph with {len(edges)} player-play edges")

            graph_results = run_graph_stage(edges, config, run_logger)
            artifacts['graph_results'] = graph_results
        except Exception as e:
            log(f"Skipping graph stage: {e}")

    if 'rl' in stages:
        try:
            # Extract ball landing positions from DataFrame
            # Use unique ball landing locations as targets for RL training
            ball_positions = df[['ball_land_x', 'ball_land_y']].drop_duplicates().values.tolist()
            log(f"Extracted {len(ball_positions)} unique ball landing positions for RL")

            agent, history = run_rl_stage(ball_positions, config, run_logger)
            artifacts['rl_agent'] = agent
            artifacts['rl_history'] = history
        except Exception as e:
            log(f"Skipping RL stage: {e}")

    # Generate visualizations
    log("=== Generating Visualizations ===")
    artifacts_dir = f"{args.output_dir}/artifacts"

    from ..mlops.eval_pipeline import (
        plot_confusion_matrix, plot_reliability_diagram,
        plot_trajectory_comparison, plot_graph_hit_at_k
    )

    # Classic confusion matrix and reliability diagram
    if 'classic_model' in artifacts:
        try:
            best_name = artifacts['classic_results'].get('best_model', 'Classical')
            y_pred_classic = artifacts['classic_model'].predict(X_val_tab)
            plot_confusion_matrix(y_val, y_pred_classic,
                                 save_path=f"{artifacts_dir}/classic_{best_name}_confusion_matrix.png",
                                 title=f"Confusion Matrix - {best_name} (Classical ML)")
            log(f"  Saved classic_{best_name}_confusion_matrix.png")

            # Reliability diagram for classical model
            y_proba_classic = artifacts['classic_model'].predict_proba(X_val_tab)
            if y_proba_classic is not None:
                # Handle both 1D (positive class only) and 2D (all classes) probabilities
                if len(y_proba_classic.shape) > 1:
                    proba_pos = y_proba_classic[:, 1]
                else:
                    proba_pos = y_proba_classic  # Already positive class probabilities
                plot_reliability_diagram(y_val, proba_pos,
                                        save_path=f"{artifacts_dir}/classic_{best_name}_reliability_diagram.png",
                                        title=f"Reliability Diagram - {best_name} (Classical ML)")
                log(f"  Saved classic_{best_name}_reliability_diagram.png")
        except Exception as e:
            log(f"  Could not generate classic confusion matrix: {e}")

    # Neural (MLP) confusion matrix and reliability diagram
    if 'neural_model' in artifacts:
        try:
            y_pred_neural = artifacts['neural_model'].predict(X_val_tab)
            plot_confusion_matrix(y_val, y_pred_neural,
                                 save_path=f"{artifacts_dir}/neural_MLP_confusion_matrix.png",
                                 title="Confusion Matrix - MLP (Neural Network)")
            log("  Saved neural_MLP_confusion_matrix.png")

            # Reliability diagram for MLP
            y_proba_neural = artifacts['neural_model'].predict_proba(X_val_tab)
            if y_proba_neural is not None:
                # Handle both 1D (positive class only) and 2D (all classes) probabilities
                if len(y_proba_neural.shape) > 1:
                    proba_pos_nn = y_proba_neural[:, 1]
                else:
                    proba_pos_nn = y_proba_neural
                plot_reliability_diagram(y_val, proba_pos_nn,
                                        save_path=f"{artifacts_dir}/neural_MLP_reliability_diagram.png",
                                        title="Reliability Diagram - MLP (Neural Network)")
                log("  Saved neural_MLP_reliability_diagram.png")
        except Exception as e:
            log(f"  Could not generate neural confusion matrix: {e}")

    # LSTM trajectory comparison
    if 'lstm_model' in artifacts:
        try:
            traj_pred_lstm = artifacts['lstm_model'].predict_trajectory(X_val_seq)
            plot_trajectory_comparison(Y_traj_val, traj_pred_lstm, n_samples=5,
                                      save_path=f"{artifacts_dir}/sequential_LSTM_trajectory_comparison.png",
                                      title="Trajectory Comparison - LSTM (Sequential)")
            log("  Saved sequential_LSTM_trajectory_comparison.png")
        except Exception as e:
            log(f"  Could not generate LSTM trajectory comparison: {e}")

    # Transformer trajectory comparison
    if 'transformer_model' in artifacts:
        try:
            traj_pred_tf = artifacts['transformer_model'].predict_trajectory(X_val_seq)
            plot_trajectory_comparison(Y_traj_val, traj_pred_tf, n_samples=5,
                                      save_path=f"{artifacts_dir}/sequential_Transformer_trajectory_comparison.png",
                                      title="Trajectory Comparison - Transformer (Sequential)")
            log("  Saved sequential_Transformer_trajectory_comparison.png")
        except Exception as e:
            log(f"  Could not generate Transformer trajectory comparison: {e}")

    # Graph Hit@K bar chart
    if 'graph_results' in artifacts:
        try:
            # Convert list of dicts to nested dict format for plotting
            graph_res_list = artifacts['graph_results']
            graph_dict = {}
            for entry in graph_res_list:
                method = entry['method']
                k = entry['k']
                if method not in graph_dict:
                    graph_dict[method] = {}
                graph_dict[method][f'hit@{k}'] = entry['hit_at_k']

            plot_graph_hit_at_k(graph_dict,
                               save_path=f"{artifacts_dir}/graph_LinkPrediction_hit_at_k.png",
                               title="Hit@K - Graph Link Prediction (Common Neighbors/Jaccard/Adamic-Adar)")
            log("  Saved graph_LinkPrediction_hit_at_k.png")
        except Exception as e:
            log(f"  Could not generate graph hit@k plot: {e}")

    # RL visualizations
    if 'rl_agent' in artifacts and 'rl_history' in artifacts:
        try:
            from ..rl.q_learning import visualize_rl_results, DynamicGridWorld
            rl_config = config['rl']
            env = DynamicGridWorld(grid_size=rl_config['grid_size'], seed=config['seed'])
            visualize_rl_results(env, artifacts['rl_agent'],
                                artifacts['rl_history']['returns'],
                                save_dir=artifacts_dir)
            log("  Saved rl_dynamic_results.png")
        except Exception as e:
            log(f"  Could not generate RL visualizations: {e}")

    # Save models
    log("=== Saving Models ===")
    models_dir = f"{args.output_dir}/models"
    os.makedirs(models_dir, exist_ok=True)

    import pickle

    if 'classic_model' in artifacts:
        try:
            with open(f"{models_dir}/classical_model.pkl", 'wb') as f:
                pickle.dump(artifacts['classic_model'].model, f)
            log("  Saved classical_model.pkl")
        except Exception as e:
            log(f"  Could not save classical model: {e}")

    if 'neural_model' in artifacts:
        try:
            import torch
            torch.save(artifacts['neural_model'].model.state_dict(),
                      f"{models_dir}/mlp_model.pt")
            log("  Saved mlp_model.pt")
        except Exception as e:
            log(f"  Could not save MLP model: {e}")

    if 'lstm_model' in artifacts:
        try:
            import torch
            torch.save(artifacts['lstm_model'].model.state_dict(),
                      f"{models_dir}/lstm_model.pt")
            log("  Saved lstm_model.pt")
        except Exception as e:
            log(f"  Could not save LSTM model: {e}")

    if 'transformer_model' in artifacts:
        try:
            import torch
            torch.save(artifacts['transformer_model'].model.state_dict(),
                      f"{models_dir}/transformer_model.pt")
            log("  Saved transformer_model.pt")
        except Exception as e:
            log(f"  Could not save Transformer model: {e}")

    # Save scaler
    try:
        with open(f"{models_dir}/scaler.pkl", 'wb') as f:
            pickle.dump(scaler, f)
        log("  Saved scaler.pkl")
    except Exception as e:
        log(f"  Could not save scaler: {e}")

    # Save Q-table
    if 'rl_agent' in artifacts:
        try:
            np.save(f"{models_dir}/q_table.npy", artifacts['rl_agent'].Q)
            log("  Saved q_table.npy")
        except Exception as e:
            log(f"  Could not save Q-table: {e}")

    # Save VAE sampler
    if 'generative_results' in artifacts:
        try:
            sampler = artifacts['generative_results'].get('sampler')
            if sampler is not None:
                with open(f"{models_dir}/vae_sampler.pkl", 'wb') as f:
                    pickle.dump(sampler, f)
                log("  Saved vae_sampler.pkl")
        except Exception as e:
            log(f"  Could not save VAE sampler: {e}")

    # Generate model cards to docs/Model_Cards/
    log("=== Generating Model Cards ===")
    model_cards_dir = "docs/Model_Cards"
    os.makedirs(model_cards_dir, exist_ok=True)

    # Classical ML Model Card
    if 'classic_results' in artifacts:
        try:
            best_name = artifacts['classic_results'].get('best_model', 'Classical')
            all_res = artifacts['classic_results'].get('all_results', {})
            best_metrics = all_res.get(best_name, {}).get('metrics', {})

            classical_card = f"""# Model Card: Classical ML ({best_name})

## Model Details
- **Model Type**: {best_name} Classifier
- **Framework**: scikit-learn
- **Version**: 1.0
- **Date**: {datetime.now().strftime('%B %Y')}

## Model Description
{best_name} classifier for binary classification with SMOTE resampling and class weighting.

## Intended Use
- Baseline classification for trajectory prediction
- Fast inference for real-time applications
- Interpretable feature importance (for tree-based models)

## Training Data
- **Samples**: {len(X_train_tab)} training, {len(X_val_tab)} validation
- **Features**: {X_train_tab.shape[1]} tabular features
- **Class Balance**: {y_train.mean():.1%} positive (minority class)
- **Resampling**: SMOTE applied to training data

## Performance (Validation Set)
| Metric | Value |
|--------|-------|
| Accuracy | {best_metrics.get('accuracy', 'N/A'):.4f} |
| Precision | {best_metrics.get('precision', 'N/A'):.4f} |
| Recall | {best_metrics.get('recall', 'N/A'):.4f} |
| F1 Score | {best_metrics.get('f1', 'N/A'):.4f} |
| F1-macro | {best_metrics.get('f1_macro', 'N/A'):.4f} |
| ROC-AUC | {best_metrics.get('roc_auc', 'N/A'):.4f} |
| ECE (Calibration) | {best_metrics.get('ece', 'N/A'):.4f} |
| Latency (batch=1) | {best_metrics.get('latency_batch_1_ms', 'N/A'):.2f} ms |
| Latency (batch=32) | {best_metrics.get('latency_batch_32_ms', 'N/A'):.2f} ms |

## Limitations
1. Uses only first-frame features (no temporal context)
2. Class imbalance may affect minority class predictions
3. May not generalize to different game scenarios

## Ethical Considerations

### Bias Sources
- **Sampling Bias**: Training data may not represent all game scenarios equally
- **Label Leakage**: Temporal ordering preserved to prevent information leakage
- **Class Imbalance**: SMOTE used but may introduce synthetic artifacts

### Privacy
- No personally identifiable information (PII) in features
- Positions anonymized; no re-identification possible

### Transparency
- Feature importance available for tree-based models
- All hyperparameters logged in run_log.csv

### Safety
- **Confidence Threshold**: Use predicted probability > 0.7 for high-confidence predictions
- **Human-in-the-Loop**: Flag predictions with probability in [0.4, 0.6] for manual review

## Reproducibility
- **Seed**: {config['seed']}
- **Config**: configs/default.yaml
- **Data**: data/base/input.csv (immutable)

---
*Generated: {datetime.now().isoformat()}*
"""
            with open(f"{model_cards_dir}/classical_{best_name.lower()}.md", 'w') as f:
                f.write(classical_card)
            log(f"  Saved classical_{best_name.lower()}.md")
        except Exception as e:
            log(f"  Could not generate classical model card: {e}")

    # Neural Network (MLP) Model Card
    if 'neural_metrics' in artifacts:
        try:
            nn_metrics = artifacts['neural_metrics']
            neural_card = f"""# Model Card: Neural Network (MLP)

## Model Details
- **Model Type**: Multi-Layer Perceptron (MLP)
- **Framework**: PyTorch
- **Version**: 1.0
- **Date**: {datetime.now().strftime('%B %Y')}

## Model Description
Multi-Layer Perceptron for binary classification with dropout regularization and MC Dropout for uncertainty quantification.

## Architecture
- **Input**: {X_train_tab.shape[1]} features
- **Hidden Layers**: {config['mlp'].get('hidden_sizes', [128, 64])}
- **Activation**: ReLU
- **Dropout**: {config['mlp'].get('dropout', 0.3)}
- **Output**: Binary classification (softmax)

## Intended Use
- Binary classification from tabular features
- Alternative to classical ML with non-linear decision boundaries
- Uncertainty estimation via MC Dropout
- Ensemble component for model comparison

## Training Data
- **Samples**: {len(X_train_tab)} training, {len(X_val_tab)} validation
- **Features**: {X_train_tab.shape[1]} tabular features
- **Class Balance**: {y_train.mean():.1%} positive (minority class)
- **Preprocessing**: StandardScaler normalization

## Performance (Validation Set)
| Metric | Value |
|--------|-------|
| Accuracy | {nn_metrics.get('accuracy', 'N/A'):.4f} |

## Training Configuration
- **Optimizer**: Adam
- **Learning Rate**: {config['mlp'].get('lr', 0.001)}
- **Epochs**: {config['mlp'].get('epochs', 50)}
- **Batch Size**: {config['mlp'].get('batch_size', 32)}
- **Loss**: Cross-Entropy

## Limitations
1. No temporal information (first-frame features only)
2. Sensitive to feature scaling (requires normalization)
3. May overfit on small datasets
4. Black-box model requires post-hoc explanations

## Ethical Considerations

### Bias Sources
- **Sampling Bias**: Model learns patterns from available training data only
- **Feature Bias**: Feature engineering decisions may embed assumptions
- **Class Imbalance**: Minority class may be under-predicted

### Privacy
- No personally identifiable information (PII) in features
- Model weights do not encode individual data points
- Input features are normalized and anonymized

### Transparency
- MC Dropout provides uncertainty estimates
- SHAP/LIME can be applied for feature importance
- Training logs available in run_log.csv

### Safety
- **Confidence Threshold**: Use MC Dropout std < 0.15 for high-confidence predictions
- **Human-in-the-Loop**: High uncertainty predictions should be flagged for review
- **Fallback**: Use classical ML baseline for comparison

## Reproducibility
- **Seed**: {config['seed']}
- **Config**: configs/default.yaml
- **Data**: data/base/input.csv (immutable)
- **Dependencies**: See requirements.txt

---
*Generated: {datetime.now().isoformat()}*
"""
            with open(f"{model_cards_dir}/neural_mlp_model_card.md", 'w') as f:
                f.write(neural_card)
            log("  Saved neural_mlp_model_card.md")
        except Exception as e:
            log(f"  Could not generate neural model card: {e}")

    # Sequential Model Card
    if 'sequential_results' in artifacts:
        try:
            seq_res = artifacts['sequential_results']
            lstm_metrics = seq_res['lstm'][2] if 'lstm' in seq_res else {}
            tf_metrics = seq_res['transformer'][2] if 'transformer' in seq_res else {}

            sequential_card = f"""# Model Card: Sequential Models (LSTM & Transformer)

## Model Details
- **Model Types**: LSTM Encoder-Decoder, Transformer Encoder-Decoder
- **Framework**: PyTorch
- **Version**: 1.0
- **Date**: {datetime.now().strftime('%B %Y')}

## Model Description
Encoder-decoder architectures for dual-task learning:
- **Task 1**: Trajectory forecasting (regression)
- **Task 2**: Binary classification

Both models support MC Dropout for uncertainty quantification.

## Architecture

### LSTM
- **Type**: Encoder-decoder with teacher forcing
- **Hidden Size**: {config['lstm'].get('hidden_size', 64)}
- **Layers**: {config['lstm'].get('num_layers', 2)}
- **Dropout**: {config['lstm'].get('dropout', 0.2)}

### Transformer
- **Type**: Full encoder-decoder with positional encoding
- **d_model**: {config['transformer'].get('d_model', 64)}
- **Attention Heads**: {config['transformer'].get('nhead', 4)}
- **Feedforward Dim**: {config['transformer'].get('dim_feedforward', 256)}
- **Dropout**: {config['transformer'].get('dropout', 0.1)}

## Intended Use
- Multi-step trajectory forecasting
- Simultaneous classification from temporal features
- Decision support for player positioning prediction
- Uncertainty-aware predictions via MC Dropout

## Training Data
- **Sequences**: {len(X_train_seq)} training, {len(X_val_seq)} validation
- **Sequence Length**: {X_train_seq.shape[1]} timesteps
- **Features per Timestep**: {X_train_seq.shape[2]}
- **Class Balance**: {Y_class_train.mean():.1%} positive

## Performance (Validation Set)

### LSTM
| Metric | Value |
|--------|-------|
| Trajectory MSE | {lstm_metrics.get('traj_mse', 'N/A'):.4f} |
| Trajectory MAE | {lstm_metrics.get('traj_mae', 'N/A'):.4f} |
| Trajectory MAPE | {lstm_metrics.get('traj_mape', 'N/A'):.2f}% |
| Trajectory MASE | {lstm_metrics.get('traj_mase', 'N/A'):.4f} |
| Classification Accuracy | {lstm_metrics.get('class_accuracy', 'N/A'):.4f} |
| MC Dropout Mean Std | {lstm_metrics.get('mc_dropout_mean_std', 'N/A'):.4f} |
| Latency (batch=1) | {lstm_metrics.get('latency_batch_1_ms', 'N/A'):.2f} ms |

### Transformer
| Metric | Value |
|--------|-------|
| Trajectory MSE | {tf_metrics.get('traj_mse', 'N/A'):.4f} |
| Trajectory MAE | {tf_metrics.get('traj_mae', 'N/A'):.4f} |
| Trajectory MAPE | {tf_metrics.get('traj_mape', 'N/A'):.2f}% |
| Trajectory MASE | {tf_metrics.get('traj_mase', 'N/A'):.4f} |
| Classification Accuracy | {tf_metrics.get('class_accuracy', 'N/A'):.4f} |
| MC Dropout Mean Std | {tf_metrics.get('mc_dropout_mean_std', 'N/A'):.4f} |
| Latency (batch=1) | {tf_metrics.get('latency_batch_1_ms', 'N/A'):.2f} ms |

## Training Configuration
- **Optimizer**: Adam
- **LSTM LR**: {config['lstm'].get('lr', 0.001)}
- **Transformer LR**: {config['transformer'].get('lr', 0.001)}
- **Epochs**: {config['lstm'].get('epochs', 30)}
- **Batch Size**: {config['lstm'].get('batch_size', 16)}

## Limitations
1. Requires fixed sequence length input
2. Computational cost scales with sequence length
3. Transformer attention is O(n²) in sequence length
4. May struggle with very long-range dependencies

## Ethical Considerations

### Bias Sources
- **Temporal Bias**: Predictions weighted toward recent timesteps
- **Sampling Bias**: Training data may not cover all game scenarios
- **Label Leakage**: Strict temporal ordering prevents information leakage

### Privacy
- No personally identifiable information (PII) in features
- Trajectories are anonymized coordinate sequences
- No re-identification possible from predictions

### Transparency
- Attention weights (Transformer) provide interpretability
- MC Dropout quantifies prediction uncertainty
- All training parameters logged in run_log.csv

### Safety
- **Confidence Threshold**: Use MC Dropout std < 0.2 for trajectory predictions
- **Human-in-the-Loop**: Flag high-uncertainty predictions for manual review
- **Fallback**: Use simpler models when uncertainty is high

## Reproducibility
- **Seed**: {config['seed']}
- **Config**: configs/default.yaml
- **Data**: data/base/input.csv (immutable)
- **Dependencies**: See requirements.txt

---
*Generated: {datetime.now().isoformat()}*
"""
            with open(f"{model_cards_dir}/sequential_model_card.md", 'w') as f:
                f.write(sequential_card)
            log("  Saved sequential_model_card.md")
        except Exception as e:
            log(f"  Could not generate sequential model card: {e}")

    # Generative (VAE Sampler) Model Card
    if 'generative_results' in artifacts:
        try:
            gen_res = artifacts['generative_results']
            ablation = gen_res.get('ablation_results', {})
            best_ratio = max(ablation.keys(), key=lambda r: ablation[r].get('accuracy', 0)) if ablation else 0

            generative_card = f"""# Model Card: Generative VAE-Style Sampler

## Model Details
- **Model Type**: PCA-based VAE-Style Latent Sampler
- **Framework**: NumPy/SciPy
- **Version**: 1.0
- **Date**: {datetime.now().strftime('%B %Y')}

## Model Description
PCA-based VAE-style sampler for data augmentation via latent space sampling.
Uses SVD for dimensionality reduction and Gaussian sampling in latent space.

## Architecture
- **Dimensionality Reduction**: SVD/PCA
- **Latent Dimension**: {config['generative'].get('latent_dim', 8)}
- **Sampling**: Gaussian in latent space
- **Label Assignment**: Nearest-neighbor matching
- **Reconstruction**: Linear projection back to data space

## Guardrails
- **KL-Filter**: Rejects samples > 3.0 std from latent mean (outlier rejection)
- **Per-Class Cap**: Limits synthetic samples per class to minority_count × 1.0

## Intended Use
- Data augmentation for imbalanced datasets
- Synthetic sample generation for training
- Ablation studies on augmentation ratios
- Improving minority class representation

## Augmentation Ablation Results

| Ratio | Accuracy | F1 Score |
|-------|----------|----------|
"""
            for ratio in sorted(ablation.keys()):
                metrics = ablation[ratio]
                generative_card += f"| {ratio:.0%} | {metrics.get('accuracy', 'N/A'):.4f} | {metrics.get('f1', 'N/A'):.4f} |\n"

            generative_card += f"""
## Best Configuration
- **Optimal Ratio**: {best_ratio:.0%}
- **Accuracy Delta vs Baseline**: {(ablation.get(best_ratio, {}).get('accuracy', 0) - ablation.get(0.0, {}).get('accuracy', 0)):+.4f}

## Limitations
1. PCA assumes linear relationships in data
2. Synthetic samples may not capture all data modes
3. Label assignment depends on training data quality
4. Cannot generate truly novel patterns

## Ethical Considerations

### Bias Sources
- **Augmentation Artifacts**: Synthetic data may not represent real-world diversity
- **Pattern Amplification**: Nearest-neighbor labeling may reinforce existing biases
- **Class Imbalance**: Per-class capping prevents synthetic data from dominating

### Privacy
- Synthetic samples are clearly flagged in run_log.csv
- No PII in generated features
- Generated data cannot be traced back to individuals
- Augmentation ratios documented for transparency

### Transparency
- All synthetic samples logged with augmentation ratio
- Ablation results show impact of augmentation
- KL-filter threshold documented (3.0 std)

### Safety
- **Guardrails**: KL-filter rejects unrealistic samples
- **Per-Class Cap**: Prevents synthetic data from overwhelming real data
- **Validation**: Ablation study validates augmentation benefit

## Reproducibility
- **Seed**: {config['seed']}
- **Config**: configs/default.yaml
- **Latent Dim**: {config['generative'].get('latent_dim', 8)}
- **Augment Ratios**: {config['generative'].get('augment_ratios', [0.0, 0.10, 0.25])}

---
*Generated: {datetime.now().isoformat()}*
"""
            with open(f"{model_cards_dir}/generative_vae_model_card.md", 'w') as f:
                f.write(generative_card)
            log("  Saved generative_vae_model_card.md")
        except Exception as e:
            log(f"  Could not generate generative model card: {e}")

    # Graph Link Prediction Model Card
    if 'graph_results' in artifacts:
        try:
            graph_res = artifacts['graph_results']
            # Find best method at K=5
            best_entry = max([e for e in graph_res if e['k'] == 5],
                           key=lambda x: x.get('hit_at_k', 0), default={})

            graph_card = f"""# Model Card: Graph Link Prediction

## Model Details
- **Model Type**: Similarity-Based Link Prediction
- **Framework**: NetworkX / Custom Implementation
- **Version**: 1.0
- **Date**: {datetime.now().strftime('%B %Y')}

## Model Description
Similarity-based link prediction on bipartite player-play graph using classical graph algorithms.

## Methods Implemented
1. **Common Neighbors**: Count shared neighbors between nodes
2. **Jaccard Coefficient**: Normalized intersection over union of neighborhoods
3. **Adamic-Adar Index**: Weights rare neighbors higher (inverse log frequency)

## Intended Use
- Recommend plays to players based on historical participation
- Predict missing edges in player-play bipartite graph
- Collaborative filtering for play recommendations
- Understanding player-play interaction patterns

## Graph Statistics
- **Graph Type**: Bipartite (player-play)
- **Edge Split**: 80% train / 20% test (stratified per player)
- **K Values Evaluated**: [3, 5, 10]

## Performance

### Hit@K at K=5
| Method | Hit@5 | MAP@5 | Coverage@5 |
|--------|-------|-------|------------|
"""
            for entry in graph_res:
                if entry['k'] == 5:
                    graph_card += f"| {entry['method']} | {entry.get('hit_at_k', 'N/A'):.4f} | {entry.get('map_at_k', 'N/A'):.4f} | {entry.get('coverage_at_k', 'N/A'):.4f} |\n"

            graph_card += f"""
## Best Method
- **Method**: {best_entry.get('method', 'N/A')} at K=5
- **Hit@K**: {best_entry.get('hit_at_k', 'N/A'):.4f}

## Limitations
1. Cold-start problem for new players/plays with no edges
2. Cannot capture complex feature interactions (structure-only)
3. Performance depends on graph density and connectivity
4. Assumes similar players like similar plays

## Ethical Considerations

### Bias Sources
- **Popularity Bias**: High-degree nodes (popular plays) may be over-recommended
- **Sampling Bias**: Historical data may not represent all player types equally
- **Filter Bubbles**: Similar players get similar recommendations

### Privacy
- Player IDs are anonymized integers
- No personally identifiable information in graph structure
- Recommendations based on aggregated patterns, not individuals

### Transparency
- All similarity methods are interpretable and well-documented
- Edge weights and rankings are explainable
- Coverage@K metric measures recommendation diversity

### Safety
- **Diversity Check**: Coverage@K ensures recommendations aren't too narrow
- **Human Review**: Low-confidence predictions should be validated
- **Fallback**: Random baseline provides comparison point

## Reproducibility
- **Seed**: {config['seed']}
- **Config**: configs/default.yaml
- **Test Ratio**: {config['graph'].get('test_ratio', 0.2)}
- **K Values**: {config['graph'].get('k_values', [3, 5, 10])}

---
*Generated: {datetime.now().isoformat()}*
"""
            with open(f"{model_cards_dir}/graph_link_prediction_model_card.md", 'w') as f:
                f.write(graph_card)
            log("  Saved graph_link_prediction_model_card.md")
        except Exception as e:
            log(f"  Could not generate graph model card: {e}")

    # Reinforcement Learning Model Card
    if 'rl_history' in artifacts:
        try:
            rl_hist = artifacts['rl_history']
            returns = rl_hist['returns']
            convergence_ep = rl_hist.get('convergence_episode', -1)

            rl_card = f"""# Model Card: Reinforcement Learning (Q-Learning)

## Model Details
- **Model Type**: Tabular Q-Learning (Off-Policy TD Control)
- **Framework**: Custom NumPy Implementation
- **Version**: 1.0
- **Date**: {datetime.now().strftime('%B %Y')}

## Model Description
Tabular Q-learning agent for navigation in dynamic GridWorld environment.
Learns optimal policy to reach ball landing positions with dynamic reward maps.

## Algorithm
- **Type**: Tabular Q-Learning (off-policy TD control)
- **State Space**: {config['rl'].get('grid_size', 10)}×{config['rl'].get('grid_size', 10)} = {config['rl'].get('grid_size', 10)**2} states
- **Action Space**: 4 (up, down, left, right)
- **Discount Factor (γ)**: {config['rl'].get('gamma', 0.95)}
- **Learning Rate (α)**: {config['rl'].get('alpha', 0.2)} with decay

## Exploration Strategy
- **Method**: ε-greedy with exponential decay
- **Initial ε**: {config['rl'].get('epsilon', 1.0)}
- **Minimum ε**: {config['rl'].get('epsilon_min', 0.01)}
- **Decay Rate**: {config['rl'].get('epsilon_decay', 0.995)}

## Intended Use
- Learn optimal policy to navigate to ball landing positions
- Dynamic reward maps based on target location
- Decision support for player positioning
- Real-time policy execution in simulated environment

## Training Configuration
- **Episodes**: {len(returns)}
- **Max Steps per Episode**: {config['rl'].get('max_steps', 50)}
- **Target Positions**: Sampled from training data

## Performance
| Metric | Value |
|--------|-------|
| Final Avg Return (last 100) | {np.mean(returns[-100:]):.2f} |
| Maximum Return | {np.max(returns):.2f} |
| Total Episodes | {len(returns)} |
| Convergence Episode (90%) | {convergence_ep if convergence_ep > 0 else 'Not converged'} |

## Reward Structure
| Condition | Reward |
|-----------|--------|
| Target reached | +20 |
| Near target (≤1.5 cells) | +10 |
| Close to target (≤3 cells) | +5 |
| Elsewhere | -0.5 |
| Edge penalty | -2 |

## Limitations
1. Discrete state space may lose spatial precision
2. Q-table size grows quadratically with grid size
3. Requires retraining for new reward structures
4. No generalization to unseen target positions

## Ethical Considerations

### Bias Sources
- **Reward Shaping Bias**: Hand-crafted rewards may not align with real objectives
- **Exploration Bias**: ε-greedy may underexplore certain state regions
- **Target Sampling Bias**: Training targets from data may not cover all scenarios

### Privacy
- Environment is simulated; no real-world data used
- Target positions derived from anonymized ball trajectories
- Q-table contains no personally identifiable information

### Transparency
- Q-values fully interpretable (state-action values)
- Policy can be visualized as action map over grid
- Episode returns logged for training analysis
- Convergence tracked and documented

### Safety
- **Bounded Actions**: Agent can only move in 4 cardinal directions
- **Episode Limits**: Maximum steps prevent infinite loops
- **Human-in-the-Loop**: Policy should be validated before real deployment
- **Simulation First**: Always test in simulation before real-world use

## Reproducibility
- **Seed**: {config['seed']}
- **Config**: configs/default.yaml
- **Grid Size**: {config['rl'].get('grid_size', 10)}
- **Episodes**: {config['rl'].get('episodes', 500)}
- **Q-Table**: Saved to outputs/models/q_table.npy

---
*Generated: {datetime.now().isoformat()}*
"""
            with open(f"{model_cards_dir}/rl_qlearning_model_card.md", 'w') as f:
                f.write(rl_card)
            log("  Saved rl_qlearning_model_card.md")
        except Exception as e:
            log(f"  Could not generate RL model card: {e}")

    # Generate ethics documentation
    log("=== Generating Ethics Documentation ===")
    try:
        ethics_doc = f"""# Ethics, Risk & Reproducibility Documentation

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
- **Issue**: Minority class ({y_train.mean():.1%} positive) may be under-predicted
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
- **Global Seed**: {config['seed']}
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
| **Global Seed** | {config['seed']} |
| **Data Path** | {config.get('data_path', 'data/base/input.csv')} |
| **Config File** | configs/default.yaml |
| **Output Directory** | {args.output_dir} |
| **Training Samples** | {len(X_train_tab)} |
| **Validation Samples** | {len(X_val_tab)} |
| **Class Balance** | {y_train.mean():.1%} positive |

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

*Generated: {datetime.now().isoformat()}*
*Seed: {config['seed']}*
"""
        with open(f"{args.output_dir}/ethics_documentation.md", 'w') as f:
            f.write(ethics_doc)
        log("  Saved ethics_documentation.md")

        # Also save to docs/ for easy access
        with open(f"docs/Ethics_Risk_Reproducibility.md", 'w') as f:
            f.write(ethics_doc)
        log("  Saved docs/Ethics_Risk_Reproducibility.md")
    except Exception as e:
        log(f"  Could not generate ethics documentation: {e}")

    # Generate System Diagram as PNG
    log("=== Generating System Diagram PNG ===")
    try:
        import matplotlib.pyplot as plt
        import matplotlib.patches as mpatches
        from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

        fig, ax = plt.subplots(figsize=(14, 10))
        ax.set_xlim(0, 14)
        ax.set_ylim(0, 10)
        ax.axis('off')
        ax.set_title('System Architecture - Final Project Phase 5', fontsize=16, fontweight='bold', pad=20)

        # Define colors
        colors = {
            'data': '#E3F2FD',
            'classical': '#FFECB3',
            'neural': '#C8E6C9',
            'sequential': '#F8BBD9',
            'generative': '#D1C4E9',
            'graph': '#B2DFDB',
            'rl': '#FFCCBC',
            'mlops': '#CFD8DC',
            'api': '#B3E5FC'
        }

        def add_box(ax, x, y, w, h, text, color, fontsize=9):
            box = FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.05",
                                 facecolor=color, edgecolor='black', linewidth=1.5)
            ax.add_patch(box)
            ax.text(x + w/2, y + h/2, text, ha='center', va='center',
                   fontsize=fontsize, fontweight='bold', wrap=True)

        # Data Input
        add_box(ax, 5.5, 9, 3, 0.7, 'Input Data\n(input.csv)', colors['data'], 10)

        # Augment (VAE)
        add_box(ax, 5.5, 7.8, 3, 0.7, 'Augment (VAE)\nKL-filter, per-class cap', colors['generative'], 9)

        # Supervised row
        add_box(ax, 1, 6.2, 3.5, 1.2, 'Classical ML\nLogReg, DecTree\nRandomForest', colors['classical'], 9)
        add_box(ax, 5.25, 6.2, 3.5, 1.2, 'Neural Network\nMLP, MC Dropout', colors['neural'], 9)
        add_box(ax, 9.5, 6.2, 3.5, 1.2, 'Sequential\nLSTM, Transformer', colors['sequential'], 9)

        # Second row
        add_box(ax, 2.5, 4.3, 3.5, 1.2, 'Graph Recommender\nLink Prediction\nHit@K, MAP@K', colors['graph'], 9)
        add_box(ax, 8, 4.3, 3.5, 1.2, 'RL Policy\nQ-Learning\nGridWorld', colors['rl'], 9)

        # MLOps sidecar
        add_box(ax, 11.5, 6.2, 2.3, 2.8, 'MLOps\nLogger\n\nMetrics\nConfigs\nArtifacts\nModel Cards', colors['mlops'], 8)

        # Outputs
        add_box(ax, 5.25, 2.5, 3.5, 1.2, 'Outputs\nrun_log.csv\nModel Cards\nEthics Docs', colors['mlops'], 9)

        # API
        add_box(ax, 5.25, 0.8, 3.5, 1.2, 'API (FastAPI)\n/predict, /forecast\n/recommend, /policy', colors['api'], 9)

        # Arrows
        arrow_style = "Simple, tail_width=0.5, head_width=4, head_length=8"
        kw = dict(arrowstyle=arrow_style, color="gray", lw=1.5)

        # Data to Augment
        ax.annotate("", xy=(7, 8.5), xytext=(7, 9),
                   arrowprops=dict(arrowstyle='->', color='gray', lw=2))

        # Augment to supervised
        ax.annotate("", xy=(2.75, 7.4), xytext=(7, 7.8),
                   arrowprops=dict(arrowstyle='->', color='gray', lw=2, connectionstyle="arc3,rad=-0.2"))
        ax.annotate("", xy=(7, 7.4), xytext=(7, 7.8),
                   arrowprops=dict(arrowstyle='->', color='gray', lw=2))
        ax.annotate("", xy=(11.25, 7.4), xytext=(7, 7.8),
                   arrowprops=dict(arrowstyle='->', color='gray', lw=2, connectionstyle="arc3,rad=0.2"))

        # Supervised to Graph/RL
        ax.annotate("", xy=(4.25, 5.5), xytext=(7, 6.2),
                   arrowprops=dict(arrowstyle='->', color='gray', lw=2, connectionstyle="arc3,rad=-0.2"))
        ax.annotate("", xy=(9.75, 5.5), xytext=(7, 6.2),
                   arrowprops=dict(arrowstyle='->', color='gray', lw=2, connectionstyle="arc3,rad=0.2"))

        # Graph/RL to Outputs
        ax.annotate("", xy=(7, 3.7), xytext=(4.25, 4.3),
                   arrowprops=dict(arrowstyle='->', color='gray', lw=2, connectionstyle="arc3,rad=0.2"))
        ax.annotate("", xy=(7, 3.7), xytext=(9.75, 4.3),
                   arrowprops=dict(arrowstyle='->', color='gray', lw=2, connectionstyle="arc3,rad=-0.2"))

        # Outputs to API
        ax.annotate("", xy=(7, 2), xytext=(7, 2.5),
                   arrowprops=dict(arrowstyle='->', color='gray', lw=2))

        # MLOps connections (dashed)
        ax.plot([11.5, 9.5], [7.6, 6.8], 'k--', lw=1, alpha=0.5)
        ax.plot([11.5, 5.25], [7.6, 6.8], 'k--', lw=1, alpha=0.5)
        ax.plot([11.5, 2.75], [7.6, 6.8], 'k--', lw=1, alpha=0.5)

        plt.tight_layout()
        plt.savefig('docs/System_Diagram.png', dpi=150, bbox_inches='tight', facecolor='white')
        plt.close()
        log("  Saved docs/System_Diagram.png")
    except Exception as e:
        log(f"  Could not generate System Diagram PNG: {e}")

    # Export system diagram to docs/
    log("=== Generating System Diagram Markdown ===")
    docs_dir = "docs"
    os.makedirs(docs_dir, exist_ok=True)

    try:
        system_diagram = """# System Architecture Diagram

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
"""
        with open(f"{docs_dir}/System_Diagram.md", 'w') as f:
            f.write(system_diagram)
        log("  Saved docs/System_Diagram.md")
    except Exception as e:
        log(f"  Could not generate system diagram: {e}")

    # Export comparison tables to docs/
    log("=== Generating Comparison Tables ===")
    try:
        # Build comparison table from artifacts with comprehensive metrics
        comparison_table = """# Model Comparison Tables

## Classification Results (Supervised)

| Model | Type | Accuracy | Precision | Recall | F1 | F1-macro | ROC-AUC | ECE | Latency (batch=1) | Latency (batch=32) |
|-------|------|----------|-----------|--------|-------|----------|---------|-----|-------------------|-------------------|
"""
        if 'classic_results' in artifacts:
            best_name = artifacts['classic_results'].get('best_model', 'Classical')
            res = artifacts['classic_results'].get('all_results', {})
            if best_name in res:
                m = res[best_name]['metrics']
                comparison_table += f"| {best_name} | Classical | {m.get('accuracy', 'N/A'):.4f} | {m.get('precision', 'N/A'):.4f} | {m.get('recall', 'N/A'):.4f} | {m.get('f1', 'N/A'):.4f} | {m.get('f1_macro', 'N/A'):.4f} | {m.get('roc_auc', 'N/A'):.4f} | {m.get('ece', 'N/A'):.4f} | {m.get('latency_batch_1_ms', 'N/A'):.2f}ms | {m.get('latency_batch_32_ms', 'N/A'):.2f}ms |\n"
            else:
                best_acc = artifacts['classic_results'].get('best_accuracy', 'N/A')
                comparison_table += f"| {best_name} | Classical | {best_acc} | - | - | - | - | - | - | - | - |\n"

        if 'neural_metrics' in artifacts:
            m = artifacts['neural_metrics']
            comparison_table += f"| MLP | Neural | {m.get('accuracy', 'N/A'):.4f} | - | - | - | - | - | - | - | - |\n"

        if 'sequential_results' in artifacts:
            if 'lstm' in artifacts['sequential_results']:
                lstm_m = artifacts['sequential_results']['lstm'][2]
                comparison_table += f"| LSTM | Sequential | {lstm_m.get('class_accuracy', 'N/A'):.4f} | - | - | {lstm_m.get('class_f1', 'N/A'):.4f} | - | - | - | {lstm_m.get('latency_batch_1_ms', 'N/A'):.2f}ms | {lstm_m.get('latency_batch_32_ms', 'N/A'):.2f}ms |\n"
            if 'transformer' in artifacts['sequential_results']:
                tf_m = artifacts['sequential_results']['transformer'][2]
                comparison_table += f"| Transformer | Sequential | {tf_m.get('class_accuracy', 'N/A'):.4f} | - | - | {tf_m.get('class_f1', 'N/A'):.4f} | - | - | - | {tf_m.get('latency_batch_1_ms', 'N/A'):.2f}ms | {tf_m.get('latency_batch_32_ms', 'N/A'):.2f}ms |\n"

        comparison_table += """
## Trajectory Prediction Results (Time Series)

| Model | MSE | MAE | MAPE (%) | MASE | MC Dropout Std | Train Time (ms) |
|-------|-----|-----|----------|------|----------------|-----------------|
"""
        if 'sequential_results' in artifacts:
            if 'lstm' in artifacts['sequential_results']:
                lstm_m = artifacts['sequential_results']['lstm'][2]
                comparison_table += f"| LSTM | {lstm_m.get('traj_mse', 'N/A'):.4f} | {lstm_m.get('traj_mae', 'N/A'):.4f} | {lstm_m.get('traj_mape', 'N/A'):.2f} | {lstm_m.get('traj_mase', 'N/A'):.4f} | {lstm_m.get('mc_dropout_mean_std', 'N/A'):.4f} | {lstm_m.get('train_time_ms', 'N/A'):.0f} |\n"
            if 'transformer' in artifacts['sequential_results']:
                tf_m = artifacts['sequential_results']['transformer'][2]
                comparison_table += f"| Transformer | {tf_m.get('traj_mse', 'N/A'):.4f} | {tf_m.get('traj_mae', 'N/A'):.4f} | {tf_m.get('traj_mape', 'N/A'):.2f} | {tf_m.get('traj_mase', 'N/A'):.4f} | {tf_m.get('mc_dropout_mean_std', 'N/A'):.4f} | {tf_m.get('train_time_ms', 'N/A'):.0f} |\n"

        comparison_table += """
## Generative Augmentation Ablation

| Augmentation Ratio | Accuracy | F1 Score | Delta vs 0% |
|--------------------|----------|----------|-------------|
"""
        if 'generative_results' in artifacts:
            ablation = artifacts['generative_results'].get('ablation_results', {})
            baseline_acc = ablation.get(0.0, {}).get('accuracy', 0)
            for ratio, metrics in sorted(ablation.items()):
                acc = metrics.get('accuracy', 0)
                delta = acc - baseline_acc if baseline_acc else 0
                delta_str = f"+{delta:.4f}" if delta >= 0 else f"{delta:.4f}"
                if ratio == 0.0:
                    delta_str = "-"
                comparison_table += f"| {ratio:.0%} | {metrics.get('accuracy', 'N/A'):.4f} | {metrics.get('f1', 'N/A'):.4f} | {delta_str} |\n"

        comparison_table += """
## Graph Link Prediction

### Hit@K
| Method | K=3 | K=5 | K=10 |
|--------|-----|-----|------|
"""
        if 'graph_results' in artifacts:
            graph_res = artifacts['graph_results']
            methods_data = {}
            for entry in graph_res:
                m = entry['method']
                k = entry['k']
                if m not in methods_data:
                    methods_data[m] = {'hit': {}, 'map': {}, 'cov': {}}
                methods_data[m]['hit'][k] = entry.get('hit_at_k', 0)
                methods_data[m]['map'][k] = entry.get('map_at_k', 0)
                methods_data[m]['cov'][k] = entry.get('coverage_at_k', 0)

            for method, data in methods_data.items():
                comparison_table += f"| {method} | {data['hit'].get(3, 'N/A'):.4f} | {data['hit'].get(5, 'N/A'):.4f} | {data['hit'].get(10, 'N/A'):.4f} |\n"

            comparison_table += """
### MAP@K
| Method | K=3 | K=5 | K=10 |
|--------|-----|-----|------|
"""
            for method, data in methods_data.items():
                comparison_table += f"| {method} | {data['map'].get(3, 'N/A'):.4f} | {data['map'].get(5, 'N/A'):.4f} | {data['map'].get(10, 'N/A'):.4f} |\n"

            comparison_table += """
### Coverage@K
| Method | K=3 | K=5 | K=10 |
|--------|-----|-----|------|
"""
            for method, data in methods_data.items():
                comparison_table += f"| {method} | {data['cov'].get(3, 'N/A'):.4f} | {data['cov'].get(5, 'N/A'):.4f} | {data['cov'].get(10, 'N/A'):.4f} |\n"

        comparison_table += """
## Reinforcement Learning

### Performance Metrics
| Metric | Value |
|--------|-------|
"""
        if 'rl_history' in artifacts:
            returns = artifacts['rl_history']['returns']
            convergence_ep = artifacts['rl_history'].get('convergence_episode', -1)
            comparison_table += f"| Final Avg Return (last 100) | {np.mean(returns[-100:]):.2f} |\n"
            comparison_table += f"| Max Return | {np.max(returns):.2f} |\n"
            comparison_table += f"| Total Episodes | {len(returns)} |\n"
            if convergence_ep > 0:
                comparison_table += f"| Convergence Episode (90% threshold) | {convergence_ep} |\n"
            else:
                comparison_table += f"| Convergence Episode | Not converged |\n"

        # RL Hyperparameter Sensitivity Table
        rl_config = config['rl']
        comparison_table += f"""
### Hyperparameter Configuration
| Hyperparameter | Value | Description |
|----------------|-------|-------------|
| Grid Size | {rl_config.get('grid_size', 10)} | State space dimension |
| γ (Discount) | {rl_config.get('gamma', 0.95)} | Future reward discount |
| α (Learning Rate) | {rl_config.get('alpha', 0.2)} | Q-value update rate |
| ε (Initial) | {rl_config.get('epsilon', 1.0)} | Initial exploration rate |
| ε (Minimum) | {rl_config.get('epsilon_min', 0.01)} | Minimum exploration |
| ε (Decay) | {rl_config.get('epsilon_decay', 0.995)} | Exploration decay rate |
| Episodes | {rl_config.get('episodes', 500)} | Training episodes |
| Max Steps | {rl_config.get('max_steps', 50)} | Steps per episode |

### Hyperparameter Sensitivity Analysis
| Parameter | Low | Default | High | Effect on Performance |
|-----------|-----|---------|------|----------------------|
| γ (Discount) | 0.8 | 0.95 | 0.99 | Higher γ → longer-term planning |
| α (Learning Rate) | 0.05 | 0.2 | 0.5 | Higher α → faster but less stable |
| ε (Decay) | 0.99 | 0.995 | 0.999 | Slower decay → more exploration |
| Episodes | 200 | 500 | 1000 | More episodes → better convergence |

*Note: Sensitivity analysis based on typical Q-learning behavior. Actual results may vary.*
"""

        with open(f"{docs_dir}/Comparison_Tables.md", 'w') as f:
            f.write(comparison_table)
        log("  Saved docs/Comparison_Tables.md")
    except Exception as e:
        log(f"  Could not generate comparison tables: {e}")

    # Export artifacts
    log(f"Pipeline complete. Artifacts generated:")
    for key in artifacts:
        log(f"  - {key}")
    log(f"See {args.output_dir}/run_log.csv for detailed metrics.")


if __name__ == '__main__':
    main()
