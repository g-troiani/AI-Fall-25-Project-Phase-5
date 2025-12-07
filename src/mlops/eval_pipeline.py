"""
Evaluation Pipeline
===================
Metrics computation and visualization for all components.
Source: Project 5 (MLOps)
"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                             f1_score, roc_auc_score, confusion_matrix,
                             mean_squared_error, mean_absolute_error)
from sklearn.calibration import calibration_curve


def compute_classification_metrics(y_true, y_pred, y_proba=None) -> dict:
    """Compute standard classification metrics."""
    metrics = {
        'accuracy': accuracy_score(y_true, y_pred),
        'precision': precision_score(y_true, y_pred, zero_division=0),
        'recall': recall_score(y_true, y_pred, zero_division=0),
        'f1': f1_score(y_true, y_pred, zero_division=0),
    }
    
    if y_proba is not None:
        try:
            metrics['roc_auc'] = roc_auc_score(y_true, y_proba)
        except ValueError:
            metrics['roc_auc'] = 0.0
    
    return metrics


def compute_regression_metrics(y_true, y_pred) -> dict:
    """Compute regression metrics for trajectory prediction."""
    mse = mean_squared_error(y_true.reshape(-1), y_pred.reshape(-1))
    mae = mean_absolute_error(y_true.reshape(-1), y_pred.reshape(-1))
    
    # Per-step metrics
    if len(y_true.shape) > 1:
        mse_per_step = [mean_squared_error(y_true[:, i], y_pred[:, i]) 
                       for i in range(y_true.shape[1])]
    else:
        mse_per_step = [mse]
    
    return {
        'mse': mse,
        'rmse': np.sqrt(mse),
        'mae': mae,
        'mse_per_step': mse_per_step
    }


def plot_confusion_matrix(y_true, y_pred, labels=None, 
                          save_path: str = None, title: str = "Confusion Matrix"):
    """Plot confusion matrix."""
    cm = confusion_matrix(y_true, y_pred)
    
    fig, ax = plt.subplots(figsize=(6, 5))
    im = ax.imshow(cm, cmap='Blues')
    plt.colorbar(im, ax=ax)
    
    if labels is None:
        labels = [str(i) for i in range(len(cm))]
    
    ax.set_xticks(range(len(labels)))
    ax.set_yticks(range(len(labels)))
    ax.set_xticklabels(labels)
    ax.set_yticklabels(labels)
    
    # Add text annotations
    for i in range(len(cm)):
        for j in range(len(cm)):
            ax.text(j, i, str(cm[i, j]), ha='center', va='center',
                   color='white' if cm[i, j] > cm.max()/2 else 'black')
    
    ax.set_xlabel('Predicted')
    ax.set_ylabel('True')
    ax.set_title(title)
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    
    plt.close()
    return fig


def plot_reliability_diagram(y_true, y_proba, n_bins: int = 10,
                             save_path: str = None, title: str = "Reliability Diagram"):
    """Plot calibration/reliability diagram."""
    try:
        prob_true, prob_pred = calibration_curve(y_true, y_proba, n_bins=n_bins)
    except ValueError:
        return None
    
    fig, ax = plt.subplots(figsize=(6, 6))
    
    ax.plot([0, 1], [0, 1], 'k--', label='Perfectly Calibrated')
    ax.plot(prob_pred, prob_true, 'o-', label='Model')
    
    ax.set_xlabel('Mean Predicted Probability')
    ax.set_ylabel('Fraction of Positives')
    ax.set_title(title)
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    
    plt.close()
    return fig


def plot_trajectory_comparison(y_true, y_pred, n_samples: int = 5,
                               save_path: str = None, title: str = "Trajectory Comparison"):
    """Plot true vs predicted trajectories."""
    fig, axes = plt.subplots(1, n_samples, figsize=(3*n_samples, 3))
    
    if n_samples == 1:
        axes = [axes]
    
    for i, ax in enumerate(axes):
        if i >= len(y_true):
            break
        
        ax.plot(y_true[i, :, 0], y_true[i, :, 1], 'g-o', label='True', markersize=4)
        ax.plot(y_pred[i, :, 0], y_pred[i, :, 1], 'r--x', label='Pred', markersize=4)
        
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_title(f'Sample {i+1}')
        if i == 0:
            ax.legend()
        ax.grid(True, alpha=0.3)
    
    plt.suptitle(title)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    
    plt.close()
    return fig


def plot_graph_hit_at_k(results: dict, save_path: str = None,
                        title: str = "Graph Link Prediction - Hit@K"):
    """Plot Hit@K comparison across methods."""
    methods = list(results.keys())
    k_values = [k for k in results[methods[0]].keys() if k.startswith('hit@')]
    
    fig, ax = plt.subplots(figsize=(8, 5))
    
    x = np.arange(len(k_values))
    width = 0.25
    
    for i, method in enumerate(methods):
        values = [results[method][k] for k in k_values]
        ax.bar(x + i*width, values, width, label=method)
    
    ax.set_xlabel('K')
    ax.set_ylabel('Hit@K')
    ax.set_title(title)
    ax.set_xticks(x + width)
    ax.set_xticklabels([k.replace('hit@', '') for k in k_values])
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    
    plt.close()
    return fig


def generate_comparison_table(results: dict) -> str:
    """Generate markdown comparison table from results."""
    lines = ["| Model | Accuracy | Precision | Recall | F1 | ROC-AUC |",
             "|-------|----------|-----------|--------|-------|---------|"]
    
    for name, metrics in results.items():
        row = f"| {name} | "
        row += f"{metrics.get('accuracy', 0):.4f} | "
        row += f"{metrics.get('precision', 0):.4f} | "
        row += f"{metrics.get('recall', 0):.4f} | "
        row += f"{metrics.get('f1', 0):.4f} | "
        row += f"{metrics.get('roc_auc', 0):.4f} |"
        lines.append(row)
    
    return "\n".join(lines)


def evaluate_trajectory(adapter, X_test, Y_traj_test, Y_class_test, name: str,
                        timer_cls=None) -> dict:
    """
    Evaluate sequential model on test set.
    
    Args:
        adapter: SequentialModelAdapter with predict_trajectory/predict_class
        X_test: Test input sequences
        Y_traj_test: True trajectory outputs
        Y_class_test: True classification labels
        name: Model name for reporting
        
    Returns:
        Dict with trajectory and classification metrics
    """
    if timer_cls is None:
        from .utils import Timer as timer_cls
    
    with timer_cls() as t:
        traj_pred = adapter.predict_trajectory(X_test)
        class_pred = adapter.predict_class(X_test)
    
    # Trajectory metrics
    l2_errors = np.sqrt(np.sum((traj_pred - Y_traj_test)**2, axis=2))
    mean_l2 = l2_errors.mean()
    final_l2 = l2_errors[:, -1].mean()
    
    # Classification metrics
    acc = accuracy_score(Y_class_test, class_pred)
    f1 = f1_score(Y_class_test, class_pred, zero_division=0)
    
    return {
        'name': name,
        'mean_l2': mean_l2,
        'final_l2': final_l2,
        'class_accuracy': acc,
        'class_f1': f1,
        'latency_ms': t.elapsed_ms
    }


def generate_final_comparison(classical_metrics: dict, mlp_metrics: dict,
                              lstm_metrics: dict, transformer_metrics: dict,
                              classical_name: str = "RandomForest") -> str:
    """Generate formatted final comparison table."""
    return f"""
================================================================================
                           FINAL MODEL COMPARISON
================================================================================

┌─────────────────────────────────────────────────────────────────────────────┐
│                           CLASSIFICATION RESULTS                            │
├─────────────────────────┬──────────┬──────────┬──────────┬─────────────────┤
│ Model                   │ Accuracy │ F1 Score │ ROC-AUC  │ Latency (ms)    │
├─────────────────────────┼──────────┼──────────┼──────────┼─────────────────┤
│ {classical_name:<23} │ {classical_metrics.get('accuracy', 0):.4f}   │ {classical_metrics.get('f1', 0):.4f}   │ {classical_metrics.get('roc_auc', 0):.4f}   │ {classical_metrics.get('latency_ms', 0):.2f}            │
│ MLP (Project 3)         │ {mlp_metrics.get('accuracy', 0):.4f}   │ {mlp_metrics.get('f1', 0):.4f}   │ {mlp_metrics.get('roc_auc', 0):.4f}   │ {mlp_metrics.get('latency_ms', 0):.2f}            │
│ LSTM (Project 4)        │ {lstm_metrics.get('class_accuracy', 0):.4f}   │ {lstm_metrics.get('class_f1', 0):.4f}   │ -        │ {lstm_metrics.get('latency_ms', 0):.2f}           │
│ Transformer (Project 4) │ {transformer_metrics.get('class_accuracy', 0):.4f}   │ {transformer_metrics.get('class_f1', 0):.4f}   │ -        │ {transformer_metrics.get('latency_ms', 0):.2f}           │
└─────────────────────────┴──────────┴──────────┴──────────┴─────────────────┘

┌─────────────────────────────────────────────────────────────────────────────┐
│                        TRAJECTORY PREDICTION RESULTS                        │
├─────────────────────────┬─────────────────┬─────────────────┬───────────────┤
│ Model                   │ Mean L2 (yards) │ Final L2 (yards)│ Latency (ms)  │
├─────────────────────────┼─────────────────┼─────────────────┼───────────────┤
│ LSTM (Project 4)        │ {lstm_metrics.get('mean_l2', 0):.4f}          │ {lstm_metrics.get('final_l2', 0):.4f}           │ {lstm_metrics.get('latency_ms', 0):.2f}         │
│ Transformer (Project 4) │ {transformer_metrics.get('mean_l2', 0):.4f}          │ {transformer_metrics.get('final_l2', 0):.4f}           │ {transformer_metrics.get('latency_ms', 0):.2f}         │
└─────────────────────────┴─────────────────┴─────────────────┴───────────────┘
"""
