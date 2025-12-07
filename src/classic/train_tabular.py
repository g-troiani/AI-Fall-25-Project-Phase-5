"""
Classical ML Training
=====================
Train classical ML models (Logistic Regression, Decision Tree, Random Forest).
Source: Projects 1-2
"""

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                             f1_score, roc_auc_score, confusion_matrix)
import time

try:
    from imblearn.over_sampling import SMOTE
    SMOTE_AVAILABLE = True
except ImportError:
    SMOTE_AVAILABLE = False


class ClassicalMLAdapter:
    """
    Adapter for classical ML models.
    Provides consistent fit/predict/predict_proba interface.
    Supports SMOTE for class imbalance.
    """

    def __init__(self, model, use_smote: bool = False, seed: int = 42):
        self.model = model
        self.use_smote = use_smote and SMOTE_AVAILABLE
        self.smote = SMOTE(random_state=seed) if self.use_smote else None

    def fit(self, X, y):
        if self.use_smote:
            X_resampled, y_resampled = self.smote.fit_resample(X, y)
            self.model.fit(X_resampled, y_resampled)
        else:
            self.model.fit(X, y)
        return self

    def predict(self, X):
        return self.model.predict(X)

    def predict_proba(self, X):
        if hasattr(self.model, 'predict_proba'):
            return self.model.predict_proba(X)[:, 1]
        return self.predict(X).astype(float)

    def get_params(self):
        return self.model.get_params()


def get_classical_models(config: dict, seed: int = 42) -> dict:
    """Create classical model adapters from config."""
    use_smote = config.get('use_smote', True)
    
    return {
        'LogisticRegression': ClassicalMLAdapter(
            LogisticRegression(random_state=seed, max_iter=1000, class_weight='balanced'),
            use_smote=use_smote, seed=seed
        ),
        'DecisionTree': ClassicalMLAdapter(
            DecisionTreeClassifier(random_state=seed, class_weight='balanced'),
            use_smote=use_smote, seed=seed
        ),
        'RandomForest': ClassicalMLAdapter(
            RandomForestClassifier(
                random_state=seed, 
                class_weight='balanced',
                n_estimators=config.get('rf_n_estimators', 100),
                max_depth=config.get('rf_max_depth', 10)
            ),
            use_smote=use_smote, seed=seed
        )
    }


def compute_ece(y_true, y_proba, n_bins: int = 10) -> float:
    """
    Compute Expected Calibration Error (ECE).
    Measures how well predicted probabilities match actual outcomes.
    """
    bin_boundaries = np.linspace(0, 1, n_bins + 1)
    ece = 0.0
    n_samples = len(y_true)

    for i in range(n_bins):
        in_bin = (y_proba > bin_boundaries[i]) & (y_proba <= bin_boundaries[i + 1])
        prop_in_bin = in_bin.mean()

        if prop_in_bin > 0:
            avg_confidence = y_proba[in_bin].mean()
            avg_accuracy = y_true[in_bin].mean()
            ece += np.abs(avg_accuracy - avg_confidence) * prop_in_bin

    return ece


def measure_latency(adapter, X, batch_sizes: list = [1, 32], n_runs: int = 10) -> dict:
    """
    Measure inference latency for different batch sizes.

    Returns:
        Dict with latency_batch_1_ms, latency_batch_32_ms
    """
    latencies = {}

    for batch_size in batch_sizes:
        times = []
        X_batch = X[:batch_size] if len(X) >= batch_size else X

        # Warmup
        adapter.predict(X_batch)

        for _ in range(n_runs):
            start = time.perf_counter()
            adapter.predict(X_batch)
            elapsed = (time.perf_counter() - start) * 1000  # ms
            times.append(elapsed)

        latencies[f'latency_batch_{batch_size}_ms'] = np.median(times)

    return latencies


def evaluate_model(adapter, X, y_true) -> dict:
    """Compute classification metrics including F1-macro and ECE."""
    y_pred = adapter.predict(X)
    y_proba = adapter.predict_proba(X)

    metrics = {
        'accuracy': accuracy_score(y_true, y_pred),
        'precision': precision_score(y_true, y_pred, zero_division=0),
        'recall': recall_score(y_true, y_pred, zero_division=0),
        'f1': f1_score(y_true, y_pred, zero_division=0),
        'f1_macro': f1_score(y_true, y_pred, average='macro', zero_division=0),
    }

    try:
        metrics['roc_auc'] = roc_auc_score(y_true, y_proba)
    except ValueError:
        metrics['roc_auc'] = 0.0

    # ECE for calibration
    metrics['ece'] = compute_ece(y_true, y_proba)

    # Latency measurements
    latencies = measure_latency(adapter, X)
    metrics.update(latencies)

    return metrics


def train_classical_models(X_train, y_train, X_val, y_val, config: dict, 
                           seed: int = 42, run_logger=None, timer_cls=None, log_fn=None):
    """
    Train and evaluate all classical models.
    
    Returns:
        best_adapter: Best performing model adapter
        results: Dict of all model results
        best_name: Name of best model
    """
    if timer_cls is None:
        from ..mlops.utils import Timer as timer_cls
    if log_fn is None:
        from ..mlops.utils import log as log_fn
    
    models = get_classical_models(config, seed)
    results = {}
    
    for name, adapter in models.items():
        log_fn(f"  Training {name}...")
        
        with timer_cls() as t:
            adapter.fit(X_train, y_train)
        
        metrics = evaluate_model(adapter, X_val, y_val)
        metrics['train_time_ms'] = t.elapsed_ms
        
        results[name] = {'adapter': adapter, 'metrics': metrics}
        
        if run_logger:
            run_logger.log(
                'classical', f'{name}_val_accuracy', metrics['accuracy'],
                latency_ms=t.elapsed_ms,
                params={'model': name, 'smote': config.get('use_smote', True)},
                notes='class_weight=balanced'
            )
    
    # Find best model by F1
    best_name = max(results, key=lambda k: results[k]['metrics']['f1'])
    
    return results[best_name]['adapter'], results, best_name


def infer_tabular(adapter, X) -> tuple:
    """Run inference on tabular data."""
    predictions = adapter.predict(X)
    probabilities = adapter.predict_proba(X)
    return predictions, probabilities
