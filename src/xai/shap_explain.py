"""
Explainable AI (XAI) Module - SHAP & LIME
=========================================
Implements SHAP and LIME explainability for model predictions.
Provides feature importance, local explanations, and visualization utilities.

Source: Project 5 - Generative Models & Responsible AI
"""

import numpy as np
import pandas as pd
from typing import Optional, List, Dict, Any, Callable, Tuple
import warnings

# Try to import SHAP and LIME - handle gracefully if not installed
try:
    import shap
    SHAP_AVAILABLE = True
except ImportError:
    SHAP_AVAILABLE = False
    warnings.warn("SHAP not installed. Install with: pip install shap")

try:
    import lime
    import lime.lime_tabular
    LIME_AVAILABLE = True
except ImportError:
    LIME_AVAILABLE = False
    warnings.warn("LIME not installed. Install with: pip install lime")


class SHAPExplainer:
    """
    SHAP-based model explainability.

    Provides:
    - Global feature importance
    - Local instance explanations
    - Feature interaction analysis
    """

    def __init__(self, model: Any, X_background: np.ndarray,
                 feature_names: Optional[List[str]] = None,
                 model_type: str = 'tree'):
        """
        Initialize SHAP explainer.

        Args:
            model: Trained model (sklearn, XGBoost, etc.)
            X_background: Background dataset for SHAP (sample of training data)
            feature_names: Names of features
            model_type: 'tree' for tree-based, 'kernel' for model-agnostic
        """
        if not SHAP_AVAILABLE:
            raise ImportError("SHAP not installed. Run: pip install shap")

        self.model = model
        self.X_background = X_background
        self.feature_names = feature_names or [f"feature_{i}" for i in range(X_background.shape[1])]
        self.model_type = model_type
        self.explainer = None
        self.shap_values = None

        self._init_explainer()

    def _init_explainer(self):
        """Initialize appropriate SHAP explainer based on model type."""
        if self.model_type == 'tree':
            try:
                self.explainer = shap.TreeExplainer(self.model)
            except Exception:
                # Fallback to kernel explainer
                self.explainer = shap.KernelExplainer(
                    self.model.predict_proba if hasattr(self.model, 'predict_proba') else self.model.predict,
                    shap.sample(self.X_background, min(100, len(self.X_background)))
                )
        elif self.model_type == 'kernel':
            predict_fn = self.model.predict_proba if hasattr(self.model, 'predict_proba') else self.model.predict
            self.explainer = shap.KernelExplainer(
                predict_fn,
                shap.sample(self.X_background, min(100, len(self.X_background)))
            )
        elif self.model_type == 'deep':
            self.explainer = shap.DeepExplainer(self.model, self.X_background)
        else:
            raise ValueError(f"Unknown model_type: {self.model_type}")

    def explain_instance(self, x: np.ndarray) -> Dict[str, Any]:
        """
        Explain a single prediction.

        Args:
            x: Single instance (1D array)

        Returns:
            Dictionary with SHAP values and feature contributions
        """
        x = x.reshape(1, -1) if x.ndim == 1 else x
        shap_values = self.explainer.shap_values(x)

        # Handle multi-class output
        if isinstance(shap_values, list):
            shap_values = shap_values[1]  # Positive class for binary

        contributions = dict(zip(self.feature_names, shap_values[0]))
        sorted_contributions = dict(sorted(contributions.items(),
                                           key=lambda x: abs(x[1]), reverse=True))

        return {
            'shap_values': shap_values[0],
            'feature_contributions': sorted_contributions,
            'base_value': self.explainer.expected_value if hasattr(self.explainer, 'expected_value') else None,
            'prediction': self.model.predict(x)[0] if hasattr(self.model, 'predict') else None
        }

    def explain_global(self, X: np.ndarray) -> Dict[str, Any]:
        """
        Compute global feature importance using SHAP.

        Args:
            X: Dataset to explain

        Returns:
            Dictionary with global importance metrics
        """
        shap_values = self.explainer.shap_values(X)

        # Handle multi-class
        if isinstance(shap_values, list):
            shap_values = shap_values[1]

        self.shap_values = shap_values

        # Mean absolute SHAP values
        mean_abs_shap = np.abs(shap_values).mean(axis=0)
        importance = dict(zip(self.feature_names, mean_abs_shap))
        sorted_importance = dict(sorted(importance.items(),
                                        key=lambda x: x[1], reverse=True))

        return {
            'feature_importance': sorted_importance,
            'shap_values_matrix': shap_values,
            'mean_abs_shap': mean_abs_shap
        }

    def get_top_features(self, X: np.ndarray, top_k: int = 5) -> List[str]:
        """Get top-k most important features."""
        global_exp = self.explain_global(X)
        return list(global_exp['feature_importance'].keys())[:top_k]

    def generate_explanation_text(self, x: np.ndarray, prediction_label: str = "positive") -> str:
        """
        Generate human-readable explanation text.

        Args:
            x: Instance to explain
            prediction_label: Label for positive prediction

        Returns:
            Human-readable explanation string
        """
        exp = self.explain_instance(x)
        top_features = list(exp['feature_contributions'].items())[:5]

        lines = [f"Prediction: {prediction_label}"]
        lines.append("\nTop contributing features:")

        for feature, value in top_features:
            direction = "increases" if value > 0 else "decreases"
            lines.append(f"  - {feature}: {direction} prediction by {abs(value):.4f}")

        return "\n".join(lines)


class LIMEExplainer:
    """
    LIME-based model explainability.

    Provides:
    - Local interpretable explanations
    - Feature importance for individual predictions
    """

    def __init__(self, X_train: np.ndarray,
                 feature_names: Optional[List[str]] = None,
                 class_names: Optional[List[str]] = None,
                 mode: str = 'classification'):
        """
        Initialize LIME explainer.

        Args:
            X_train: Training data for LIME
            feature_names: Names of features
            class_names: Names of classes
            mode: 'classification' or 'regression'
        """
        if not LIME_AVAILABLE:
            raise ImportError("LIME not installed. Run: pip install lime")

        self.X_train = X_train
        self.feature_names = feature_names or [f"feature_{i}" for i in range(X_train.shape[1])]
        self.class_names = class_names or ['negative', 'positive']
        self.mode = mode

        self.explainer = lime.lime_tabular.LimeTabularExplainer(
            training_data=X_train,
            feature_names=self.feature_names,
            class_names=self.class_names,
            mode=self.mode,
            discretize_continuous=True
        )

    def explain_instance(self, model: Any, x: np.ndarray,
                         num_features: int = 10) -> Dict[str, Any]:
        """
        Explain a single prediction using LIME.

        Args:
            model: Trained model with predict_proba method
            x: Instance to explain
            num_features: Number of features in explanation

        Returns:
            Dictionary with LIME explanation
        """
        x = x.flatten() if x.ndim > 1 else x

        predict_fn = model.predict_proba if hasattr(model, 'predict_proba') else model.predict

        explanation = self.explainer.explain_instance(
            x, predict_fn, num_features=num_features
        )

        feature_weights = dict(explanation.as_list())

        return {
            'feature_weights': feature_weights,
            'local_prediction': explanation.local_pred[0] if hasattr(explanation, 'local_pred') else None,
            'intercept': explanation.intercept[1] if self.mode == 'classification' else explanation.intercept,
            'score': explanation.score
        }

    def generate_explanation_text(self, model: Any, x: np.ndarray) -> str:
        """Generate human-readable LIME explanation."""
        exp = self.explain_instance(model, x)

        lines = ["LIME Explanation:"]
        lines.append(f"Local model fit: {exp['score']:.3f}")
        lines.append("\nFeature contributions:")

        for feature, weight in list(exp['feature_weights'].items())[:5]:
            direction = "+" if weight > 0 else ""
            lines.append(f"  - {feature}: {direction}{weight:.4f}")

        return "\n".join(lines)


class XAIReport:
    """
    Generate comprehensive XAI reports combining SHAP and LIME.
    """

    def __init__(self, model: Any, X_train: np.ndarray,
                 feature_names: Optional[List[str]] = None):
        """
        Initialize XAI report generator.

        Args:
            model: Trained model
            X_train: Training data
            feature_names: Feature names
        """
        self.model = model
        self.X_train = X_train
        self.feature_names = feature_names or [f"feature_{i}" for i in range(X_train.shape[1])]

        self.shap_explainer = None
        self.lime_explainer = None

        self._init_explainers()

    def _init_explainers(self):
        """Initialize both SHAP and LIME explainers."""
        if SHAP_AVAILABLE:
            try:
                self.shap_explainer = SHAPExplainer(
                    self.model, self.X_train,
                    self.feature_names, model_type='tree'
                )
            except Exception as e:
                warnings.warn(f"SHAP initialization failed: {e}")

        if LIME_AVAILABLE:
            try:
                self.lime_explainer = LIMEExplainer(
                    self.X_train, self.feature_names
                )
            except Exception as e:
                warnings.warn(f"LIME initialization failed: {e}")

    def generate_instance_report(self, x: np.ndarray) -> Dict[str, Any]:
        """
        Generate comprehensive explanation for a single instance.

        Args:
            x: Instance to explain

        Returns:
            Combined SHAP and LIME explanations
        """
        report = {'instance': x.tolist()}

        if self.shap_explainer:
            report['shap'] = self.shap_explainer.explain_instance(x)

        if self.lime_explainer:
            report['lime'] = self.lime_explainer.explain_instance(self.model, x)

        # Agreement analysis
        if 'shap' in report and 'lime' in report:
            shap_top = set(list(report['shap']['feature_contributions'].keys())[:5])
            lime_features = [f.split()[0] for f in report['lime']['feature_weights'].keys()]
            lime_top = set(lime_features[:5])
            report['agreement'] = {
                'overlap': len(shap_top & lime_top),
                'shap_top5': list(shap_top),
                'lime_top5': list(lime_top)
            }

        return report

    def generate_global_report(self, X: np.ndarray) -> Dict[str, Any]:
        """
        Generate global feature importance report.

        Args:
            X: Dataset to analyze

        Returns:
            Global importance metrics
        """
        report = {}

        if self.shap_explainer:
            report['shap_global'] = self.shap_explainer.explain_global(X)

        return report

    def save_report(self, report: Dict[str, Any], filepath: str):
        """Save report to JSON file."""
        import json
        with open(filepath, 'w') as f:
            # Convert numpy arrays to lists
            def convert(obj):
                if isinstance(obj, np.ndarray):
                    return obj.tolist()
                elif isinstance(obj, dict):
                    return {k: convert(v) for k, v in obj.items()}
                elif isinstance(obj, list):
                    return [convert(i) for i in obj]
                return obj

            json.dump(convert(report), f, indent=2)


def explain_prediction(model: Any, X_train: np.ndarray, x_instance: np.ndarray,
                       feature_names: Optional[List[str]] = None,
                       method: str = 'both') -> str:
    """
    Quick function to explain a single prediction.

    Args:
        model: Trained model
        X_train: Training data for background
        x_instance: Instance to explain
        feature_names: Feature names
        method: 'shap', 'lime', or 'both'

    Returns:
        Human-readable explanation string
    """
    lines = ["=" * 50]
    lines.append("EXPLAINABLE AI REPORT")
    lines.append("=" * 50)

    if method in ['shap', 'both'] and SHAP_AVAILABLE:
        try:
            shap_exp = SHAPExplainer(model, X_train, feature_names, 'tree')
            lines.append("\n--- SHAP Explanation ---")
            lines.append(shap_exp.generate_explanation_text(x_instance))
        except Exception as e:
            lines.append(f"\nSHAP failed: {e}")

    if method in ['lime', 'both'] and LIME_AVAILABLE:
        try:
            lime_exp = LIMEExplainer(X_train, feature_names)
            lines.append("\n--- LIME Explanation ---")
            lines.append(lime_exp.generate_explanation_text(model, x_instance))
        except Exception as e:
            lines.append(f"\nLIME failed: {e}")

    lines.append("\n" + "=" * 50)
    return "\n".join(lines)


# Fallback implementations when libraries not available
class SimpleSHAPFallback:
    """
    Simple permutation-based feature importance (fallback when SHAP unavailable).
    """

    def __init__(self, model: Any, X: np.ndarray, feature_names: List[str]):
        self.model = model
        self.X = X
        self.feature_names = feature_names

    def compute_importance(self, n_repeats: int = 10) -> Dict[str, float]:
        """Compute permutation importance."""
        baseline = self.model.predict(self.X)
        if hasattr(baseline[0], '__len__'):
            baseline = baseline[:, 1]  # Binary classification

        importances = {}
        for i, name in enumerate(self.feature_names):
            scores = []
            for _ in range(n_repeats):
                X_perm = self.X.copy()
                np.random.shuffle(X_perm[:, i])
                perm_pred = self.model.predict(X_perm)
                if hasattr(perm_pred[0], '__len__'):
                    perm_pred = perm_pred[:, 1]
                score = np.mean(np.abs(baseline - perm_pred))
                scores.append(score)
            importances[name] = np.mean(scores)

        return dict(sorted(importances.items(), key=lambda x: x[1], reverse=True))


if __name__ == "__main__":
    print("XAI Module Demo")
    print("=" * 40)
    print(f"SHAP available: {SHAP_AVAILABLE}")
    print(f"LIME available: {LIME_AVAILABLE}")

    # Demo with simple model
    from sklearn.ensemble import RandomForestClassifier

    np.random.seed(42)
    X_demo = np.random.rand(100, 5)
    y_demo = (X_demo[:, 0] + X_demo[:, 1] > 1).astype(int)

    model = RandomForestClassifier(n_estimators=10, random_state=42)
    model.fit(X_demo, y_demo)

    feature_names = ['x_position', 'y_position', 'speed', 'acceleration', 'direction']

    if SHAP_AVAILABLE:
        print("\n--- SHAP Demo ---")
        shap_exp = SHAPExplainer(model, X_demo, feature_names, 'tree')
        exp = shap_exp.explain_instance(X_demo[0])
        print("Top features:", list(exp['feature_contributions'].keys())[:3])

    if LIME_AVAILABLE:
        print("\n--- LIME Demo ---")
        lime_exp = LIMEExplainer(X_demo, feature_names)
        exp = lime_exp.explain_instance(model, X_demo[0])
        print("LIME score:", exp['score'])

    print("\n--- Fallback Demo ---")
    fallback = SimpleSHAPFallback(model, X_demo, feature_names)
    importance = fallback.compute_importance(n_repeats=5)
    print("Permutation importance:", list(importance.keys())[:3])
