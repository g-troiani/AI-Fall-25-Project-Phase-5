"""
Classical ML Inference
======================
Inference wrapper for trained classical ML models.
Provides batch prediction, probability estimation, and ensemble methods.
"""

import numpy as np
import pandas as pd
import pickle
from pathlib import Path
from typing import Union, List, Optional, Dict
import time


class TabularInference:
    """
    Inference wrapper for classical ML models.
    Supports single model prediction and soft-voting ensemble.
    """

    def __init__(self, model_path: Union[str, Path] = None, model=None):
        """
        Initialize inference wrapper.

        Args:
            model_path: Path to pickled model file
            model: Pre-loaded model object
        """
        if model_path:
            self.model = self.load_model(model_path)
        elif model:
            self.model = model
        else:
            raise ValueError("Either model_path or model must be provided")

        self.latency_ms = None

    def load_model(self, model_path: Union[str, Path]):
        """Load a pickled model from disk."""
        with open(model_path, 'rb') as f:
            return pickle.load(f)

    def predict(self, X: Union[np.ndarray, pd.DataFrame],
                batch_size: int = 1000) -> np.ndarray:
        """
        Make predictions on input data.

        Args:
            X: Input features (n_samples, n_features)
            batch_size: Batch size for inference

        Returns:
            predictions: Array of predictions
        """
        start = time.perf_counter()

        # Convert DataFrame to numpy if needed
        if isinstance(X, pd.DataFrame):
            X = X.values

        # Batch prediction for large datasets
        if len(X) > batch_size:
            predictions = []
            for i in range(0, len(X), batch_size):
                batch = X[i:i + batch_size]
                predictions.append(self.model.predict(batch))
            predictions = np.concatenate(predictions)
        else:
            predictions = self.model.predict(X)

        self.latency_ms = (time.perf_counter() - start) * 1000
        return predictions

    def predict_proba(self, X: Union[np.ndarray, pd.DataFrame],
                      batch_size: int = 1000) -> np.ndarray:
        """
        Predict class probabilities.

        Args:
            X: Input features
            batch_size: Batch size for inference

        Returns:
            probabilities: Array of class probabilities
        """
        start = time.perf_counter()

        # Convert DataFrame to numpy if needed
        if isinstance(X, pd.DataFrame):
            X = X.values

        # Check if model supports predict_proba
        if not hasattr(self.model, 'predict_proba'):
            # Fallback to predictions as 0/1
            predictions = self.predict(X, batch_size)
            probabilities = np.zeros((len(predictions), 2))
            probabilities[predictions == 0, 0] = 1
            probabilities[predictions == 1, 1] = 1
            return probabilities

        # Batch prediction
        if len(X) > batch_size:
            probabilities = []
            for i in range(0, len(X), batch_size):
                batch = X[i:i + batch_size]
                probabilities.append(self.model.predict_proba(batch))
            probabilities = np.vstack(probabilities)
        else:
            probabilities = self.model.predict_proba(X)

        self.latency_ms = (time.perf_counter() - start) * 1000
        return probabilities

    def get_latency(self) -> Optional[float]:
        """Return the latency of the last prediction in milliseconds."""
        return self.latency_ms


class EnsembleInference:
    """
    Ensemble inference with soft voting.
    Combines predictions from multiple models.
    """

    def __init__(self, model_paths: List[Union[str, Path]] = None,
                 models: List = None, weights: Optional[List[float]] = None):
        """
        Initialize ensemble inference.

        Args:
            model_paths: List of paths to pickled models
            models: List of pre-loaded models
            weights: Optional weights for soft voting
        """
        if model_paths:
            self.models = [self._load_model(p) for p in model_paths]
        elif models:
            self.models = models
        else:
            raise ValueError("Either model_paths or models must be provided")

        self.weights = weights if weights else [1.0] * len(self.models)
        self.weights = np.array(self.weights) / sum(self.weights)
        self.latency_ms = None

    def _load_model(self, model_path: Union[str, Path]):
        """Load a pickled model from disk."""
        with open(model_path, 'rb') as f:
            return pickle.load(f)

    def predict(self, X: Union[np.ndarray, pd.DataFrame]) -> np.ndarray:
        """
        Make ensemble predictions using soft voting.

        Args:
            X: Input features

        Returns:
            predictions: Ensemble predictions
        """
        probas = self.predict_proba(X)
        return (probas[:, 1] > 0.5).astype(int)

    def predict_proba(self, X: Union[np.ndarray, pd.DataFrame]) -> np.ndarray:
        """
        Predict class probabilities using weighted soft voting.

        Args:
            X: Input features

        Returns:
            probabilities: Weighted ensemble probabilities
        """
        start = time.perf_counter()

        # Convert DataFrame to numpy if needed
        if isinstance(X, pd.DataFrame):
            X = X.values

        # Get predictions from all models
        all_probas = []
        for model in self.models:
            if hasattr(model, 'predict_proba'):
                proba = model.predict_proba(X)
            else:
                # Fallback for models without predict_proba
                preds = model.predict(X)
                proba = np.zeros((len(preds), 2))
                proba[preds == 0, 0] = 1
                proba[preds == 1, 1] = 1
            all_probas.append(proba)

        # Weighted average
        ensemble_proba = np.average(all_probas, axis=0, weights=self.weights)

        self.latency_ms = (time.perf_counter() - start) * 1000
        return ensemble_proba

    def get_latency(self) -> Optional[float]:
        """Return the latency of the last prediction in milliseconds."""
        return self.latency_ms


def load_and_predict(model_path: Union[str, Path],
                     X: Union[np.ndarray, pd.DataFrame],
                     return_proba: bool = False) -> Union[np.ndarray, tuple]:
    """
    Convenience function to load model and make predictions.

    Args:
        model_path: Path to pickled model
        X: Input features
        return_proba: Whether to return probabilities

    Returns:
        predictions or (predictions, probabilities)
    """
    inferencer = TabularInference(model_path=model_path)
    predictions = inferencer.predict(X)

    if return_proba:
        probabilities = inferencer.predict_proba(X)
        return predictions, probabilities

    return predictions


if __name__ == "__main__":
    # Example usage
    print("Classical ML Inference Module")
    print("=" * 50)

    # Example: Single model inference
    # model = TabularInference(model_path="outputs/models/random_forest.pkl")
    # predictions = model.predict(X_test)
    # probabilities = model.predict_proba(X_test)
    # print(f"Latency: {model.get_latency():.2f} ms")

    # Example: Ensemble inference
    # ensemble = EnsembleInference(
    #     model_paths=[
    #         "outputs/models/random_forest.pkl",
    #         "outputs/models/xgboost.pkl",
    #         "outputs/models/logistic_regression.pkl"
    #     ],
    #     weights=[0.4, 0.4, 0.2]
    # )
    # ensemble_predictions = ensemble.predict(X_test)
    # ensemble_probabilities = ensemble.predict_proba(X_test)

    print("\nInference module ready for use!")
