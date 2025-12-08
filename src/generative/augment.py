"""
Geometric Data Augmentation Module
==================================
Implements rotation, flip, and noise-based augmentations for tabular trajectory data.
Treats 2D position features (x, y) as spatial coordinates that can be transformed.

Source: Project 5 - Generative Models & Responsible AI
"""

import numpy as np
from typing import Tuple, List, Optional, Callable
import pandas as pd


def rotate_coordinates(x: float, y: float, angle_degrees: float,
                       center_x: float = 50.0, center_y: float = 26.5) -> Tuple[float, float]:
    """
    Rotate (x, y) coordinates around a center point.

    Args:
        x, y: Original coordinates
        angle_degrees: Rotation angle in degrees
        center_x, center_y: Center of rotation (default: field center)

    Returns:
        Rotated (x, y) coordinates
    """
    angle_rad = np.radians(angle_degrees)
    cos_a, sin_a = np.cos(angle_rad), np.sin(angle_rad)

    # Translate to origin
    x_centered = x - center_x
    y_centered = y - center_y

    # Rotate
    x_rot = x_centered * cos_a - y_centered * sin_a
    y_rot = x_centered * sin_a + y_centered * cos_a

    # Translate back
    return x_rot + center_x, y_rot + center_y


def flip_horizontal(x: float, field_width: float = 100.0) -> float:
    """Flip x-coordinate horizontally across field center."""
    return field_width - x


def flip_vertical(y: float, field_height: float = 53.3) -> float:
    """Flip y-coordinate vertically across field center."""
    return field_height - y


def add_gaussian_noise(value: float, std: float = 0.5) -> float:
    """Add Gaussian noise to a value."""
    return value + np.random.normal(0, std)


class GeometricAugmenter:
    """
    Geometric augmentation for trajectory/position data.

    Supports:
    - Rotations (90, 180, 270 degrees or custom)
    - Horizontal flip
    - Vertical flip
    - Gaussian noise injection
    - Combined transformations
    """

    def __init__(self,
                 x_col: str = 'x',
                 y_col: str = 'y',
                 field_width: float = 100.0,
                 field_height: float = 53.3,
                 seed: int = 42):
        """
        Initialize augmenter.

        Args:
            x_col: Name of x-coordinate column
            y_col: Name of y-coordinate column
            field_width: Width of coordinate space
            field_height: Height of coordinate space
            seed: Random seed for reproducibility
        """
        self.x_col = x_col
        self.y_col = y_col
        self.field_width = field_width
        self.field_height = field_height
        self.seed = seed
        np.random.seed(seed)

        self.augmentation_log = []

    def rotate(self, X: np.ndarray, angle: float = 90.0,
               x_idx: int = 0, y_idx: int = 1) -> np.ndarray:
        """
        Rotate position features by given angle.

        Args:
            X: Feature array (n_samples, n_features)
            angle: Rotation angle in degrees
            x_idx, y_idx: Column indices for x, y coordinates

        Returns:
            Augmented feature array
        """
        X_aug = X.copy()
        center_x = self.field_width / 2
        center_y = self.field_height / 2

        for i in range(len(X_aug)):
            x_new, y_new = rotate_coordinates(
                X_aug[i, x_idx], X_aug[i, y_idx],
                angle, center_x, center_y
            )
            X_aug[i, x_idx] = x_new
            X_aug[i, y_idx] = y_new

        self.augmentation_log.append(f"rotate_{angle}deg")
        return X_aug

    def flip_h(self, X: np.ndarray, x_idx: int = 0) -> np.ndarray:
        """
        Flip horizontally (mirror across y-axis).

        Args:
            X: Feature array
            x_idx: Column index for x coordinate

        Returns:
            Flipped feature array
        """
        X_aug = X.copy()
        X_aug[:, x_idx] = self.field_width - X_aug[:, x_idx]
        self.augmentation_log.append("flip_horizontal")
        return X_aug

    def flip_v(self, X: np.ndarray, y_idx: int = 1) -> np.ndarray:
        """
        Flip vertically (mirror across x-axis).

        Args:
            X: Feature array
            y_idx: Column index for y coordinate

        Returns:
            Flipped feature array
        """
        X_aug = X.copy()
        X_aug[:, y_idx] = self.field_height - X_aug[:, y_idx]
        self.augmentation_log.append("flip_vertical")
        return X_aug

    def add_noise(self, X: np.ndarray, std: float = 0.5,
                  feature_indices: Optional[List[int]] = None) -> np.ndarray:
        """
        Add Gaussian noise to features.

        Args:
            X: Feature array
            std: Standard deviation of noise
            feature_indices: Which features to add noise to (default: all)

        Returns:
            Noisy feature array
        """
        X_aug = X.copy()

        if feature_indices is None:
            feature_indices = list(range(X.shape[1]))

        for idx in feature_indices:
            noise = np.random.normal(0, std, size=X_aug.shape[0])
            X_aug[:, idx] = X_aug[:, idx] + noise

        self.augmentation_log.append(f"noise_std{std}")
        return X_aug

    def augment_batch(self, X: np.ndarray, y: np.ndarray,
                      transforms: List[str] = ['rotate_90', 'flip_h', 'flip_v'],
                      x_idx: int = 0, y_idx: int = 1) -> Tuple[np.ndarray, np.ndarray]:
        """
        Apply multiple augmentations and combine with original data.

        Args:
            X: Original features
            y: Original labels
            transforms: List of transforms to apply
            x_idx, y_idx: Coordinate column indices

        Returns:
            (X_augmented, y_augmented) with original + augmented samples
        """
        all_X = [X]
        all_y = [y]

        for transform in transforms:
            if transform == 'rotate_90':
                X_aug = self.rotate(X, 90.0, x_idx, y_idx)
            elif transform == 'rotate_180':
                X_aug = self.rotate(X, 180.0, x_idx, y_idx)
            elif transform == 'rotate_270':
                X_aug = self.rotate(X, 270.0, x_idx, y_idx)
            elif transform == 'flip_h':
                X_aug = self.flip_h(X, x_idx)
            elif transform == 'flip_v':
                X_aug = self.flip_v(X, y_idx)
            elif transform.startswith('noise_'):
                std = float(transform.split('_')[1])
                X_aug = self.add_noise(X, std, [x_idx, y_idx])
            else:
                continue

            all_X.append(X_aug)
            all_y.append(y.copy())

        return np.vstack(all_X), np.concatenate(all_y)

    def get_augmentation_summary(self) -> dict:
        """Return summary of applied augmentations."""
        return {
            'transforms_applied': self.augmentation_log,
            'total_transforms': len(self.augmentation_log),
            'field_dimensions': (self.field_width, self.field_height)
        }


def run_augmentation_ablation(X_train: np.ndarray, y_train: np.ndarray,
                               X_val: np.ndarray, y_val: np.ndarray,
                               train_fn: Callable,
                               seed: int = 42) -> dict:
    """
    Run ablation study comparing augmentation strategies.

    Args:
        X_train, y_train: Training data
        X_val, y_val: Validation data
        train_fn: Function that trains a model and returns accuracy
        seed: Random seed

    Returns:
        Dictionary with ablation results
    """
    augmenter = GeometricAugmenter(seed=seed)
    results = {}

    # Baseline (no augmentation)
    baseline_acc = train_fn(X_train, y_train, X_val, y_val)
    results['baseline'] = {'accuracy': baseline_acc, 'n_samples': len(X_train)}

    # Test different augmentation strategies
    strategies = [
        ('rotate_90', ['rotate_90']),
        ('rotate_180', ['rotate_180']),
        ('flip_h', ['flip_h']),
        ('flip_v', ['flip_v']),
        ('flip_both', ['flip_h', 'flip_v']),
        ('all_rotations', ['rotate_90', 'rotate_180', 'rotate_270']),
        ('full_augment', ['rotate_90', 'rotate_180', 'flip_h', 'flip_v']),
    ]

    for name, transforms in strategies:
        X_aug, y_aug = augmenter.augment_batch(X_train, y_train, transforms)
        acc = train_fn(X_aug, y_aug, X_val, y_val)
        results[name] = {
            'accuracy': acc,
            'n_samples': len(X_aug),
            'transforms': transforms,
            'delta_vs_baseline': acc - baseline_acc
        }

    return results


# Convenience function for quick augmentation
def quick_augment(X: np.ndarray, y: np.ndarray,
                  strategy: str = 'flip_both',
                  seed: int = 42) -> Tuple[np.ndarray, np.ndarray]:
    """
    Quick augmentation with preset strategies.

    Args:
        X: Features
        y: Labels
        strategy: One of 'flip_h', 'flip_v', 'flip_both', 'rotate', 'full'
        seed: Random seed

    Returns:
        Augmented (X, y)
    """
    augmenter = GeometricAugmenter(seed=seed)

    strategies = {
        'flip_h': ['flip_h'],
        'flip_v': ['flip_v'],
        'flip_both': ['flip_h', 'flip_v'],
        'rotate': ['rotate_90', 'rotate_180', 'rotate_270'],
        'full': ['rotate_90', 'rotate_180', 'flip_h', 'flip_v']
    }

    transforms = strategies.get(strategy, ['flip_h', 'flip_v'])
    return augmenter.augment_batch(X, y, transforms)


if __name__ == "__main__":
    # Demo usage
    print("Geometric Augmentation Demo")
    print("=" * 40)

    # Create sample data (simulating player positions)
    np.random.seed(42)
    X_sample = np.random.rand(10, 4) * np.array([100, 53.3, 10, 5])  # x, y, speed, accel
    y_sample = np.array([0, 1, 0, 0, 1, 0, 1, 0, 0, 1])

    augmenter = GeometricAugmenter(seed=42)

    print(f"Original shape: {X_sample.shape}")
    print(f"Original x range: [{X_sample[:, 0].min():.1f}, {X_sample[:, 0].max():.1f}]")

    # Apply augmentations
    X_aug, y_aug = augmenter.augment_batch(
        X_sample, y_sample,
        transforms=['rotate_90', 'flip_h', 'flip_v']
    )

    print(f"\nAugmented shape: {X_aug.shape}")
    print(f"Augmentation multiplier: {len(X_aug) / len(X_sample):.1f}x")
    print(f"\nAugmentation summary: {augmenter.get_augmentation_summary()}")
