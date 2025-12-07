"""
Generative VAE-style Sampler
============================
VAE-style data augmentation using PCA-based latent space sampling.
Source: Project 5
"""

import numpy as np


class VAELatentSampler:
    """
    VAE-style sampler using PCA for dimensionality reduction.
    Learns latent distribution from training data and samples new points.
    """

    def __init__(self, latent_dim: int = 8, random_state: int = 42):
        self.latent_dim = latent_dim
        self.random_state = random_state
        self.rng = np.random.default_rng(random_state)

        # Will be set during fit
        self.mean = None
        self.std = None
        self.U = None
        self.S = None
        self.Vt = None
        self.latent_mean = None
        self.latent_std = None
        self.original_shape = None

    def fit(self, X):
        """Fit the sampler to training data."""
        self.original_shape = X.shape[1:]  # (seq_len, n_features) or (n_features,)

        # Flatten sequences
        X_flat = X.reshape(X.shape[0], -1)

        # Normalize
        self.mean = X_flat.mean(axis=0)
        self.std = X_flat.std(axis=0) + 1e-8
        X_norm = (X_flat - self.mean) / self.std

        # SVD for PCA
        self.U, self.S, self.Vt = np.linalg.svd(X_norm, full_matrices=False)

        # Project to latent space
        Z = X_norm @ self.Vt[:self.latent_dim].T

        # Learn latent distribution
        self.latent_mean = Z.mean(axis=0)
        self.latent_std = Z.std(axis=0) + 1e-8

        return self

    def sample(self, n_samples: int, kl_threshold: float = None):
        """
        Generate synthetic samples with optional KL-filter for outlier rejection.

        Args:
            n_samples: Number of samples to generate
            kl_threshold: If set, reject samples with latent KL > threshold (default: 3.0)
        """
        if kl_threshold is None:
            kl_threshold = 3.0  # Default: reject samples > 3 std from mean

        valid_samples = []
        max_attempts = n_samples * 5  # Prevent infinite loops
        attempts = 0

        while len(valid_samples) < n_samples and attempts < max_attempts:
            batch_size = min(n_samples * 2, max_attempts - attempts)
            Z = self.rng.normal(self.latent_mean, self.latent_std, (batch_size, self.latent_dim))

            # KL-filter: reject outliers in latent space
            # Compute Mahalanobis-like distance (standardized distance from mean)
            z_normalized = (Z - self.latent_mean) / self.latent_std
            distances = np.sqrt(np.sum(z_normalized ** 2, axis=1))

            # Keep samples within threshold
            valid_mask = distances <= kl_threshold
            valid_Z = Z[valid_mask]

            for z in valid_Z:
                if len(valid_samples) < n_samples:
                    valid_samples.append(z)

            attempts += batch_size

        Z_valid = np.array(valid_samples[:n_samples])

        # Decode: project back to data space
        X_norm = Z_valid @ self.Vt[:self.latent_dim]

        # Denormalize
        X_flat = X_norm * self.std + self.mean

        # Reshape to original shape
        return X_flat.reshape(n_samples, *self.original_shape)

    def sample_with_labels(self, n_samples: int, X_train, Y_train,
                           kl_threshold: float = 3.0, cap_per_class: float = 1.0):
        """
        Generate samples with labels via nearest-neighbor matching.

        Guardrails:
        - KL-filter: Reject latent outliers (> kl_threshold std from mean)
        - Per-class cap: Limit synthetic samples per class to minority_count * cap_per_class

        Args:
            n_samples: Total synthetic samples to generate
            X_train: Training features
            Y_train: Training labels
            kl_threshold: KL-filter threshold (default 3.0)
            cap_per_class: Cap synthetic per class at minority_count * this value
        """
        X_synth = self.sample(n_samples, kl_threshold=kl_threshold)

        # Assign labels by nearest neighbor in training set
        X_train_flat = X_train.reshape(X_train.shape[0], -1)
        X_synth_flat = X_synth.reshape(n_samples, -1)

        Y_synth = []
        for x in X_synth_flat:
            dists = np.linalg.norm(X_train_flat - x, axis=1)
            nearest_idx = np.argmin(dists)
            Y_synth.append(Y_train[nearest_idx])

        Y_synth = np.array(Y_synth)

        # Per-class capping: limit synthetic samples per class
        unique_classes, class_counts = np.unique(Y_train, return_counts=True)
        minority_count = class_counts.min()
        max_per_class = int(minority_count * cap_per_class)

        # Filter to respect per-class cap
        keep_mask = np.ones(len(Y_synth), dtype=bool)
        for cls in unique_classes:
            cls_indices = np.where(Y_synth == cls)[0]
            if len(cls_indices) > max_per_class:
                # Randomly select which to keep
                remove_indices = self.rng.choice(cls_indices, len(cls_indices) - max_per_class, replace=False)
                keep_mask[remove_indices] = False

        X_synth_filtered = X_synth[keep_mask]
        Y_synth_filtered = Y_synth[keep_mask]

        return X_synth_filtered, Y_synth_filtered


def augment_with_vae(X_train, Y_train, augment_ratio: float = 0.1, 
                     latent_dim: int = 8, seed: int = 42):
    """
    Augment training data with VAE-generated synthetic samples.
    
    Args:
        X_train: Training features
        Y_train: Training labels
        augment_ratio: Ratio of synthetic samples to add
        latent_dim: VAE latent dimension
        seed: Random seed
        
    Returns:
        X_aug: Augmented features
        Y_aug: Augmented labels
    """
    n_synth = int(len(X_train) * augment_ratio)
    if n_synth == 0:
        return X_train, Y_train
    
    sampler = VAELatentSampler(latent_dim=latent_dim, random_state=seed)
    sampler.fit(X_train)
    
    X_synth, Y_synth = sampler.sample_with_labels(n_synth, X_train, Y_train)
    
    X_aug = np.concatenate([X_train, X_synth], axis=0)
    Y_aug = np.concatenate([Y_train, Y_synth], axis=0)
    
    return X_aug, Y_aug


def run_augmentation_ablation(train_fn, X_train, Y_train, X_val, Y_val,
                              augment_ratios: list = [0.0, 0.10, 0.25],
                              latent_dim: int = 8, seed: int = 42,
                              run_logger=None, log_fn=None):
    """
    Run ablation study comparing different augmentation ratios.

    Args:
        train_fn: Function(X, Y) -> metrics dict
        augment_ratios: List of ratios to test

    Returns:
        results: Dict with 'ablation_results' and 'sampler'
    """
    if log_fn is None:
        from ..mlops.utils import log as log_fn

    ablation_results = {}

    # Create and fit a sampler for saving
    sampler = VAELatentSampler(latent_dim=latent_dim, random_state=seed)
    sampler.fit(X_train)

    for ratio in augment_ratios:
        log_fn(f"Testing augmentation ratio: {ratio:.0%}")

        X_aug, Y_aug = augment_with_vae(X_train, Y_train, ratio, latent_dim, seed)
        metrics = train_fn(X_aug, Y_aug, X_val, Y_val)

        ablation_results[ratio] = metrics

        if run_logger:
            run_logger.log('generative', f'augment_{ratio:.0%}_accuracy',
                          metrics.get('accuracy', 0),
                          params={'ratio': ratio, 'n_synth': int(len(X_train) * ratio)})

    return {'ablation_results': ablation_results, 'sampler': sampler}
