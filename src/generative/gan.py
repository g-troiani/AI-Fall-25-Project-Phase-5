"""
GAN Scaffold for Synthetic Data Generation
==========================================
Generative Adversarial Network implementation for trajectory/tabular data synthesis.
Adapted for NFL player trajectory prediction pipeline.

Source: Project 5 - Generative Models & Responsible AI
"""

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from typing import Tuple, Optional, List
from dataclasses import dataclass


@dataclass
class GANConfig:
    """Configuration for GAN training."""
    latent_dim: int = 32
    hidden_dim: int = 128
    n_features: int = 12
    lr_generator: float = 0.0002
    lr_discriminator: float = 0.0002
    beta1: float = 0.5
    beta2: float = 0.999
    n_epochs: int = 200
    batch_size: int = 32
    label_smoothing: float = 0.1
    seed: int = 42


class Generator(nn.Module):
    """
    Generator network for GAN.

    Maps random noise to synthetic data samples.
    """

    def __init__(self, latent_dim: int, hidden_dim: int, output_dim: int):
        super().__init__()

        self.model = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.LeakyReLU(0.2),

            nn.Linear(hidden_dim, hidden_dim * 2),
            nn.BatchNorm1d(hidden_dim * 2),
            nn.LeakyReLU(0.2),

            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.LeakyReLU(0.2),

            nn.Linear(hidden_dim, output_dim),
            nn.Tanh()  # Output in [-1, 1], rescale as needed
        )

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        return self.model(z)


class Discriminator(nn.Module):
    """
    Discriminator network for GAN.

    Classifies samples as real or fake.
    """

    def __init__(self, input_dim: int, hidden_dim: int):
        super().__init__()

        self.model = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.3),

            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.3),

            nn.Linear(hidden_dim // 2, 1),
            nn.Sigmoid()
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)


class TabularGAN:
    """
    GAN for tabular/trajectory data synthesis.

    Features:
    - Label-smoothing for stable training
    - Spectral normalization option
    - Training history logging
    - Sample quality metrics
    """

    def __init__(self, config: GANConfig):
        self.config = config
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        # Set seeds
        torch.manual_seed(config.seed)
        np.random.seed(config.seed)

        # Initialize networks
        self.generator = Generator(
            config.latent_dim, config.hidden_dim, config.n_features
        ).to(self.device)

        self.discriminator = Discriminator(
            config.n_features, config.hidden_dim
        ).to(self.device)

        # Optimizers
        self.opt_g = optim.Adam(
            self.generator.parameters(),
            lr=config.lr_generator,
            betas=(config.beta1, config.beta2)
        )
        self.opt_d = optim.Adam(
            self.discriminator.parameters(),
            lr=config.lr_discriminator,
            betas=(config.beta1, config.beta2)
        )

        # Loss
        self.criterion = nn.BCELoss()

        # Training history
        self.history = {
            'g_loss': [],
            'd_loss': [],
            'd_real_acc': [],
            'd_fake_acc': []
        }

        # Data normalization params
        self.data_mean = None
        self.data_std = None

    def _normalize(self, X: np.ndarray) -> np.ndarray:
        """Normalize data to [-1, 1] range."""
        if self.data_mean is None:
            self.data_mean = X.mean(axis=0)
            self.data_std = X.std(axis=0) + 1e-8
        return (X - self.data_mean) / self.data_std

    def _denormalize(self, X: np.ndarray) -> np.ndarray:
        """Denormalize data back to original range."""
        return X * self.data_std + self.data_mean

    def train(self, X_train: np.ndarray, verbose: bool = True) -> dict:
        """
        Train the GAN on real data.

        Args:
            X_train: Training data (n_samples, n_features)
            verbose: Print progress

        Returns:
            Training history dictionary
        """
        # Normalize data
        X_norm = self._normalize(X_train)
        X_tensor = torch.FloatTensor(X_norm).to(self.device)

        n_samples = len(X_train)
        n_batches = n_samples // self.config.batch_size

        for epoch in range(self.config.n_epochs):
            epoch_g_loss = 0.0
            epoch_d_loss = 0.0
            epoch_d_real = 0.0
            epoch_d_fake = 0.0

            # Shuffle data
            perm = torch.randperm(n_samples)
            X_shuffled = X_tensor[perm]

            for batch_idx in range(n_batches):
                start = batch_idx * self.config.batch_size
                end = start + self.config.batch_size
                real_batch = X_shuffled[start:end]
                batch_size = real_batch.size(0)

                # Labels with smoothing
                real_labels = torch.ones(batch_size, 1).to(self.device) * (1 - self.config.label_smoothing)
                fake_labels = torch.zeros(batch_size, 1).to(self.device)

                # ---------------------
                # Train Discriminator
                # ---------------------
                self.opt_d.zero_grad()

                # Real samples
                real_pred = self.discriminator(real_batch)
                d_loss_real = self.criterion(real_pred, real_labels)

                # Fake samples
                z = torch.randn(batch_size, self.config.latent_dim).to(self.device)
                fake_batch = self.generator(z)
                fake_pred = self.discriminator(fake_batch.detach())
                d_loss_fake = self.criterion(fake_pred, fake_labels)

                d_loss = d_loss_real + d_loss_fake
                d_loss.backward()
                self.opt_d.step()

                # ---------------------
                # Train Generator
                # ---------------------
                self.opt_g.zero_grad()

                z = torch.randn(batch_size, self.config.latent_dim).to(self.device)
                fake_batch = self.generator(z)
                fake_pred = self.discriminator(fake_batch)

                # Generator wants discriminator to think fake is real
                g_loss = self.criterion(fake_pred, torch.ones(batch_size, 1).to(self.device))
                g_loss.backward()
                self.opt_g.step()

                # Accumulate metrics
                epoch_g_loss += g_loss.item()
                epoch_d_loss += d_loss.item()
                epoch_d_real += (real_pred > 0.5).float().mean().item()
                epoch_d_fake += (fake_pred < 0.5).float().mean().item()

            # Average metrics
            self.history['g_loss'].append(epoch_g_loss / n_batches)
            self.history['d_loss'].append(epoch_d_loss / n_batches)
            self.history['d_real_acc'].append(epoch_d_real / n_batches)
            self.history['d_fake_acc'].append(epoch_d_fake / n_batches)

            if verbose and (epoch + 1) % 50 == 0:
                print(f"Epoch {epoch+1}/{self.config.n_epochs} | "
                      f"G Loss: {self.history['g_loss'][-1]:.4f} | "
                      f"D Loss: {self.history['d_loss'][-1]:.4f} | "
                      f"D Real: {self.history['d_real_acc'][-1]:.2%} | "
                      f"D Fake: {self.history['d_fake_acc'][-1]:.2%}")

        return self.history

    def generate(self, n_samples: int) -> np.ndarray:
        """
        Generate synthetic samples.

        Args:
            n_samples: Number of samples to generate

        Returns:
            Synthetic data array (n_samples, n_features)
        """
        self.generator.eval()
        with torch.no_grad():
            z = torch.randn(n_samples, self.config.latent_dim).to(self.device)
            fake_samples = self.generator(z).cpu().numpy()

        # Denormalize
        return self._denormalize(fake_samples)

    def evaluate_quality(self, X_real: np.ndarray, n_synthetic: int = 1000) -> dict:
        """
        Evaluate quality of generated samples.

        Args:
            X_real: Real data for comparison
            n_synthetic: Number of synthetic samples to generate

        Returns:
            Quality metrics dictionary
        """
        X_fake = self.generate(n_synthetic)

        # Statistical comparison
        real_mean = X_real.mean(axis=0)
        fake_mean = X_fake.mean(axis=0)
        real_std = X_real.std(axis=0)
        fake_std = X_fake.std(axis=0)

        # Mean absolute error of statistics
        mean_mae = np.abs(real_mean - fake_mean).mean()
        std_mae = np.abs(real_std - fake_std).mean()

        # Coverage: % of real data range covered by fake
        real_min, real_max = X_real.min(axis=0), X_real.max(axis=0)
        fake_min, fake_max = X_fake.min(axis=0), X_fake.max(axis=0)
        coverage = np.mean((fake_min <= real_min) & (fake_max >= real_max))

        return {
            'mean_mae': mean_mae,
            'std_mae': std_mae,
            'coverage': coverage,
            'real_mean': real_mean,
            'fake_mean': fake_mean
        }

    def save(self, path: str):
        """Save model state."""
        torch.save({
            'generator': self.generator.state_dict(),
            'discriminator': self.discriminator.state_dict(),
            'config': self.config,
            'data_mean': self.data_mean,
            'data_std': self.data_std,
            'history': self.history
        }, path)

    def load(self, path: str):
        """Load model state."""
        checkpoint = torch.load(path, map_location=self.device)
        self.generator.load_state_dict(checkpoint['generator'])
        self.discriminator.load_state_dict(checkpoint['discriminator'])
        self.data_mean = checkpoint['data_mean']
        self.data_std = checkpoint['data_std']
        self.history = checkpoint['history']


class ConditionalGAN(TabularGAN):
    """
    Conditional GAN that generates samples conditioned on class labels.

    Useful for generating class-specific synthetic data.
    """

    def __init__(self, config: GANConfig, n_classes: int = 2):
        self.n_classes = n_classes
        super().__init__(config)

        # Reinitialize with conditioning
        self.generator = self._build_conditional_generator()
        self.discriminator = self._build_conditional_discriminator()

        # Reinitialize optimizers
        self.opt_g = optim.Adam(
            self.generator.parameters(),
            lr=config.lr_generator,
            betas=(config.beta1, config.beta2)
        )
        self.opt_d = optim.Adam(
            self.discriminator.parameters(),
            lr=config.lr_discriminator,
            betas=(config.beta1, config.beta2)
        )

    def _build_conditional_generator(self) -> nn.Module:
        """Build generator with class embedding."""
        class CondGenerator(nn.Module):
            def __init__(self, latent_dim, hidden_dim, output_dim, n_classes):
                super().__init__()
                self.label_emb = nn.Embedding(n_classes, latent_dim)
                self.model = nn.Sequential(
                    nn.Linear(latent_dim * 2, hidden_dim),
                    nn.BatchNorm1d(hidden_dim),
                    nn.LeakyReLU(0.2),
                    nn.Linear(hidden_dim, hidden_dim * 2),
                    nn.BatchNorm1d(hidden_dim * 2),
                    nn.LeakyReLU(0.2),
                    nn.Linear(hidden_dim * 2, output_dim),
                    nn.Tanh()
                )

            def forward(self, z, labels):
                label_emb = self.label_emb(labels)
                x = torch.cat([z, label_emb], dim=1)
                return self.model(x)

        return CondGenerator(
            self.config.latent_dim, self.config.hidden_dim,
            self.config.n_features, self.n_classes
        ).to(self.device)

    def _build_conditional_discriminator(self) -> nn.Module:
        """Build discriminator with class conditioning."""
        class CondDiscriminator(nn.Module):
            def __init__(self, input_dim, hidden_dim, n_classes):
                super().__init__()
                self.label_emb = nn.Embedding(n_classes, input_dim)
                self.model = nn.Sequential(
                    nn.Linear(input_dim * 2, hidden_dim),
                    nn.LeakyReLU(0.2),
                    nn.Dropout(0.3),
                    nn.Linear(hidden_dim, hidden_dim // 2),
                    nn.LeakyReLU(0.2),
                    nn.Linear(hidden_dim // 2, 1),
                    nn.Sigmoid()
                )

            def forward(self, x, labels):
                label_emb = self.label_emb(labels)
                x = torch.cat([x, label_emb], dim=1)
                return self.model(x)

        return CondDiscriminator(
            self.config.n_features, self.config.hidden_dim, self.n_classes
        ).to(self.device)

    def generate_class(self, n_samples: int, class_label: int) -> np.ndarray:
        """Generate samples for a specific class."""
        self.generator.eval()
        with torch.no_grad():
            z = torch.randn(n_samples, self.config.latent_dim).to(self.device)
            labels = torch.full((n_samples,), class_label, dtype=torch.long).to(self.device)
            fake_samples = self.generator(z, labels).cpu().numpy()
        return self._denormalize(fake_samples)


def augment_with_gan(X_train: np.ndarray, y_train: np.ndarray,
                     augment_ratio: float = 0.25,
                     config: Optional[GANConfig] = None,
                     verbose: bool = False) -> Tuple[np.ndarray, np.ndarray]:
    """
    Augment training data using GAN.

    Args:
        X_train: Training features
        y_train: Training labels
        augment_ratio: Ratio of synthetic samples to add
        config: GAN configuration
        verbose: Print training progress

    Returns:
        Augmented (X, y) arrays
    """
    if config is None:
        config = GANConfig(n_features=X_train.shape[1])
    else:
        config.n_features = X_train.shape[1]

    n_synthetic = int(len(X_train) * augment_ratio)

    if n_synthetic == 0:
        return X_train, y_train

    # Train GAN
    gan = TabularGAN(config)
    gan.train(X_train, verbose=verbose)

    # Generate synthetic samples
    X_synthetic = gan.generate(n_synthetic)

    # Assign labels based on nearest neighbor
    from sklearn.neighbors import NearestNeighbors
    nn = NearestNeighbors(n_neighbors=1)
    nn.fit(X_train)
    _, indices = nn.kneighbors(X_synthetic)
    y_synthetic = y_train[indices.flatten()]

    # Combine
    X_aug = np.vstack([X_train, X_synthetic])
    y_aug = np.concatenate([y_train, y_synthetic])

    return X_aug, y_aug


if __name__ == "__main__":
    print("GAN Scaffold Demo")
    print("=" * 40)

    # Create sample data
    np.random.seed(42)
    X_demo = np.random.rand(200, 12) * np.array([100, 53.3, 10, 5, 360, 100, 10, 10, 180, 20, 10, 10])
    y_demo = (X_demo[:, 0] + X_demo[:, 1] > 75).astype(int)

    # Configure and train
    config = GANConfig(
        latent_dim=16,
        hidden_dim=64,
        n_features=12,
        n_epochs=100,
        batch_size=32
    )

    gan = TabularGAN(config)
    print("Training GAN...")
    history = gan.train(X_demo, verbose=True)

    # Generate samples
    X_synthetic = gan.generate(50)
    print(f"\nGenerated {len(X_synthetic)} synthetic samples")
    print(f"Shape: {X_synthetic.shape}")

    # Evaluate quality
    quality = gan.evaluate_quality(X_demo)
    print(f"\nQuality Metrics:")
    print(f"  Mean MAE: {quality['mean_mae']:.4f}")
    print(f"  Std MAE: {quality['std_mae']:.4f}")
    print(f"  Coverage: {quality['coverage']:.2%}")
