"""
CNN for Vision Tasks
=====================
Train Convolutional Neural Networks for image classification.
Includes data augmentation, MC Dropout, and calibration.
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from typing import Tuple, Optional, List
import time


class SimpleCNN(nn.Module):
    """Simple CNN for image classification."""

    def __init__(self, input_channels: int = 3, num_classes: int = 10,
                 dropout_rate: float = 0.5):
        super().__init__()

        # Convolutional layers
        self.conv1 = nn.Conv2d(input_channels, 32, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(32)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(64)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm2d(128)

        self.pool = nn.MaxPool2d(2, 2)
        self.dropout = nn.Dropout(dropout_rate)

        # Fully connected layers (size depends on input image size)
        # Assuming 32x32 input -> after 3 pools: 4x4
        self.fc1 = nn.Linear(128 * 4 * 4, 256)
        self.fc2 = nn.Linear(256, 128)
        self.fc3 = nn.Linear(128, num_classes)

    def forward(self, x, mc_dropout: bool = False):
        # Conv block 1
        x = self.pool(F.relu(self.bn1(self.conv1(x))))

        # Conv block 2
        x = self.pool(F.relu(self.bn2(self.conv2(x))))

        # Conv block 3
        x = self.pool(F.relu(self.bn3(self.conv3(x))))

        # Flatten
        x = x.view(x.size(0), -1)

        # FC layers with dropout
        x = F.relu(self.fc1(x))
        if mc_dropout or self.training:
            x = self.dropout(x)

        x = F.relu(self.fc2(x))
        if mc_dropout or self.training:
            x = self.dropout(x)

        x = self.fc3(x)
        return x


class VisionCNNTrainer:
    """Trainer for CNN models."""

    def __init__(self, model: nn.Module, device: str = 'cuda'):
        self.model = model.to(device)
        self.device = device
        self.history = {
            'train_loss': [],
            'train_acc': [],
            'val_loss': [],
            'val_acc': []
        }

    def train_epoch(self, dataloader: DataLoader, optimizer, criterion):
        """Train for one epoch."""
        self.model.train()
        total_loss = 0
        correct = 0
        total = 0

        for images, labels in dataloader:
            images = images.to(self.device)
            labels = labels.to(self.device)

            optimizer.zero_grad()
            outputs = self.model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()

        avg_loss = total_loss / len(dataloader)
        accuracy = 100. * correct / total
        return avg_loss, accuracy

    def validate(self, dataloader: DataLoader, criterion):
        """Validate on validation set."""
        self.model.eval()
        total_loss = 0
        correct = 0
        total = 0

        with torch.no_grad():
            for images, labels in dataloader:
                images = images.to(self.device)
                labels = labels.to(self.device)

                outputs = self.model(images)
                loss = criterion(outputs, labels)

                total_loss += loss.item()
                _, predicted = outputs.max(1)
                total += labels.size(0)
                correct += predicted.eq(labels).sum().item()

        avg_loss = total_loss / len(dataloader)
        accuracy = 100. * correct / total
        return avg_loss, accuracy

    def fit(self, train_loader: DataLoader, val_loader: DataLoader,
            epochs: int, lr: float = 0.001, patience: int = 10):
        """Train the model with early stopping."""
        optimizer = torch.optim.Adam(self.model.parameters(), lr=lr)
        criterion = nn.CrossEntropyLoss()

        best_val_acc = 0
        patience_counter = 0

        for epoch in range(epochs):
            train_loss, train_acc = self.train_epoch(train_loader, optimizer, criterion)
            val_loss, val_acc = self.validate(val_loader, criterion)

            self.history['train_loss'].append(train_loss)
            self.history['train_acc'].append(train_acc)
            self.history['val_loss'].append(val_loss)
            self.history['val_acc'].append(val_acc)

            print(f"Epoch {epoch+1}/{epochs}")
            print(f"  Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%")
            print(f"  Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%")

            # Early stopping based on validation accuracy
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                patience_counter = 0
                torch.save(self.model.state_dict(), 'outputs/models/best_cnn.pt')
                print(f"  ✓ New best model saved!")
            else:
                patience_counter += 1
                if patience_counter >= patience:
                    print(f"Early stopping at epoch {epoch+1}")
                    break

        # Load best model
        self.model.load_state_dict(torch.load('outputs/models/best_cnn.pt'))
        print(f"\nBest validation accuracy: {best_val_acc:.2f}%")

    def predict_with_mc_dropout(self, dataloader: DataLoader,
                                n_samples: int = 50) -> Tuple[np.ndarray, np.ndarray]:
        """
        Predict with MC Dropout for uncertainty estimation.

        Returns:
            mean_predictions: Mean predictions across MC samples
            uncertainty: Standard deviation across MC samples
        """
        self.model.train()  # Keep dropout active
        all_predictions = []

        with torch.no_grad():
            for _ in range(n_samples):
                batch_predictions = []
                for images, _ in dataloader:
                    images = images.to(self.device)
                    outputs = self.model(images, mc_dropout=True)
                    probs = F.softmax(outputs, dim=1)
                    batch_predictions.append(probs.cpu().numpy())
                all_predictions.append(np.vstack(batch_predictions))

        all_predictions = np.array(all_predictions)  # (n_samples, n_data, n_classes)

        mean_predictions = all_predictions.mean(axis=0)
        uncertainty = all_predictions.std(axis=0)

        return mean_predictions, uncertainty

    def predict(self, dataloader: DataLoader) -> Tuple[np.ndarray, np.ndarray]:
        """Make standard predictions."""
        self.model.eval()
        all_predictions = []
        all_labels = []

        with torch.no_grad():
            for images, labels in dataloader:
                images = images.to(self.device)
                outputs = self.model(images)
                probs = F.softmax(outputs, dim=1)
                all_predictions.append(probs.cpu().numpy())
                all_labels.append(labels.numpy())

        return np.vstack(all_predictions), np.concatenate(all_labels)

    def measure_latency(self, input_shape: Tuple[int, int, int, int],
                       n_runs: int = 100) -> dict:
        """
        Measure inference latency.

        Args:
            input_shape: (batch_size, channels, height, width)
            n_runs: Number of inference runs

        Returns:
            Dictionary with latency statistics
        """
        self.model.eval()
        dummy_input = torch.randn(input_shape).to(self.device)

        # Warmup
        with torch.no_grad():
            for _ in range(10):
                _ = self.model(dummy_input)

        # Measure
        latencies = []
        with torch.no_grad():
            for _ in range(n_runs):
                start = time.perf_counter()
                _ = self.model(dummy_input)
                torch.cuda.synchronize() if torch.cuda.is_available() else None
                end = time.perf_counter()
                latencies.append((end - start) * 1000)  # Convert to ms

        return {
            'mean_ms': np.mean(latencies),
            'std_ms': np.std(latencies),
            'min_ms': np.min(latencies),
            'max_ms': np.max(latencies),
            'batch_size': input_shape[0]
        }


def compute_ece(predictions: np.ndarray, labels: np.ndarray, n_bins: int = 10) -> float:
    """
    Compute Expected Calibration Error (ECE).

    Args:
        predictions: Predicted probabilities (n_samples, n_classes)
        labels: True labels (n_samples,)
        n_bins: Number of bins

    Returns:
        ECE value
    """
    confidences = predictions.max(axis=1)
    pred_labels = predictions.argmax(axis=1)
    accuracies = (pred_labels == labels).astype(float)

    ece = 0
    bin_boundaries = np.linspace(0, 1, n_bins + 1)

    for i in range(n_bins):
        bin_lower = bin_boundaries[i]
        bin_upper = bin_boundaries[i + 1]

        in_bin = (confidences > bin_lower) & (confidences <= bin_upper)
        prop_in_bin = in_bin.mean()

        if prop_in_bin > 0:
            accuracy_in_bin = accuracies[in_bin].mean()
            avg_confidence_in_bin = confidences[in_bin].mean()
            ece += np.abs(avg_confidence_in_bin - accuracy_in_bin) * prop_in_bin

    return ece


if __name__ == "__main__":
    print("Vision CNN Training Module")
    print("=" * 50)
    print("Ready for image classification with CNNs")
