"""
Time Series Forecasting
========================
Train LSTM/Transformer models for time-series forecasting.
Implements windowing, forecasting horizons, and prediction intervals.
"""

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from typing import Tuple, Optional, List
import matplotlib.pyplot as plt
from pathlib import Path


class TimeSeriesDataset(Dataset):
    """Dataset for time series with sliding window."""

    def __init__(self, data: np.ndarray, window_size: int, horizon: int, stride: int = 1):
        """
        Args:
            data: Time series data (n_samples, n_features)
            window_size: Input sequence length
            horizon: Forecast horizon
            stride: Stride for sliding window
        """
        self.data = data
        self.window_size = window_size
        self.horizon = horizon
        self.stride = stride

        self.sequences = []
        self.targets = []
        self._create_sequences()

    def _create_sequences(self):
        """Create sequences using sliding window."""
        for i in range(0, len(self.data) - self.window_size - self.horizon + 1, self.stride):
            seq = self.data[i:i + self.window_size]
            target = self.data[i + self.window_size:i + self.window_size + self.horizon]
            self.sequences.append(seq)
            self.targets.append(target)

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
        return (torch.FloatTensor(self.sequences[idx]),
                torch.FloatTensor(self.targets[idx]))


class LSTMForecaster(nn.Module):
    """LSTM model for time series forecasting."""

    def __init__(self, input_size: int, hidden_size: int, num_layers: int,
                 horizon: int, dropout: float = 0.3):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.horizon = horizon

        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout if num_layers > 1 else 0,
            batch_first=True
        )

        self.fc = nn.Linear(hidden_size, horizon * input_size)

    def forward(self, x):
        # x: (batch, seq_len, input_size)
        lstm_out, _ = self.lstm(x)
        # Take last hidden state
        last_hidden = lstm_out[:, -1, :]
        # Predict horizon steps
        output = self.fc(last_hidden)
        # Reshape to (batch, horizon, input_size)
        output = output.view(-1, self.horizon, x.size(-1))
        return output


class TransformerForecaster(nn.Module):
    """Transformer model for time series forecasting."""

    def __init__(self, input_size: int, d_model: int, nhead: int,
                 num_layers: int, horizon: int, dropout: float = 0.1):
        super().__init__()
        self.input_size = input_size
        self.d_model = d_model
        self.horizon = horizon

        self.input_embedding = nn.Linear(input_size, d_model)
        self.pos_encoder = PositionalEncoding(d_model, dropout)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=d_model * 4,
            dropout=dropout,
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers)

        self.fc = nn.Linear(d_model, horizon * input_size)

    def forward(self, x):
        # x: (batch, seq_len, input_size)
        x = self.input_embedding(x)
        x = self.pos_encoder(x)
        x = self.transformer(x)
        # Take last position
        last_hidden = x[:, -1, :]
        output = self.fc(last_hidden)
        output = output.view(-1, self.horizon, self.input_size)
        return output


class PositionalEncoding(nn.Module):
    """Positional encoding for transformer."""

    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 5000):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-np.log(10000.0) / d_model))
        pe = torch.zeros(max_len, 1, d_model)
        pe[:, 0, 0::2] = torch.sin(position * div_term)
        pe[:, 0, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:x.size(1)].transpose(0, 1)
        return self.dropout(x)


class TimeSeriesTrainer:
    """Trainer for time series forecasting models."""

    def __init__(self, model: nn.Module, device: str = 'cuda'):
        self.model = model.to(device)
        self.device = device
        self.history = {'train_loss': [], 'val_loss': []}

    def train_epoch(self, dataloader: DataLoader, optimizer, criterion):
        """Train for one epoch."""
        self.model.train()
        total_loss = 0
        for batch_x, batch_y in dataloader:
            batch_x = batch_x.to(self.device)
            batch_y = batch_y.to(self.device)

            optimizer.zero_grad()
            outputs = self.model(batch_x)
            loss = criterion(outputs, batch_y)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        return total_loss / len(dataloader)

    def validate(self, dataloader: DataLoader, criterion):
        """Validate on validation set."""
        self.model.eval()
        total_loss = 0
        with torch.no_grad():
            for batch_x, batch_y in dataloader:
                batch_x = batch_x.to(self.device)
                batch_y = batch_y.to(self.device)

                outputs = self.model(batch_x)
                loss = criterion(outputs, batch_y)
                total_loss += loss.item()

        return total_loss / len(dataloader)

    def fit(self, train_loader: DataLoader, val_loader: DataLoader,
            epochs: int, lr: float = 0.001, patience: int = 10):
        """Train the model with early stopping."""
        optimizer = torch.optim.Adam(self.model.parameters(), lr=lr)
        criterion = nn.MSELoss()

        best_val_loss = float('inf')
        patience_counter = 0

        for epoch in range(epochs):
            train_loss = self.train_epoch(train_loader, optimizer, criterion)
            val_loss = self.validate(val_loader, criterion)

            self.history['train_loss'].append(train_loss)
            self.history['val_loss'].append(val_loss)

            print(f"Epoch {epoch+1}/{epochs} - Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}")

            # Early stopping
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_counter = 0
                # Save best model
                torch.save(self.model.state_dict(), 'outputs/models/best_forecaster.pt')
            else:
                patience_counter += 1
                if patience_counter >= patience:
                    print(f"Early stopping at epoch {epoch+1}")
                    break

        # Load best model
        self.model.load_state_dict(torch.load('outputs/models/best_forecaster.pt'))

    def predict(self, dataloader: DataLoader) -> Tuple[np.ndarray, np.ndarray]:
        """Make predictions."""
        self.model.eval()
        predictions = []
        actuals = []

        with torch.no_grad():
            for batch_x, batch_y in dataloader:
                batch_x = batch_x.to(self.device)
                outputs = self.model(batch_x)
                predictions.append(outputs.cpu().numpy())
                actuals.append(batch_y.numpy())

        return np.vstack(predictions), np.vstack(actuals)

    def plot_forecast(self, predictions: np.ndarray, actuals: np.ndarray,
                      save_path: Optional[str] = None):
        """Plot forecast vs actual."""
        plt.figure(figsize=(12, 6))
        plt.plot(actuals.flatten(), label='Actual', alpha=0.7)
        plt.plot(predictions.flatten(), label='Predicted', alpha=0.7)
        plt.xlabel('Time Step')
        plt.ylabel('Value')
        plt.title('Time Series Forecast')
        plt.legend()
        plt.grid(True)

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()


def compute_metrics(predictions: np.ndarray, actuals: np.ndarray) -> dict:
    """Compute forecasting metrics."""
    mae = np.mean(np.abs(predictions - actuals))
    mse = np.mean((predictions - actuals) ** 2)
    rmse = np.sqrt(mse)
    mape = np.mean(np.abs((actuals - predictions) / (actuals + 1e-8))) * 100

    return {
        'mae': mae,
        'mse': mse,
        'rmse': rmse,
        'mape': mape
    }


if __name__ == "__main__":
    print("Time Series Forecasting Module")
    print("=" * 50)
    print("Ready for training LSTM/Transformer forecasters")
