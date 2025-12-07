"""
LSTM Encoder-Decoder Training
==============================
LSTM for trajectory prediction with classification head.
Source: Project 4
"""

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import accuracy_score, f1_score
import time


class LSTMTrajectoryModel(nn.Module):
    """
    LSTM Encoder-Decoder for trajectory prediction.
    
    Architecture (from Project 4):
        - Encoder: Multi-layer LSTM
        - Decoder: Autoregressive LSTM with teacher forcing
        - Dual output: trajectory (x,y) + classification logits
    """

    def __init__(self, input_size: int, hidden_size: int = 64, num_layers: int = 2,
                 output_seq_len: int = 10, dropout: float = 0.2):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.output_seq_len = output_seq_len

        # Encoder LSTM
        self.encoder = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0
        )

        # Decoder LSTM for trajectory
        self.decoder = nn.LSTM(
            input_size=2,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0
        )

        # Output heads
        self.fc_traj = nn.Linear(hidden_size, 2)  # Trajectory (x, y)
        self.fc_class = nn.Linear(hidden_size, 2)  # Classification
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, target=None, teacher_forcing_ratio=0.5):
        batch_size = x.size(0)

        # Encode
        encoder_out, (hidden, cell) = self.encoder(x)

        # Trajectory decoding - start with zeros
        decoder_input = torch.zeros(batch_size, 1, 2).to(x.device)
        traj_outputs = []

        for t in range(self.output_seq_len):
            decoder_out, (hidden, cell) = self.decoder(decoder_input, (hidden, cell))
            pred = self.fc_traj(self.dropout(decoder_out))
            traj_outputs.append(pred)

            # Teacher forcing
            if target is not None and np.random.random() < teacher_forcing_ratio:
                decoder_input = target[:, t:t+1, :]
            else:
                decoder_input = pred

        trajectory = torch.cat(traj_outputs, dim=1)

        # Classification from encoder's final output
        class_logits = self.fc_class(self.dropout(encoder_out[:, -1, :]))

        return trajectory, class_logits


class SequentialModelAdapter:
    """
    Adapter for sequential models (LSTM, Transformer).
    Handles dual-task training: trajectory (MSE) + classification (CrossEntropy).
    """

    def __init__(self, model: nn.Module, lr: float = 0.001, epochs: int = 30, 
                 batch_size: int = 16, traj_weight: float = 1.0, 
                 class_weight: float = 0.5, device: str = 'cpu'):
        self.model = model.to(device)
        self.lr = lr
        self.epochs = epochs
        self.batch_size = batch_size
        self.traj_weight = traj_weight
        self.class_weight = class_weight
        self.device = device
        self.history = {'train_loss': [], 'val_loss': [], 'traj_loss': [], 'class_acc': []}

    def fit(self, X_train, Y_traj_train, Y_class_train, 
            X_val=None, Y_traj_val=None, Y_class_val=None):
        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr)
        mse_loss = nn.MSELoss()
        ce_loss = nn.CrossEntropyLoss()

        X_train_t = torch.FloatTensor(X_train).to(self.device)
        Y_traj_t = torch.FloatTensor(Y_traj_train).to(self.device)
        Y_class_t = torch.LongTensor(Y_class_train).to(self.device)

        dataset = TensorDataset(X_train_t, Y_traj_t, Y_class_t)
        loader = DataLoader(dataset, batch_size=self.batch_size, shuffle=True)

        teacher_forcing_ratio = 0.5

        for epoch in range(self.epochs):
            self.model.train()
            epoch_loss = 0

            for X_batch, Y_traj_batch, Y_class_batch in loader:
                optimizer.zero_grad()

                traj_pred, class_logits = self.model(X_batch, Y_traj_batch, teacher_forcing_ratio)

                loss_traj = mse_loss(traj_pred, Y_traj_batch)
                loss_class = ce_loss(class_logits, Y_class_batch)
                loss = self.traj_weight * loss_traj + self.class_weight * loss_class

                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                optimizer.step()

                epoch_loss += loss.item()

            teacher_forcing_ratio = max(0.1, teacher_forcing_ratio - 0.01)
            self.history['train_loss'].append(epoch_loss / len(loader))

            if X_val is not None:
                val_metrics = self._validate(X_val, Y_traj_val, Y_class_val, mse_loss, ce_loss)
                self.history['val_loss'].append(val_metrics['loss'])
                self.history['traj_loss'].append(val_metrics['traj_loss'])
                self.history['class_acc'].append(val_metrics['class_acc'])

        return self

    def _validate(self, X_val, Y_traj_val, Y_class_val, mse_loss, ce_loss):
        self.model.eval()
        with torch.no_grad():
            X_val_t = torch.FloatTensor(X_val).to(self.device)
            Y_traj_t = torch.FloatTensor(Y_traj_val).to(self.device)
            Y_class_t = torch.LongTensor(Y_class_val).to(self.device)

            traj_pred, class_logits = self.model(X_val_t)

            loss_traj = mse_loss(traj_pred, Y_traj_t).item()
            loss_class = ce_loss(class_logits, Y_class_t).item()

            class_pred = class_logits.argmax(dim=1)
            class_acc = (class_pred == Y_class_t).float().mean().item()

        return {'loss': loss_traj + loss_class, 'traj_loss': loss_traj, 'class_acc': class_acc}

    def predict_trajectory(self, X):
        self.model.eval()
        with torch.no_grad():
            X_t = torch.FloatTensor(X).to(self.device)
            traj_pred, _ = self.model(X_t)
            return traj_pred.cpu().numpy()

    def predict_class(self, X):
        self.model.eval()
        with torch.no_grad():
            X_t = torch.FloatTensor(X).to(self.device)
            _, class_logits = self.model(X_t)
            return class_logits.argmax(dim=1).cpu().numpy()

    def predict_with_mc_dropout(self, X, n_samples: int = 30):
        """
        MC Dropout for uncertainty estimation.
        Returns mean prediction and prediction intervals (std).
        """
        self.model.train()  # Keep dropout active
        predictions = []

        with torch.no_grad():
            X_t = torch.FloatTensor(X).to(self.device)
            for _ in range(n_samples):
                traj_pred, _ = self.model(X_t)
                predictions.append(traj_pred.cpu().numpy())

        predictions = np.array(predictions)  # (n_samples, batch, seq, 2)
        mean_pred = predictions.mean(axis=0)
        std_pred = predictions.std(axis=0)

        self.model.eval()
        return mean_pred, std_pred


def compute_time_series_metrics(y_true, y_pred) -> dict:
    """
    Compute comprehensive time series metrics.

    Args:
        y_true: Ground truth trajectory (batch, seq_len, 2)
        y_pred: Predicted trajectory (batch, seq_len, 2)

    Returns:
        Dict with MSE, MAE, MAPE, MASE
    """
    # Flatten for per-step metrics
    y_true_flat = y_true.reshape(-1)
    y_pred_flat = y_pred.reshape(-1)

    # MSE
    mse = np.mean((y_true_flat - y_pred_flat) ** 2)

    # MAE
    mae = np.mean(np.abs(y_true_flat - y_pred_flat))

    # MAPE (avoiding division by zero)
    mask = np.abs(y_true_flat) > 1e-8
    if mask.any():
        mape = np.mean(np.abs((y_true_flat[mask] - y_pred_flat[mask]) / y_true_flat[mask])) * 100
    else:
        mape = 0.0

    # MASE: scaled by naive forecast error (using last value as prediction)
    # For trajectory: compare to persistence model (predict same as previous)
    naive_errors = []
    for i in range(len(y_true)):
        if y_true.shape[1] > 1:
            naive_pred = y_true[i, :-1, :]  # Shift: previous step predicts next
            naive_true = y_true[i, 1:, :]
            naive_errors.append(np.mean(np.abs(naive_true - naive_pred)))
    naive_mae = np.mean(naive_errors) if naive_errors else 1.0
    mase = mae / naive_mae if naive_mae > 1e-8 else mae

    return {
        'mse': mse,
        'mae': mae,
        'mape': mape,
        'mase': mase
    }


def measure_neural_latency(adapter, X, batch_sizes: list = [1, 32], n_runs: int = 10) -> dict:
    """Measure neural network inference latency."""
    latencies = {}

    for batch_size in batch_sizes:
        times = []
        X_batch = X[:batch_size] if len(X) >= batch_size else X

        # Warmup
        adapter.predict_trajectory(X_batch)

        for _ in range(n_runs):
            start = time.perf_counter()
            adapter.predict_trajectory(X_batch)
            elapsed = (time.perf_counter() - start) * 1000
            times.append(elapsed)

        latencies[f'latency_batch_{batch_size}_ms'] = np.median(times)

    return latencies


def train_lstm(X_train, Y_traj_train, Y_class_train,
               X_val, Y_traj_val, Y_class_val,
               config: dict, seed: int = 42, run_logger=None, 
               timer_cls=None, log_fn=None):
    """
    Train LSTM model.
    
    Returns:
        adapter: Trained SequentialModelAdapter
        history: Training history
        metrics: Validation metrics
    """
    if timer_cls is None:
        from ..mlops.utils import Timer as timer_cls
    if log_fn is None:
        from ..mlops.utils import log as log_fn
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    input_size = X_train.shape[2]
    output_seq_len = Y_traj_train.shape[1]
    
    model = LSTMTrajectoryModel(
        input_size=input_size,
        hidden_size=config.get('hidden_size', 64),
        num_layers=config.get('num_layers', 2),
        output_seq_len=output_seq_len,
        dropout=config.get('dropout', 0.2)
    )
    
    adapter = SequentialModelAdapter(
        model,
        lr=config.get('lr', 0.001),
        epochs=config.get('epochs', 30),
        batch_size=config.get('batch_size', 16),
        device=device
    )
    
    log_fn(f"Training LSTM on {device}...")
    
    with timer_cls() as t:
        adapter.fit(X_train, Y_traj_train, Y_class_train,
                    X_val, Y_traj_val, Y_class_val)
    
    # Evaluate with comprehensive metrics
    traj_pred = adapter.predict_trajectory(X_val)
    class_pred = adapter.predict_class(X_val)

    # Time series metrics (MSE, MAE, MAPE, MASE)
    ts_metrics = compute_time_series_metrics(Y_traj_val, traj_pred)

    class_acc = accuracy_score(Y_class_val, class_pred)
    class_f1 = f1_score(Y_class_val, class_pred, zero_division=0)

    # MC Dropout prediction intervals
    mc_mean, mc_std = adapter.predict_with_mc_dropout(X_val, n_samples=30)
    mc_mean_std = np.mean(mc_std)  # Average uncertainty

    # Latency measurements
    latencies = measure_neural_latency(adapter, X_val)

    metrics = {
        'traj_mse': ts_metrics['mse'],
        'traj_mae': ts_metrics['mae'],
        'traj_mape': ts_metrics['mape'],
        'traj_mase': ts_metrics['mase'],
        'mc_dropout_mean_std': mc_mean_std,
        'class_accuracy': class_acc,
        'class_f1': class_f1,
        'train_time_ms': t.elapsed_ms,
        **latencies
    }

    if run_logger:
        run_logger.log('sequential', 'lstm_val_traj_mse', ts_metrics['mse'],
                       latency_ms=t.elapsed_ms,
                       params={'hidden': config.get('hidden_size', 64),
                               'layers': config.get('num_layers', 2)})
        run_logger.log('sequential', 'lstm_val_mae', ts_metrics['mae'])
        run_logger.log('sequential', 'lstm_val_mape', ts_metrics['mape'])
        run_logger.log('sequential', 'lstm_val_mase', ts_metrics['mase'])
        run_logger.log('sequential', 'lstm_val_class_acc', class_acc)
        run_logger.log('sequential', 'lstm_mc_dropout_std', mc_mean_std)

    return adapter, adapter.history, metrics
