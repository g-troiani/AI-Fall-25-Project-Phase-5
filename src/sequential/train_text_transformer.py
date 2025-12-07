"""
Transformer Training
====================
Transformer for trajectory prediction with classification head.
Source: Project 4
"""

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import accuracy_score, f1_score

from ..neural.train_text_lstm import (SequentialModelAdapter,
                                       compute_time_series_metrics,
                                       measure_neural_latency)


class PositionalEncoding(nn.Module):
    """Sinusoidal positional encoding."""

    def __init__(self, d_model: int, max_len: int = 100, dropout: float = 0.1):
        super().__init__()
        self.dropout = nn.Dropout(dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-np.log(10000.0) / d_model))

        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)

        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:, :x.size(1), :]
        return self.dropout(x)


class TransformerTrajectoryModel(nn.Module):
    """
    Transformer for trajectory prediction.
    
    Architecture (from Project 4):
        - Full encoder-decoder Transformer
        - Positional encoding with sqrt(d_model) scaling
        - Autoregressive decoding with teacher forcing
        - Dual output: trajectory + classification
    """

    def __init__(self, input_size: int, d_model: int = 64, nhead: int = 4,
                 num_layers: int = 2, dim_feedforward: int = 256,
                 output_seq_len: int = 10, dropout: float = 0.1):
        super().__init__()
        self.d_model = d_model
        self.output_seq_len = output_seq_len

        # Input projections
        self.input_proj = nn.Linear(input_size, d_model)
        self.output_proj = nn.Linear(2, d_model)

        # Positional encoding
        self.pos_encoder = PositionalEncoding(d_model, dropout=dropout)

        # Full Transformer
        self.transformer = nn.Transformer(
            d_model=d_model,
            nhead=nhead,
            num_encoder_layers=num_layers,
            num_decoder_layers=num_layers,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True
        )

        # Output heads
        self.fc_traj = nn.Linear(d_model, 2)
        self.fc_class = nn.Linear(d_model, 2)

        self.attention_weights = None

    def generate_square_subsequent_mask(self, sz: int):
        """Generate causal mask."""
        mask = torch.triu(torch.ones(sz, sz), diagonal=1).bool()
        return mask.to(next(self.parameters()).device)

    def forward(self, src, tgt=None, teacher_forcing_ratio=0.5):
        batch_size = src.size(0)

        # Encode source with sqrt(d_model) scaling
        src_emb = self.input_proj(src) * np.sqrt(self.d_model)
        src_emb = self.pos_encoder(src_emb)

        # Autoregressive trajectory decoding
        decoder_input = torch.zeros(batch_size, 1, 2).to(src.device)
        traj_outputs = []

        for t in range(self.output_seq_len):
            tgt_emb = self.output_proj(decoder_input) * np.sqrt(self.d_model)
            tgt_emb = self.pos_encoder(tgt_emb)

            tgt_mask = self.generate_square_subsequent_mask(tgt_emb.size(1))

            out = self.transformer(src_emb, tgt_emb, tgt_mask=tgt_mask)
            pred = self.fc_traj(out[:, -1:, :])
            traj_outputs.append(pred)

            # Teacher forcing
            if tgt is not None and np.random.random() < teacher_forcing_ratio:
                next_input = tgt[:, t:t+1, :]
            else:
                next_input = pred

            decoder_input = torch.cat([decoder_input, next_input], dim=1)

        trajectory = torch.cat(traj_outputs, dim=1)

        # Classification from mean-pooled encoder output
        encoder_out = self.transformer.encoder(src_emb)
        class_logits = self.fc_class(encoder_out.mean(dim=1))

        return trajectory, class_logits


def train_transformer(X_train, Y_traj_train, Y_class_train,
                      X_val, Y_traj_val, Y_class_val,
                      config: dict, seed: int = 42, run_logger=None,
                      timer_cls=None, log_fn=None):
    """
    Train Transformer model.
    
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
    
    model = TransformerTrajectoryModel(
        input_size=input_size,
        d_model=config.get('d_model', 64),
        nhead=config.get('nhead', 4),
        num_layers=config.get('num_layers', 2),
        dim_feedforward=config.get('dim_feedforward', 256),
        output_seq_len=output_seq_len,
        dropout=config.get('dropout', 0.1)
    )
    
    adapter = SequentialModelAdapter(
        model,
        lr=config.get('lr', 0.001),
        epochs=config.get('epochs', 30),
        batch_size=config.get('batch_size', 16),
        device=device
    )
    
    log_fn(f"Training Transformer on {device}...")
    
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
    mc_mean_std = np.mean(mc_std)

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
        run_logger.log('sequential', 'transformer_val_traj_mse', ts_metrics['mse'],
                       latency_ms=t.elapsed_ms,
                       params={'d_model': config.get('d_model', 64),
                               'nhead': config.get('nhead', 4)})
        run_logger.log('sequential', 'transformer_val_mae', ts_metrics['mae'])
        run_logger.log('sequential', 'transformer_val_mape', ts_metrics['mape'])
        run_logger.log('sequential', 'transformer_val_mase', ts_metrics['mase'])
        run_logger.log('sequential', 'transformer_val_class_acc', class_acc)
        run_logger.log('sequential', 'transformer_mc_dropout_std', mc_mean_std)

    return adapter, adapter.history, metrics
