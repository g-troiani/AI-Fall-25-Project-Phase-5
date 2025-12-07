"""
Neural Network Training (MLP)
=============================
MLP Classifier with MC Dropout for uncertainty estimation.
Source: Project 3
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score


class MLPClassifier(nn.Module):
    """
    MLP Classifier with MC Dropout.
    
    Architecture (from Project 3):
        Linear(d_in, hidden1) -> ReLU -> Dropout -> 
        Linear(hidden1, hidden2) -> ReLU -> Linear(hidden2, d_out)
    """

    def __init__(self, d_in: int, d_out: int = 2, hidden1: int = 128, 
                 hidden2: int = 64, dropout: float = 0.2):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_in, hidden1),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden1, hidden2),
            nn.ReLU(),
            nn.Linear(hidden2, d_out)
        )

    def forward(self, x):
        return self.net(x)

    def mc_predict(self, x, n_mc: int = 20):
        """
        MC Dropout inference for uncertainty estimation.
        Returns mean and variance of predictions.
        """
        self.train()  # Keep dropout on
        all_mc_probs = []

        with torch.no_grad():
            for _ in range(n_mc):
                logits = self.forward(x)
                probs = F.softmax(logits, dim=1)
                all_mc_probs.append(probs.cpu().numpy())

        all_mc_probs = np.stack(all_mc_probs, axis=0)  # [n_mc, N, C]
        mean_probs = all_mc_probs.mean(axis=0)
        var_probs = all_mc_probs.var(axis=0)

        self.eval()
        return torch.FloatTensor(mean_probs), torch.FloatTensor(var_probs)


class NeuralNetAdapter:
    """
    Adapter for PyTorch neural networks.
    Provides consistent fit/predict/predict_proba interface.
    """

    def __init__(self, model: nn.Module, lr: float = 0.001, epochs: int = 50, 
                 batch_size: int = 32, device: str = 'cpu'):
        self.model = model.to(device)
        self.lr = lr
        self.epochs = epochs
        self.batch_size = batch_size
        self.device = device
        self.history = {'train_loss': [], 'val_loss': [], 'val_acc': []}

    def fit(self, X_train, y_train, X_val=None, y_val=None):
        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr)
        criterion = nn.CrossEntropyLoss()

        X_train_t = torch.FloatTensor(X_train).to(self.device)
        y_train_t = torch.LongTensor(y_train).to(self.device)

        dataset = TensorDataset(X_train_t, y_train_t)
        loader = DataLoader(dataset, batch_size=self.batch_size, shuffle=True)

        for epoch in range(self.epochs):
            self.model.train()
            epoch_loss = 0

            for X_batch, y_batch in loader:
                optimizer.zero_grad()
                outputs = self.model(X_batch)
                loss = criterion(outputs, y_batch)
                loss.backward()
                optimizer.step()
                epoch_loss += loss.item()

            self.history['train_loss'].append(epoch_loss / len(loader))

            if X_val is not None:
                val_loss, val_acc = self._validate(X_val, y_val, criterion)
                self.history['val_loss'].append(val_loss)
                self.history['val_acc'].append(val_acc)

        return self

    def _validate(self, X_val, y_val, criterion):
        self.model.eval()
        with torch.no_grad():
            X_val_t = torch.FloatTensor(X_val).to(self.device)
            y_val_t = torch.LongTensor(y_val).to(self.device)
            outputs = self.model(X_val_t)
            loss = criterion(outputs, y_val_t).item()
            preds = outputs.argmax(dim=1)
            acc = (preds == y_val_t).float().mean().item()
        return loss, acc

    def predict(self, X):
        self.model.eval()
        with torch.no_grad():
            X_t = torch.FloatTensor(X).to(self.device)
            outputs = self.model(X_t)
            return outputs.argmax(dim=1).cpu().numpy()

    def predict_proba(self, X):
        self.model.eval()
        with torch.no_grad():
            X_t = torch.FloatTensor(X).to(self.device)
            outputs = self.model(X_t)
            probs = F.softmax(outputs, dim=1)
            return probs[:, 1].cpu().numpy()

    def predict_with_uncertainty(self, X, n_samples: int = 10):
        """MC Dropout inference for uncertainty estimation."""
        X_t = torch.FloatTensor(X).to(self.device)
        mean_pred, var_pred = self.model.mc_predict(X_t, n_samples)
        return mean_pred.cpu().numpy(), var_pred.cpu().numpy()


def train_mlp(X_train, y_train, X_val, y_val, config: dict, 
              seed: int = 42, run_logger=None, timer_cls=None, log_fn=None):
    """
    Train MLP model.
    
    Returns:
        adapter: Trained NeuralNetAdapter
        history: Training history dict
    """
    if timer_cls is None:
        from ..mlops.utils import Timer as timer_cls
    if log_fn is None:
        from ..mlops.utils import log as log_fn
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    d_in = X_train.shape[1]
    
    hidden_sizes = config.get('hidden_sizes', [128, 64])
    model = MLPClassifier(
        d_in=d_in, d_out=2,
        hidden1=hidden_sizes[0], hidden2=hidden_sizes[1],
        dropout=config.get('dropout', 0.3)
    )
    
    adapter = NeuralNetAdapter(
        model, 
        lr=config.get('lr', 0.001),
        epochs=config.get('epochs', 50),
        batch_size=config.get('batch_size', 32),
        device=device
    )
    
    log_fn(f"Training MLP on {device}...")
    
    with timer_cls() as t:
        adapter.fit(X_train, y_train, X_val, y_val)
    
    # Evaluate
    val_pred = adapter.predict(X_val)
    val_proba = adapter.predict_proba(X_val)
    
    metrics = {
        'accuracy': accuracy_score(y_val, val_pred),
        'precision': precision_score(y_val, val_pred, zero_division=0),
        'recall': recall_score(y_val, val_pred, zero_division=0),
        'f1': f1_score(y_val, val_pred, zero_division=0),
        'train_time_ms': t.elapsed_ms
    }
    
    try:
        metrics['roc_auc'] = roc_auc_score(y_val, val_proba)
    except ValueError:
        metrics['roc_auc'] = 0.0
    
    if run_logger:
        run_logger.log('neural', 'mlp_val_accuracy', metrics['accuracy'],
                       latency_ms=t.elapsed_ms, 
                       params={'hidden': hidden_sizes, 'dropout': config.get('dropout', 0.3)})
    
    return adapter, adapter.history, metrics
