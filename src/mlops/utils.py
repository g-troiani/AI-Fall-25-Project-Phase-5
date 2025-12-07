"""
MLOps Utilities
===============
Logging, timing, hashing, and reproducibility utilities for the ML pipeline.
"""

import os
import csv
import json
import time
import hashlib
import subprocess
import pathlib
from datetime import datetime


def sha256(path: str) -> str:
    """Compute SHA256 hash of a file."""
    h = hashlib.sha256()
    h.update(pathlib.Path(path).read_bytes())
    return h.hexdigest()


def sha256_partial(path: str, max_bytes: int = 10000) -> str:
    """Compute SHA256 hash of first N bytes of a file."""
    if not os.path.exists(path):
        return 'file_not_found'
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        h.update(f.read(max_bytes))
    return h.hexdigest()[:16]


def hash_dict(d: dict) -> str:
    """Compute SHA256 hash of a dictionary."""
    return hashlib.sha256(
        json.dumps(d, sort_keys=True, default=str).encode()
    ).hexdigest()[:16]


def git_commit() -> str:
    """Get current git commit hash (short)."""
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"]
        ).decode().strip()
    except Exception:
        return "nogit"


class Timer:
    """Context manager for timing code blocks."""
    
    def __enter__(self):
        self.start = time.perf_counter()
        return self

    def __exit__(self, *args):
        self.elapsed = time.perf_counter() - self.start
        self.elapsed_ms = self.elapsed * 1000


class RunLogger:
    """
    MLOps run logger - appends experiment metrics to CSV.
    
    Schema:
        run_id, timestamp, git_commit, data_hash, config_hash, seed,
        component, metric_name, metric_value, latency_ms, params_json, notes
    """
    
    FIELDNAMES = [
        'run_id', 'timestamp', 'git_commit', 'data_hash', 'config_hash',
        'seed', 'component', 'metric_name', 'metric_value', 'latency_ms',
        'params_json', 'notes'
    ]

    def __init__(self, log_path: str, data_path: str = None, config: dict = None, seed: int = 42):
        self.log_path = log_path
        self.seed = seed
        self.run_id = datetime.now().strftime('%Y%m%d_%H%M%S')
        self.data_hash = sha256_partial(data_path) if data_path else 'N/A'
        self.config_hash = hash_dict(config) if config else 'N/A'
        self._git_commit = git_commit()
        
        # Initialize CSV if not exists
        if not os.path.exists(log_path):
            os.makedirs(os.path.dirname(log_path), exist_ok=True)
            with open(log_path, 'w', newline='') as f:
                writer = csv.DictWriter(f, fieldnames=self.FIELDNAMES)
                writer.writeheader()

    def log(self, component: str, metric_name: str, metric_value, 
            latency_ms: float = 0, params: dict = None, notes: str = ''):
        """Append a metric row to the run log."""
        row = {
            'run_id': self.run_id,
            'timestamp': datetime.now().isoformat(),
            'git_commit': self._git_commit,
            'data_hash': self.data_hash,
            'config_hash': self.config_hash,
            'seed': self.seed,
            'component': component,
            'metric_name': metric_name,
            'metric_value': f'{metric_value:.6f}' if isinstance(metric_value, float) else metric_value,
            'latency_ms': f'{latency_ms:.2f}',
            'params_json': json.dumps(params)[:200] if params else '{}',
            'notes': notes
        }
        with open(self.log_path, 'a', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=self.FIELDNAMES)
            writer.writerow(row)


def log(msg: str):
    """Simple timestamped logging to stdout."""
    print(f"[{datetime.now().strftime('%H:%M:%S')}] {msg}")


def set_seeds(seed: int = 42):
    """Set random seeds for reproducibility."""
    import numpy as np
    import random
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    except ImportError:
        pass
