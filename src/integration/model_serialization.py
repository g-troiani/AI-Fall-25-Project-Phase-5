"""
Model Serialization
===================
Save and load trained models.
Source: Project 5 (Results)
"""

import os
import pickle
import numpy as np

try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False


def save_classical_model(model, filepath: str):
    """Save sklearn/classical model."""
    with open(filepath, 'wb') as f:
        pickle.dump(model, f)


def load_classical_model(filepath: str):
    """Load sklearn/classical model."""
    with open(filepath, 'rb') as f:
        return pickle.load(f)


def save_pytorch_model(model, filepath: str):
    """Save PyTorch model state dict."""
    if not TORCH_AVAILABLE:
        raise ImportError("PyTorch not available")
    torch.save(model.state_dict(), filepath)


def load_pytorch_model(model, filepath: str):
    """Load PyTorch model state dict."""
    if not TORCH_AVAILABLE:
        raise ImportError("PyTorch not available")
    model.load_state_dict(torch.load(filepath))
    return model


def save_scaler(scaler, filepath: str):
    """Save sklearn scaler."""
    with open(filepath, 'wb') as f:
        pickle.dump(scaler, f)


def load_scaler(filepath: str):
    """Load sklearn scaler."""
    with open(filepath, 'rb') as f:
        return pickle.load(f)


def save_q_table(Q: np.ndarray, filepath: str):
    """Save Q-table as numpy array."""
    np.save(filepath, Q)


def load_q_table(filepath: str) -> np.ndarray:
    """Load Q-table from numpy file."""
    return np.load(filepath)


def save_all_models(output_dir: str, models: dict, scaler=None, 
                    vae_sampler=None, q_table=None, log_fn=None):
    """
    Save all trained models to output directory.
    
    Args:
        output_dir: Base output directory
        models: Dict with keys like 'classical', 'mlp', 'lstm', 'transformer'
        scaler: Fitted StandardScaler
        vae_sampler: VAELatentSampler instance
        q_table: Q-learning Q-table array
    """
    if log_fn:
        log_fn("Saving models...")
    
    models_dir = os.path.join(output_dir, 'models')
    os.makedirs(models_dir, exist_ok=True)
    
    saved_files = []
    
    # Classical model
    if 'classical' in models:
        filepath = os.path.join(models_dir, 'classical_model.pkl')
        save_classical_model(models['classical'], filepath)
        saved_files.append(filepath)
    
    # PyTorch models
    if TORCH_AVAILABLE:
        if 'mlp' in models:
            filepath = os.path.join(models_dir, 'mlp_model.pt')
            save_pytorch_model(models['mlp'], filepath)
            saved_files.append(filepath)
        
        if 'lstm' in models:
            filepath = os.path.join(models_dir, 'lstm_model.pt')
            save_pytorch_model(models['lstm'], filepath)
            saved_files.append(filepath)
        
        if 'transformer' in models:
            filepath = os.path.join(models_dir, 'transformer_model.pt')
            save_pytorch_model(models['transformer'], filepath)
            saved_files.append(filepath)
    
    # Scaler
    if scaler is not None:
        filepath = os.path.join(models_dir, 'scaler.pkl')
        save_scaler(scaler, filepath)
        saved_files.append(filepath)
    
    # VAE Sampler
    if vae_sampler is not None:
        filepath = os.path.join(models_dir, 'vae_sampler.pkl')
        with open(filepath, 'wb') as f:
            pickle.dump(vae_sampler, f)
        saved_files.append(filepath)
    
    # Q-table
    if q_table is not None:
        filepath = os.path.join(models_dir, 'q_table.npy')
        save_q_table(q_table, filepath)
        saved_files.append(filepath)
    
    if log_fn:
        log_fn(f"Saved {len(saved_files)} model files")
        for f in saved_files:
            size = os.path.getsize(f)
            log_fn(f"  {os.path.basename(f)}: {size/1024:.1f} KB")
    
    return saved_files


def load_all_models(output_dir: str, model_classes: dict = None):
    """
    Load all saved models from output directory.
    
    Args:
        output_dir: Base output directory
        model_classes: Dict mapping model names to their classes for PyTorch models
        
    Returns:
        Dict of loaded models
    """
    models_dir = os.path.join(output_dir, 'models')
    loaded = {}
    
    # Classical model
    classical_path = os.path.join(models_dir, 'classical_model.pkl')
    if os.path.exists(classical_path):
        loaded['classical'] = load_classical_model(classical_path)
    
    # Scaler
    scaler_path = os.path.join(models_dir, 'scaler.pkl')
    if os.path.exists(scaler_path):
        loaded['scaler'] = load_scaler(scaler_path)
    
    # VAE Sampler
    vae_path = os.path.join(models_dir, 'vae_sampler.pkl')
    if os.path.exists(vae_path):
        with open(vae_path, 'rb') as f:
            loaded['vae_sampler'] = pickle.load(f)
    
    # Q-table
    q_path = os.path.join(models_dir, 'q_table.npy')
    if os.path.exists(q_path):
        loaded['q_table'] = load_q_table(q_path)
    
    # PyTorch models (need model classes to instantiate)
    if TORCH_AVAILABLE and model_classes:
        for name, model_class in model_classes.items():
            filepath = os.path.join(models_dir, f'{name}_model.pt')
            if os.path.exists(filepath):
                # Note: caller must provide properly configured model instance
                pass
    
    return loaded
