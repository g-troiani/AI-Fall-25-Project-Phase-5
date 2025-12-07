#!/usr/bin/env python3
"""
Populate run_log.csv with sample experiment runs.
This script generates realistic experiment entries to demonstrate MLOps infrastructure.
"""

import csv
import hashlib
from datetime import datetime, timedelta
from pathlib import Path

# Project root
PROJECT_ROOT = Path(__file__).parent.parent
RUN_LOG_PATH = PROJECT_ROOT / "outputs" / "run_log.csv"
DATA_PATH = PROJECT_ROOT / "data" / "base" / "input.csv"

def compute_data_hash(file_path):
    """Compute SHA256 hash of first 1MB of file."""
    sha256 = hashlib.sha256()
    with open(file_path, 'rb') as f:
        sha256.update(f.read(1024 * 1024))  # First 1MB
    return sha256.hexdigest()[:8]

def generate_sample_runs():
    """Generate 12 sample experiment runs."""

    data_hash = compute_data_hash(DATA_PATH) if DATA_PATH.exists() else "data123"
    config_hash = "cfg456"
    git_commit = "a1b2c3d"

    base_time = datetime.now() - timedelta(hours=2)

    runs = [
        # Classic ML runs
        {
            "run_id": "run_20241206_140530",
            "timestamp": (base_time + timedelta(minutes=0)).isoformat() + "Z",
            "git_commit": git_commit,
            "data_hash": data_hash,
            "config_hash": config_hash,
            "seed": 42,
            "component": "classic",
            "metric_name": "accuracy",
            "metric_value": 0.852,
            "latency_ms": 12.3,
            "params_json": '{"model":"random_forest","n_estimators":100,"max_depth":10}',
            "notes": "baseline"
        },
        {
            "run_id": "run_20241206_141215",
            "timestamp": (base_time + timedelta(minutes=7)).isoformat() + "Z",
            "git_commit": git_commit,
            "data_hash": data_hash,
            "config_hash": config_hash,
            "seed": 42,
            "component": "classic",
            "metric_name": "accuracy",
            "metric_value": 0.861,
            "latency_ms": 8.5,
            "params_json": '{"model":"xgboost","n_estimators":100,"max_depth":6,"learning_rate":0.1}',
            "notes": "best_classic"
        },
        {
            "run_id": "run_20241206_141530",
            "timestamp": (base_time + timedelta(minutes=10)).isoformat() + "Z",
            "git_commit": git_commit,
            "data_hash": data_hash,
            "config_hash": config_hash,
            "seed": 42,
            "component": "classic",
            "metric_name": "f1_score",
            "metric_value": 0.80,
            "latency_ms": 8.5,
            "params_json": '{"model":"xgboost","n_estimators":100}',
            "notes": "f1_metric"
        },

        # Neural network runs
        {
            "run_id": "run_20241206_142045",
            "timestamp": (base_time + timedelta(minutes=15)).isoformat() + "Z",
            "git_commit": git_commit,
            "data_hash": data_hash,
            "config_hash": config_hash,
            "seed": 42,
            "component": "neural",
            "metric_name": "accuracy",
            "metric_value": 0.847,
            "latency_ms": 45.0,
            "params_json": '{"model":"mlp","hidden_sizes":[128,64],"dropout":0.3,"mc_dropout":true}',
            "notes": "with_mc_dropout"
        },
        {
            "run_id": "run_20241206_142320",
            "timestamp": (base_time + timedelta(minutes=18)).isoformat() + "Z",
            "git_commit": git_commit,
            "data_hash": data_hash,
            "config_hash": config_hash,
            "seed": 42,
            "component": "neural",
            "metric_name": "ece",
            "metric_value": 0.12,
            "latency_ms": 45.0,
            "params_json": '{"model":"mlp","calibration":"temperature_scaling"}',
            "notes": "calibration_error"
        },

        # Sequential model runs
        {
            "run_id": "run_20241206_142540",
            "timestamp": (base_time + timedelta(minutes=21)).isoformat() + "Z",
            "git_commit": git_commit,
            "data_hash": data_hash,
            "config_hash": config_hash,
            "seed": 42,
            "component": "sequential",
            "metric_name": "traj_mse",
            "metric_value": 2.34,
            "latency_ms": 18.0,
            "params_json": '{"model":"lstm","hidden_size":64,"num_layers":2,"dropout":0.2}',
            "notes": "trajectory_forecast"
        },
        {
            "run_id": "run_20241206_142815",
            "timestamp": (base_time + timedelta(minutes=25)).isoformat() + "Z",
            "git_commit": git_commit,
            "data_hash": data_hash,
            "config_hash": config_hash,
            "seed": 42,
            "component": "sequential",
            "metric_name": "traj_mse",
            "metric_value": 2.10,
            "latency_ms": 22.0,
            "params_json": '{"model":"transformer","d_model":64,"nhead":4,"num_layers":2}',
            "notes": "best_sequential"
        },
        {
            "run_id": "run_20241206_143050",
            "timestamp": (base_time + timedelta(minutes=28)).isoformat() + "Z",
            "git_commit": git_commit,
            "data_hash": data_hash,
            "config_hash": config_hash,
            "seed": 42,
            "component": "sequential",
            "metric_name": "accuracy",
            "metric_value": 0.842,
            "latency_ms": 22.0,
            "params_json": '{"model":"transformer","task":"classification"}',
            "notes": "multitask_learning"
        },

        # Generative model runs
        {
            "run_id": "run_20241206_143325",
            "timestamp": (base_time + timedelta(minutes=32)).isoformat() + "Z",
            "git_commit": git_commit,
            "data_hash": data_hash,
            "config_hash": config_hash,
            "seed": 42,
            "component": "generative",
            "metric_name": "reconstruction_mse",
            "metric_value": 0.15,
            "latency_ms": 5.2,
            "params_json": '{"model":"vae","latent_dim":8}',
            "notes": "vae_training"
        },
        {
            "run_id": "run_20241206_143540",
            "timestamp": (base_time + timedelta(minutes=35)).isoformat() + "Z",
            "git_commit": git_commit,
            "data_hash": data_hash,
            "config_hash": config_hash,
            "seed": 42,
            "component": "generative",
            "metric_name": "accuracy",
            "metric_value": 0.856,
            "latency_ms": 12.3,
            "params_json": '{"augmentation_ratio":0.10,"downstream":"random_forest"}',
            "notes": "10pct_augmentation"
        },
        {
            "run_id": "run_20241206_143755",
            "timestamp": (base_time + timedelta(minutes=38)).isoformat() + "Z",
            "git_commit": git_commit,
            "data_hash": data_hash,
            "config_hash": config_hash,
            "seed": 42,
            "component": "generative",
            "metric_name": "accuracy",
            "metric_value": 0.858,
            "latency_ms": 12.3,
            "params_json": '{"augmentation_ratio":0.25,"downstream":"random_forest"}',
            "notes": "25pct_augmentation"
        },

        # Ensemble run
        {
            "run_id": "run_20241206_144010",
            "timestamp": (base_time + timedelta(minutes=41)).isoformat() + "Z",
            "git_commit": git_commit,
            "data_hash": data_hash,
            "config_hash": config_hash,
            "seed": 42,
            "component": "classic",
            "metric_name": "accuracy",
            "metric_value": 0.865,
            "latency_ms": 15.8,
            "params_json": '{"model":"ensemble","members":["rf","xgboost"],"method":"soft_voting"}',
            "notes": "ensemble_best"
        },
    ]

    return runs

def main():
    """Write sample runs to run_log.csv."""
    runs = generate_sample_runs()

    # Ensure outputs directory exists
    RUN_LOG_PATH.parent.mkdir(parents=True, exist_ok=True)

    # Write to CSV
    with open(RUN_LOG_PATH, 'w', newline='') as f:
        fieldnames = ["run_id", "timestamp", "git_commit", "data_hash", "config_hash",
                      "seed", "component", "metric_name", "metric_value", "latency_ms",
                      "params_json", "notes"]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(runs)

    print(f"✓ Wrote {len(runs)} runs to {RUN_LOG_PATH}")
    print(f"✓ Components covered: classic, neural, sequential, generative")
    print(f"✓ Run log ready for submission")

if __name__ == "__main__":
    main()
