"""
Smoke Integration Tests
========================
End-to-end smoke tests for the orchestration pipeline.
Quick sanity checks that the system can run without crashing.
"""

import pytest
import sys
from pathlib import Path
import importlib
import yaml

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))


class TestModuleImports:
    """Test that all modules can be imported without errors."""

    def test_import_classic_modules(self):
        """Test importing classic ML modules."""
        try:
            from src.classic import train_tabular, infer_tabular
            assert True
        except ImportError as e:
            pytest.fail(f"Failed to import classic modules: {e}")

    def test_import_neural_modules(self):
        """Test importing neural network modules."""
        try:
            from src.neural import train_mlp, train_vision_cnn
            assert True
        except ImportError as e:
            pytest.fail(f"Failed to import neural modules: {e}")

    def test_import_sequential_modules(self):
        """Test importing sequential modules."""
        try:
            from src.sequential import train_text_transformer, train_timeseries
            assert True
        except ImportError as e:
            pytest.fail(f"Failed to import sequential modules: {e}")

    def test_import_generative_modules(self):
        """Test importing generative modules."""
        try:
            from src.generative import vae_synth
            assert True
        except ImportError as e:
            pytest.fail(f"Failed to import generative modules: {e}")

    def test_import_graph_modules(self):
        """Test importing graph modules."""
        try:
            from src.graph import gnn_link_pred
            assert True
        except ImportError as e:
            pytest.fail(f"Failed to import graph modules: {e}")

    def test_import_rl_modules(self):
        """Test importing RL modules."""
        try:
            from src.rl import q_learning
            assert True
        except ImportError as e:
            pytest.fail(f"Failed to import RL modules: {e}")

    def test_import_mlops_modules(self):
        """Test importing MLOps modules."""
        try:
            from src.mlops import eval_pipeline, utils
            assert True
        except ImportError as e:
            pytest.fail(f"Failed to import MLOps modules: {e}")

    def test_import_integration_modules(self):
        """Test importing integration modules."""
        try:
            from src.integration import orchestrate, data_pipeline, model_serialization, model_cards
            assert True
        except ImportError as e:
            pytest.fail(f"Failed to import integration modules: {e}")

    def test_import_api_service(self):
        """Test importing API service."""
        try:
            from src.api import service
            assert True
        except ImportError as e:
            # FastAPI is optional - skip if not installed
            if "fastapi" in str(e).lower():
                pytest.skip("FastAPI not installed - skipping API service import test")
            else:
                pytest.fail(f"Failed to import API service: {e}")


class TestConfigurationFiles:
    """Test that configuration files exist and are valid YAML."""

    @pytest.fixture
    def configs_dir(self):
        """Get configs directory."""
        return Path(__file__).parent.parent / "configs"

    def test_default_config_exists(self, configs_dir):
        """Test that default.yaml exists and is valid."""
        config_path = configs_dir / "default.yaml"
        assert config_path.exists(), "default.yaml should exist"

        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        assert config is not None, "default.yaml should be valid YAML"

    def test_all_required_configs_exist(self, configs_dir):
        """Test that all required config files exist."""
        required_configs = [
            "default.yaml",
            "classic.yaml",
            "neural.yaml",
            "sequential.yaml",
            "generative.yaml",
            "graph.yaml",
            "rl.yaml"
        ]

        for config_name in required_configs:
            config_path = configs_dir / config_name
            assert config_path.exists(), f"{config_name} should exist"

            # Validate YAML syntax
            with open(config_path, 'r') as f:
                config = yaml.safe_load(f)
            assert config is not None, f"{config_name} should be valid YAML"
            print(f"✓ {config_name} is valid")

    def test_configs_have_expected_structure(self, configs_dir):
        """Test that configs have expected top-level keys."""
        # Test default.yaml has required keys
        with open(configs_dir / "default.yaml", 'r') as f:
            default_config = yaml.safe_load(f)

        assert "paths" in default_config or "random_seed" in default_config, \
            "default.yaml should have paths or seed configuration"


class TestDirectoryStructure:
    """Test that directory structure is correct."""

    @pytest.fixture
    def project_root(self):
        """Get project root."""
        return Path(__file__).parent.parent

    def test_src_structure(self, project_root):
        """Test that src/ has all required subdirectories."""
        src_dir = project_root / "src"
        assert src_dir.exists()

        required_dirs = [
            "classic",
            "neural",
            "sequential",
            "generative",
            "graph",
            "rl",
            "mlops",
            "api",
            "integration"
        ]

        for dir_name in required_dirs:
            dir_path = src_dir / dir_name
            assert dir_path.exists(), f"src/{dir_name} should exist"
            assert dir_path.is_dir(), f"src/{dir_name} should be a directory"

            # Check for __init__.py
            init_file = dir_path / "__init__.py"
            assert init_file.exists(), f"src/{dir_name}/__init__.py should exist"

    def test_data_structure(self, project_root):
        """Test that data/ has required subdirectories."""
        data_dir = project_root / "data"
        assert data_dir.exists()

        assert (data_dir / "base").exists(), "data/base should exist"
        assert (data_dir / "derived").exists(), "data/derived should exist"

    def test_outputs_structure(self, project_root):
        """Test that outputs/ has required subdirectories."""
        outputs_dir = project_root / "outputs"
        assert outputs_dir.exists()

        required_dirs = ["models", "logs", "artifacts"]
        for dir_name in required_dirs:
            dir_path = outputs_dir / dir_name
            assert dir_path.exists(), f"outputs/{dir_name} should exist"

    def test_tests_directory(self, project_root):
        """Test that tests/ directory exists with required files."""
        tests_dir = project_root / "tests"
        assert tests_dir.exists()

        required_tests = [
            "test_data_contracts.py",
            "test_api_contract.py",
            "test_smoke_integration.py"
        ]

        for test_file in required_tests:
            test_path = tests_dir / test_file
            assert test_path.exists(), f"tests/{test_file} should exist"


class TestMLOpsInfrastructure:
    """Test MLOps infrastructure components."""

    @pytest.fixture
    def project_root(self):
        """Get project root."""
        return Path(__file__).parent.parent

    def test_runlog_schema_exists(self, project_root):
        """Test that runlog schema file exists."""
        schema_path = project_root / "src" / "mlops" / "runlog_schema.json"
        assert schema_path.exists(), "runlog_schema.json should exist"

        import json
        with open(schema_path, 'r') as f:
            schema = json.load(f)
        assert schema is not None
        assert "properties" in schema
        print("✓ runlog_schema.json is valid JSON Schema")

    def test_runlog_csv_can_be_created(self, project_root):
        """Test that run_log.csv can be created."""
        import csv
        import tempfile

        # Try to create a run log in temp directory
        with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as f:
            writer = csv.DictWriter(f, fieldnames=[
                "run_id", "timestamp", "git_commit", "data_hash",
                "config_hash", "seed", "component", "metric_name",
                "metric_value", "latency_ms", "params_json", "notes"
            ])
            writer.writeheader()
            writer.writerow({
                "run_id": "test_run",
                "timestamp": "2025-01-06T12:00:00Z",
                "git_commit": "abc123",
                "data_hash": "hash1",
                "config_hash": "hash2",
                "seed": 42,
                "component": "classic",
                "metric_name": "accuracy",
                "metric_value": 0.85,
                "latency_ms": 10.5,
                "params_json": "{}",
                "notes": "test"
            })

        assert True  # If we got here, CSV creation works
        print("✓ run_log.csv format is valid")


class TestQuickSanityChecks:
    """Quick sanity checks for basic functionality."""

    def test_numpy_available(self):
        """Test that numpy is available."""
        import numpy as np
        arr = np.array([1, 2, 3])
        assert len(arr) == 3

    def test_pandas_available(self):
        """Test that pandas is available."""
        import pandas as pd
        df = pd.DataFrame({'a': [1, 2, 3]})
        assert len(df) == 3

    def test_torch_available(self):
        """Test that PyTorch is available."""
        try:
            import torch
            tensor = torch.tensor([1.0, 2.0, 3.0])
            assert len(tensor) == 3
        except ImportError:
            pytest.skip("PyTorch not installed")

    def test_sklearn_available(self):
        """Test that scikit-learn is available."""
        from sklearn.ensemble import RandomForestClassifier
        model = RandomForestClassifier(n_estimators=10)
        assert model is not None


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
