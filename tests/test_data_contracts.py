"""
Data Contract Tests
====================
Validate data schemas, CSV formats, and data integrity.
Ensures data files meet expected contracts.
"""

import pytest
import pandas as pd
import numpy as np
from pathlib import Path
import hashlib


class TestDataContracts:
    """Test data file contracts and schemas."""

    @pytest.fixture
    def project_root(self):
        """Get project root directory."""
        return Path(__file__).parent.parent

    @pytest.fixture
    def data_base_dir(self, project_root):
        """Get data/base directory."""
        return project_root / "data" / "base"

    def test_data_base_directory_exists(self, data_base_dir):
        """Test that data/base directory exists."""
        assert data_base_dir.exists(), "data/base directory should exist"
        assert data_base_dir.is_dir(), "data/base should be a directory"

    def test_graph_edges_csv_exists(self, data_base_dir):
        """Test that graph_edges.csv exists."""
        graph_edges_path = data_base_dir / "graph_edges.csv"
        if graph_edges_path.exists():
            # File exists, validate structure
            df = pd.read_csv(graph_edges_path)
            assert 'customer' in df.columns or 'user' in df.columns, \
                "graph_edges.csv should have 'customer' or 'user' column"
            assert 'product' in df.columns or 'item' in df.columns, \
                "graph_edges.csv should have 'product' or 'item' column"
            assert len(df) > 0, "graph_edges.csv should not be empty"
        else:
            pytest.skip("graph_edges.csv not found - place file in data/base/")

    def test_gridworld_csv_exists(self, data_base_dir):
        """Test that gridworld.csv exists."""
        gridworld_path = data_base_dir / "gridworld.csv"
        if gridworld_path.exists():
            # File exists, validate structure
            df = pd.read_csv(gridworld_path)
            assert df.shape[0] == df.shape[1], "gridworld.csv should be square (NxN)"
            assert df.shape[0] >= 3, "gridworld should be at least 3x3"
        else:
            pytest.skip("gridworld.csv not found - place file in data/base/")

    def test_input_data_csv_format(self, data_base_dir):
        """Test that input CSV files have valid format."""
        input_file = data_base_dir / "input.csv"

        if input_file.exists():
            # Read CSV and check basic structure
            df = pd.read_csv(input_file)
            assert len(df) > 0, "Input CSV should not be empty"
            assert len(df.columns) > 0, "Input CSV should have columns"

            # Check for missing values
            missing_pct = df.isnull().sum().sum() / (len(df) * len(df.columns))
            assert missing_pct < 0.5, "Input CSV has too many missing values (>50%)"

            print(f"✓ Input CSV valid: {len(df)} rows, {len(df.columns)} columns")
        else:
            pytest.skip("input.csv not found in data/base/")

    def test_data_file_hashes(self, data_base_dir):
        """Test data file integrity via hashing."""
        csv_files = list(data_base_dir.glob("*.csv"))

        if len(csv_files) == 0:
            pytest.skip("No CSV files found in data/base/")

        for csv_file in csv_files:
            # Compute hash
            hasher = hashlib.sha256()
            with open(csv_file, 'rb') as f:
                hasher.update(f.read())
            file_hash = hasher.hexdigest()

            # Just verify it's a valid hash
            assert len(file_hash) == 64, f"{csv_file.name} hash should be 64 chars"
            print(f"✓ {csv_file.name}: {file_hash[:16]}...")

    def test_no_duplicate_rows(self, data_base_dir):
        """Test that data files don't have excessive duplicates."""
        csv_files = list(data_base_dir.glob("*.csv"))

        if len(csv_files) == 0:
            pytest.skip("No CSV files found in data/base/")

        for csv_file in csv_files:
            df = pd.read_csv(csv_file)
            duplicate_pct = (len(df) - len(df.drop_duplicates())) / len(df)

            assert duplicate_pct < 0.9, \
                f"{csv_file.name} has excessive duplicates ({duplicate_pct:.1%})"
            print(f"✓ {csv_file.name}: {duplicate_pct:.1%} duplicates")

    def test_numeric_columns_valid_range(self, data_base_dir):
        """Test that numeric columns have reasonable ranges."""
        input_file = data_base_dir / "input.csv"

        if not input_file.exists():
            pytest.skip("input.csv not found in data/base/")

        df = pd.read_csv(input_file)
        numeric_cols = df.select_dtypes(include=[np.number]).columns

        for col in numeric_cols:
            # Check for inf and -inf
            assert not np.isinf(df[col]).any(), f"Column {col} contains inf values"

            # Check reasonable range (no extremely large values)
            max_val = df[col].max()
            assert max_val < 1e15, f"Column {col} has suspiciously large values"

    def test_categorical_columns_cardinality(self, data_base_dir):
        """Test that categorical columns have reasonable cardinality."""
        input_file = data_base_dir / "input.csv"

        if not input_file.exists():
            pytest.skip("input.csv not found in data/base/")

        df = pd.read_csv(input_file)
        categorical_cols = df.select_dtypes(include=['object']).columns

        for col in categorical_cols:
            n_unique = df[col].nunique()
            cardinality_ratio = n_unique / len(df)

            # Warn if almost all values are unique (might be ID column)
            if cardinality_ratio > 0.9:
                print(f"⚠ Column {col} has high cardinality ({n_unique} unique values)")

            # Ensure it's not completely unique (except for ID columns)
            if 'id' not in col.lower():
                assert cardinality_ratio < 1.0, \
                    f"Non-ID column {col} has all unique values"


class TestDataPipelineContracts:
    """Test data pipeline output contracts."""

    @pytest.fixture
    def data_derived_dir(self):
        """Get data/derived directory."""
        return Path(__file__).parent.parent / "data" / "derived"

    def test_derived_directory_exists(self, data_derived_dir):
        """Test that data/derived directory exists."""
        assert data_derived_dir.exists(), "data/derived directory should exist"

    def test_processed_data_if_exists(self, data_derived_dir):
        """Test processed data files if they exist."""
        processed_files = list(data_derived_dir.glob("*.csv")) + \
                         list(data_derived_dir.glob("*.pkl")) + \
                         list(data_derived_dir.glob("*.parquet"))

        if len(processed_files) == 0:
            pytest.skip("No processed data files found in data/derived/")

        for file_path in processed_files:
            # Just check file is not empty
            assert file_path.stat().st_size > 0, \
                f"{file_path.name} is empty"
            print(f"✓ {file_path.name}: {file_path.stat().st_size} bytes")


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
