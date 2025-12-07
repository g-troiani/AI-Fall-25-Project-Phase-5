"""
Tests for data pipeline
"""
import pytest
import pandas as pd
import numpy as np
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from integration.data_pipeline import (
    load_data,
    engineer_features,
    construct_sequences,
    split_data,
    normalize_data,
    NUMERICAL_FEATURES
)


def test_numerical_features_defined():
    """Test that NUMERICAL_FEATURES is defined correctly"""
    assert NUMERICAL_FEATURES is not None
    assert isinstance(NUMERICAL_FEATURES, list)
    assert len(NUMERICAL_FEATURES) > 0
    assert 'x' in NUMERICAL_FEATURES
    assert 'y' in NUMERICAL_FEATURES


def test_load_data_function_exists():
    """Test that load_data function exists"""
    assert callable(load_data)


def test_engineer_features_function_exists():
    """Test that engineer_features function exists"""
    assert callable(engineer_features)


def test_construct_sequences_function_exists():
    """Test that construct_sequences function exists"""
    assert callable(construct_sequences)


def test_split_data_function_exists():
    """Test that split_data function exists"""
    assert callable(split_data)


def test_normalize_data_function_exists():
    """Test that normalize_data function exists"""
    assert callable(normalize_data)
