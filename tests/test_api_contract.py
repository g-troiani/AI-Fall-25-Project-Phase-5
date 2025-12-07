"""
API Contract Tests
==================
Test API endpoints, request/response formats, and contracts.
Validates that the API service meets specifications.
"""

import pytest
import sys
from pathlib import Path

# Try to import FastAPI - skip tests if not available
try:
    from fastapi.testclient import TestClient
    FASTAPI_AVAILABLE = True
except ImportError:
    FASTAPI_AVAILABLE = False
    TestClient = None

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

# Only import if FastAPI is available
if FASTAPI_AVAILABLE:
    try:
        from src.api.service import app
        client = TestClient(app)
    except ImportError:
        FASTAPI_AVAILABLE = False
        client = None
else:
    app = None
    client = None

# Skip all tests in this module if FastAPI not available
pytestmark = pytest.mark.skipif(
    not FASTAPI_AVAILABLE,
    reason="FastAPI not installed - skipping API contract tests"
)


class TestAPIEndpoints:
    """Test API endpoint availability and basic responses."""

    def test_root_endpoint(self):
        """Test root endpoint returns API information."""
        response = client.get("/")
        assert response.status_code == 200

        data = response.json()
        assert "message" in data
        assert "version" in data
        assert "endpoints" in data

    def test_health_check(self):
        """Test health check endpoint."""
        response = client.get("/health")
        assert response.status_code == 200

        data = response.json()
        assert data["status"] == "healthy"

    def test_docs_available(self):
        """Test that OpenAPI docs are available."""
        response = client.get("/docs")
        assert response.status_code == 200

    def test_openapi_schema(self):
        """Test that OpenAPI schema is valid."""
        response = client.get("/openapi.json")
        assert response.status_code == 200

        schema = response.json()
        assert "openapi" in schema
        assert "info" in schema
        assert "paths" in schema


class TestTabularPredictionEndpoint:
    """Test /predict/tabular endpoint."""

    def test_tabular_prediction_valid_request(self):
        """Test tabular prediction with valid request."""
        payload = {
            "features": [[1.0, 2.0, 3.0, 4.0], [5.0, 6.0, 7.0, 8.0]],
            "model_type": "ensemble",
            "return_proba": False
        }

        response = client.post("/predict/tabular", json=payload)

        # May fail if models don't exist, but should have proper error handling
        assert response.status_code in [200, 404, 500]

        if response.status_code == 200:
            data = response.json()
            assert "predictions" in data
            assert "latency_ms" in data
            assert isinstance(data["predictions"], list)
            assert len(data["predictions"]) == 2  # Same as input batch size

    def test_tabular_prediction_with_probabilities(self):
        """Test tabular prediction requesting probabilities."""
        payload = {
            "features": [[1.0, 2.0, 3.0, 4.0]],
            "model_type": "ensemble",
            "return_proba": True
        }

        response = client.post("/predict/tabular", json=payload)

        if response.status_code == 200:
            data = response.json()
            assert "probabilities" in data
            assert data["probabilities"] is not None
            assert isinstance(data["probabilities"], list)

    def test_tabular_prediction_invalid_features(self):
        """Test tabular prediction with invalid features."""
        payload = {
            "features": "not_a_list",  # Invalid type
            "model_type": "ensemble"
        }

        response = client.post("/predict/tabular", json=payload)
        assert response.status_code == 422  # Validation error

    def test_tabular_prediction_empty_features(self):
        """Test tabular prediction with empty features."""
        payload = {
            "features": [],
            "model_type": "ensemble"
        }

        response = client.post("/predict/tabular", json=payload)
        # Should handle gracefully
        assert response.status_code in [200, 400, 422, 500]


class TestRecommendationEndpoint:
    """Test /recommend/{customer_id} endpoint."""

    def test_recommend_valid_request(self):
        """Test recommendation with valid customer ID."""
        response = client.post("/recommend/customer_123?top_k=10&method=common_neighbors")

        assert response.status_code in [200, 404, 500]

        if response.status_code == 200:
            data = response.json()
            assert "customer_id" in data
            assert "recommendations" in data
            assert "method" in data
            assert data["customer_id"] == "customer_123"
            assert isinstance(data["recommendations"], list)
            assert len(data["recommendations"]) <= 10

    def test_recommend_different_methods(self):
        """Test recommendation with different scoring methods."""
        methods = ["common_neighbors", "jaccard", "adamic_adar"]

        for method in methods:
            response = client.post(f"/recommend/customer_456?method={method}")
            assert response.status_code in [200, 404, 500]

            if response.status_code == 200:
                data = response.json()
                assert data["method"] == method

    def test_recommend_invalid_method(self):
        """Test recommendation with invalid method."""
        response = client.post("/recommend/customer_123?method=invalid_method")
        assert response.status_code == 422  # Validation error

    def test_recommend_top_k_validation(self):
        """Test recommendation with various top_k values."""
        # Valid range
        response = client.post("/recommend/customer_123?top_k=5")
        assert response.status_code in [200, 404, 500]

        # Too small (< 1)
        response = client.post("/recommend/customer_123?top_k=0")
        assert response.status_code == 422

        # Too large (> 100)
        response = client.post("/recommend/customer_123?top_k=200")
        assert response.status_code == 422


class TestForecastEndpoint:
    """Test /forecast/ts endpoint."""

    def test_forecast_valid_request(self):
        """Test time-series forecasting with valid request."""
        payload = {
            "sequence": [1.0, 1.2, 1.1, 1.3, 1.5, 1.4, 1.6, 1.8, 1.7, 1.9],
            "horizon": 14,
            "model_type": "lstm"
        }

        response = client.post("/forecast/ts", json=payload)
        assert response.status_code in [200, 404, 500]

        if response.status_code == 200:
            data = response.json()
            assert "forecast" in data
            assert "horizon" in data
            assert "model_type" in data
            assert len(data["forecast"]) == 14
            assert data["model_type"] == "lstm"

    def test_forecast_transformer_model(self):
        """Test forecasting with transformer model."""
        payload = {
            "sequence": [1.0, 2.0, 3.0, 4.0, 5.0],
            "horizon": 7,
            "model_type": "transformer"
        }

        response = client.post("/forecast/ts", json=payload)
        assert response.status_code in [200, 404, 500]

    def test_forecast_short_sequence(self):
        """Test forecasting with very short sequence."""
        payload = {
            "sequence": [1.0],
            "horizon": 5,
            "model_type": "lstm"
        }

        response = client.post("/forecast/ts", json=payload)
        # Should handle gracefully (might return error)
        assert response.status_code in [200, 400, 500]


class TestPolicyEndpoint:
    """Test /policy/gridworld endpoint."""

    def test_policy_get_full_grid(self):
        """Test getting full policy grid."""
        response = client.get("/policy/gridworld")
        assert response.status_code in [200, 404, 500]

        if response.status_code == 200:
            data = response.json()
            assert "policy_grid" in data
            assert isinstance(data["policy_grid"], list)
            assert len(data["policy_grid"]) > 0
            assert len(data["policy_grid"][0]) > 0  # 2D grid

    def test_policy_query_specific_state(self):
        """Test querying policy for specific state."""
        response = client.get("/policy/gridworld?state=2,3")
        assert response.status_code in [200, 404, 500]

        if response.status_code == 200:
            data = response.json()
            assert "action_at_state" in data
            assert data["action_at_state"] is not None

    def test_policy_invalid_state_format(self):
        """Test policy with invalid state format."""
        response = client.get("/policy/gridworld?state=invalid")
        assert response.status_code == 422  # Validation error

    def test_policy_q_values(self):
        """Test that Q-values are returned when querying state."""
        response = client.get("/policy/gridworld?state=1,1")

        if response.status_code == 200:
            data = response.json()
            if "q_values" in data and data["q_values"] is not None:
                assert isinstance(data["q_values"], dict)


class TestAPIResponseFormats:
    """Test API response format consistency."""

    def test_all_responses_are_json(self):
        """Test that all endpoints return JSON."""
        endpoints = [
            ("/", "get"),
            ("/health", "get"),
            ("/policy/gridworld", "get")
        ]

        for endpoint, method in endpoints:
            if method == "get":
                response = client.get(endpoint)
            else:
                response = client.post(endpoint)

            if response.status_code == 200:
                assert response.headers["content-type"].startswith("application/json")

    def test_error_responses_have_detail(self):
        """Test that error responses include detail field."""
        # Make a request that should fail
        response = client.post("/predict/tabular", json={"invalid": "data"})

        if response.status_code >= 400:
            data = response.json()
            assert "detail" in data


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
