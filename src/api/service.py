"""
API Service Facade
==================
FastAPI service for model serving and inference.
Provides REST endpoints for tabular prediction, graph recommendation,
time-series forecasting, and RL policy queries.
"""

from fastapi import FastAPI, HTTPException, Query
from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any
import numpy as np
import pickle
from pathlib import Path
import sys

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.classic.infer_tabular import TabularInference, EnsembleInference

app = FastAPI(
    title="Final Project Phase 5 API",
    description="ML System API for tabular, graph, sequential, and RL models",
    version="1.0.0"
)


# ============================================================================
# Request/Response Models
# ============================================================================

class TabularPredictRequest(BaseModel):
    """Request for tabular prediction."""
    features: List[List[float]] = Field(..., description="Input features (batch_size, n_features)")
    model_type: Optional[str] = Field("ensemble", description="Model type: random_forest, xgboost, or ensemble")
    return_proba: bool = Field(False, description="Return probabilities instead of predictions")

    class Config:
        json_schema_extra = {
            "example": {
                "features": [[1.2, 3.4, 5.6, 7.8], [2.1, 4.3, 6.5, 8.7]],
                "model_type": "ensemble",
                "return_proba": True
            }
        }


class TabularPredictResponse(BaseModel):
    """Response for tabular prediction."""
    predictions: List[int] = Field(..., description="Predicted classes")
    probabilities: Optional[List[List[float]]] = Field(None, description="Class probabilities")
    latency_ms: float = Field(..., description="Inference latency in milliseconds")


class RecommendRequest(BaseModel):
    """Request for graph-based recommendations."""
    customer_id: str = Field(..., description="Customer ID")
    top_k: int = Field(10, description="Number of recommendations to return")
    method: str = Field("common_neighbors", description="Scoring method: common_neighbors, jaccard, adamic_adar")

    class Config:
        json_schema_extra = {
            "example": {
                "customer_id": "customer_123",
                "top_k": 10,
                "method": "common_neighbors"
            }
        }


class RecommendResponse(BaseModel):
    """Response for recommendations."""
    customer_id: str
    recommendations: List[Dict[str, Any]] = Field(..., description="List of {product_id, score}")
    method: str


class ForecastRequest(BaseModel):
    """Request for time-series forecasting."""
    sequence: List[float] = Field(..., description="Input time series sequence")
    horizon: int = Field(14, description="Forecast horizon (number of steps)")
    model_type: str = Field("lstm", description="Model type: lstm or transformer")

    class Config:
        json_schema_extra = {
            "example": {
                "sequence": [1.0, 1.2, 1.1, 1.3, 1.5, 1.4, 1.6, 1.8, 1.7, 1.9],
                "horizon": 14,
                "model_type": "lstm"
            }
        }


class ForecastResponse(BaseModel):
    """Response for forecasting."""
    forecast: List[float] = Field(..., description="Forecasted values")
    horizon: int
    model_type: str


class PolicyRequest(BaseModel):
    """Request for RL policy query."""
    state: Optional[List[int]] = Field(None, description="Optional state position [row, col]")

    class Config:
        json_schema_extra = {
            "example": {
                "state": [2, 3]
            }
        }


class PolicyResponse(BaseModel):
    """Response for RL policy."""
    policy_grid: List[List[str]] = Field(..., description="Policy grid (actions at each state)")
    action_at_state: Optional[str] = Field(None, description="Action at queried state")
    q_values: Optional[Dict[str, float]] = Field(None, description="Q-values for actions at state")


# ============================================================================
# Global Model Storage
# ============================================================================

class ModelRegistry:
    """Registry for loaded models."""

    def __init__(self):
        self.models = {}
        self.model_dir = Path("outputs/models")

    def load_tabular_model(self, model_type: str):
        """Load tabular model if not already loaded."""
        if model_type not in self.models:
            model_path = self.model_dir / f"{model_type}.pkl"
            if not model_path.exists():
                raise HTTPException(status_code=404, detail=f"Model {model_type} not found")
            self.models[model_type] = TabularInference(model_path=model_path)
        return self.models[model_type]

    def load_ensemble(self):
        """Load ensemble of tabular models."""
        if "ensemble" not in self.models:
            model_paths = [
                self.model_dir / "random_forest.pkl",
                self.model_dir / "xgboost.pkl"
            ]
            # Check if models exist
            existing_paths = [p for p in model_paths if p.exists()]
            if not existing_paths:
                raise HTTPException(status_code=404, detail="No ensemble models found")

            self.models["ensemble"] = EnsembleInference(model_paths=existing_paths)
        return self.models["ensemble"]

    def load_graph_recommender(self):
        """Load graph link prediction recommender."""
        if "graph_recommender" not in self.models:
            model_path = self.model_dir / "graph_recommender.pkl"
            if not model_path.exists():
                raise HTTPException(status_code=404, detail="Graph recommender not found")
            with open(model_path, 'rb') as f:
                self.models["graph_recommender"] = pickle.load(f)
        return self.models["graph_recommender"]

    def load_sequential_model(self, model_type: str):
        """Load LSTM or Transformer model."""
        key = f"sequential_{model_type}"
        if key not in self.models:
            model_path = self.model_dir / f"{model_type}.pkl"
            if not model_path.exists():
                raise HTTPException(status_code=404, detail=f"Sequential model {model_type} not found")
            with open(model_path, 'rb') as f:
                self.models[key] = pickle.load(f)
        return self.models[key]

    def load_q_table(self):
        """Load Q-learning Q-table and policy."""
        if "q_learning" not in self.models:
            model_path = self.model_dir / "q_learning.pkl"
            if not model_path.exists():
                raise HTTPException(status_code=404, detail="Q-learning model not found")
            with open(model_path, 'rb') as f:
                self.models["q_learning"] = pickle.load(f)
        return self.models["q_learning"]


# Initialize registry
registry = ModelRegistry()


# ============================================================================
# API Endpoints
# ============================================================================

@app.get("/")
async def root():
    """Root endpoint with API information."""
    return {
        "message": "Final Project Phase 5 API",
        "version": "1.0.0",
        "endpoints": {
            "tabular": "/predict/tabular",
            "recommendations": "/recommend/{customer_id}",
            "forecasting": "/forecast/ts",
            "rl_policy": "/policy/gridworld"
        }
    }


@app.post("/predict/tabular", response_model=TabularPredictResponse)
async def predict_tabular(request: TabularPredictRequest):
    """
    Predict using classical/neural tabular models.
    Supports ensemble soft voting.
    """
    try:
        X = np.array(request.features)

        # Load model
        if request.model_type == "ensemble":
            model = registry.load_ensemble()
        else:
            model = registry.load_tabular_model(request.model_type)

        # Make predictions
        if request.return_proba:
            probabilities = model.predict_proba(X)
            predictions = (probabilities[:, 1] > 0.5).astype(int).tolist()
            probabilities_list = probabilities.tolist()
        else:
            predictions = model.predict(X).tolist()
            probabilities_list = None

        return TabularPredictResponse(
            predictions=predictions,
            probabilities=probabilities_list,
            latency_ms=model.get_latency()
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/recommend/{customer_id}", response_model=RecommendResponse)
async def recommend_products(
    customer_id: str,
    top_k: int = Query(10, ge=1, le=100),
    method: str = Query("common_neighbors", pattern="^(common_neighbors|jaccard|adamic_adar)$")
):
    """
    Get product recommendations for a customer using graph-based link prediction.
    Uses LinkPredictionScorer with common_neighbors, jaccard, or adamic_adar methods.
    """
    try:
        # Load graph recommender model
        recommender = registry.load_graph_recommender()

        # Get recommendations using specified method
        recs = recommender.recommend(customer_id, k=top_k, method=method)

        # Convert to response format
        recommendations = [
            {"product_id": str(product), "score": float(score)}
            for product, score in recs
        ]

        return RecommendResponse(
            customer_id=customer_id,
            recommendations=recommendations,
            method=method
        )

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Recommendation error: {str(e)}")


@app.post("/forecast/ts", response_model=ForecastResponse)
async def forecast_timeseries(request: ForecastRequest):
    """
    Forecast future trajectory positions using LSTM or Transformer models.
    Input sequence should be recent player positions (x, y coordinates).
    """
    try:
        sequence = np.array(request.sequence)
        if len(sequence) < 2:
            raise HTTPException(status_code=400, detail="Sequence too short (minimum 2 timesteps)")

        # Load sequential model (LSTM or Transformer)
        model = registry.load_sequential_model(request.model_type)

        # Reshape sequence for model input: (1, seq_len, n_features)
        # For trajectory forecasting, we need at least 2D features (x, y)
        if len(sequence.shape) == 1:
            # Assume 1D sequence is a time series, duplicate for x,y coordinates
            sequence_2d = np.stack([sequence, sequence], axis=-1)
        else:
            sequence_2d = sequence

        # Ensure 3D shape: (batch=1, seq_len, features)
        if len(sequence_2d.shape) == 2:
            sequence_2d = sequence_2d[np.newaxis, :, :]

        # Get forecast from model (returns trajectory predictions)
        # Model predicts next N positions as (x, y) coordinates
        forecast_array = model.predict(sequence_2d, horizon=request.horizon)

        # Flatten to 1D if input was 1D
        if len(sequence.shape) == 1:
            forecast = forecast_array[0, :, 0].tolist()  # Use only x-coordinate
        else:
            forecast = forecast_array[0].tolist()  # Keep 2D structure

        return ForecastResponse(
            forecast=forecast,
            horizon=request.horizon,
            model_type=request.model_type
        )

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Forecasting error: {str(e)}")


@app.get("/policy/gridworld", response_model=PolicyResponse)
async def get_policy(state: Optional[str] = Query(None, pattern=r"^\d+,\d+$")):
    """
    Get Q-learning policy for gridworld.
    Returns optimal action for each grid state based on learned Q-values.
    Optionally query specific state with format "row,col" (e.g., "2,3").
    """
    try:
        # Load Q-learning agent with Q-table
        agent = registry.load_q_table()

        # Extract Q-table and grid size
        Q = agent.Q  # Shape: (n_states, n_actions)
        grid_size = int(np.sqrt(Q.shape[0]))
        action_names = ['↑', '↓', '←', '→']  # up, down, left, right

        # Convert Q-table to policy grid
        policy = np.argmax(Q, axis=1)  # Greedy policy
        policy_grid = []

        for row in range(grid_size):
            policy_row = []
            for col in range(grid_size):
                state_idx = row * grid_size + col
                action_idx = policy[state_idx]
                policy_row.append(action_names[action_idx])
            policy_grid.append(policy_row)

        response = PolicyResponse(policy_grid=policy_grid)

        # If specific state queried, return action and Q-values
        if state:
            row, col = map(int, state.split(","))
            if 0 <= row < grid_size and 0 <= col < grid_size:
                state_idx = row * grid_size + col
                response.action_at_state = policy_grid[row][col]

                # Return Q-values for all actions at this state
                q_vals = Q[state_idx]
                response.q_values = {
                    "up": float(q_vals[0]),
                    "down": float(q_vals[1]),
                    "left": float(q_vals[2]),
                    "right": float(q_vals[3])
                }
            else:
                raise HTTPException(
                    status_code=400,
                    detail=f"State ({row}, {col}) out of bounds for {grid_size}x{grid_size} grid"
                )

        return response

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Policy query error: {str(e)}")


@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {"status": "healthy"}


# ============================================================================
# Main Entry Point
# ============================================================================

if __name__ == "__main__":
    import uvicorn

    print("Starting API server...")
    print("Documentation available at: http://localhost:8000/docs")

    uvicorn.run(app, host="0.0.0.0", port=8000, log_level="info")
