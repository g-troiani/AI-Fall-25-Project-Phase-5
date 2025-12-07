# Final Project Phase 5 - Modular ML System

A comprehensive machine learning system implementing multiple model architectures for predictive analytics.

## Project Structure

```
project5_final_system/
├── src/
│   ├── classic/           # P1-P2: Traditional ML models
│   │   ├── train_tabular.py
│   │   └── infer_tabular.py
│   ├── neural/            # P3: Neural network models
│   │   ├── train_mlp.py
│   │   ├── train_vision_cnn.py
│   │   └── train_text_lstm.py
│   ├── sequential/        # P4: Sequential models
│   │   ├── train_text_transformer.py
│   │   └── train_timeseries.py
│   ├── generative/        # Generative models (VAE for augmentation)
│   │   └── vae_synth.py
│   ├── graph/             # Graph models (GNN for link prediction)
│   │   └── gnn_link_pred.py
│   ├── rl/                # Reinforcement learning (Q-Learning)
│   │   └── q_learning.py
│   ├── mlops/             # MLOps infrastructure
│   │   ├── eval_pipeline.py
│   │   ├── utils.py
│   │   └── runlog_schema.json
│   ├── api/               # API service facade
│   │   └── service.py
│   └── integration/       # Orchestration and pipeline
│       ├── data_pipeline.py
│       ├── model_serialization.py
│       ├── orchestrate.py
│       └── model_cards.py
├── configs/               # YAML configuration files
│   ├── default.yaml       # Global settings
│   ├── classic.yaml
│   ├── neural.yaml
│   ├── sequential.yaml
│   ├── generative.yaml
│   ├── graph.yaml
│   └── rl.yaml
├── data/
│   ├── base/              # Raw immutable data (graph_edges.csv, gridworld.csv)
│   └── derived/           # Processed data (gitignored)
├── docs/
│   ├── Model_Cards/       # Model documentation (7+ cards)
│   ├── Final_Report_Template.md # 10-12 page report
│   ├── Presentation_Outline.md # 10-15 min presentation
│   └── System_Diagram_Description.md # Architecture diagram description
├── tests/                 # Unit tests
│   ├── test_data_contracts.py
│   ├── test_api_contract.py
│   └── test_smoke_integration.py
├── outputs/
│   ├── models/            # Trained models (.pkl, .pt)
│   ├── logs/              # Training logs
│   ├── artifacts/         # Plots, confusion matrices
│   └── run_log.csv        # MLOps experiment tracking
├── requirements.txt
└── README.md
```

## Features

### Implemented Models
- **Classic ML** (P1-P2): Random Forest, XGBoost, Logistic Regression
- **Neural Networks** (P3): MLP, CNN for vision
- **Sequential Models** (P4): LSTM, Transformer for text and time-series
- **Generative**: VAE for data augmentation
- **Graph**: GNN for link prediction and recommendations
- **Reinforcement Learning**: Q-Learning for gridworld

### Key Components
- **Modular Architecture**: Separate modules for each model type
- **Data Pipeline**: Preprocessing, feature engineering, train/val/test splits
- **MLOps Infrastructure**: Run logging, experiment tracking, reproducibility
- **Model Serialization**: Save/load trained models
- **API Service**: FastAPI endpoints for inference
- **Orchestration**: End-to-end DAG pipeline
- **Testing**: Data contracts, API contracts, smoke tests
- **Model Cards**: Documentation for transparency and ethics

## Installation

```bash
# Clone the repository
git clone <repository-url>
cd Final_Project_Phase_5

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

## Usage

### 1. Prepare Data
Place required data files in `data/base/`:
```bash
# Move main dataset and rename to input.csv
mv your_dataset.csv data/base/input.csv

# Add assignment-provided files
# - graph_edges.csv (customer-product edges)
# - gridworld.csv (5x5 reward map)
```

See `data/README.md` for detailed data requirements.

### 2. Configure Models
Edit configuration files in `configs/`:
- `default.yaml` - Global settings (paths, seeds, MLOps)
- `classic.yaml` - Random Forest, XGBoost hyperparameters
- `neural.yaml` - MLP, CNN hyperparameters
- `sequential.yaml` - LSTM, Transformer hyperparameters
- `generative.yaml` - VAE augmentation settings
- `graph.yaml` - Link prediction settings
- `rl.yaml` - Q-learning hyperparameters

### 3. Run Training Pipeline
```bash
# Run orchestrator with specific stages
python src/integration/orchestrate.py --stages classic,neural --config configs/default.yaml

# Or run with all stages
python src/integration/orchestrate.py --stages generative,neural,sequential,graph,rl

# Or train individual models
python src/classic/train_tabular.py
python src/neural/train_mlp.py
python src/sequential/train_text_lstm.py
python src/generative/vae_synth.py
python src/graph/gnn_link_pred.py
python src/rl/q_learning.py
```

### 4. Evaluate Models
```bash
python src/mlops/eval_pipeline.py
```

### 5. Start API Service
```bash
# Start FastAPI server
uvicorn src.api.service:app --reload

# Access API documentation at http://localhost:8000/docs

# Test endpoints
curl -X POST "http://localhost:8000/predict/tabular" \
  -H "Content-Type: application/json" \
  -d '{"features": [[1.0, 2.0, 3.0, 4.0]], "model_type": "ensemble", "return_proba": true}'
```

### 6. Generate Model Cards
```bash
python src/integration/model_cards.py
```

## Configuration

### Data Configuration (`configs/data.yaml`)
- Feature selection and engineering
- Preprocessing strategies
- Train/validation/test splits

### Model Configuration (`configs/models.yaml`)
- Hyperparameters for each model type
- Architecture specifications
- Optimization settings

### Training Configuration (`configs/training.yaml`)
- Early stopping criteria
- Checkpointing strategy
- Logging and monitoring

## Testing

```bash
# Run all tests
pytest tests/ -v

# Run specific test suites
pytest tests/test_data_contracts.py -v
pytest tests/test_api_contract.py -v
pytest tests/test_smoke_integration.py -v

# Run with coverage
pytest --cov=src tests/

# Run quick smoke test
pytest tests/test_smoke_integration.py::TestModuleImports -v
```

## MLOps & Experiment Tracking

All experiments are logged to `outputs/run_log.csv` with full reproducibility information:
- Run ID and timestamp
- Git commit hash
- Data file hash (SHA256)
- Config file hash (SHA256)
- Random seed
- Component (classic, neural, sequential, graph, rl)
- Metrics (accuracy, F1, MAE, RMSE, etc.)
- Inference latency
- Hyperparameters (JSON)

Example log entry:
```csv
run_20250106_123045,2025-01-06T12:30:45Z,a1b2c3d,hash1,hash2,42,classic,accuracy,0.8523,12.34,{"model":"rf","n_estimators":100},baseline
```

View the schema: `src/mlops/runlog_schema.json`

## Model Performance

Results are logged in `outputs/` and include:
- **run_log.csv**: All experiment metrics
- **logs/**: Training/validation loss curves
- **artifacts/**: Confusion matrices, ROC curves, attention maps
- **models/**: Serialized trained models (.pkl, .pt)

## Model Cards

Detailed documentation for each model is available in `docs/Model_Cards/`. Each card includes:
- Model architecture and design choices
- Training data and preprocessing
- Performance metrics
- Intended use cases
- Limitations and ethical considerations

## API Endpoints

The FastAPI service (`src/api/service.py`) provides the following endpoints:

### POST /predict/tabular
Tabular prediction with ensemble or single models
```bash
curl -X POST "http://localhost:8000/predict/tabular" \
  -H "Content-Type: application/json" \
  -d '{"features": [[1.0, 2.0, 3.0, 4.0]], "model_type": "ensemble", "return_proba": true}'
```

### POST /recommend/{customer_id}
Graph-based product recommendations
```bash
curl -X POST "http://localhost:8000/recommend/customer_123?top_k=10&method=common_neighbors"
```

### POST /forecast/ts
Time-series forecasting
```bash
curl -X POST "http://localhost:8000/forecast/ts" \
  -H "Content-Type: application/json" \
  -d '{"sequence": [1.0, 1.2, 1.1, 1.3], "horizon": 14, "model_type": "lstm"}'
```

### GET /policy/gridworld
Q-learning policy for gridworld
```bash
curl "http://localhost:8000/policy/gridworld?state=2,3"
```

## System Architecture

See `docs/System_Diagram.png` for a visual overview of the system components and data flow.

## API Service (Optional)

To serve models via REST API:
```bash
# Start the API server
uvicorn src.api.service:app --reload

# Access API documentation
# http://localhost:8000/docs
```

## Outputs

All outputs are organized in the `outputs/` directory:
- `models/` - Serialized model files (.pkl, .pt)
- `logs/` - Training logs and metrics
- `predictions/` - Model predictions on test data
- `visualizations/` - Performance plots and charts

## Dependencies

Key libraries:
- PyTorch - Deep learning framework
- scikit-learn - Classic ML algorithms
- XGBoost - Gradient boosting
- PyTorch Geometric - Graph neural networks
- pandas, numpy - Data processing
- matplotlib, seaborn - Visualization

See `requirements.txt` for complete list.

## Contributing

1. Follow PEP 8 style guidelines
2. Add unit tests for new features
3. Update documentation and model cards
4. Use type hints where applicable

## License

[Specify your license here]

## Authors

Gianmaria Troiani

## Acknowledgments

- Assignment requirements from MDC AI Program
- Course instructor: Ryan
