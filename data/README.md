# Data Directory

## Structure

- `base/` - Original raw data files (immutable, version controlled for small files)
- `derived/` - Processed and transformed data (gitignored)

## Required Files for Assignment

Please place the following files in the `data/base/` directory:

### 1. graph_edges.csv
**Purpose**: Customer-product bipartite graph for link prediction
**Format**: CSV with columns: `customer`, `product` (or `user`, `item`)
**Example**:
```csv
customer,product
customer_001,product_A
customer_001,product_B
customer_002,product_A
customer_002,product_C
```

**Source**: Provided in assignment materials or download from course repository.

### 2. gridworld.csv
**Purpose**: 5x5 reward map for Q-learning reinforcement learning
**Format**: CSV representing an NxN grid (typically 5x5) with reward values
**Example** (5x5 grid):
```csv
-1,-1,-1,-1,-1
-1,-10,-1,-1,-1
-1,-1,-1,-10,-1
-1,-1,-1,-1,-1
-1,-1,-1,-1,100
```

**Source**: Provided in assignment materials or download from course repository.

### 3. input.csv (your main dataset)
**Purpose**: Main tabular dataset for supervised learning
**Location**: Should be in `data/base/input.csv`
**Command**:
```bash
# Rename your dataset to input.csv and move to data/base/
mv your_dataset.csv data/base/input.csv
```

## Data Pipeline Flow

```
data/base/              →    src/integration/         →    data/derived/
(raw data)                  (data_pipeline.py)            (processed data)
                                    ↓
                             Feature engineering
                             Train/val/test split
                             Normalization
                                    ↓
                             data/derived/
                             - X_train.pkl
                             - X_val.pkl
                             - X_test.pkl
                             - y_train.pkl
                             - y_val.pkl
                             - y_test.pkl
```

## Usage

1. **Place raw data in `base/`**:
   - graph_edges.csv
   - gridworld.csv
   - input.csv (your main dataset)

2. **Run data pipeline**:
   ```bash
   python src/integration/data_pipeline.py
   ```

3. **Processed data appears in `derived/`**:
   - Preprocessed features
   - Train/validation/test splits
   - Normalized data
   - Encoded categorical variables

## Data Contracts

All data files should meet these contracts:
- **No missing required files**: graph_edges.csv and gridworld.csv must be present
- **Valid CSV format**: Proper headers, no corrupted rows
- **Reasonable size**: Files should load in reasonable time (<1min for base datasets)
- **Schema compliance**: Columns should match expected names

Run tests to validate:
```bash
pytest tests/test_data_contracts.py -v
```

## Git Tracking

- `base/` - **Tracked** (for small/toy datasets provided in assignment)
- `derived/` - **Not tracked** (gitignored, regenerated from pipeline)

Large datasets (>100MB) should not be committed to git. Use git-lfs or external storage instead.
