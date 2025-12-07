# Model Card: Graph Link Prediction

## Model Details
- **Model Type**: Similarity-Based Link Prediction
- **Framework**: NetworkX / Custom Implementation
- **Version**: 1.0
- **Date**: December 2025

## Model Description
Similarity-based link prediction on bipartite player-play graph using classical graph algorithms.

## Methods Implemented
1. **Common Neighbors**: Count shared neighbors between nodes
2. **Jaccard Coefficient**: Normalized intersection over union of neighborhoods
3. **Adamic-Adar Index**: Weights rare neighbors higher (inverse log frequency)

## Intended Use
- Recommend plays to players based on historical participation
- Predict missing edges in player-play bipartite graph
- Collaborative filtering for play recommendations
- Understanding player-play interaction patterns

## Graph Statistics
- **Graph Type**: Bipartite (player-play)
- **Edge Split**: 80% train / 20% test (stratified per player)
- **K Values Evaluated**: [3, 5, 10]

## Performance

### Hit@K at K=5
| Method | Hit@5 | MAP@5 | Coverage@5 |
|--------|-------|-------|------------|
| common_neighbors | 0.7485 | 0.5118 | 0.9870 |
| jaccard | 0.7466 | 0.5192 | 0.9913 |
| adamic_adar | 0.7563 | 0.5334 | 0.9913 |

## Best Method
- **Method**: adamic_adar at K=5
- **Hit@K**: 0.7563

## Limitations
1. Cold-start problem for new players/plays with no edges
2. Cannot capture complex feature interactions (structure-only)
3. Performance depends on graph density and connectivity
4. Assumes similar players like similar plays

## Ethical Considerations

### Bias Sources
- **Popularity Bias**: High-degree nodes (popular plays) may be over-recommended
- **Sampling Bias**: Historical data may not represent all player types equally
- **Filter Bubbles**: Similar players get similar recommendations

### Privacy
- Player IDs are anonymized integers
- No personally identifiable information in graph structure
- Recommendations based on aggregated patterns, not individuals

### Transparency
- All similarity methods are interpretable and well-documented
- Edge weights and rankings are explainable
- Coverage@K metric measures recommendation diversity

### Safety
- **Diversity Check**: Coverage@K ensures recommendations aren't too narrow
- **Human Review**: Low-confidence predictions should be validated
- **Fallback**: Random baseline provides comparison point

## Reproducibility
- **Seed**: 42
- **Config**: configs/default.yaml
- **Test Ratio**: 0.2
- **K Values**: [3, 5, 10]

---
*Generated: 2025-12-07T09:11:35.020772*
