# Model Card: Graph Link Prediction

## Model Description
Similarity-based link prediction on bipartite player-play graph.

## Methods Implemented
1. **Common Neighbors**: Count shared neighbors
2. **Jaccard Coefficient**: Normalized intersection over union
3. **Adamic-Adar Index**: Weights rare neighbors higher

## Intended Use
- Recommend plays to players based on historical participation
- Predict missing edges in player-play bipartite graph
- Collaborative filtering for play recommendations

## Graph Statistics
- Edge split: 80% train / 20% test (per player)
- K values evaluated: [3, 5, 10]

## Performance (Hit@K at K=5)
- common_neighbors: Hit@5 = 0.7447, MAP@5 = 0.5105, Coverage@5 = 0.9870
- jaccard: Hit@5 = 0.7485, MAP@5 = 0.5109, Coverage@5 = 0.9913
- adamic_adar: Hit@5 = 0.7563, MAP@5 = 0.5313, Coverage@5 = 0.9913

## Best Method
- adamic_adar at K=5 with Hit@K = 0.7563

## Limitations
- Cold-start problem for new players/plays
- Cannot capture complex feature interactions
- Performance depends on graph density

## Ethical Considerations
- Recommendations may reinforce existing patterns
- Diversity should be considered alongside accuracy
- Fair representation across player groups should be monitored
