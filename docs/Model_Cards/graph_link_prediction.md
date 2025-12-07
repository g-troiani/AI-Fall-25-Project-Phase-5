# Model Card: Graph Link Prediction

## Model Details

**Model Type**: Graph Neural Network (GNN) for Link Prediction
**Framework**: PyTorch Geometric
**Version**: 1.0
**Date**: December 2024

### Architecture

- **Graph Type**: Bipartite graph (customers ↔ products)
- **Node Features**: Customer/product embeddings
- **GNN Layers**: 2 Graph Convolutional layers
- **Scoring Methods**: Common Neighbors, Jaccard, Adamic-Adar
- **Output**: Top-K product recommendations per customer

## Intended Use

- **Recommendation System**: Suggest products to customers based on graph structure
- **Cold Start**: Handle new customers/products with minimal data
- **Explainability**: Graph structure provides interpretable recommendations

**Note**: Requires graph_edges.csv (customer-product interactions) which is not included in current dataset.

## Training Data

- **Expected**: Customer-product bipartite edges
- **Current**: Placeholder (graph_edges.csv not available)
- **Split**: 80% train edges, 20% test edges (edge masking)

## Performance Metrics

*Placeholder (no graph data available)*

Expected metrics:
- Hit@10: ~0.65 (65% of test edges in top-10 recommendations)
- MAP@10: ~0.45 (Mean Average Precision)
- Coverage: ~0.80 (80% of products recommended at least once)

## Limitations

1. **Data Dependency**: Requires graph_edges.csv (not in current project)
2. **Cold Start**: New customers with no edges get random recommendations
3. **Popularity Bias**: Tends to recommend popular products
4. **Scalability**: GNN computation scales with graph size

## Ethical Considerations

### Bias
- **Popularity Bias**: Popular products over-recommended
- **Filter Bubble**: May create echo chambers (only similar products recommended)
- **Demographic Bias**: If customers are demographically skewed, recommendations reflect that

### Mitigation
- Diversity-aware re-ranking
- Explore/exploit balance (ε-greedy recommendations)
- Fairness constraints (ensure minority products get exposure)

### Privacy
- Customer purchase history is sensitive data
- Aggregated graph structure should not reveal individual preferences

---

**Last Updated**: December 6, 2024
**Status**: Placeholder (no graph data available)
