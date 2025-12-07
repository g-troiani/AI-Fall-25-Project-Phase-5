"""
Graph Link Prediction
=====================
Link prediction using similarity measures on bipartite graphs.
Source: Project 5 (Cell 36, 37)
"""

import numpy as np
from collections import defaultdict


class LinkPredictionScorer:
    """
    Link prediction using similarity measures on bipartite graph.
    Exact match to notebook Cell 36.
    """

    def __init__(self, train_edges: list):
        """
        Initialize with training edges.
        
        Args:
            train_edges: List of (player, play) tuples
        """
        self.player_to_plays = defaultdict(set)
        self.play_to_players = defaultdict(set)

        for player, play in train_edges:
            self.player_to_plays[player].add(play)
            self.play_to_players[play].add(player)

        self.all_players = set(self.player_to_plays.keys())
        self.all_plays = set(self.play_to_players.keys())

    def common_neighbors(self, player, play) -> int:
        """Count common neighbors between player and play."""
        player_plays = self.player_to_plays[player]
        play_players = self.play_to_players[play]

        common = 0
        for other_play in player_plays:
            common += len(self.play_to_players[other_play] & play_players)
        return common

    def jaccard(self, player, play) -> float:
        """Jaccard similarity."""
        player_plays = self.player_to_plays[player]
        play_players = self.play_to_players[play]

        player_neighbors = set()
        for p in player_plays:
            player_neighbors.update(self.play_to_players[p])

        if not player_neighbors or not play_players:
            return 0.0

        intersection = len(player_neighbors & play_players)
        union = len(player_neighbors | play_players)
        return intersection / union if union > 0 else 0.0

    def adamic_adar(self, player, play) -> float:
        """Adamic-Adar index (weights rare neighbors higher)."""
        player_plays = self.player_to_plays[player]
        play_players = self.play_to_players[play]

        score = 0.0
        for other_play in player_plays:
            common_players = self.play_to_players[other_play] & play_players
            for cp in common_players:
                degree = len(self.player_to_plays[cp])
                if degree > 1:
                    score += 1.0 / np.log(degree)
        return score

    def recommend_plays(self, player, k: int = 5, method: str = 'common_neighbors') -> list:
        """
        Recommend top-K plays for a player.
        
        Returns:
            List of (play, score) tuples sorted by score descending
        """
        scorer = getattr(self, method)
        player_plays = self.player_to_plays[player]
        candidate_plays = self.all_plays - player_plays

        scores = [(play, scorer(player, play)) for play in candidate_plays]
        scores.sort(key=lambda x: x[1], reverse=True)
        return scores[:k]

    # Alias for generic usage
    def recommend(self, node, k: int = 5, method: str = 'common_neighbors') -> list:
        """Alias for recommend_plays."""
        return self.recommend_plays(node, k, method)


def split_edges(edges: list, test_ratio: float = 0.2, seed: int = 42) -> tuple:
    """
    Split edges into train/test per player (80/20 per player).
    Matches notebook Cell 37 exactly.
    
    Returns:
        train_edges, test_edges
    """
    rng = np.random.default_rng(seed)
    
    # Group by player
    player_edges = defaultdict(list)
    for player, play in edges:
        player_edges[player].append(play)
    
    train_edges = []
    test_edges = []
    
    for player, plays in player_edges.items():
        if len(plays) < 2:
            train_edges.extend([(player, p) for p in plays])
            continue
        
        plays = list(plays)
        rng.shuffle(plays)
        n_test = max(1, int(len(plays) * test_ratio))
        
        test_edges.extend([(player, p) for p in plays[:n_test]])
        train_edges.extend([(player, p) for p in plays[n_test:]])
    
    return train_edges, test_edges


def hit_at_k(scorer: LinkPredictionScorer, test_edges: list, 
             k: int, method: str) -> float:
    """
    Compute Hit@K - fraction of test edges appearing in top-K recommendations.
    Exact match to notebook Cell 37.
    """
    player_test = defaultdict(set)
    for player, play in test_edges:
        player_test[player].add(play)
    
    hits = 0
    total = 0
    for player, true_plays in player_test.items():
        if player not in scorer.all_players:
            continue
        recs = {r[0] for r in scorer.recommend_plays(player, k, method)}
        for play in true_plays:
            if play in recs:
                hits += 1
            total += 1
    return hits / total if total > 0 else 0.0


def map_at_k(scorer: LinkPredictionScorer, test_edges: list,
             k: int, method: str) -> float:
    """
    Compute Mean Average Precision at K.
    Exact match to notebook Cell 37.
    """
    player_test = defaultdict(set)
    for player, play in test_edges:
        player_test[player].add(play)

    ap_scores = []
    for player, true_plays in player_test.items():
        if player not in scorer.all_players:
            continue
        recs = scorer.recommend_plays(player, k, method)
        hits = 0
        prec_sum = 0.0
        for i, (play, _) in enumerate(recs):
            if play in true_plays:
                hits += 1
                prec_sum += hits / (i + 1)
        ap = prec_sum / min(len(true_plays), k) if true_plays else 0.0
        ap_scores.append(ap)
    return np.mean(ap_scores) if ap_scores else 0.0


def coverage_at_k(scorer: LinkPredictionScorer, test_edges: list,
                  k: int, method: str) -> float:
    """
    Compute Coverage@K - fraction of total items recommended at least once.
    Measures catalog diversity of recommendations.
    """
    player_test = defaultdict(set)
    for player, play in test_edges:
        player_test[player].add(play)

    recommended_items = set()
    for player in player_test.keys():
        if player not in scorer.all_players:
            continue
        recs = scorer.recommend_plays(player, k, method)
        for play, _ in recs:
            recommended_items.add(play)

    total_items = len(scorer.all_plays)
    return len(recommended_items) / total_items if total_items > 0 else 0.0


def evaluate_link_prediction(scorer: LinkPredictionScorer, test_edges: list,
                             k_values: list = [3, 5, 10],
                             methods: list = ['common_neighbors', 'jaccard', 'adamic_adar'],
                             run_logger=None) -> list:
    """
    Evaluate link prediction using Hit@K, MAP@K, and Coverage@K.

    Returns:
        results: List of dicts with method, k, hit_at_k, map_at_k, coverage_at_k
    """
    results = []

    for method in methods:
        for k in k_values:
            hit_k = hit_at_k(scorer, test_edges, k, method)
            map_k = map_at_k(scorer, test_edges, k, method)
            cov_k = coverage_at_k(scorer, test_edges, k, method)

            results.append({
                'method': method,
                'k': k,
                'hit_at_k': hit_k,
                'map_at_k': map_k,
                'coverage_at_k': cov_k
            })

            if run_logger:
                run_logger.log('graph', f'{method}_hit@{k}', hit_k,
                              params={'k': k}, notes=method)
                run_logger.log('graph', f'{method}_map@{k}', map_k,
                              params={'k': k}, notes=method)
                run_logger.log('graph', f'{method}_coverage@{k}', cov_k,
                              params={'k': k}, notes=method)

    return results


def build_graph_from_data(df, player_col: str = 'player_id', 
                          play_col: str = 'play_key') -> list:
    """
    Build edge list from DataFrame.
    
    Args:
        df: DataFrame with player and play columns
        player_col: Column name for player identifier
        play_col: Column name for play identifier
        
    Returns:
        edges: List of (player, play) tuples
    """
    edges = df[[player_col, play_col]].drop_duplicates()
    return list(edges.itertuples(index=False, name=None))


def run_graph_evaluation(edges: list, test_ratio: float = 0.2, 
                         k_values: list = [3, 5, 10], seed: int = 42,
                         run_logger=None, log_fn=None) -> dict:
    """
    Full graph evaluation pipeline.
    
    Returns:
        results: List of evaluation metrics
    """
    if log_fn:
        log_fn("Evaluating link prediction methods...")
    
    # Split per player
    train_edges, test_edges = split_edges(edges, test_ratio, seed)
    
    if log_fn:
        log_fn(f"Train edges: {len(train_edges)}, Test edges: {len(test_edges)}")
    
    # Initialize scorer
    scorer = LinkPredictionScorer(train_edges)
    
    # Evaluate
    results = evaluate_link_prediction(scorer, test_edges, k_values, 
                                       run_logger=run_logger)
    
    return results
