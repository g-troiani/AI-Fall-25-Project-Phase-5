"""
Data Pipeline
=============
Data loading, feature engineering, sequence construction, and splitting.
Source: Projects 1-5
"""

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


# Default numerical features used in the NFL trajectory prediction
NUMERICAL_FEATURES = ['x', 'y', 's', 'a', 'dir', 'o', 's_x', 's_y', 'a_x', 'a_y',
                      'ball_land_x', 'ball_land_y']

# NFL Player Position Acronyms Reference (Cell 32)
POSITION_GLOSSARY = """
NFL Player Position Acronyms:
═══════════════════════════════════════════════════════════════════════════
Acronym  Full Name           Role
───────────────────────────────────────────────────────────────────────────
WR       Wide Receiver       Catches passes, lines up wide near sidelines
TE       Tight End           Hybrid blocker/receiver, lines up next to OL
RB       Running Back        Carries ball on runs, catches short passes
FB       Fullback            Blocking back, occasional short-yardage carrier
QB       Quarterback         Throws passes, calls plays
CB       Cornerback          Defends against wide receivers
S        Safety              Deep defensive back (FS/SS variants)
LB       Linebacker          Middle defender (MLB, OLB, ILB variants)
DE       Defensive End       Pass rusher on edge of defensive line
DT       Defensive Tackle    Interior defensive lineman
OL       Offensive Line      Blockers (includes C, G, T)
K        Kicker              Kicks field goals and extra points
P        Punter              Punts on 4th down
UNK      Unknown             Missing/unspecified position in data
═══════════════════════════════════════════════════════════════════════════
Note: WR, TE, and RB are most relevant for trajectory prediction (pass catchers)
"""


def load_data(data_path: str, log_fn=None):
    """Load raw data from CSV."""
    if log_fn:
        log_fn(f"Loading data from {data_path}...")
    df = pd.read_csv(data_path)
    if log_fn:
        log_fn(f"Loaded {len(df)} rows, {len(df.columns)} columns")
    return df


def engineer_features(df: pd.DataFrame, catch_radius: float = 6.0, log_fn=None):
    """
    Create derived features from raw tracking data.
    
    Features created:
        - dir_rad: Direction in radians
        - s_x, s_y: Velocity components
        - a_x, a_y: Acceleration components
        - dist_to_ball: Distance to ball landing
        - within_catch_radius: Binary classification target
    """
    if log_fn:
        log_fn("Engineering features...")
    
    df = df.copy()
    
    # Create derived features
    df['dir_rad'] = np.deg2rad(df['dir'])
    df['s_x'] = df['s'] * np.cos(df['dir_rad'])
    df['s_y'] = df['s'] * np.sin(df['dir_rad'])
    df['a_x'] = df['a'] * np.cos(df['dir_rad'])
    df['a_y'] = df['a'] * np.sin(df['dir_rad'])
    
    # Create classification target
    df['dist_to_ball'] = np.sqrt(
        (df['x'] - df['ball_land_x'])**2 + 
        (df['y'] - df['ball_land_y'])**2
    )
    df['within_catch_radius'] = (df['dist_to_ball'] <= catch_radius).astype(int)
    
    if log_fn:
        log_fn(f"Class distribution: {df['within_catch_radius'].mean():.1%} positive")
    
    return df


def construct_sequences(df: pd.DataFrame, input_seq_len: int = 20, 
                        output_seq_len: int = 5, catch_radius: float = 6.0,
                        features: list = None, log_fn=None):
    """
    Create input/output sequences for sequential models.
    
    Args:
        df: DataFrame with tracking data
        input_seq_len: Number of input frames
        output_seq_len: Number of output frames to predict
        catch_radius: Threshold for classification
        features: List of feature columns to use
        
    Returns:
        sequences: List of dicts with X, Y_traj, Y_class, metadata
    """
    if log_fn:
        log_fn("Constructing sequences...")
    
    if features is None:
        features = NUMERICAL_FEATURES
    
    total_seq_len = input_seq_len + output_seq_len
    
    # Create play key for grouping
    df['play_key'] = df['game_id'].astype(str) + '_' + df['play_id'].astype(str)
    
    sequences = []
    metadata = []
    
    for play_key, group in df.groupby('play_key'):
        # Filter for prediction target players if column exists
        if 'player_to_predict' in group.columns:
            target_players = group[group['player_to_predict'] == True]
        else:
            target_players = group
        
        # Group by player
        player_col = 'nfl_id' if 'nfl_id' in target_players.columns else None
        
        if player_col:
            player_groups = target_players.groupby(player_col)
        else:
            player_groups = [(0, target_players)]
        
        for player_id, player_data in player_groups:
            # Sort by frame
            frame_col = 'frame_id' if 'frame_id' in player_data.columns else 'frame'
            player_data = player_data.sort_values(frame_col)
            
            if len(player_data) >= total_seq_len:
                # Extract features
                feature_data = player_data[features].values[:total_seq_len]
                
                # Input: first input_seq_len frames
                X_seq = feature_data[:input_seq_len]
                
                # Output trajectory: next output_seq_len frames (x, y only)
                Y_traj = feature_data[input_seq_len:, :2]
                
                # Classification label
                final_dist = player_data.iloc[total_seq_len - 1]['dist_to_ball']
                Y_class = int(final_dist <= catch_radius)
                
                # Ball landing position
                ball_land_x = player_data.iloc[0]['ball_land_x']
                ball_land_y = player_data.iloc[0]['ball_land_y']
                
                sequences.append({
                    'X': X_seq,
                    'Y_traj': Y_traj,
                    'Y_class': Y_class,
                    'ball_land_x': ball_land_x,
                    'ball_land_y': ball_land_y
                })
                
                # Metadata for fairness analysis
                meta = {
                    'play_key': play_key,
                    'player_id': player_id
                }
                if 'player_position' in player_data.columns:
                    meta['player_position'] = player_data.iloc[0]['player_position']
                metadata.append(meta)
    
    if log_fn:
        log_fn(f"Created {len(sequences)} sequences")
    
    return sequences, metadata


def split_data(sequences: list, metadata: list, test_size: float = 0.2, 
               val_size: float = 0.15, seed: int = 42, log_fn=None):
    """
    Stratified train/val/test split.
    
    Returns:
        Dict with train/val/test arrays for X, Y_traj, Y_class, metadata, ball_lands
    """
    if log_fn:
        log_fn("Splitting data...")
    
    # Convert to arrays
    X_seq = np.array([s['X'] for s in sequences])
    Y_traj = np.array([s['Y_traj'] for s in sequences])
    Y_class = np.array([s['Y_class'] for s in sequences])
    ball_lands = np.array([[s['ball_land_x'], s['ball_land_y']] for s in sequences])
    
    # First split: train+val vs test
    split1 = train_test_split(
        X_seq, Y_traj, Y_class, metadata, ball_lands,
        test_size=test_size, random_state=seed, stratify=Y_class
    )
    X_temp, X_test = split1[0], split1[1]
    Y_traj_temp, Y_traj_test = split1[2], split1[3]
    Y_class_temp, Y_class_test = split1[4], split1[5]
    meta_temp, meta_test = split1[6], split1[7]
    ball_temp, ball_test = split1[8], split1[9]
    
    # Second split: train vs val
    val_ratio = val_size / (1 - test_size)
    split2 = train_test_split(
        X_temp, Y_traj_temp, Y_class_temp, meta_temp, ball_temp,
        test_size=val_ratio, random_state=seed, stratify=Y_class_temp
    )
    X_train, X_val = split2[0], split2[1]
    Y_traj_train, Y_traj_val = split2[2], split2[3]
    Y_class_train, Y_class_val = split2[4], split2[5]
    meta_train, meta_val = split2[6], split2[7]
    ball_train, ball_val = split2[8], split2[9]
    
    if log_fn:
        log_fn(f"Train: {len(X_train)} ({Y_class_train.mean():.1%} positive)")
        log_fn(f"Val: {len(X_val)} ({Y_class_val.mean():.1%} positive)")
        log_fn(f"Test: {len(X_test)} ({Y_class_test.mean():.1%} positive)")
    
    return {
        'X_train': X_train, 'X_val': X_val, 'X_test': X_test,
        'Y_traj_train': Y_traj_train, 'Y_traj_val': Y_traj_val, 'Y_traj_test': Y_traj_test,
        'Y_class_train': Y_class_train, 'Y_class_val': Y_class_val, 'Y_class_test': Y_class_test,
        'meta_train': meta_train, 'meta_val': meta_val, 'meta_test': meta_test,
        'ball_train': ball_train, 'ball_val': ball_val, 'ball_test': ball_test
    }


def normalize_data(data: dict, log_fn=None):
    """
    Normalize features using StandardScaler fit on training data.
    
    Returns:
        data: Updated dict with scaled arrays
        scaler: Fitted StandardScaler
    """
    if log_fn:
        log_fn("Normalizing features...")
    
    X_train = data['X_train']
    X_val = data['X_val']
    X_test = data['X_test']
    
    n_train, seq_len, n_features = X_train.shape
    
    # Fit scaler on training data
    scaler = StandardScaler()
    X_train_flat = X_train.reshape(-1, n_features)
    scaler.fit(X_train_flat)
    
    # Transform all sets
    data['X_train_scaled'] = scaler.transform(X_train.reshape(-1, n_features)).reshape(n_train, seq_len, n_features)
    data['X_val_scaled'] = scaler.transform(X_val.reshape(-1, n_features)).reshape(len(X_val), seq_len, n_features)
    data['X_test_scaled'] = scaler.transform(X_test.reshape(-1, n_features)).reshape(len(X_test), seq_len, n_features)
    
    # First-frame features for classical ML
    data['X_train_classical'] = data['X_train_scaled'][:, 0, :]
    data['X_val_classical'] = data['X_val_scaled'][:, 0, :]
    data['X_test_classical'] = data['X_test_scaled'][:, 0, :]
    
    return data, scaler


def build_graph_edges(df: pd.DataFrame, log_fn=None):
    """
    Build player-play edges for graph link prediction.
    
    Returns:
        edges_list: List of (player_id, play_key) tuples
    """
    if log_fn:
        log_fn("Building graph from player-play interactions...")
    
    df = df.copy()
    
    # Create identifiers
    df['play_key'] = df['game_id'].astype(str) + '_' + df['play_id'].astype(str)
    df['player_id'] = df['player_position'] + '_' + df['game_id'].astype(str).str[-4:]
    
    # Unique edges
    edges = df[['player_id', 'play_key']].drop_duplicates()
    edges_list = list(edges.itertuples(index=False, name=None))
    
    if log_fn:
        log_fn(f"Total edges: {len(edges_list)}")
        log_fn(f"Unique players: {edges['player_id'].nunique()}")
        log_fn(f"Unique plays: {edges['play_key'].nunique()}")
    
    return edges_list


def run_full_pipeline(data_path: str, config: dict, log_fn=None):
    """
    Run the complete data pipeline.
    
    Returns:
        data: Dict with all processed arrays
        scaler: Fitted StandardScaler
        edges: Graph edges list
        df: Original DataFrame with engineered features
    """
    # Load
    df = load_data(data_path, log_fn)
    
    # Feature engineering
    df = engineer_features(df, config.get('catch_radius', 6.0), log_fn)
    
    # Sequence construction
    sequences, metadata = construct_sequences(
        df, 
        input_seq_len=config.get('input_seq_len', 20),
        output_seq_len=config.get('output_seq_len', 5),
        catch_radius=config.get('catch_radius', 6.0),
        log_fn=log_fn
    )
    
    # Split
    data = split_data(
        sequences, metadata,
        test_size=config.get('test_size', 0.2),
        val_size=config.get('val_size', 0.15),
        seed=config.get('seed', 42),
        log_fn=log_fn
    )
    
    # Normalize
    data, scaler = normalize_data(data, log_fn)
    
    # Build graph edges
    edges = build_graph_edges(df, log_fn)
    
    return data, scaler, edges, df
