# Model Card: Reinforcement Learning (Q-Learning)

## Model Details
- **Model Type**: Tabular Q-Learning (Off-Policy TD Control)
- **Framework**: Custom NumPy Implementation
- **Version**: 1.0
- **Date**: December 2025

## Model Description
Tabular Q-learning agent for navigation in dynamic GridWorld environment.
Learns optimal policy to reach ball landing positions with dynamic reward maps.

## Algorithm
- **Type**: Tabular Q-Learning (off-policy TD control)
- **State Space**: 10×10 = 100 states
- **Action Space**: 4 (up, down, left, right)
- **Discount Factor (γ)**: 0.95
- **Learning Rate (α)**: 0.2 with decay

## Exploration Strategy
- **Method**: ε-greedy with exponential decay
- **Initial ε**: 1.0
- **Minimum ε**: 0.01
- **Decay Rate**: 0.995

## Intended Use
- Learn optimal policy to navigate to ball landing positions
- Dynamic reward maps based on target location
- Decision support for player positioning
- Real-time policy execution in simulated environment

## Training Configuration
- **Episodes**: 500
- **Max Steps per Episode**: 50
- **Target Positions**: Sampled from training data

## Performance
| Metric | Value |
|--------|-------|
| Final Avg Return (last 100) | 81.11 |
| Maximum Return | 485.00 |
| Total Episodes | 500 |
| Convergence Episode (90%) | 306 |

## Reward Structure
| Condition | Reward |
|-----------|--------|
| Target reached | +20 |
| Near target (≤1.5 cells) | +10 |
| Close to target (≤3 cells) | +5 |
| Elsewhere | -0.5 |
| Edge penalty | -2 |

## Limitations
1. Discrete state space may lose spatial precision
2. Q-table size grows quadratically with grid size
3. Requires retraining for new reward structures
4. No generalization to unseen target positions

## Ethical Considerations

### Bias Sources
- **Reward Shaping Bias**: Hand-crafted rewards may not align with real objectives
- **Exploration Bias**: ε-greedy may underexplore certain state regions
- **Target Sampling Bias**: Training targets from data may not cover all scenarios

### Privacy
- Environment is simulated; no real-world data used
- Target positions derived from anonymized ball trajectories
- Q-table contains no personally identifiable information

### Transparency
- Q-values fully interpretable (state-action values)
- Policy can be visualized as action map over grid
- Episode returns logged for training analysis
- Convergence tracked and documented

### Safety
- **Bounded Actions**: Agent can only move in 4 cardinal directions
- **Episode Limits**: Maximum steps prevent infinite loops
- **Human-in-the-Loop**: Policy should be validated before real deployment
- **Simulation First**: Always test in simulation before real-world use

## Reproducibility
- **Seed**: 42
- **Config**: configs/default.yaml
- **Grid Size**: 10
- **Episodes**: 500
- **Q-Table**: Saved to outputs/models/q_table.npy

---
*Generated: 2025-12-07T09:11:35.020864*
