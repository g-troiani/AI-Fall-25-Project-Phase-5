# Model Card: Q-Learning for Gridworld

## Model Details

**Model Type**: Tabular Q-Learning (Reinforcement Learning)
**Framework**: NumPy
**Version**: 1.0
**Date**: December 2024

### Algorithm

- **Environment**: 5x5 gridworld with rewards
- **State Space**: 25 states (grid cells)
- **Action Space**: 4 actions (up, down, left, right)
- **Q-Table**: 25 × 4 matrix
- **Exploration**: ε-greedy (ε starts at 1.0, decays to 0.01)
- **Learning Rate**: α = 0.2
- **Discount Factor**: γ = 0.95

## Intended Use

- **Policy Learning**: Find optimal path to goal in gridworld
- **Demonstration**: Show RL concepts (exploration vs. exploitation)
- **API Endpoint**: Serve learned policy via GET /policy/gridworld

**Note**: Requires gridworld.csv (5x5 reward map) which is not included in current dataset.

## Training Data

- **Expected**: 5x5 reward grid with goal state (reward=+100) and obstacles (reward=-10)
- **Current**: Placeholder (gridworld.csv not available)
- **Training**: 500 episodes, max 50 steps per episode

## Performance Metrics

*Placeholder (no gridworld data available)*

Expected metrics:
- Average Episodic Return: ~85 (converged)
- Convergence Time: ~300 episodes
- Policy Accuracy: 100% (optimal path found)

### Example Learned Policy

```
Goal is at (4, 4), obstacles at (1, 2) and (3, 2)

→ → → ↓ ↓
→ → X ↓ ↓
→ → → ↓ ↓
→ → X ↓ ↓
→ → → → G

X = obstacle, G = goal, arrows = optimal actions
```

## Limitations

1. **Tabular Method**: Doesn't scale to large state spaces (use DQN for > 10k states)
2. **Discrete Actions**: Only 4 cardinal directions (no diagonal movement)
3. **Static Environment**: Assumes rewards don't change over time
4. **No Generalization**: Must retrain for new gridworld layouts

## Ethical Considerations

### Bias
- Reward function bias: If goal placement favors certain paths, policy inherits that bias
- Exploration bias: ε-greedy may under-explore certain states

### Safety
- **Simulation Only**: This is a toy environment, not safety-critical
- **Real-World RL**: Requires extensive safety testing (reward hacking, unintended behaviors)

### Transparency
- **Interpretable Policy**: Q-table is fully interpretable (inspect action values)
- **Reproducibility**: Fixed seed ensures deterministic learning

---

**Last Updated**: December 6, 2024
**Status**: Placeholder (no gridworld data available)
