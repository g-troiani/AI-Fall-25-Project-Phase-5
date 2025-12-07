# Model Card: Reinforcement Learning (Q-Learning)

## Model Description
Tabular Q-learning agent for navigation in dynamic GridWorld environment.

## Algorithm
- Type: Tabular Q-Learning (off-policy TD control)
- State Space: 10x10 = 100 states
- Action Space: 4 (up, down, left, right)
- Discount Factor (γ): 0.95
- Learning Rate (α): 0.2 with decay

## Exploration Strategy
- ε-greedy with decay
- Initial ε: 1.0
- Minimum ε: 0.01
- Decay rate: 0.995

## Intended Use
- Learn optimal policy to navigate to ball landing positions
- Dynamic reward maps based on target location
- Decision support for player positioning

## Training Configuration
- Episodes: 500
- Max steps per episode: 50
- Target positions: Sampled from training data

## Performance
- Final Average Return (last 100 episodes): 81.11
- Maximum Return: 485.00
- Convergence Episode (90% threshold): 306

## Reward Structure
- Target reached: +20
- Near target (≤1.5 cells): +10
- Close to target (≤3 cells): +5
- Elsewhere: -0.5
- Edge penalty: -2

## Limitations
- Discrete state space may lose precision
- Q-table size grows quadratically with grid size
- Requires retraining for new reward structures

## Ethical Considerations
- Policy should be validated before real-world deployment
- Agent behavior should be interpretable
- Safety constraints should be incorporated for critical applications
