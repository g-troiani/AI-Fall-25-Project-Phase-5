"""
Q-Learning for GridWorld
========================
Tabular Q-learning with dynamic rewards based on ball landing positions.
Source: Project 5
"""

import numpy as np
import matplotlib.pyplot as plt


class DynamicGridWorld:
    """
    GridWorld where rewards are dynamically set based on target position.
    
    Features:
        - High reward at target location
        - Reward decays with distance from target
        - Edge penalties for out of bounds
    """

    def __init__(self, grid_size: int = 10, 
                 field_x_range: tuple = (0, 120), 
                 field_y_range: tuple = (0, 53.3), 
                 seed: int = 42):
        self.grid_size = grid_size
        self.n_states = grid_size * grid_size
        self.n_actions = 4  # up, down, left, right
        self.action_names = ['↑', '↓', '←', '→']
        self.rng = np.random.default_rng(seed)

        self.field_x_range = field_x_range
        self.field_y_range = field_y_range

        self.rewards = np.zeros((grid_size, grid_size))
        self.ball_grid_pos = None  # Match notebook naming
        self.target_pos = None  # Alias for compatibility
        self.state = None

    def set_ball_landing(self, ball_land_x: float, ball_land_y: float):
        """
        Set reward grid based on ball landing position.
        Called before each episode to create play-specific rewards.
        """
        # Map ball landing coordinates to grid position
        grid_x = int((ball_land_x - self.field_x_range[0]) /
                     (self.field_x_range[1] - self.field_x_range[0]) * (self.grid_size - 1))
        grid_y = int((ball_land_y - self.field_y_range[0]) /
                     (self.field_y_range[1] - self.field_y_range[0]) * (self.grid_size - 1))

        grid_x = np.clip(grid_x, 0, self.grid_size - 1)
        grid_y = np.clip(grid_y, 0, self.grid_size - 1)

        self.ball_grid_pos = (grid_y, grid_x)
        self.target_pos = self.ball_grid_pos  # Alias for compatibility

        # Create reward map
        self.rewards = np.zeros((self.grid_size, self.grid_size))

        for i in range(self.grid_size):
            for j in range(self.grid_size):
                dist = np.sqrt((i - grid_y)**2 + (j - grid_x)**2)

                if dist == 0:
                    self.rewards[i, j] = 20.0
                elif dist <= 1.5:
                    self.rewards[i, j] = 10.0
                elif dist <= 3:
                    self.rewards[i, j] = 5.0
                else:
                    self.rewards[i, j] = -0.5

        # Edge penalties
        self.rewards[0, :] -= 2.0
        self.rewards[-1, :] -= 2.0
        self.rewards[:, 0] -= 2.0
        self.rewards[:, -1] -= 2.0

        return self

    def reset(self, start_x: float = None, start_y: float = None) -> int:
        """Reset to starting position."""
        if start_x is not None and start_y is not None:
            grid_x = int((start_x - self.field_x_range[0]) /
                        (self.field_x_range[1] - self.field_x_range[0]) * (self.grid_size - 1))
            grid_y = int((start_y - self.field_y_range[0]) /
                        (self.field_y_range[1] - self.field_y_range[0]) * (self.grid_size - 1))
            grid_x = np.clip(grid_x, 0, self.grid_size - 1)
            grid_y = np.clip(grid_y, 0, self.grid_size - 1)
            self.state = grid_y * self.grid_size + grid_x
        else:
            row = self.rng.integers(self.grid_size // 2, self.grid_size)
            col = self.rng.integers(0, self.grid_size)
            self.state = row * self.grid_size + col
        return self.state

    def step(self, action: int) -> tuple:
        """Take action, return (next_state, reward, done)."""
        row, col = self.state // self.grid_size, self.state % self.grid_size

        if action == 0:  # up
            row = max(0, row - 1)
        elif action == 1:  # down
            row = min(self.grid_size - 1, row + 1)
        elif action == 2:  # left
            col = max(0, col - 1)
        elif action == 3:  # right
            col = min(self.grid_size - 1, col + 1)

        self.state = row * self.grid_size + col
        reward = self.rewards[row, col]
        done = (row, col) == self.ball_grid_pos

        return self.state, reward, done

    # Alias for compatibility
    def set_target(self, target_x: float, target_y: float):
        """Alias for set_ball_landing."""
        return self.set_ball_landing(target_x, target_y)


class QLearningAgent:
    """
    Tabular Q-learning agent.
    
    Features:
        - ε-greedy exploration with decay
        - Learning rate α with decay: α_t = α₀/(1 + t/τ)
    """

    def __init__(self, n_states: int, n_actions: int, 
                 gamma: float = 0.95, alpha: float = 0.2,
                 epsilon: float = 1.0, epsilon_min: float = 0.01, 
                 epsilon_decay: float = 0.995,
                 alpha_decay_tau: int = 500, seed: int = 42):
        self.n_states = n_states
        self.n_actions = n_actions
        self.gamma = gamma
        self.alpha_0 = alpha
        self.alpha = alpha
        self.epsilon = epsilon
        self.epsilon_min = epsilon_min
        self.epsilon_decay = epsilon_decay
        self.alpha_decay_tau = alpha_decay_tau
        self.rng = np.random.default_rng(seed)

        self.Q = np.zeros((n_states, n_actions))
        self.step_count = 0

    def select_action(self, state: int) -> int:
        """ε-greedy action selection."""
        if self.rng.random() < self.epsilon:
            return self.rng.integers(0, self.n_actions)
        return np.argmax(self.Q[state])

    def update(self, state: int, action: int, reward: float, 
               next_state: int, done: bool):
        """Q-learning update with α decay."""
        self.step_count += 1
        self.alpha = self.alpha_0 / (1 + self.step_count / self.alpha_decay_tau)

        if done:
            target = reward
        else:
            target = reward + self.gamma * np.max(self.Q[next_state])

        self.Q[state, action] += self.alpha * (target - self.Q[state, action])

    def decay_epsilon(self):
        """Decay exploration rate."""
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)

    def get_policy(self) -> np.ndarray:
        """Get greedy policy."""
        return np.argmax(self.Q, axis=1)

    def get_value(self) -> np.ndarray:
        """Get state values."""
        return np.max(self.Q, axis=1)


def compute_convergence_time(returns: list, threshold_pct: float = 0.9,
                              window: int = 50) -> int:
    """
    Compute convergence time (episode) for RL training.

    Convergence is defined as the first episode where the moving average
    of returns exceeds threshold_pct * max moving average.

    Args:
        returns: List of episode returns
        threshold_pct: Fraction of max return to consider converged (default 0.9)
        window: Window size for moving average

    Returns:
        Episode number where convergence first occurred, or -1 if never converged
    """
    if len(returns) < window:
        return -1

    ma = np.convolve(returns, np.ones(window) / window, mode='valid')
    max_ma = np.max(ma)
    threshold = threshold_pct * max_ma

    for i, val in enumerate(ma):
        if val >= threshold:
            return i + window  # Account for valid mode offset

    return -1


def train_q_learning(env: DynamicGridWorld, agent: QLearningAgent,
                     target_positions: list, episodes: int = 500,
                     max_steps: int = 50, log_fn=None) -> dict:
    """
    Train Q-learning agent on dynamic gridworld.
    
    Args:
        env: GridWorld environment
        agent: Q-learning agent
        target_positions: List of (x, y) target positions
        episodes: Number of training episodes
        max_steps: Max steps per episode
        
    Returns:
        history: Training history with returns
    """
    if log_fn is None:
        from ..mlops.utils import log as log_fn
    
    returns = []
    
    for ep in range(episodes):
        # Sample a target position
        idx = ep % len(target_positions)
        target_x, target_y = target_positions[idx]
        env.set_ball_landing(target_x, target_y)
        
        state = env.reset()
        total_reward = 0
        
        for step in range(max_steps):
            action = agent.select_action(state)
            next_state, reward, done = env.step(action)
            
            agent.update(state, action, reward, next_state, done)
            
            state = next_state
            total_reward += reward
            
            if done:
                break
        
        agent.decay_epsilon()
        returns.append(total_reward)
        
        if (ep + 1) % 100 == 0:
            avg_return = np.mean(returns[-100:])
            log_fn(f"Episode {ep+1}: Avg Return = {avg_return:.2f}, ε = {agent.epsilon:.3f}")
    
    # Compute convergence time: episode where moving average first exceeds threshold
    convergence_episode = compute_convergence_time(returns, threshold_pct=0.9, window=50)

    return {'returns': returns, 'convergence_episode': convergence_episode}


def plot_policy(agent: QLearningAgent, grid_size: int, 
                save_path: str = None, title: str = "Q-Learning Policy"):
    """Plot the learned policy as arrows."""
    policy = agent.get_policy().reshape(grid_size, grid_size)
    values = agent.get_value().reshape(grid_size, grid_size)
    
    fig, ax = plt.subplots(figsize=(8, 8))
    
    # Value heatmap
    im = ax.imshow(values, cmap='RdYlGn', origin='upper')
    plt.colorbar(im, ax=ax, label='State Value')
    
    # Policy arrows
    action_arrows = {0: (0, -0.3), 1: (0, 0.3), 2: (-0.3, 0), 3: (0.3, 0)}
    
    for i in range(grid_size):
        for j in range(grid_size):
            action = policy[i, j]
            dx, dy = action_arrows[action]
            ax.arrow(j, i, dx, dy, head_width=0.2, head_length=0.1, 
                    fc='black', ec='black')
    
    ax.set_title(title)
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    
    plt.close()
    return fig


def plot_learning_curve(returns: list, save_path: str = None,
                        window: int = 50, title: str = "RL Learning Curve"):
    """Plot learning curve with moving average."""
    fig, ax = plt.subplots(figsize=(10, 4))
    
    ax.plot(returns, alpha=0.3, label='Episode Return')
    
    # Moving average
    if len(returns) >= window:
        ma = np.convolve(returns, np.ones(window)/window, mode='valid')
        ax.plot(range(window-1, len(returns)), ma, label=f'{window}-Episode MA')
    
    ax.set_xlabel('Episode')
    ax.set_ylabel('Return')
    ax.set_title(title)
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    
    plt.close()
    return fig


def test_rl_agent(env: DynamicGridWorld, agent: QLearningAgent,
                  test_targets: np.ndarray, test_starts: np.ndarray = None,
                  max_steps: int = 50, log_fn=None) -> dict:
    """
    Test trained RL agent on unseen target positions.
    
    Args:
        env: GridWorld environment
        agent: Trained Q-learning agent
        test_targets: Array of (target_x, target_y) positions
        test_starts: Optional array of (start_x, start_y) positions
        max_steps: Maximum steps per episode
        
    Returns:
        Dict with test metrics
    """
    if log_fn:
        log_fn(f"Testing RL agent on {len(test_targets)} unseen positions...")
    
    test_returns = []
    goals_reached = 0
    
    for i in range(len(test_targets)):
        target_x, target_y = test_targets[i]
        
        # Set start position if provided
        if test_starts is not None:
            start_x, start_y = test_starts[i]
        else:
            start_x, start_y = None, None
        
        env.set_ball_landing(target_x, target_y)
        state = env.reset(start_x, start_y)
        
        episode_return = 0
        
        # Use greedy policy (no exploration)
        for step in range(max_steps):
            action = np.argmax(agent.Q[state])  # Greedy
            next_state, reward, done = env.step(action)
            episode_return += reward
            state = next_state
            if done:
                goals_reached += 1
                break
        
        test_returns.append(episode_return)
    
    results = {
        'avg_return': np.mean(test_returns),
        'goal_rate': goals_reached / len(test_targets),
        'max_return': max(test_returns),
        'min_return': min(test_returns),
        'std_return': np.std(test_returns),
        'n_tests': len(test_targets)
    }
    
    if log_fn:
        log_fn(f"Test Results:")
        log_fn(f"  Average Return: {results['avg_return']:.2f}")
        log_fn(f"  Goal Reach Rate: {results['goal_rate']:.1%}")
    
    return results


def visualize_rl_results(env: DynamicGridWorld, agent: QLearningAgent,
                         episode_returns: list, save_dir: str = None,
                         last_target: tuple = None):
    """
    Create comprehensive RL visualization.
    
    Args:
        env: GridWorld environment
        agent: Trained Q-learning agent
        episode_returns: List of episode returns during training
        save_dir: Directory to save plots
        last_target: (x, y) of last target position used
    """
    fig, axes = plt.subplots(2, 2, figsize=(14, 12))
    
    # 1. Episode returns
    ax = axes[0, 0]
    ax.plot(episode_returns, 'b-', alpha=0.3)
    window = 50
    if len(episode_returns) >= window:
        ma = np.convolve(episode_returns, np.ones(window)/window, mode='valid')
        ax.plot(range(window-1, len(episode_returns)), ma, 'r-', linewidth=2, label=f'MA({window})')
    ax.set_xlabel('Episode')
    ax.set_ylabel('Return')
    ax.set_title('Episode Returns', fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # 2. Reward map (if target set)
    ax = axes[0, 1]
    im = ax.imshow(env.rewards, cmap='RdYlGn', origin='upper')
    if env.target_pos:
        ax.scatter(env.target_pos[1], env.target_pos[0], c='blue', s=200, marker='*',
                   label='Target', zorder=5)
    title = 'Reward Map'
    if last_target:
        title += f'\n(Target at {last_target[0]:.1f}, {last_target[1]:.1f})'
    ax.set_title(title, fontweight='bold')
    ax.set_xlabel('Grid X')
    ax.set_ylabel('Grid Y')
    plt.colorbar(im, ax=ax, label='Reward')
    ax.legend()
    
    # 3. Value function
    ax = axes[1, 0]
    value_grid = agent.get_value().reshape(env.grid_size, env.grid_size)
    im = ax.imshow(value_grid, cmap='viridis', origin='upper')
    ax.set_title('Learned Value Function V(s)', fontweight='bold')
    ax.set_xlabel('Grid X')
    ax.set_ylabel('Grid Y')
    plt.colorbar(im, ax=ax, label='Value')
    
    # 4. Policy arrows
    ax = axes[1, 1]
    policy = agent.get_policy().reshape(env.grid_size, env.grid_size)
    ax.imshow(np.ones((env.grid_size, env.grid_size)) * 0.9, cmap='gray', vmin=0, vmax=1, origin='upper')
    arrows = ['↑', '↓', '←', '→']
    for i in range(env.grid_size):
        for j in range(env.grid_size):
            ax.text(j, i, arrows[policy[i, j]], ha='center', va='center', 
                   fontsize=12, fontweight='bold', color='blue')
    ax.set_title('Learned Policy', fontweight='bold')
    ax.set_xlabel('Grid X')
    ax.set_ylabel('Grid Y')
    
    plt.tight_layout()
    
    if save_dir:
        plt.savefig(f"{save_dir}/rl_dynamic_results.png", dpi=150, bbox_inches='tight')
    
    plt.close()
    return fig
