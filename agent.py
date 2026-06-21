import numpy as np


class QLearningAgent:
    """
    Tabular Q-Learning agent.

    Update rule (off-policy TD):
        Q(s,a) <- Q(s,a) + alpha * [r + gamma * max_a' Q(s',a') - Q(s,a)]

    Exploration uses epsilon-greedy with multiplicative decay per episode.
    """

    def __init__(
        self,
        n_states: int,
        n_actions: int,
        alpha: float = 0.1,
        gamma: float = 0.99,
        epsilon: float = 1.0,
        epsilon_decay: float = 0.9995,
        epsilon_min: float = 0.01,
        seed: int = 42,
    ):
        self.n_states = n_states
        self.n_actions = n_actions
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.epsilon_min = epsilon_min

        np.random.seed(seed)
        self.Q = np.zeros((n_states, n_actions), dtype=np.float64)

    # ------------------------------------------------------------------
    # Action selection
    # ------------------------------------------------------------------

    def choose_action(self, state: int) -> int:
        """Epsilon-greedy action selection."""
        if np.random.random() < self.epsilon:
            return np.random.randint(self.n_actions)
        return int(np.argmax(self.Q[state]))

    # ------------------------------------------------------------------
    # Learning
    # ------------------------------------------------------------------

    def update(self, state: int, action: int, reward: float, next_state: int, done: bool):
        """
        Apply the Q-Learning update equation:
            Q(s,a) <- Q(s,a) + alpha * [r + gamma * max Q(s',a') - Q(s,a)]

        When done=True the bootstrap target collapses to just `r`.
        """
        max_next_q = 0.0 if done else float(np.max(self.Q[next_state]))
        td_error = reward + self.gamma * max_next_q - self.Q[state, action]
        self.Q[state, action] += self.alpha * td_error

    def decay_epsilon(self):
        """Multiply epsilon by the decay factor, clipping at epsilon_min."""
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)

    # ------------------------------------------------------------------
    # Policy extraction
    # ------------------------------------------------------------------

    def get_policy(self) -> np.ndarray:
        """Return a length-n_states array of greedy actions."""
        return np.argmax(self.Q, axis=1)

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def save(self, path: str):
        np.save(path, self.Q)

    def load(self, path: str):
        self.Q = np.load(path)
