"""
train.py — Train a Q-Learning agent on the custom Frozen Lake environment.

Bonus Option B: Training performance is visualised and saved to results/.
"""

import json
import os

import matplotlib.pyplot as plt
import numpy as np

from agent import QLearningAgent
from environment import FrozenLakeEnv

# ---------------------------------------------------------------------------
# Hyperparameters
# ---------------------------------------------------------------------------
ALPHA = 0.1          # Learning rate
GAMMA = 0.99         # Discount factor
EPSILON = 1.0        # Initial exploration rate
EPSILON_DECAY = 0.9999
EPSILON_MIN = 0.01
N_EPISODES = 50000
MAX_STEPS = 200
RESULTS_DIR = "results"

ACTION_SYMBOLS = {0: "←", 1: "↓", 2: "→", 3: "↑"}


# ---------------------------------------------------------------------------
# Training loop
# ---------------------------------------------------------------------------

def train(alpha=ALPHA, gamma=GAMMA, epsilon=EPSILON,
          epsilon_decay=EPSILON_DECAY, epsilon_min=EPSILON_MIN,
          n_episodes=N_EPISODES, max_steps=MAX_STEPS):

    env = QLearningAgent.__new__(QLearningAgent)   # avoid import shadowing
    env = FrozenLakeEnv()
    agent = QLearningAgent(
        n_states=env.n_states,
        n_actions=env.n_actions,
        alpha=alpha,
        gamma=gamma,
        epsilon=epsilon,
        epsilon_decay=epsilon_decay,
        epsilon_min=epsilon_min,
    )

    episode_rewards = []
    episode_successes = []
    epsilon_history = []

    for episode in range(n_episodes):
        state = env.reset()
        total_reward = 0.0

        for _ in range(max_steps):
            action = agent.choose_action(state)
            next_state, reward, done, _ = env.step(action)
            agent.update(state, action, reward, next_state, done)
            state = next_state
            total_reward += reward
            if done:
                break

        agent.decay_epsilon()

        episode_rewards.append(total_reward)
        episode_successes.append(1 if total_reward > 0 else 0)
        epsilon_history.append(agent.epsilon)

        if (episode + 1) % 5000 == 0:
            win_rate = sum(episode_successes[-5000:]) / 5000 * 100
            print(
                f"  Episode {episode + 1:>6}/{n_episodes} | "
                f"Success (last 5k): {win_rate:5.1f}% | "
                f"ε = {agent.epsilon:.4f}"
            )

    return agent, episode_rewards, episode_successes, epsilon_history


# ---------------------------------------------------------------------------
# Plotting  (Bonus Option B)
# ---------------------------------------------------------------------------

def plot_training(rewards, successes, epsilons, output_dir=RESULTS_DIR):
    os.makedirs(output_dir, exist_ok=True)
    window = 500

    fig, axes = plt.subplots(3, 1, figsize=(12, 10))
    fig.suptitle("Q-Learning Training on Frozen Lake (8×8)", fontsize=14, fontweight="bold")

    # --- Reward moving average ---
    ma_rewards = np.convolve(rewards, np.ones(window) / window, mode="valid")
    axes[0].plot(ma_rewards, color="steelblue", linewidth=1.5)
    axes[0].set_title(f"Episode Rewards  (moving avg, window={window})")
    axes[0].set_xlabel("Episode")
    axes[0].set_ylabel("Reward")
    axes[0].grid(True, alpha=0.3)

    # --- Success rate moving average ---
    ma_success = np.convolve(successes, np.ones(window) / window, mode="valid") * 100
    axes[1].plot(ma_success, color="seagreen", linewidth=1.5)
    axes[1].set_title(f"Success Rate %  (moving avg, window={window})")
    axes[1].set_xlabel("Episode")
    axes[1].set_ylabel("Success Rate (%)")
    axes[1].set_ylim(0, 105)
    axes[1].grid(True, alpha=0.3)

    # --- Epsilon decay ---
    axes[2].plot(epsilons, color="darkorange", linewidth=1.5)
    axes[2].set_title("Epsilon Decay Over Training")
    axes[2].set_xlabel("Episode")
    axes[2].set_ylabel("Epsilon")
    axes[2].grid(True, alpha=0.3)

    plt.tight_layout()
    path = os.path.join(output_dir, "training_curves.png")
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"\nTraining curves saved → {path}")


# ---------------------------------------------------------------------------
# Policy display
# ---------------------------------------------------------------------------

def print_policy(agent, env):
    policy = agent.get_policy()
    print("\nLearned Policy Grid:")
    print("+" + "----+" * 8)
    for r in range(8):
        row = "|"
        for c in range(8):
            s = r * 8 + c
            cell = env.get_cell_type(s)
            if cell == "H":
                row += "  H |"
            elif cell == "G":
                row += "  G |"
            else:
                row += f"  {ACTION_SYMBOLS[policy[s]]} |"
        print(row)
    print("+" + "----+" * 8)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    os.makedirs(RESULTS_DIR, exist_ok=True)

    print("=" * 60)
    print("  Q-Learning — Frozen Lake (8×8)  |  Training")
    print("=" * 60)
    print(f"  α={ALPHA}  γ={GAMMA}  ε₀={EPSILON}  decay={EPSILON_DECAY}"
          f"  ε_min={EPSILON_MIN}  episodes={N_EPISODES}")
    print()

    agent, rewards, successes, epsilons = train()

    # Persist Q-table
    q_path = os.path.join(RESULTS_DIR, "q_table.npy")
    agent.save(q_path)
    print(f"Q-table saved → {q_path}")

    # Save training statistics
    stats = {
        "hyperparameters": {
            "alpha": ALPHA,
            "gamma": GAMMA,
            "epsilon_initial": EPSILON,
            "epsilon_decay": EPSILON_DECAY,
            "epsilon_min": EPSILON_MIN,
            "n_episodes": N_EPISODES,
            "max_steps": MAX_STEPS,
        },
        "total_successes": int(sum(successes)),
        "overall_success_rate_pct": round(sum(successes) / N_EPISODES * 100, 2),
        "last_5000_success_rate_pct": round(sum(successes[-5000:]) / 5000 * 100, 2),
        "final_epsilon": round(epsilons[-1], 6),
    }
    stats_path = os.path.join(RESULTS_DIR, "train_stats.json")
    with open(stats_path, "w") as f:
        json.dump(stats, f, indent=2)
    print(f"Training stats saved → {stats_path}")

    # Plot (Bonus B)
    plot_training(rewards, successes, epsilons)

    # Show final policy
    env = FrozenLakeEnv()
    print_policy(agent, env)
