"""
evaluate.py — Evaluate a trained Q-Learning agent on the Frozen Lake environment.

Runs the agent greedily (epsilon=0) for N_EVAL_EPISODES episodes and reports:
  - Success Rate (%)
  - Average Reward
  - Number of Successful Runs
  - Number of Failures
"""

import json
import os

import numpy as np

from agent import QLearningAgent
from environment import FrozenLakeEnv

N_EVAL_EPISODES = 500
MAX_STEPS = 200
RESULTS_DIR = "results"
Q_TABLE_PATH = os.path.join(RESULTS_DIR, "q_table.npy")

ACTION_SYMBOLS = {0: "←", 1: "↓", 2: "→", 3: "↑"}


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------

def evaluate(q_table_path=Q_TABLE_PATH, n_episodes=N_EVAL_EPISODES,
             max_steps=MAX_STEPS, render_example=False):
    """
    Evaluate the trained agent greedily over `n_episodes` episodes.

    Returns a dict with evaluation metrics.
    """
    if not os.path.exists(q_table_path):
        raise FileNotFoundError(
            f"Q-table not found at '{q_table_path}'. Run train.py first."
        )

    env = FrozenLakeEnv()
    agent = QLearningAgent(n_states=env.n_states, n_actions=env.n_actions)
    agent.load(q_table_path)
    agent.epsilon = 0.0   # purely greedy during evaluation

    all_rewards = []
    successes = 0
    failures = 0

    for episode in range(n_episodes):
        state = env.reset()
        total_reward = 0.0

        if render_example and episode == 0:
            print("\nRendering first evaluation episode:")
            env.render()

        for _ in range(max_steps):
            action = agent.choose_action(state)
            next_state, reward, done, _ = env.step(action)
            state = next_state
            total_reward += reward

            if render_example and episode == 0:
                env.render()

            if done:
                break

        all_rewards.append(total_reward)
        if total_reward > 0:
            successes += 1
        else:
            failures += 1

    success_rate = successes / n_episodes * 100
    avg_reward = float(np.mean(all_rewards))

    metrics = {
        "n_episodes": n_episodes,
        "successes": successes,
        "failures": failures,
        "success_rate_pct": round(success_rate, 2),
        "avg_reward": round(avg_reward, 6),
    }
    return metrics


# ---------------------------------------------------------------------------
# Policy display
# ---------------------------------------------------------------------------

def print_policy(q_table_path=Q_TABLE_PATH):
    env = FrozenLakeEnv()
    agent = QLearningAgent(n_states=env.n_states, n_actions=env.n_actions)
    agent.load(q_table_path)

    policy = agent.get_policy()
    print("\nExtracted Policy Grid  (← ↓ → ↑  |  H=Hole  G=Goal):")
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
    print("=" * 50)
    print("  Q-Learning — Frozen Lake (8×8)  |  Evaluation")
    print("=" * 50)

    metrics = evaluate(render_example=True)

    print(f"\n  Episodes evaluated : {metrics['n_episodes']}")
    print(f"  Successful runs    : {metrics['successes']}")
    print(f"  Failures           : {metrics['failures']}")
    print(f"  Success Rate       : {metrics['success_rate_pct']:.2f}%")
    print(f"  Average Reward     : {metrics['avg_reward']:.6f}")
    print("=" * 50)

    # Persist evaluation results
    os.makedirs(RESULTS_DIR, exist_ok=True)
    out_path = os.path.join(RESULTS_DIR, "eval_results.json")
    with open(out_path, "w") as f:
        json.dump(metrics, f, indent=2)
    print(f"\nEvaluation results saved → {out_path}")

    print_policy()
