# Frozen Lake Q-Learning from First Principles

DSCD 614 – Reinforcement Learning · Assignment 1  

---

## Introduction

### What is Reinforcement Learning?

According to the Reinforcement Learning (RL) paradigm, an autonomous agent learns how to make decisions by interacting with its surroundings. The agent receives a reward signal together with a transition to a new state at each discrete time step after observing the current state and choosing an action based on its policy. Finding a policy that maximizes the anticipated total of discounted future rewards is the goal.

### What is Frozen Lake?

Frozen Lake is a benchmark Reinforcement Learning environment in which an agent learns to navigate a slippery frozen surface from a starting position to a goal while avoiding holes. It is widely used to illustrate how algorithms such as Q-Learning learn optimal decision-making through trial and error, making it an excellent educational tool for understanding the fundamentals of Reinforcement Learning. The map used in this assignment is:

```
S F F F F F F F
F F F F F F F F
F F F H F F F F
F F F H F F F F
F F F H F F F F
F H H F F F H F
F H F F H F H F
F F F H F F F G
```

- **S** — Start state (top-left, position 0)
- **F** — Frozen (safe to stand on)
- **H** — Hole (terminal; the agent falls in and the episode ends)
- **G** — Goal (terminal; reward of +1)

---

## Environment Design

Environment design refers to the process of creating and defining the world in which a reinforcement learning (RL) agent operates. The environment is the setting in which the agent interacts while learning to make decisions and achieve specific goals. It provides the agent with information about its current situation, responds to the agent's actions, and generates rewards or penalties that guide the learning process.

### State Representation

Each of the 64 grid cells is encoded as a single integer `state = row × 8 + col`. This gives a flat state space of size 64, which maps directly to rows in the Q-table.

### Action Representation

| Index | Direction | Row delta | Col delta |
|-------|-----------|-----------|-----------|
| 0     | Left      | 0         | −1        |
| 1     | Down      | +1        | 0         |
| 2     | Right     | 0         | +1        |
| 3     | Up        | −1        | 0         |

Movement at grid boundaries is clamped, the agent stays in place if it would move off the grid.

### Reward Structure

| Event        | Reward |
|--------------|--------|
| Reach Goal   | +1.0   |
| Fall in Hole | −1.0   |
| Safe step    |  0.0   |

A negative reward for holes is essential on this map: with only sparse positive reward (+1 at the goal), the agent would rarely discover the goal through random exploration. The hole penalty provides an immediate negative signal that propagates through the Q-table via the Bellman update, enabling the agent to learn hole-avoidance efficiently.

---

## Q-Learning Algorithm

Implemented in `agent.py` as the `QLearningAgent` class.

### Description



### Updated Equation

```
Q(s, a) ← Q(s, a) + α [ r + γ · max_a' Q(s', a') − Q(s, a) ]
```

- `α` — learning rate: controls how much new information overwrites old estimates  
- `γ` — discount factor: weights future rewards relative to immediate rewards  
- `r` — reward received after taking action `a` in state `s`  
- `s'` — next state  
- `max_a' Q(s', a')` — best known value of the next state (bootstrap target)

When the episode ends (`done=True`), the bootstrap term collapses to zero: the target is simply `r`.

### Exploration Strategy

**Epsilon-greedy with multiplicative decay:**

- With probability `ε`, choose a random action (explore).
- With probability `1 − ε`, choose `argmax_a Q(s, a)` (exploit).
- After each episode: `ε ← max(ε_min, ε × decay_rate)`.

This ensures the agent explores broadly early in training (when `ε ≈ 1`) and gradually shifts to exploitation as the Q-table matures.

---

## Training Procedure

Implemented in `train.py`. Run with:

```bash
python3 train.py
```

### Hyperparameters

| Parameter       | Value  | Rationale                                          |
|-----------------|--------|----------------------------------------------------|
| Learning rate α | 0.1    | Standard value; balances speed and stability       |
| Discount factor γ | 0.99 | High value; rewards far into the future count      |
| Initial ε       | 1.0    | Full exploration at the start                      |
| ε decay rate    | 0.9999 | Slow decay keeps exploration active for ~23k eps   |
| ε minimum       | 0.01   | Retains a small amount of exploration at all times |
| Episodes        | 50,000 | Sufficient for convergence on the 8×8 map         |
| Max steps/ep    | 200    | Prevents infinite loops in early training          |

### Training Statistics Recorded

- Episode rewards (`results/train_stats.json`)
- Success rate per window of 5,000 episodes
- Epsilon value over time
- Training curves plot (`results/training_curves.png`)

---

## Results

### Training Convergence

| Episode range | Success Rate |
|---------------|-------------|
| 0 – 5,000     | 12.3%        |
| 5,001 – 10,000 | 31.9%       |
| 10,001 – 15,000 | 53.9%      |
| 15,001 – 20,000 | 70.5%      |
| 25,001 – 30,000 | 88.6%      |
| 45,001 – 50,000 | **98.5%**  |

### Final Evaluation (500 episodes, greedy policy)

| Metric          | Value     |
|-----------------|-----------|
| Success Rate    | **100%**  |
| Average Reward  | 1.000000  |
| Successes       | 500 / 500 |
| Failures        | 0         |

### Learned Policy

```
+----+----+----+----+----+----+----+----+
|  ↓ |  ↓ |  ↓ |  ↓ |  ↓ |  ↓ |  ↓ |  ↓ |
|  → |  → |  → |  → |  ↓ |  ↓ |  ↓ |  ↓ |
|  → |  → |  ↑ |  H |  ↓ |  ↓ |  ↓ |  ↓ |
|  ↑ |  ↑ |  ↑ |  H |  ↓ |  ↓ |  ↓ |  ↓ |
|  ↑ |  ↑ |  ← |  H |  → |  → |  → |  ↓ |
|  ↑ |  H |  H |  → |  ↑ |  ↑ |  H |  ↓ |
|  ↑ |  H |  ↓ |  ← |  H |  ↑ |  H |  ↓ |
|  ↑ |  ← |  ← |  H |  ↓ |  ↑ |  → |  G |
+----+----+----+----+----+----+----+----+
```

The policy routes the agent around the column-3 holes via the right side of the grid and navigates the bottom-row obstacles to reach the goal.

### Discussion

The training results in results/training_curves.png show that the agent's performance improved steadily throughout the 50,000 training episodes. Setting the epsilon decay rate to 0.9999 allowed the agent to continue exploring the environment for a longer period, giving it enough opportunities to find the goal and learn the safest paths. As training progressed, the agent gradually relied more on what it had learned and eventually converged to an optimal policy. The final policy achieved a 100% success rate, showing that Q-Learning can effectively solve a deterministic Frozen Lake environment when given sufficient exploration and an appropriate reward structure.

---

## Execution Instructions

### Requirements

```bash
pip3 install numpy matplotlib reportlab
```

Or install from the provided file:

```bash
pip3 install -r requirements.txt
```

### Run Training

```bash
python3 train.py
```

Trains the agent for 50,000 episodes, saves the Q-table to `results/q_table.npy`, and writes training curves to `results/training_curves.png`.

### Run Evaluation

```bash
python3 evaluate.py
```

Loads the saved Q-table and evaluates the greedy policy over 500 episodes. Results are printed to stdout and saved to `results/eval_results.json`.

### Generate Report

```bash
python3 generate_report.py
```

Produces `report.pdf` using ReportLab.

---

## Repository Structure

```
frozen-lake-qlearning/
├── environment.py      # FrozenLakeEnv class (Part A)
├── agent.py            # QLearningAgent class (Part B)
├── train.py            # Training loop + plots (Parts C, D, Bonus B)
├── evaluate.py         # Evaluation (Part E)
├── generate_report.py  # PDF report generation
├── requirements.txt
├── README.md
├── report.pdf
└── results/
    ├── q_table.npy
    ├── train_stats.json
    ├── eval_results.json
    └── training_curves.png
```
