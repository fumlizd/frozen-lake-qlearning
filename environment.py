import numpy as np

MAP = [
    "SFFFFFFF",
    "FFFFFFFF",
    "FFFHFFFF",
    "FFFHFFFF",
    "FFFHFFFF",
    "FHHFFFHF",
    "FHFFHFHF",
    "FFFHFFFG",
]

# Action indices: 0=Left, 1=Down, 2=Right, 3=Up
ACTION_NAMES = {0: "Left", 1: "Down", 2: "Right", 3: "Up"}
ACTION_DELTAS = {
    0: (0, -1),
    1: (1,  0),
    2: (0,  1),
    3: (-1, 0),
}


class FrozenLakeEnv:
    """
    Custom 8x8 Frozen Lake environment.

    State space : 64 integer indices (row * 8 + col)
    Action space: 4  (0=Left, 1=Down, 2=Right, 3=Up)

    Rewards:
        +1.0  reaching the Goal
         0.0  all other transitions (including falling in a Hole)
    """

    def __init__(self):
        self.grid = [list(row) for row in MAP]
        self.n_rows = 8
        self.n_cols = 8
        self.n_states = self.n_rows * self.n_cols
        self.n_actions = 4
        self.start_state = self._find_cell("S")
        self.state = self.start_state

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _find_cell(self, char):
        for r in range(self.n_rows):
            for c in range(self.n_cols):
                if self.grid[r][c] == char:
                    return self._pos_to_state(r, c)
        raise ValueError(f"Cell '{char}' not found in map")

    def _state_to_pos(self, state):
        return state // self.n_cols, state % self.n_cols

    def _pos_to_state(self, row, col):
        return row * self.n_cols + col

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def reset(self):
        """Reset the environment to the start state and return it."""
        self.state = self.start_state
        return self.state

    def step(self, action):
        """
        Take one step in the environment.

        Parameters
        ----------
        action : int  (0=Left, 1=Down, 2=Right, 3=Up)

        Returns
        -------
        next_state : int
        reward     : float
        done       : bool
        info       : dict
        """
        if self.is_terminal():
            raise RuntimeError("Cannot call step() on a terminal state. Call reset() first.")

        row, col = self._state_to_pos(self.state)
        dr, dc = ACTION_DELTAS[action]

        # Clamp to grid boundaries (wall = stay in place)
        new_row = max(0, min(self.n_rows - 1, row + dr))
        new_col = max(0, min(self.n_cols - 1, col + dc))

        self.state = self._pos_to_state(new_row, new_col)
        cell = self.grid[new_row][new_col]

        if cell == "H":
            return self.state, -1.0, True, {"outcome": "hole"}
        if cell == "G":
            return self.state, 1.0, True, {"outcome": "goal"}
        return self.state, 0.0, False, {"outcome": "frozen"}

    def render(self):
        """Print a text representation of the grid with the agent marked as 'A'."""
        agent_row, agent_col = self._state_to_pos(self.state)
        print()
        for r in range(self.n_rows):
            row_str = ""
            for c in range(self.n_cols):
                if r == agent_row and c == agent_col:
                    row_str += "A "
                else:
                    row_str += self.grid[r][c] + " "
            print(row_str)
        print()

    def get_state(self):
        """Return the current state index."""
        return self.state

    def is_terminal(self):
        """Return True if the current state is a terminal state (Hole or Goal)."""
        row, col = self._state_to_pos(self.state)
        return self.grid[row][col] in ("H", "G")

    def get_cell_type(self, state):
        """Return the cell character ('S','F','H','G') for the given state index."""
        row, col = self._state_to_pos(state)
        return self.grid[row][col]
