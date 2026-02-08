"""
Warehouse storage rack placement environment.
Manages state representation, objective function, and neighborhood generation
for optimizing the placement of 20 storage racks in a 20x20 warehouse grid.
"""
import random
from typing import List, Tuple
import numpy as np


class RackState:
    """Represents the placement of 20 storage racks in a 20x20 warehouse."""
    
    GRID_SIZE = 20
    NUM_RACKS = 20
    DEPOT = (10, 10)
    CONGESTION_THRESHOLD = 5
    CONGESTION_WEIGHT = 2.0

    def __init__(self, positions: List[Tuple[int, int]] = None):
        """
        Initialize a rack placement state.
        Args:
            positions: List of (x, y) tuples representing rack positions.
                      If None, generates random placement.
        """
        if positions is None:
            self.positions = self._generate_random_placement()
        else:
            self.positions = list(set(positions))  # Ensure uniqueness
            if len(self.positions) != self.NUM_RACKS:
                raise ValueError(
                    f"Must have exactly {self.NUM_RACKS} unique positions"
                )

    def _generate_random_placement(self) -> List[Tuple[int, int]]:
        """Generate a random valid placement of racks."""
        positions = set()
        while len(positions) < self.NUM_RACKS:
            x = random.randint(0, self.GRID_SIZE - 1)
            y = random.randint(0, self.GRID_SIZE - 1)
            # Avoid placing racks at depot
            if (x, y) != self.DEPOT:
                positions.add((x, y))
        return sorted(list(positions))

    @staticmethod
    def manhattan_distance(pos1: Tuple[int, int],
                          pos2: Tuple[int, int]) -> int:
        """Compute Manhattan distance between two positions."""
        return abs(pos1[0] - pos2[0]) + abs(pos1[1] - pos2[1])

    def objective_function(self) -> float:
        """
        Compute the objective function value.
        f(s) = (1/20) * sum(distances from depot) + 2.0 * congestion_count
        Returns:
            float: Objective value to minimize.
        """
        # Average travel distance to depot
        total_distance = sum(
            self.manhattan_distance(self.DEPOT, pos)
            for pos in self.positions
        )
        avg_distance = total_distance / self.NUM_RACKS

        # Congestion penalty: count racks within distance threshold of depot
        congestion_count = sum(
            1 for pos in self.positions
            if self.manhattan_distance(self.DEPOT, pos) < self.CONGESTION_THRESHOLD
        )
        congestion_penalty = self.CONGESTION_WEIGHT * congestion_count

        return avg_distance + congestion_penalty

    def get_neighbors(self) -> List['RackState']:
        """
        Generate all neighbors by moving one rack by ±1 in x or y.
        Returns:
            List of neighboring RackState objects.
        """
        neighbors = []
        for i, (x, y) in enumerate(self.positions):
            # Try all four directions
            for dx, dy in [(1, 0), (-1, 0), (0, 1), (0, -1)]:
                new_x = x + dx
                new_y = y + dy
                # Check bounds
                if not (0 <= new_x < self.GRID_SIZE and
                        0 <= new_y < self.GRID_SIZE):
                    continue
                # Check not at depot
                if (new_x, new_y) == self.DEPOT:
                    continue
                # Check uniqueness
                new_pos = (new_x, new_y)
                if new_pos in self.positions:
                    continue
                # Create neighbor
                neighbor_positions = self.positions.copy()
                neighbor_positions[i] = new_pos
                neighbors.append(RackState(neighbor_positions))
        return neighbors

    def copy(self) -> 'RackState':
        """Create a deep copy of this state."""
        return RackState(self.positions.copy())

    def to_grid(self) -> np.ndarray:
        """
        Convert to a 2D grid representation.
        Returns:
            20x20 array where 1 = rack, 2 = depot, 0 = empty.
        """
        grid = np.zeros((self.GRID_SIZE, self.GRID_SIZE), dtype=int)
        for x, y in self.positions:
            grid[y, x] = 1
        grid[self.DEPOT[1], self.DEPOT[0]] = 2
        return grid

    def __repr__(self) -> str:
        return f"RackState(obj={self.objective_function():.2f})"
