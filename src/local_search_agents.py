"""
Local search algorithms for warehouse rack placement optimization.
Implements three algorithms:
- Steepest-ascent hill-climbing (minimize objective)
- Simulated annealing with exponential cooling schedule
- Genetic algorithm with tournament selection, crossover, and mutation
"""
import random
import math
from typing import List, Tuple
from warehouse_env_racks import RackState


class HillClimber:
    """Steepest-ascent hill-climbing (minimization)."""

    def __init__(self, initial_state: RackState = None, max_iterations: int = 1000):
        """
        Initialize the hill climber.
        Args:
            initial_state: Starting state. If None, generates random.
            max_iterations: Maximum iterations to run.
        """
        self.initial_state = initial_state or RackState()
        self.max_iterations = max_iterations
        self.best_state = self.initial_state.copy()
        self.history = [self.best_state.objective_function()]

    def optimize(self) -> Tuple[RackState, List[float]]:
        """
        Run hill climbing algorithm.
        Returns:
            Tuple of (best_state, history of objective values).
        """
        current_state = self.initial_state.copy()
        for iteration in range(self.max_iterations):
            neighbors = current_state.get_neighbors()
            if not neighbors:
                break
            # Find best neighbor (steepest ascent / descent in minimize)
            best_neighbor = min(neighbors,
                              key=lambda s: s.objective_function())
            best_neighbor_value = best_neighbor.objective_function()
            current_value = current_state.objective_function()
            # Move to better neighbor
            if best_neighbor_value < current_value:
                current_state = best_neighbor
                self.history.append(best_neighbor_value)
                # Track best overall
                if best_neighbor_value < self.best_state.objective_function():
                    self.best_state = best_neighbor.copy()
            else:
                # Local minimum reached
                break
        return self.best_state, self.history


class SimulatedAnnealing:
    """Simulated annealing with exponential cooling schedule."""

    def __init__(self, initial_state: RackState = None,
                 initial_temp: float = 100.0,
                 cooling_rate: float = 0.995,
                 max_iterations: int = 1000):
        """
        Initialize simulated annealing.
        Args:
            initial_state: Starting state. If None, generates random.
            initial_temp: Initial temperature.
            cooling_rate: Temperature multiplier per iteration (< 1).
            max_iterations: Maximum iterations to run.
        """
        self.initial_state = initial_state or RackState()
        self.initial_temp = initial_temp
        self.cooling_rate = cooling_rate
        self.max_iterations = max_iterations
        self.best_state = self.initial_state.copy()
        self.history = [self.best_state.objective_function()]

    def acceptance_probability(self, current_value: float,
                              neighbor_value: float,
                              temperature: float) -> float:
        """
        Compute acceptance probability.
        Args:
            current_value: Current objective value.
            neighbor_value: Neighbor objective value.
            temperature: Current temperature.
        Returns:
            Probability of accepting move (0 to 1).
        """
        if neighbor_value < current_value:
            return 1.0
        # Accept worse solution with probability based on temperature
        return math.exp(-(neighbor_value - current_value) / temperature)

    def optimize(self) -> Tuple[RackState, List[float]]:
        """
        Run simulated annealing algorithm.
        Returns:
            Tuple of (best_state, history of objective values).
        """
        current_state = self.initial_state.copy()
        current_value = current_state.objective_function()
        temperature = self.initial_temp
        for iteration in range(self.max_iterations):
            neighbors = current_state.get_neighbors()
            if not neighbors:
                break
            # Pick random neighbor
            neighbor = random.choice(neighbors)
            neighbor_value = neighbor.objective_function()
            # Accept or reject move
            if self.acceptance_probability(current_value,
                                          neighbor_value,
                                          temperature) > random.random():
                current_state = neighbor
                current_value = neighbor_value
                self.history.append(neighbor_value)
                # Track best overall
                if neighbor_value < self.best_state.objective_function():
                    self.best_state = neighbor.copy()
            else:
                self.history.append(current_value)
            # Cool down
            temperature *= self.cooling_rate
        return self.best_state, self.history


class GeneticAlgorithm:
    """Genetic algorithm for optimization."""

    def __init__(self, population_size: int = 50,
                 mutation_rate: float = 0.2,
                 max_generations: int = 50):
        """
        Initialize genetic algorithm.
        Args:
            population_size: Size of population.
            mutation_rate: Probability of mutation per individual.
            max_generations: Maximum generations to evolve.
        """
        self.population_size = population_size
        self.mutation_rate = mutation_rate
        self.max_generations = max_generations
        self.best_state = None
        self.history = []

    def create_population(self, size: int) -> List[RackState]:
        """Create initial population of random states."""
        return [RackState() for _ in range(size)]

    def tournament_selection(self, population: List[RackState],
                            tournament_size: int = 3) -> RackState:
        """
        Tournament selection: randomly pick tournament_size individuals,
        return the best one.
        Args:
            population: Population to select from.
            tournament_size: Size of tournament.
        Returns:
            Selected individual.
        """
        tournament = random.sample(population, tournament_size)
        return min(tournament, key=lambda s: s.objective_function())

    def crossover(self, parent1: RackState, parent2: RackState) -> RackState:
        """
        Single-point crossover: split positions at random point.
        Args:
            parent1: First parent.
            parent2: Second parent.
        Returns:
            Child with genes from both parents.
        """
        # Single-point crossover
        crossover_point = random.randint(1, len(parent1.positions) - 1)
        child_positions = (parent1.positions[:crossover_point] +
                          parent2.positions[crossover_point:])
        # Ensure uniqueness by removing duplicates and filling gaps
        unique_positions = list(dict.fromkeys(child_positions))
        while len(unique_positions) < RackState.NUM_RACKS:
            # Add random positions to fill gaps
            x = random.randint(0, RackState.GRID_SIZE - 1)
            y = random.randint(0, RackState.GRID_SIZE - 1)
            if (x, y) not in unique_positions and (x, y) != RackState.DEPOT:
                unique_positions.append((x, y))
        # Trim if too many
        unique_positions = unique_positions[:RackState.NUM_RACKS]
        return RackState(unique_positions)

    def mutate(self, state: RackState) -> RackState:
        """
        Mutation: move a random rack by ±1 in random direction.
        Args:
            state: State to mutate.
        Returns:
            Mutated state.
        """
        if random.random() > self.mutation_rate:
            return state.copy()
        # Try up to 10 times to find valid mutation
        for _ in range(10):
            state_copy = state.copy()
            rack_idx = random.randint(0, RackState.NUM_RACKS - 1)
            x, y = state_copy.positions[rack_idx]
            dx = random.choice([-1, 0, 1])
            dy = random.choice([-1, 0, 1])
            if dx == 0 and dy == 0:
                continue
            new_x = x + dx
            new_y = y + dy
            # Check validity
            if (0 <= new_x < RackState.GRID_SIZE and
                0 <= new_y < RackState.GRID_SIZE and
                (new_x, new_y) != RackState.DEPOT and
                (new_x, new_y) not in state_copy.positions):
                state_copy.positions[rack_idx] = (new_x, new_y)
                return state_copy
        return state.copy()

    def optimize(self) -> Tuple[RackState, List[float]]:
        """
        Run genetic algorithm.
        Returns:
            Tuple of (best_state, history of objective values).
        """
        # Initialize population
        population = self.create_population(self.population_size)
        self.best_state = min(population,
                             key=lambda s: s.objective_function())
        self.history = [self.best_state.objective_function()]
        for generation in range(self.max_generations):
            # Evaluate fitness (lower objective = better)
            # Create new population through selection, crossover, mutation
            new_population = []
            for _ in range(self.population_size):
                # Tournament selection
                parent1 = self.tournament_selection(population)
                parent2 = self.tournament_selection(population)
                # Crossover
                child = self.crossover(parent1, parent2)
                # Mutation
                child = self.mutate(child)
                new_population.append(child)
            population = new_population
            # Track best in current generation
            gen_best = min(population,
                          key=lambda s: s.objective_function())
            gen_best_value = gen_best.objective_function()
            if gen_best_value < self.best_state.objective_function():
                self.best_state = gen_best.copy()
            self.history.append(gen_best_value)
        return self.best_state, self.history
