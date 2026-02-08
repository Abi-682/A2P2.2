"""
Warehouse Storage Rack Placement Optimization
Using Local Search Algorithms: Hill-Climbing, Simulated Annealing, and Genetic Algorithm
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from copy import deepcopy
import random
from typing import List, Tuple, Dict
import json


class WarehouseState:
    """Represents a warehouse state with 20 racks on a 20x20 grid"""
    
    GRID_SIZE = 20
    NUM_RACKS = 20
    DEPOT = (10, 10)
    LAMBDA = 2.0
    CONGESTION_RADIUS = 5
    
    def __init__(self, positions: List[Tuple[int, int]] = None):
        """
        Initialize warehouse state
        Args:
            positions: List of 20 unique (x, y) positions. If None, generates random valid state.
        """
        if positions is None:
            self.positions = self._generate_random_state()
        else:
            self.positions = positions
        self._validate()
    
    @staticmethod
    def _generate_random_state() -> List[Tuple[int, int]]:
        """Generate a random valid state with 20 unique positions"""
        all_positions = [(x, y) for x in range(WarehouseState.GRID_SIZE) 
                         for y in range(WarehouseState.GRID_SIZE)]
        selected = random.sample(all_positions, WarehouseState.NUM_RACKS)
        return sorted(selected)
    
    def _validate(self):
        """Validate state: exactly 20 unique positions within bounds"""
        assert len(self.positions) == self.NUM_RACKS
        assert len(set(self.positions)) == self.NUM_RACKS
        for x, y in self.positions:
            assert 0 <= x < self.GRID_SIZE and 0 <= y < self.GRID_SIZE
    
    def manhattan_distance(self, pos1: Tuple[int, int], pos2: Tuple[int, int]) -> int:
        """Calculate Manhattan distance between two positions"""
        return abs(pos1[0] - pos2[0]) + abs(pos1[1] - pos2[1])
    
    def compute_objective(self) -> float:
        """
        Compute objective function value
        f(s) = (1/20) * sum(distances) + Lambda * congestion_count
        Lower is better (minimization problem)
        """
        # Compute average Manhattan distance from depot
        total_distance = sum(self.manhattan_distance(pos, self.DEPOT) 
                           for pos in self.positions)
        avg_distance = total_distance / self.NUM_RACKS
        
        # Count racks within congestion radius of depot
        congestion_count = sum(1 for pos in self.positions 
                             if self.manhattan_distance(pos, self.DEPOT) < self.CONGESTION_RADIUS)
        
        # Objective function
        objective = avg_distance + self.LAMBDA * congestion_count
        return objective
    
    def get_neighbors(self) -> List['WarehouseState']:
        """
        Generate all valid neighbor states by moving one rack by ±1 in x or y
        Maintains uniqueness constraint
        """
        neighbors = []
        
        for i in range(self.NUM_RACKS):
            x, y = self.positions[i]
            
            # Try all four possible moves: up, down, left, right
            for dx, dy in [(1, 0), (-1, 0), (0, 1), (0, -1)]:
                new_x, new_y = x + dx, y + dy
                
                # Check bounds
                if not (0 <= new_x < self.GRID_SIZE and 0 <= new_y < self.GRID_SIZE):
                    continue
                
                new_pos = (new_x, new_y)
                
                # Check uniqueness
                if new_pos in self.positions:
                    continue
                
                # Create neighbor state
                new_positions = self.positions.copy()
                new_positions[i] = new_pos
                new_positions.sort()
                
                neighbors.append(WarehouseState(new_positions))
        
        return neighbors
    
    def copy(self) -> 'WarehouseState':
        """Create a deep copy of the state"""
        return WarehouseState(self.positions.copy())
    
    def get_visualization_data(self) -> Tuple[np.ndarray, List[Tuple[int, int]]]:
        """Return grid and positions for visualization"""
        grid = np.zeros((self.GRID_SIZE, self.GRID_SIZE))
        for x, y in self.positions:
            grid[y, x] += 1
        return grid, self.positions


class HillClimbingSearch:
    """Steepest-ascent hill climbing (minimization version)"""
    
    def __init__(self, initial_state: WarehouseState, max_iterations: int = 1000):
        self.current_state = initial_state
        self.best_state = initial_state.copy()
        self.max_iterations = max_iterations
        self.history = [initial_state.compute_objective()]
        self.iteration_count = 0
    
    def search(self) -> WarehouseState:
        """Run hill climbing search"""
        current_objective = self.current_state.compute_objective()
        
        for iteration in range(self.max_iterations):
            self.iteration_count = iteration
            
            # Generate all neighbors
            neighbors = self.current_state.get_neighbors()
            
            if not neighbors:
                break
            
            # Find best neighbor
            best_neighbor = min(neighbors, key=lambda s: s.compute_objective())
            best_neighbor_objective = best_neighbor.compute_objective()
            
            # Check if improvement found
            if best_neighbor_objective < current_objective:
                self.current_state = best_neighbor
                current_objective = best_neighbor_objective
                self.best_state = best_neighbor.copy()
                self.history.append(current_objective)
            else:
                # Local optimum reached
                break
        
        return self.best_state
    
    def get_history(self) -> List[float]:
        """Return convergence history"""
        return self.history


class SimulatedAnnealing:
    """Simulated annealing with exponential cooling schedule"""
    
    def __init__(self, initial_state: WarehouseState, max_iterations: int = 1000, 
                 initial_temperature: float = 10.0, cooling_rate: float = 0.99):
        self.current_state = initial_state
        self.best_state = initial_state.copy()
        self.max_iterations = max_iterations
        self.initial_temperature = initial_temperature
        self.cooling_rate = cooling_rate
        self.history = [initial_state.compute_objective()]
        self.iteration_count = 0
    
    def search(self) -> WarehouseState:
        """Run simulated annealing search"""
        current_objective = self.current_state.compute_objective()
        best_objective = current_objective
        temperature = self.initial_temperature
        
        for iteration in range(self.max_iterations):
            self.iteration_count = iteration
            
            # Generate random neighbor
            neighbors = self.current_state.get_neighbors()
            if not neighbors:
                break
            
            neighbor = random.choice(neighbors)
            neighbor_objective = neighbor.compute_objective()
            
            # Acceptance criterion
            delta = neighbor_objective - current_objective
            if delta < 0 or random.random() < np.exp(-delta / max(temperature, 1e-10)):
                self.current_state = neighbor
                current_objective = neighbor_objective
                
                # Update best if improved
                if current_objective < best_objective:
                    best_objective = current_objective
                    self.best_state = neighbor.copy()
            
            self.history.append(best_objective)
            
            # Cool down
            temperature *= self.cooling_rate
        
        return self.best_state
    
    def get_history(self) -> List[float]:
        """Return convergence history"""
        return self.history


class GeneticAlgorithm:
    """Genetic Algorithm with selection, crossover, and mutation"""
    
    def __init__(self, population_size: int = 50, generations: int = 100, 
                 mutation_rate: float = 0.1, elite_size: int = 5):
        self.population_size = population_size
        self.generations = generations
        self.mutation_rate = mutation_rate
        self.elite_size = elite_size
        self.history = []
        self.best_state = None
    
    @staticmethod
    def create_individual() -> WarehouseState:
        """Create random individual"""
        return WarehouseState()
    
    @staticmethod
    def crossover(parent1: WarehouseState, parent2: WarehouseState) -> WarehouseState:
        """
        Crossover: take first 10 racks from parent1, 
        fill remaining with racks from parent2 not in parent1
        """
        positions1 = parent1.positions
        positions2 = parent2.positions
        
        # Take first 10 from parent1
        offspring_positions = list(positions1[:10])
        used = set(offspring_positions)
        
        # Fill remaining from parent2
        for pos in positions2:
            if len(offspring_positions) >= WarehouseState.NUM_RACKS:
                break
            if pos not in used:
                offspring_positions.append(pos)
                used.add(pos)
        
        # If still need more, add random positions
        while len(offspring_positions) < WarehouseState.NUM_RACKS:
            pos = (random.randint(0, WarehouseState.GRID_SIZE - 1),
                   random.randint(0, WarehouseState.GRID_SIZE - 1))
            if pos not in used:
                offspring_positions.append(pos)
                used.add(pos)
        
        offspring_positions.sort()
        return WarehouseState(offspring_positions)
    
    def mutate(self, individual: WarehouseState) -> WarehouseState:
        """Mutation: randomly move racks or swap positions"""
        mutated = individual.copy()
        
        if random.random() < self.mutation_rate:
            i = random.randint(0, WarehouseState.NUM_RACKS - 1)
            
            # Try random moves until valid position found
            max_attempts = 10
            for _ in range(max_attempts):
                new_x = random.randint(0, WarehouseState.GRID_SIZE - 1)
                new_y = random.randint(0, WarehouseState.GRID_SIZE - 1)
                new_pos = (new_x, new_y)
                
                if new_pos not in mutated.positions:
                    positions = mutated.positions.copy()
                    positions[i] = new_pos
                    positions.sort()
                    mutated = WarehouseState(positions)
                    break
        
        return mutated
    
    def search(self) -> WarehouseState:
        """Run genetic algorithm"""
        # Initialize population
        population = [self.create_individual() for _ in range(self.population_size)]
        
        for generation in range(self.generations):
            # Evaluate fitness (objective value)
            fitnesses = [(individual, individual.compute_objective()) 
                        for individual in population]
            fitnesses.sort(key=lambda x: x[1])  # Sort by objective (minimize)
            
            # Track best
            best_objective = fitnesses[0][1]
            self.history.append(best_objective)
            self.best_state = fitnesses[0][0].copy()
            
            # Elite selection
            elite = [individual for individual, _ in fitnesses[:self.elite_size]]
            
            # Create new population
            new_population = elite.copy()
            
            while len(new_population) < self.population_size:
                # Tournament selection
                parent1 = self._tournament_selection(fitnesses)
                parent2 = self._tournament_selection(fitnesses)
                
                # Crossover
                offspring = self.crossover(parent1, parent2)
                
                # Mutation
                offspring = self.mutate(offspring)
                
                new_population.append(offspring)
            
            population = new_population[:self.population_size]
        
        return self.best_state
    
    @staticmethod
    def _tournament_selection(fitnesses: List[Tuple[WarehouseState, float]], 
                            tournament_size: int = 3) -> WarehouseState:
        """Tournament selection"""
        tournament = random.sample(fitnesses, min(tournament_size, len(fitnesses)))
        winner = min(tournament, key=lambda x: x[1])
        return winner[0].copy()
    
    def get_history(self) -> List[float]:
        """Return convergence history"""
        return self.history


def visualize_layout(state: WarehouseState, title: str = "Warehouse Layout", 
                     objective_value: float = None) -> plt.Figure:
    """Create visualization of warehouse layout"""
    fig, ax = plt.subplots(figsize=(8, 8))
    
    grid, positions = state.get_visualization_data()
    
    # Show grid
    ax.imshow(grid, cmap='YlOrRd', origin='lower', alpha=0.6)
    
    # Plot racks
    for x, y in positions:
        ax.plot(x, y, 'bs', markersize=8, label='Rack' if x == positions[0][0] and y == positions[0][1] else '')
    
    # Plot depot
    depot_x, depot_y = WarehouseState.DEPOT
    ax.plot(depot_x, depot_y, 'r*', markersize=20, label='Depot', zorder=5)
    
    # Draw congestion area (radius 5 from depot)
    circle = plt.Circle(WarehouseState.DEPOT, WarehouseState.CONGESTION_RADIUS, 
                       color='red', fill=False, linestyle='--', linewidth=2, label='Congestion Zone')
    ax.add_patch(circle)
    
    ax.set_xlim(-0.5, WarehouseState.GRID_SIZE - 0.5)
    ax.set_ylim(-0.5, WarehouseState.GRID_SIZE - 0.5)
    ax.set_aspect('equal')
    ax.set_xlabel('X Coordinate')
    ax.set_ylabel('Y Coordinate')
    
    if objective_value is not None:
        title = f"{title}\nObjective Value: {objective_value:.4f}"
    
    ax.set_title(title)
    ax.legend(loc='upper right')
    ax.grid(True, alpha=0.3)
    
    return fig


def plot_convergence(histories: Dict[str, List[float]], title: str = "Convergence Curves") -> plt.Figure:
    """Plot convergence curves for multiple algorithms"""
    fig, ax = plt.subplots(figsize=(10, 6))
    
    for algorithm_name, history in histories.items():
        ax.plot(history, marker='o', markersize=3, label=algorithm_name, linewidth=2, alpha=0.7)
    
    ax.set_xlabel('Iteration')
    ax.set_ylabel('Objective Value')
    ax.set_title(title)
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    return fig


def run_experiment(num_trials: int = 20, max_iterations: int = 500) -> Dict:
    """Run complete experiment with multiple initial states"""
    results = {
        'hill_climbing': {'objectives': [], 'histories': [], 'best_state': None, 'best_objective': float('inf')},
        'simulated_annealing': {'objectives': [], 'histories': [], 'best_state': None, 'best_objective': float('inf')},
        'genetic_algorithm': {'objectives': [], 'histories': [], 'best_state': None, 'best_objective': float('inf')}
    }
    
    print(f"Running experiment with {num_trials} trials and {max_iterations} iterations per algorithm...")
    
    for trial in range(num_trials):
        print(f"\nTrial {trial + 1}/{num_trials}")
        
        # Use same initial state for fair comparison
        initial_state = WarehouseState()
        initial_objective = initial_state.compute_objective()
        print(f"  Initial objective: {initial_objective:.4f}")
        
        # Hill Climbing
        hc = HillClimbingSearch(initial_state.copy(), max_iterations=max_iterations)
        hc_best = hc.search()
        hc_objective = hc_best.compute_objective()
        results['hill_climbing']['objectives'].append(hc_objective)
        results['hill_climbing']['histories'].append(hc.get_history())
        if hc_objective < results['hill_climbing']['best_objective']:
            results['hill_climbing']['best_objective'] = hc_objective
            results['hill_climbing']['best_state'] = hc_best
        print(f"  Hill Climbing: {hc_objective:.4f}")
        
        # Simulated Annealing
        sa = SimulatedAnnealing(initial_state.copy(), max_iterations=max_iterations)
        sa_best = sa.search()
        sa_objective = sa_best.compute_objective()
        results['simulated_annealing']['objectives'].append(sa_objective)
        results['simulated_annealing']['histories'].append(sa.get_history())
        if sa_objective < results['simulated_annealing']['best_objective']:
            results['simulated_annealing']['best_objective'] = sa_objective
            results['simulated_annealing']['best_state'] = sa_best
        print(f"  Simulated Annealing: {sa_objective:.4f}")
        
        # Genetic Algorithm
        ga = GeneticAlgorithm(population_size=50, generations=max_iterations // 5, mutation_rate=0.2)
        ga_best = ga.search()
        ga_objective = ga_best.compute_objective()
        results['genetic_algorithm']['objectives'].append(ga_objective)
        results['genetic_algorithm']['histories'].append(ga.get_history())
        if ga_objective < results['genetic_algorithm']['best_objective']:
            results['genetic_algorithm']['best_objective'] = ga_objective
            results['genetic_algorithm']['best_state'] = ga_best
        print(f"  Genetic Algorithm: {ga_objective:.4f}")
    
    return results


def print_summary(results: Dict):
    """Print summary statistics"""
    print("\n" + "="*80)
    print("EXPERIMENT SUMMARY")
    print("="*80)
    
    for algorithm_name, data in results.items():
        objectives = data['objectives']
        print(f"\n{algorithm_name.upper().replace('_', ' ')}")
        print(f"  Best: {min(objectives):.4f}")
        print(f"  Worst: {max(objectives):.4f}")
        print(f"  Mean: {np.mean(objectives):.4f}")
        print(f"  Std Dev: {np.std(objectives):.4f}")


if __name__ == "__main__":
    # Run experiment
    results = run_experiment(num_trials=20, max_iterations=500)
    
    # Print summary
    print_summary(results)
    
    # Create convergence plot
    avg_histories = {}
    for algorithm_name in results.keys():
        histories = results[algorithm_name]['histories']
        # Pad histories to same length
        max_len = max(len(h) for h in histories)
        padded = []
        for h in histories:
            if len(h) < max_len:
                padded.append(h + [h[-1]] * (max_len - len(h)))
            else:
                padded.append(h)
        avg_histories[algorithm_name] = np.mean(padded, axis=0).tolist()
    
    fig_convergence = plot_convergence(avg_histories, "Average Convergence Curves (20 Trials)")
    plt.tight_layout()
    plt.savefig('convergence_curves.png', dpi=150, bbox_inches='tight')
    print("\nConvergence plot saved as 'convergence_curves.png'")
    
    # Create layout visualizations
    fig = plt.figure(figsize=(15, 5))
    
    algorithms_to_plot = ['hill_climbing', 'simulated_annealing', 'genetic_algorithm']
    for idx, algorithm_name in enumerate(algorithms_to_plot, 1):
        best_state = results[algorithm_name]['best_state']
        best_objective = results[algorithm_name]['best_objective']
        
        ax = fig.add_subplot(1, 3, idx)
        grid, positions = best_state.get_visualization_data()
        ax.imshow(grid, cmap='YlOrRd', origin='lower', alpha=0.6)
        
        for x, y in positions:
            ax.plot(x, y, 'bs', markersize=6)
        
        depot_x, depot_y = WarehouseState.DEPOT
        ax.plot(depot_x, depot_y, 'r*', markersize=16)
        
        circle = plt.Circle(WarehouseState.DEPOT, WarehouseState.CONGESTION_RADIUS, 
                           color='red', fill=False, linestyle='--', linewidth=2)
        ax.add_patch(circle)
        
        ax.set_xlim(-0.5, WarehouseState.GRID_SIZE - 0.5)
        ax.set_ylim(-0.5, WarehouseState.GRID_SIZE - 0.5)
        ax.set_aspect('equal')
        ax.set_title(f"{algorithm_name.replace('_', ' ').title()}\nObj: {best_objective:.4f}")
        ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('best_layouts.png', dpi=150, bbox_inches='tight')
    print("Best layouts plot saved as 'best_layouts.png'")
    
    print("\nExperiment complete!")
