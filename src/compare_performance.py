"""Compare local search algorithms on warehouse rack placement optimization."""

import random
import numpy as np
from typing import Dict, List, Tuple
import matplotlib.pyplot as plt
from warehouse_env_racks import RackState
from local_search_agents import HillClimber, SimulatedAnnealing, GeneticAlgorithm
from rack_visualization import RackVisualizer


def run_comparison(num_runs: int = 20, seed: int = 42) -> Dict:
    """
    Run all algorithms on multiple random initial states.
    Args:
        num_runs: Number of random initial states to test.
        seed: Random seed for reproducibility.
    Returns:
        Dictionary with results from all algorithms.
    """
    random.seed(seed)
    np.random.seed(seed)

    results = {
        'hill_climbing': {
            'best_values': [],
            'final_objectives': [],
            'convergence_histories': [],
            'best_state': None,
        },
        'simulated_annealing': {
            'best_values': [],
            'final_objectives': [],
            'convergence_histories': [],
            'best_state': None,
        },
        'genetic_algorithm': {
            'best_values': [],
            'final_objectives': [],
            'convergence_histories': [],
            'best_state': None,
        },
    }

    print(f"Running {num_runs} trials with 3 algorithms...")
    print("-" * 70)

    for trial in range(num_runs):
        # Generate random initial state
        initial_state = RackState()

        # Hill Climbing
        hc = HillClimber(initial_state.copy(), max_iterations=200)
        hc_best, hc_history = hc.optimize()
        results['hill_climbing']['best_values'].append(hc_best.objective_function())
        results['hill_climbing']['convergence_histories'].append(hc_history)
        if (results['hill_climbing']['best_state'] is None or
            hc_best.objective_function() <
            results['hill_climbing']['best_state'].objective_function()):
            results['hill_climbing']['best_state'] = hc_best.copy()

        # Simulated Annealing
        sa = SimulatedAnnealing(initial_state.copy(),
                               initial_temp=250.0,
                               cooling_rate=0.995,
                               max_iterations=2200)
        sa_best, sa_history = sa.optimize()
        results['simulated_annealing']['best_values'].append(sa_best.objective_function())
        results['simulated_annealing']['convergence_histories'].append(sa_history)
        if (results['simulated_annealing']['best_state'] is None or
            sa_best.objective_function() <
            results['simulated_annealing']['best_state'].objective_function()):
            results['simulated_annealing']['best_state'] = sa_best.copy()

        # Genetic Algorithm
        ga = GeneticAlgorithm(population_size=30,
                            mutation_rate=0.2,
                            max_generations=60)
        ga_best, ga_history = ga.optimize()
        results['genetic_algorithm']['best_values'].append(ga_best.objective_function())
        results['genetic_algorithm']['convergence_histories'].append(ga_history)
        if (results['genetic_algorithm']['best_state'] is None or
            ga_best.objective_function() <
            results['genetic_algorithm']['best_state'].objective_function()):
            results['genetic_algorithm']['best_state'] = ga_best.copy()

        if (trial + 1) % 5 == 0:
            print(f"Completed {trial + 1}/{num_runs} trials")

    print("-" * 70)
    return results


def print_summary_statistics(results: Dict) -> None:
    """
    Print summary statistics for all algorithms.
    Args:
        results: Results dictionary from run_comparison.
    """
    print("\n" + "=" * 70)
    print("SUMMARY STATISTICS")
    print("=" * 70)
    for algorithm_name, data in results.items():
        best_values = np.array(data['best_values'])
        print(f"\n{algorithm_name.replace('_', ' ').title()}:")
        print(f"  Mean objective value:    {best_values.mean():.4f}")
        print(f"  Std deviation:           {best_values.std():.4f}")
        print(f"  Min (best) objective:    {best_values.min():.4f}")
        print(f"  Max (worst) objective:   {best_values.max():.4f}")
        print(f"  Best solution found:     {data['best_state'].objective_function():.4f}")


def average_convergence_history(convergence_histories: List[List[float]]) -> List[float]:
    """
    Average convergence histories, padding shorter ones.
    Args:
        convergence_histories: List of convergence history lists.
    Returns:
        Averaged history.
    """
    max_length = max(len(h) for h in convergence_histories)
    # Pad histories to max length by repeating last value
    padded = []
    for history in convergence_histories:
        if len(history) < max_length:
            padded.append(history + [history[-1]] * (max_length - len(history)))
        else:
            padded.append(history)
    # Average
    return np.mean(padded, axis=0).tolist()


if __name__ == "__main__":
    # Execute Performance Comparison
    results = run_comparison(num_runs=20, seed=42)
    print_summary_statistics(results)

    # Plot Convergence Curves
    convergence_data = {
        'Hill Climbing': average_convergence_history(
            results['hill_climbing']['convergence_histories']),
        'Simulated Annealing': average_convergence_history(
            results['simulated_annealing']['convergence_histories']),
        'Genetic Algorithm': average_convergence_history(
            results['genetic_algorithm']['convergence_histories']),
    }
    fig1 = RackVisualizer.plot_convergence(convergence_data, figsize=(10, 6))
    plt.draw()

    # Plot Best Layouts
    best_solutions = {
        'Hill Climbing': results['hill_climbing']['best_state'],
        'Simulated Annealing': results['simulated_annealing']['best_state'],
        'Genetic Algorithm': results['genetic_algorithm']['best_state'],
    }
    fig2 = RackVisualizer.plot_comparison_layouts(best_solutions, figsize=(15, 5))
    plt.show()

    # Display Results Summary
    print("\n" + "=" * 70)
    print("BEST SOLUTIONS FOUND")
    print("=" * 70)
    for algo_name, state in best_solutions.items():
        print(f"{algo_name}: {state.objective_function():.4f}")
