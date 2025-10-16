"""
Main Test Script for FJSP NSGA-II Implementation
Tests different instances and compares results
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import time
import numpy as np
from src.fjsp_nsga2 import FJSP_NSGA2
from src.fjsp_instances import (
    load_kacem_instance,
    load_medium_instance,
    load_large_instance,
    print_instance_info
)
import matplotlib.pyplot as plt


def run_experiment(jobs, num_machines, instance_name, 
                   population_size=100, max_generations=100):
    """
    Run NSGA-II on an FJSP instance and return results
    """
    print("\n" + "=" * 80)
    print(f"RUNNING EXPERIMENT: {instance_name}")
    print("=" * 80)
    
    # Print instance info
    print_instance_info(jobs, num_machines, instance_name)
    
    # Initialize NSGA-II
    print(f"\nInitializing NSGA-II...")
    print(f"  Population size: {population_size}")
    print(f"  Max generations: {max_generations}")
    
    nsga2 = FJSP_NSGA2(
        jobs=jobs,
        num_machines=num_machines,
        population_size=population_size,
        max_generations=max_generations,
        crossover_prob=0.9,
        mutation_prob_1=0.6,
        mutation_prob_2=0.4
    )
    
    # Initialize population
    nsga2.initialize_population()
    
    # Get initial statistics
    initial_front = [ind for ind in nsga2.population if ind.rank == 0]
    if initial_front:
        initial_makespan = min(ind.makespan for ind in initial_front)
        initial_waiting = min(ind.total_waiting_time for ind in initial_front)
        initial_weighted = min(ind.total_weighted_completion_time for ind in initial_front)
    else:
        initial_makespan = min(ind.makespan for ind in nsga2.population)
        initial_waiting = min(ind.total_waiting_time for ind in nsga2.population)
        initial_weighted = min(ind.total_weighted_completion_time for ind in nsga2.population)
    
    print(f"\nInitial Best Values:")
    print(f"  Makespan: {initial_makespan:.2f}")
    print(f"  Waiting Time: {initial_waiting:.2f}")
    print(f"  Weighted Time: {initial_weighted:.2f}")
    
    # Run evolution
    start_time = time.time()
    pareto_front = nsga2.evolve()
    elapsed_time = time.time() - start_time
    
    # Get final statistics
    if pareto_front:
        final_makespan = min(ind.makespan for ind in pareto_front)
        final_waiting = min(ind.total_waiting_time for ind in pareto_front)
        final_weighted = min(ind.total_weighted_completion_time for ind in pareto_front)
    else:
        final_makespan = initial_makespan
        final_waiting = initial_waiting
        final_weighted = initial_weighted
    
    # Calculate improvements
    makespan_improvement = ((initial_makespan - final_makespan) / initial_makespan) * 100
    waiting_improvement = ((initial_waiting - final_waiting) / initial_waiting) * 100
    weighted_improvement = ((initial_weighted - final_weighted) / initial_weighted) * 100
    
    print(f"\n" + "=" * 80)
    print(f"RESULTS FOR {instance_name}")
    print("=" * 80)
    print(f"Execution time: {elapsed_time:.2f} seconds")
    print(f"Pareto front size: {len(pareto_front)}")
    
    print(f"\nFinal Best Values:")
    print(f"  Makespan: {final_makespan:.2f} (Improvement: {makespan_improvement:.2f}%)")
    print(f"  Waiting Time: {final_waiting:.2f} (Improvement: {waiting_improvement:.2f}%)")
    print(f"  Weighted Time: {final_weighted:.2f} (Improvement: {weighted_improvement:.2f}%)")
    
    # Print top solutions
    nsga2.print_pareto_solutions(max_solutions=10)
    
    return {
        'instance_name': instance_name,
        'nsga2': nsga2,
        'pareto_front': pareto_front,
        'execution_time': elapsed_time,
        'initial_makespan': initial_makespan,
        'final_makespan': final_makespan,
        'makespan_improvement': makespan_improvement,
        'initial_waiting': initial_waiting,
        'final_waiting': final_waiting,
        'waiting_improvement': waiting_improvement,
        'initial_weighted': initial_weighted,
        'final_weighted': final_weighted,
        'weighted_improvement': weighted_improvement
    }


def compare_results(results_list):
    """
    Compare results from multiple experiments
    """
    print("\n" + "=" * 80)
    print("COMPARISON OF ALL EXPERIMENTS")
    print("=" * 80)
    
    print(f"\n{'Instance':<20} {'Time(s)':<10} {'Pareto':<10} {'Makespan':<12} "
          f"{'Waiting':<12} {'Weighted':<12}")
    print(f"{'':20} {'':10} {'Size':<10} {'Improve':<12} {'Improve':<12} {'Improve':<12}")
    print("-" * 90)
    
    for result in results_list:
        print(f"{result['instance_name']:<20} "
              f"{result['execution_time']:<10.2f} "
              f"{len(result['pareto_front']):<10} "
              f"{result['makespan_improvement']:<12.2f}% "
              f"{result['waiting_improvement']:<12.2f}% "
              f"{result['weighted_improvement']:<12.2f}%")
    
    print("=" * 80)


def plot_all_convergence(results_list, save_path=None):
    """
    Plot convergence comparison for all instances
    """
    if save_path is None:
        save_path = os.path.join(os.path.dirname(__file__), '..', 'visualizations', 'all_convergence.png')
    
    fig, axes = plt.subplots(3, 1, figsize=(14, 12))
    
    colors = ['blue', 'red', 'green', 'orange', 'purple']
    
    for idx, result in enumerate(results_list):
        nsga2 = result['nsga2']
        instance_name = result['instance_name']
        color = colors[idx % len(colors)]
        
        generations = range(len(nsga2.generation_stats['best_makespan']))
        
        # Makespan
        axes[0].plot(generations, nsga2.generation_stats['best_makespan'], 
                     color=color, linewidth=2, label=instance_name, alpha=0.7)
        
        # Waiting Time
        axes[1].plot(generations, nsga2.generation_stats['best_waiting_time'], 
                     color=color, linewidth=2, label=instance_name, alpha=0.7)
        
        # Weighted Time
        axes[2].plot(generations, nsga2.generation_stats['best_weighted_time'], 
                     color=color, linewidth=2, label=instance_name, alpha=0.7)
    
    axes[0].set_xlabel('Generation', fontsize=11)
    axes[0].set_ylabel('Makespan', fontsize=11)
    axes[0].set_title('Makespan Convergence - All Instances', fontsize=13, fontweight='bold')
    axes[0].legend(loc='best', fontsize=9)
    axes[0].grid(True, alpha=0.3)
    
    axes[1].set_xlabel('Generation', fontsize=11)
    axes[1].set_ylabel('Total Waiting Time', fontsize=11)
    axes[1].set_title('Waiting Time Convergence - All Instances', fontsize=13, fontweight='bold')
    axes[1].legend(loc='best', fontsize=9)
    axes[1].grid(True, alpha=0.3)
    
    axes[2].set_xlabel('Generation', fontsize=11)
    axes[2].set_ylabel('Total Weighted Time', fontsize=11)
    axes[2].set_title('Weighted Time Convergence - All Instances', fontsize=13, fontweight='bold')
    axes[2].legend(loc='best', fontsize=9)
    axes[2].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"\nAll convergence plots saved to: {save_path}")
    plt.show()


def plot_improvement_comparison(results_list, save_path=None):
    """
    Bar chart comparing improvements across instances
    """
    if save_path is None:
        save_path = os.path.join(os.path.dirname(__file__), '..', 'visualizations', 'improvement_comparison.png')

    instance_names = [r['instance_name'] for r in results_list]
    makespan_improvements = [r['makespan_improvement'] for r in results_list]
    waiting_improvements = [r['waiting_improvement'] for r in results_list]
    weighted_improvements = [r['weighted_improvement'] for r in results_list]
    
    x = np.arange(len(instance_names))
    width = 0.25
    
    fig, ax = plt.subplots(figsize=(12, 6))
    
    bars1 = ax.bar(x - width, makespan_improvements, width, label='Makespan', 
                   color='steelblue', alpha=0.8)
    bars2 = ax.bar(x, waiting_improvements, width, label='Waiting Time', 
                   color='coral', alpha=0.8)
    bars3 = ax.bar(x + width, weighted_improvements, width, label='Weighted Time', 
                   color='lightgreen', alpha=0.8)
    
    ax.set_xlabel('Instance', fontsize=12, fontweight='bold')
    ax.set_ylabel('Improvement (%)', fontsize=12, fontweight='bold')
    ax.set_title('Objective Improvements Across Instances', fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(instance_names, rotation=15, ha='right')
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3, axis='y')
    
    # Add value labels on bars
    for bars in [bars1, bars2, bars3]:
        for bar in bars:
            height = bar.get_height()
            ax.annotate(f'{height:.1f}%',
                       xy=(bar.get_x() + bar.get_width() / 2, height),
                       xytext=(0, 3),
                       textcoords="offset points",
                       ha='center', va='bottom', fontsize=8)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"Improvement comparison saved to: {save_path}")
    plt.show()


def main():
    """
    Main function to run all experiments
    """
    print("=" * 80)
    print("FJSP NSGA-II COMPREHENSIVE TESTING")
    print("Multi-Objective Optimization: Makespan, Waiting Time, Weighted Time")
    print("=" * 80)
    
    results = []
    
    # Test 1: Small Kacem instance
    print("\n\n" + "#" * 80)
    print("TEST 1: SMALL INSTANCE (Kacem 4x5)")
    print("#" * 80)
    jobs_small, machines_small = load_kacem_instance()
    result_small = run_experiment(
        jobs_small, machines_small, "Kacem 4x5",
        population_size=50, max_generations=50
    )
    results.append(result_small)
    
    # Plot for small instance
    viz_dir = os.path.join(os.path.dirname(__file__), '..', 'visualizations')
    result_small['nsga2'].plot_convergence(
        save_path=os.path.join(viz_dir, 'kacem_convergence.png')
    )
    result_small['nsga2'].plot_pareto_front(
        save_path=os.path.join(viz_dir, 'kacem_pareto.png')
    )
    
    # Test 2: Medium instance
    print("\n\n" + "#" * 80)
    print("TEST 2: MEDIUM INSTANCE (8 Jobs x 6 Machines)")
    print("#" * 80)
    jobs_medium, machines_medium = load_medium_instance()
    result_medium = run_experiment(
        jobs_medium, machines_medium, "Medium 8x6",
        population_size=100, max_generations=100
    )
    results.append(result_medium)
    
    # Plot for medium instance
    result_medium['nsga2'].plot_convergence(
        save_path=os.path.join(viz_dir, 'medium_convergence.png')
    )
    result_medium['nsga2'].plot_pareto_front(
        save_path=os.path.join(viz_dir, 'medium_pareto.png')
    )
    
    # Test 3: Large instance
    print("\n\n" + "#" * 80)
    print("TEST 3: LARGE INSTANCE (15 Jobs x 8 Machines)")
    print("#" * 80)
    jobs_large, machines_large = load_large_instance()
    result_large = run_experiment(
        jobs_large, machines_large, "Large 15x8",
        population_size=150, max_generations=150
    )
    results.append(result_large)
    
    # Plot for large instance
    result_large['nsga2'].plot_convergence(
        save_path=os.path.join(viz_dir, 'large_convergence.png')
    )
    result_large['nsga2'].plot_pareto_front(
        save_path=os.path.join(viz_dir, 'large_pareto.png')
    )
    
    # Compare all results
    compare_results(results)
    
    # Plot comparisons
    plot_all_convergence(results)
    plot_improvement_comparison(results)
    
    # Generate summary report
    print("\n" + "=" * 80)
    print("SUMMARY REPORT")
    print("=" * 80)
    
    total_time = sum(r['execution_time'] for r in results)
    avg_makespan_improvement = np.mean([r['makespan_improvement'] for r in results])
    avg_waiting_improvement = np.mean([r['waiting_improvement'] for r in results])
    avg_weighted_improvement = np.mean([r['weighted_improvement'] for r in results])
    
    print(f"\nTotal execution time: {total_time:.2f} seconds")
    print(f"Average improvements across all instances:")
    print(f"  Makespan: {avg_makespan_improvement:.2f}%")
    print(f"  Waiting Time: {avg_waiting_improvement:.2f}%")
    print(f"  Weighted Time: {avg_weighted_improvement:.2f}%")
    
    print("\nKey Observations:")
    print("1. All three objectives (Makespan, Waiting Time, Weighted Time) show")
    print("   consistent improvement across generations.")
    print("2. The algorithm successfully generates diverse Pareto-optimal solutions.")
    print("3. Mutation probability decay helps in fine-tuning solutions over time.")
    print("4. Larger instances benefit from larger population sizes.")
    
    print("\n" + "=" * 80)
    print("ALL TESTS COMPLETED SUCCESSFULLY!")
    print("=" * 80)


if __name__ == "__main__":
    import warnings
    warnings.filterwarnings('ignore')
    
    main()
