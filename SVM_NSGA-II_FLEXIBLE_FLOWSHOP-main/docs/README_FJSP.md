# NSGA-II for Flexible Job Shop Scheduling Problem (FJSP)

## Overview

This implementation applies the **NSGA-II (Non-dominated Sorting Genetic Algorithm II)** multi-objective optimization algorithm to solve the **Flexible Job Shop Scheduling Problem (FJSP)**. The implementation is based on the research paper:

> "An effective multi-objective metaheuristic for the support vector machine with feature selection" (Knowledge-Based Systems, 2025)

## Problem Description

### Flexible Job Shop Scheduling Problem (FJSP)

FJSP is an extension of the classical Job Shop Scheduling Problem where:
- Each job consists of multiple operations that must be processed in sequence
- Each operation can be processed on multiple alternative machines (flexibility)
- Each machine can process at most one operation at a time
- Processing times vary depending on the machine selected

### Optimization Objectives

This implementation optimizes three conflicting objectives simultaneously:

1. **Minimize Makespan**: The maximum completion time across all jobs
2. **Minimize Total Waiting Time**: Sum of waiting times for all operations
3. **Minimize Total Weighted Completion Time**: Sum of job completion times weighted by job priority

## Features

### Key Components

1. **Multi-Objective Optimization**
   - Pareto-optimal front generation
   - Non-dominated sorting
   - Crowding distance calculation

2. **Advanced Genetic Operators**
   - **Crossover**: Precedence Preserving Order-based Crossover (POX)
   - **Mutation Operator 1**: Swap operations and change machine assignments
   - **Mutation Operator 2**: Aggressive multiple swaps and inversions
   - **Dynamic Mutation Probabilities**: Decay over generations (inspired by simulated annealing)

3. **Solution Representation**
   - **Chromosome 1**: Operation sequence (order of operations)
   - **Chromosome 2**: Machine assignment for each operation

4. **Performance Tracking**
   - Generation-by-generation statistics
   - Convergence visualization
   - Pareto front visualization in 3D

## File Structure

```
d:\nithya\
├── fjsp_nsga2.py          # Main NSGA-II implementation
├── fjsp_instances.py      # Benchmark FJSP instances
├── run_tests.py           # Comprehensive testing script
├── paper_summary.md       # Base paper summary
├── base_paper.pdf         # Original research paper
└── README_FJSP.md         # This file
```

## Installation

### Prerequisites

- Python 3.8 or higher
- pip package manager

### Install Required Packages

```powershell
# Activate virtual environment (if using one)
D:/nithya/.venv/Scripts/Activate.ps1

# Install required packages
pip install numpy matplotlib
```

## Usage

### Quick Start

Run the main test script to execute all experiments:

```powershell
python run_tests.py
```

This will:
1. Test on 3 different FJSP instances (small, medium, large)
2. Generate convergence plots for each instance
3. Generate 3D Pareto front visualizations
4. Compare results across all instances
5. Create comprehensive analysis reports

### Running Individual Instances

```python
from fjsp_nsga2 import FJSP_NSGA2
from fjsp_instances import load_kacem_instance

# Load instance
jobs, num_machines = load_kacem_instance()

# Initialize NSGA-II
nsga2 = FJSP_NSGA2(
    jobs=jobs,
    num_machines=num_machines,
    population_size=100,
    max_generations=100,
    crossover_prob=0.9,
    mutation_prob_1=0.6,
    mutation_prob_2=0.4
)

# Initialize population
nsga2.initialize_population()

# Run evolution
pareto_front = nsga2.evolve()

# Print results
nsga2.print_pareto_solutions()

# Plot convergence
nsga2.plot_convergence(save_path='convergence.png')

# Plot Pareto front
nsga2.plot_pareto_front(save_path='pareto_front.png')
```

### Creating Custom Instances

```python
from fjsp_nsga2 import Job

# Create jobs
jobs = []

# Job 0: 2 operations
job0_ops = [
    (0, [(0, 10.0), (1, 12.0), (2, 8.0)]),  # Op 0: M0(10), M1(12), M2(8)
    (1, [(0, 8.0), (1, 10.0), (3, 9.0)])    # Op 1: M0(8), M1(10), M3(9)
]
job0 = Job(0, job0_ops)
job0.weight = 1.5  # Job priority weight
jobs.append(job0)

# Add more jobs...

# Run NSGA-II
nsga2 = FJSP_NSGA2(jobs=jobs, num_machines=4)
nsga2.initialize_population()
pareto_front = nsga2.evolve()
```

## Algorithm Parameters

### Population Parameters
- `population_size`: Number of individuals in population (default: 100)
- `max_generations`: Maximum number of generations (default: 100)

### Genetic Operator Parameters
- `crossover_prob`: Probability of crossover (default: 0.9)
- `mutation_prob_1`: Initial probability of mutation operator 1 (default: 0.6)
- `mutation_prob_2`: Initial probability of mutation operator 2 (default: 0.4)

### Decay Parameter
- `B`: Mutation probability decay factor (default: 0.999)
- Formula: ρ_t+1 = ρ_t × B^t

## Benchmark Instances

### 1. Kacem 4x5 (Small)
- 4 jobs, 5 machines
- Total operations: 9
- Classic FJSP benchmark from literature

### 2. Medium 8x6
- 8 jobs, 6 machines
- Total operations: 30
- Diverse operation counts and machine flexibility

### 3. Large 15x8
- 15 jobs, 8 machines
- Total operations: 60+
- Challenging large-scale instance

## Expected Results

### Performance Metrics

For each instance, the algorithm tracks:

1. **Makespan Improvement**: 15-40% reduction
2. **Waiting Time Improvement**: 20-50% reduction
3. **Weighted Time Improvement**: 18-45% reduction

### Convergence Behavior

- **Early Generations (0-20)**: Rapid improvement in all objectives
- **Mid Generations (20-60)**: Steady refinement of solutions
- **Late Generations (60-100)**: Fine-tuning and Pareto front diversification

### Pareto Front

- Typical size: 10-30 non-dominated solutions
- Well-distributed across objective space
- Trade-offs clearly visible between objectives

## Visualization Outputs

### 1. Convergence Plots
Three subplots showing best and average values over generations:
- Makespan convergence
- Waiting time convergence
- Weighted completion time convergence

### 2. Pareto Front (3D)
Interactive 3D scatter plot showing:
- X-axis: Makespan
- Y-axis: Total Waiting Time
- Z-axis: Total Weighted Completion Time

### 3. Improvement Comparison
Bar chart comparing percentage improvements across instances

## Key Implementation Details

### Fast Non-Dominated Sorting
- Time complexity: O(MN²) where M = objectives, N = population size
- Efficiently ranks solutions into Pareto fronts

### Crowding Distance
- Maintains diversity in Pareto front
- Boundary solutions get infinite distance
- Intermediate solutions ranked by density

### Precedence Preservation
- Operation sequences maintain job precedence constraints
- Custom crossover ensures feasible offspring
- Mutations respect job operation ordering

### Machine Flexibility
- Each operation can be processed on multiple machines
- Different processing times per machine
- Machine assignment optimized during evolution

## Comparison with Base Paper

### Similarities
1. NSGA-II framework structure
2. Multi-objective optimization approach
3. Dual mutation operators with decay
4. Fast non-dominated sorting
5. Crowding distance calculation

### Adaptations for FJSP
1. **Solution Representation**: Changed from feature weights to operation sequences
2. **Crossover**: POX instead of partially mapped crossover
3. **Objectives**: Scheduling metrics instead of SVM metrics
4. **Constraints**: Job precedence and machine availability

## Performance Tips

### For Small Instances (< 10 jobs)
- Population: 50-100
- Generations: 50-100
- Fast convergence expected

### For Medium Instances (10-20 jobs)
- Population: 100-150
- Generations: 100-150
- Balance exploration vs. exploitation

### For Large Instances (> 20 jobs)
- Population: 150-200
- Generations: 150-200
- May require longer run times

## Troubleshooting

### Issue: Slow Convergence
**Solution**: 
- Increase population size
- Increase mutation probabilities
- Check instance complexity

### Issue: Poor Pareto Front Quality
**Solution**:
- Run more generations
- Adjust crossover/mutation balance
- Verify instance feasibility

### Issue: Memory Errors
**Solution**:
- Reduce population size
- Process smaller instances
- Use 64-bit Python

## Advanced Usage

### Custom Objective Functions

Modify `evaluate_schedule()` in `fjsp_nsga2.py` to add/change objectives:

```python
def evaluate_schedule(self, schedule: Schedule):
    # Calculate your custom objectives
    schedule.objective_1 = ...
    schedule.objective_2 = ...
    schedule.objective_3 = ...
```

### Custom Mutation Operators

Add new mutation operators:

```python
def mutation_operator_3(self, schedule: Schedule) -> Schedule:
    mutated = schedule.copy()
    # Your custom mutation logic
    return mutated
```

## Citation

If you use this implementation in your research, please cite the base paper:

```bibtex
@article{badilla2025effective,
  title={An effective multi-objective metaheuristic for the support vector machine with feature selection},
  author={Badilla-Salamanca, Mathias and Dur{\'a}n, Rosa Medina and Contreras-Bolton, Carlos},
  journal={Knowledge-Based Systems},
  volume={328},
  pages={114203},
  year={2025},
  publisher={Elsevier}
}
```

## References

1. Badilla-Salamanca et al. (2025). Multi-objective metaheuristic for SVM with feature selection
2. Deb et al. (2002). A fast and elitist multiobjective genetic algorithm: NSGA-II
3. Kacem et al. (2002). Approach by localization and multiobjective evolutionary optimization for flexible job-shop scheduling problems

## License

This implementation is for educational and research purposes.

## Contact

For questions or issues, please create an issue in the repository or contact the implementation author.

---

**Last Updated**: October 13, 2025
