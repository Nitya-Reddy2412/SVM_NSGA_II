"""
NSGA-II Implementation for Flexible Job Shop Scheduling Problem (FJSP)
Based on the paper: "An effective multi-objective metaheuristic for the support 
vector machine with feature selection"

Objectives:
1. Minimize Makespan (maximum completion time)
2. Minimize Total Waiting Time
3. Minimize Total Weighted Completion Time
"""

import numpy as np
import random
import copy
import matplotlib.pyplot as plt
from typing import List, Tuple, Dict
import time


class Job:
    """Represents a job with multiple operations"""
    def __init__(self, job_id: int, operations: List[Tuple[int, List[Tuple[int, float]]]]):
        """
        Args:
            job_id: Unique identifier for the job
            operations: List of (operation_id, [(machine_id, processing_time), ...])
        """
        self.job_id = job_id
        self.operations = operations  # List of operations
        self.weight = 1.0  # Weight for weighted completion time
        
    def __repr__(self):
        return f"Job{self.job_id}"


class Operation:
    """Represents an operation that can be processed on multiple machines"""
    def __init__(self, job_id: int, op_id: int, machine_times: Dict[int, float]):
        """
        Args:
            job_id: Parent job ID
            op_id: Operation ID within the job
            machine_times: {machine_id: processing_time}
        """
        self.job_id = job_id
        self.op_id = op_id
        self.machine_times = machine_times
        
    def get_machines(self) -> List[int]:
        """Return list of machines that can process this operation"""
        return list(self.machine_times.keys())
    
    def get_processing_time(self, machine_id: int) -> float:
        """Get processing time on a specific machine"""
        return self.machine_times.get(machine_id, float('inf'))
    
    def __repr__(self):
        return f"J{self.job_id}O{self.op_id}"


class Schedule:
    """Represents a complete schedule (individual/solution)"""
    def __init__(self, jobs: List[Job], num_machines: int):
        self.jobs = jobs
        self.num_machines = num_machines
        self.num_jobs = len(jobs)
        self.total_operations = sum(len(job.operations) for job in jobs)
        
        # Chromosome 1: Operation sequence (order of operations)
        self.operation_sequence = []
        
        # Chromosome 2: Machine assignment for each operation
        self.machine_assignment = []
        
        # Calculated objectives
        self.makespan = float('inf')
        self.total_waiting_time = float('inf')
        self.total_weighted_completion_time = float('inf')
        
        # Pareto ranking
        self.rank = 0
        self.crowding_distance = 0.0
        self.dominated_count = 0
        self.dominated_solutions = []
        
    def copy(self):
        """Create a deep copy of the schedule"""
        new_schedule = Schedule(self.jobs, self.num_machines)
        new_schedule.operation_sequence = self.operation_sequence.copy()
        new_schedule.machine_assignment = self.machine_assignment.copy()
        new_schedule.makespan = self.makespan
        new_schedule.total_waiting_time = self.total_waiting_time
        new_schedule.total_weighted_completion_time = self.total_weighted_completion_time
        new_schedule.rank = self.rank
        new_schedule.crowding_distance = self.crowding_distance
        return new_schedule
    
    def dominates(self, other) -> bool:
        """Check if this solution dominates another"""
        better_in_one = False
        for obj1, obj2 in [(self.makespan, other.makespan),
                           (self.total_waiting_time, other.total_waiting_time),
                           (self.total_weighted_completion_time, other.total_weighted_completion_time)]:
            if obj1 > obj2:
                return False
            if obj1 < obj2:
                better_in_one = True
        return better_in_one
    
    def get_objectives(self) -> Tuple[float, float, float]:
        """Return all three objectives"""
        return (self.makespan, self.total_waiting_time, 
                self.total_weighted_completion_time)


class FJSP_NSGA2:
    """NSGA-II Algorithm for Flexible Job Shop Scheduling"""
    
    def __init__(self, jobs: List[Job], num_machines: int, 
                 population_size: int = 50,
                 max_generations: int = 100,
                 crossover_prob: float = 0.9,
                 mutation_prob_1: float = 0.5,
                 mutation_prob_2: float = 0.3):
        """
        Initialize NSGA-II for FJSP
        
        Args:
            jobs: List of Job objects
            num_machines: Number of available machines
            population_size: Size of population
            max_generations: Maximum number of generations
            crossover_prob: Probability of crossover
            mutation_prob_1: Initial probability of mutation operator 1
            mutation_prob_2: Initial probability of mutation operator 2
        """
        self.jobs = jobs
        self.num_machines = num_machines
        self.num_jobs = len(jobs)
        self.population_size = population_size
        self.max_generations = max_generations
        self.crossover_prob = crossover_prob
        self.mutation_prob_1 = mutation_prob_1
        self.mutation_prob_2 = mutation_prob_2
        
        # Build operation list
        self.operations = []
        for job in jobs:
            for op_idx, (op_id, machine_times) in enumerate(job.operations):
                machine_time_dict = {m: t for m, t in machine_times}
                self.operations.append(Operation(job.job_id, op_idx, machine_time_dict))
        
        self.total_operations = len(self.operations)
        
        # Population
        self.population = []
        self.offspring_population = []
        
        # Statistics for tracking progress
        self.generation_stats = {
            'best_makespan': [],
            'best_waiting_time': [],
            'best_weighted_time': [],
            'avg_makespan': [],
            'avg_waiting_time': [],
            'avg_weighted_time': []
        }
        
        # Decay factor for mutation probabilities (from paper)
        self.B = 0.999
        
    def initialize_population(self):
        """Generate initial population with random feasible solutions"""
        print("Initializing population...")
        self.population = []
        
        for _ in range(self.population_size):
            schedule = self._create_random_schedule()
            self.evaluate_schedule(schedule)
            self.population.append(schedule)
        
        print(f"Initial population created: {len(self.population)} individuals")
    
    def _create_random_schedule(self) -> Schedule:
        """Create a random feasible schedule"""
        schedule = Schedule(self.jobs, self.num_machines)
        
        # Chromosome 1: Random operation sequence (respecting job precedence)
        operation_sequence = []
        job_op_counters = [0] * self.num_jobs
        
        # Create a list of all operations with their job indices
        available_ops = []
        for job_idx, job in enumerate(self.jobs):
            if len(job.operations) > 0:
                available_ops.append(job_idx)
        
        while available_ops:
            # Randomly select a job
            job_idx = random.choice(available_ops)
            op_idx = job_op_counters[job_idx]
            
            # Add operation to sequence
            operation_sequence.append((job_idx, op_idx))
            
            # Update counter
            job_op_counters[job_idx] += 1
            
            # Remove job if all operations are scheduled
            if job_op_counters[job_idx] >= len(self.jobs[job_idx].operations):
                available_ops.remove(job_idx)
        
        schedule.operation_sequence = operation_sequence
        
        # Chromosome 2: Random machine assignment
        machine_assignment = []
        for job_idx, op_idx in operation_sequence:
            job = self.jobs[job_idx]
            _, machine_times = job.operations[op_idx]
            # Randomly select a machine from available machines
            available_machines = [m for m, _ in machine_times]
            selected_machine = random.choice(available_machines)
            machine_assignment.append(selected_machine)
        
        schedule.machine_assignment = machine_assignment
        
        return schedule
    
    def evaluate_schedule(self, schedule: Schedule):
        """
        Evaluate a schedule and calculate all three objectives
        - Makespan
        - Total Waiting Time
        - Total Weighted Completion Time
        """
        # Initialize timing structures
        job_completion_times = [0.0] * self.num_jobs
        job_operation_end_times = [[0.0] * len(job.operations) for job in self.jobs]
        machine_available_times = [0.0] * self.num_machines
        
        total_waiting_time = 0.0
        operation_start_times = []
        operation_end_times = []
        
        # Process each operation in sequence
        for idx, (job_idx, op_idx) in enumerate(schedule.operation_sequence):
            job = self.jobs[job_idx]
            machine_id = schedule.machine_assignment[idx]
            
            # Get processing time
            _, machine_times = job.operations[op_idx]
            processing_time = next(t for m, t in machine_times if m == machine_id)
            
            # Calculate start time (max of job precedence and machine availability)
            if op_idx == 0:
                # First operation of the job
                job_ready_time = 0.0
            else:
                # Wait for previous operation of the same job
                job_ready_time = job_operation_end_times[job_idx][op_idx - 1]
            
            machine_ready_time = machine_available_times[machine_id]
            start_time = max(job_ready_time, machine_ready_time)
            end_time = start_time + processing_time
            
            # Calculate waiting time
            waiting_time = start_time - job_ready_time
            total_waiting_time += waiting_time
            
            # Update times
            job_operation_end_times[job_idx][op_idx] = end_time
            machine_available_times[machine_id] = end_time
            
            operation_start_times.append(start_time)
            operation_end_times.append(end_time)
            
            # Update job completion time
            if op_idx == len(job.operations) - 1:
                job_completion_times[job_idx] = end_time
        
        # Calculate objectives
        makespan = max(machine_available_times)
        
        # Total weighted completion time
        total_weighted_completion_time = sum(
            job.weight * job_completion_times[job.job_id]
            for job in self.jobs
        )
        
        # Update schedule objectives
        schedule.makespan = makespan
        schedule.total_waiting_time = total_waiting_time
        schedule.total_weighted_completion_time = total_weighted_completion_time
    
    def fast_non_dominated_sort(self, population: List[Schedule]) -> List[List[Schedule]]:
        """
        Fast non-dominated sorting algorithm from NSGA-II
        Returns fronts as list of lists
        """
        fronts = [[]]
        
        for p in population:
            p.dominated_solutions = []
            p.dominated_count = 0
            
            for q in population:
                if p.dominates(q):
                    p.dominated_solutions.append(q)
                elif q.dominates(p):
                    p.dominated_count += 1
            
            if p.dominated_count == 0:
                p.rank = 0
                fronts[0].append(p)
        
        i = 0
        while fronts[i]:
            next_front = []
            for p in fronts[i]:
                for q in p.dominated_solutions:
                    q.dominated_count -= 1
                    if q.dominated_count == 0:
                        q.rank = i + 1
                        next_front.append(q)
            i += 1
            fronts.append(next_front)
        
        return fronts[:-1]  # Remove empty last front
    
    def calculate_crowding_distance(self, front: List[Schedule]):
        """Calculate crowding distance for solutions in a front"""
        if len(front) == 0:
            return
        
        # Initialize crowding distance
        for solution in front:
            solution.crowding_distance = 0.0
        
        # For each objective
        objectives = ['makespan', 'total_waiting_time', 'total_weighted_completion_time']
        
        for obj in objectives:
            # Sort by objective
            front.sort(key=lambda x: getattr(x, obj))
            
            # Set boundary points to infinity
            front[0].crowding_distance = float('inf')
            front[-1].crowding_distance = float('inf')
            
            # Calculate crowding distance for intermediate solutions
            obj_min = getattr(front[0], obj)
            obj_max = getattr(front[-1], obj)
            
            if obj_max - obj_min > 0:
                for i in range(1, len(front) - 1):
                    distance = (getattr(front[i + 1], obj) - 
                               getattr(front[i - 1], obj)) / (obj_max - obj_min)
                    front[i].crowding_distance += distance
    
    def tournament_selection(self, population: List[Schedule]) -> Schedule:
        """Tournament selection based on rank and crowding distance"""
        p1 = random.choice(population)
        p2 = random.choice(population)
        
        # Compare based on rank first
        if p1.rank < p2.rank:
            return p1
        elif p2.rank < p1.rank:
            return p2
        else:
            # Same rank, compare crowding distance
            if p1.crowding_distance > p2.crowding_distance:
                return p1
            else:
                return p2
    
    def crossover_operation_sequence(self, parent1: Schedule, parent2: Schedule) -> Schedule:
        """
        Precedence Preserving Order-based Crossover (POX)
        Maintains job precedence constraints
        """
        offspring = Schedule(self.jobs, self.num_machines)
        
        # Select random job set
        num_jobs_to_select = random.randint(1, self.num_jobs)
        selected_jobs = set(random.sample(range(self.num_jobs), num_jobs_to_select))
        
        # Copy operations from parent1 for selected jobs
        offspring_sequence = []
        offspring_machines = []
        
        p1_remaining = []
        p1_machines_remaining = []
        
        for idx, (job_idx, op_idx) in enumerate(parent1.operation_sequence):
            if job_idx in selected_jobs:
                offspring_sequence.append((job_idx, op_idx))
                offspring_machines.append(parent1.machine_assignment[idx])
            else:
                p1_remaining.append((job_idx, op_idx))
                p1_machines_remaining.append(parent1.machine_assignment[idx])
        
        # Fill remaining positions from parent2
        for idx, (job_idx, op_idx) in enumerate(parent2.operation_sequence):
            if (job_idx, op_idx) in p1_remaining:
                offspring_sequence.append((job_idx, op_idx))
                # Use machine from parent2
                offspring_machines.append(parent2.machine_assignment[idx])
        
        offspring.operation_sequence = offspring_sequence
        offspring.machine_assignment = offspring_machines
        
        return offspring
    
    def mutation_operator_1(self, schedule: Schedule) -> Schedule:
        """
        Mutation Operator 1: Swap operations and change machine assignments
        Similar to the paper's mutation that modifies values
        """
        mutated = schedule.copy()
        
        # With 33% probability each:
        # 1. Swap two operations (maintaining precedence)
        if random.random() < 0.33:
            # Find two operations from different jobs
            if len(mutated.operation_sequence) > 1:
                for _ in range(10):  # Try up to 10 times
                    idx1 = random.randint(0, len(mutated.operation_sequence) - 1)
                    idx2 = random.randint(0, len(mutated.operation_sequence) - 1)
                    
                    job1, op1 = mutated.operation_sequence[idx1]
                    job2, op2 = mutated.operation_sequence[idx2]
                    
                    # Check if swap maintains precedence
                    if job1 != job2:
                        # Swap operations
                        mutated.operation_sequence[idx1], mutated.operation_sequence[idx2] = \
                            mutated.operation_sequence[idx2], mutated.operation_sequence[idx1]
                        mutated.machine_assignment[idx1], mutated.machine_assignment[idx2] = \
                            mutated.machine_assignment[idx2], mutated.machine_assignment[idx1]
                        break
        
        # 2. Change machine assignment for a random operation
        if random.random() < 0.33:
            idx = random.randint(0, len(mutated.operation_sequence) - 1)
            job_idx, op_idx = mutated.operation_sequence[idx]
            job = self.jobs[job_idx]
            _, machine_times = job.operations[op_idx]
            available_machines = [m for m, _ in machine_times]
            
            if len(available_machines) > 1:
                current_machine = mutated.machine_assignment[idx]
                available_machines.remove(current_machine)
                mutated.machine_assignment[idx] = random.choice(available_machines)
        
        # 3. Shift a subsequence (change operation order)
        if random.random() < 0.33:
            if len(mutated.operation_sequence) > 3:
                start_idx = random.randint(0, len(mutated.operation_sequence) - 3)
                end_idx = random.randint(start_idx + 1, len(mutated.operation_sequence) - 1)
                insert_idx = random.randint(0, len(mutated.operation_sequence) - 1)
                
                if insert_idx not in range(start_idx, end_idx + 1):
                    subsequence = mutated.operation_sequence[start_idx:end_idx + 1]
                    machines_sub = mutated.machine_assignment[start_idx:end_idx + 1]
                    
                    del mutated.operation_sequence[start_idx:end_idx + 1]
                    del mutated.machine_assignment[start_idx:end_idx + 1]
                    
                    if insert_idx > start_idx:
                        insert_idx -= len(subsequence)
                    
                    mutated.operation_sequence[insert_idx:insert_idx] = subsequence
                    mutated.machine_assignment[insert_idx:insert_idx] = machines_sub
        
        return mutated
    
    def mutation_operator_2(self, schedule: Schedule) -> Schedule:
        """
        Mutation Operator 2: More aggressive - multiple swaps or inversions
        """
        mutated = schedule.copy()
        
        # Random number of operations to affect (1-20% of operations)
        num_mutations = random.randint(1, max(1, len(mutated.operation_sequence) // 5))
        
        for _ in range(num_mutations):
            mutation_type = random.choice(['swap', 'reverse', 'machine'])
            
            if mutation_type == 'swap' and len(mutated.operation_sequence) > 1:
                # Swap two random operations
                idx1 = random.randint(0, len(mutated.operation_sequence) - 1)
                idx2 = random.randint(0, len(mutated.operation_sequence) - 1)
                
                job1, _ = mutated.operation_sequence[idx1]
                job2, _ = mutated.operation_sequence[idx2]
                
                if job1 != job2:
                    mutated.operation_sequence[idx1], mutated.operation_sequence[idx2] = \
                        mutated.operation_sequence[idx2], mutated.operation_sequence[idx1]
                    mutated.machine_assignment[idx1], mutated.machine_assignment[idx2] = \
                        mutated.machine_assignment[idx2], mutated.machine_assignment[idx1]
            
            elif mutation_type == 'reverse' and len(mutated.operation_sequence) > 2:
                # Reverse a subsequence
                start = random.randint(0, len(mutated.operation_sequence) - 2)
                end = random.randint(start + 1, len(mutated.operation_sequence))
                
                mutated.operation_sequence[start:end] = reversed(mutated.operation_sequence[start:end])
                mutated.machine_assignment[start:end] = reversed(mutated.machine_assignment[start:end])
            
            elif mutation_type == 'machine':
                # Change machine for random operation
                idx = random.randint(0, len(mutated.operation_sequence) - 1)
                job_idx, op_idx = mutated.operation_sequence[idx]
                job = self.jobs[job_idx]
                _, machine_times = job.operations[op_idx]
                available_machines = [m for m, _ in machine_times]
                
                if len(available_machines) > 1:
                    mutated.machine_assignment[idx] = random.choice(available_machines)
        
        return mutated
    
    def evolve(self):
        """Main evolution loop - NSGA-II algorithm"""
        print("\nStarting NSGA-II evolution...")
        print(f"Population size: {self.population_size}")
        print(f"Max generations: {self.max_generations}")
        print("-" * 80)
        
        start_time = time.time()
        
        for generation in range(self.max_generations):
            # Create offspring population
            offspring = []
            
            while len(offspring) < self.population_size:
                # Selection
                parent1 = self.tournament_selection(self.population)
                parent2 = self.tournament_selection(self.population)
                
                # Crossover
                if random.random() < self.crossover_prob:
                    child = self.crossover_operation_sequence(parent1, parent2)
                else:
                    child = parent1.copy()
                
                # Mutation operator 1 (with decaying probability)
                if random.random() < self.mutation_prob_1:
                    child = self.mutation_operator_1(child)
                
                # Mutation operator 2 (with decaying probability)
                if random.random() < self.mutation_prob_2:
                    child = self.mutation_operator_2(child)
                
                # Evaluate offspring
                self.evaluate_schedule(child)
                offspring.append(child)
            
            # Combine parent and offspring populations
            combined_population = self.population + offspring
            
            # Fast non-dominated sorting
            fronts = self.fast_non_dominated_sort(combined_population)
            
            # Calculate crowding distance for each front
            for front in fronts:
                self.calculate_crowding_distance(front)
            
            # Select new population
            new_population = []
            for front in fronts:
                if len(new_population) + len(front) <= self.population_size:
                    new_population.extend(front)
                else:
                    # Sort by crowding distance and select best
                    front.sort(key=lambda x: x.crowding_distance, reverse=True)
                    remaining = self.population_size - len(new_population)
                    new_population.extend(front[:remaining])
                    break
            
            self.population = new_population
            
            # Update mutation probabilities (decay)
            self.mutation_prob_1 *= (self.B ** generation)
            self.mutation_prob_2 *= (self.B ** generation)
            
            # Track statistics
            self._update_statistics(generation)
            
            # Print progress every 10 generations
            if (generation + 1) % 10 == 0 or generation == 0:
                self._print_progress(generation, time.time() - start_time)
        
        print("\n" + "=" * 80)
        print("Evolution completed!")
        print(f"Total time: {time.time() - start_time:.2f} seconds")
        print("=" * 80)
        
        return self.get_pareto_front()
    
    def _update_statistics(self, generation: int):
        """Update statistics for the current generation"""
        front_0 = [ind for ind in self.population if ind.rank == 0]
        
        if front_0:
            makespans = [ind.makespan for ind in front_0]
            waiting_times = [ind.total_waiting_time for ind in front_0]
            weighted_times = [ind.total_weighted_completion_time for ind in front_0]
            
            self.generation_stats['best_makespan'].append(min(makespans))
            self.generation_stats['best_waiting_time'].append(min(waiting_times))
            self.generation_stats['best_weighted_time'].append(min(weighted_times))
        
        # Average of entire population
        self.generation_stats['avg_makespan'].append(
            np.mean([ind.makespan for ind in self.population]))
        self.generation_stats['avg_waiting_time'].append(
            np.mean([ind.total_waiting_time for ind in self.population]))
        self.generation_stats['avg_weighted_time'].append(
            np.mean([ind.total_weighted_completion_time for ind in self.population]))
    
    def _print_progress(self, generation: int, elapsed_time: float):
        """Print progress information"""
        front_0 = [ind for ind in self.population if ind.rank == 0]
        
        if front_0:
            best_makespan = min(ind.makespan for ind in front_0)
            best_waiting = min(ind.total_waiting_time for ind in front_0)
            best_weighted = min(ind.total_weighted_completion_time for ind in front_0)
            
            print(f"Generation {generation + 1:3d} | "
                  f"Pareto Front Size: {len(front_0):3d} | "
                  f"Best Makespan: {best_makespan:8.2f} | "
                  f"Best Waiting: {best_waiting:8.2f} | "
                  f"Best Weighted: {best_weighted:10.2f} | "
                  f"Time: {elapsed_time:6.2f}s")
    
    def get_pareto_front(self) -> List[Schedule]:
        """Return the Pareto optimal front (rank 0 solutions)"""
        return [ind for ind in self.population if ind.rank == 0]
    
    def plot_convergence(self, save_path: str = None):
        """Plot convergence of objectives over generations"""
        fig, axes = plt.subplots(3, 1, figsize=(12, 10))
        generations = range(len(self.generation_stats['best_makespan']))
        
        # Plot Makespan
        axes[0].plot(generations, self.generation_stats['best_makespan'], 
                     'b-', linewidth=2, label='Best in Pareto Front')
        axes[0].plot(generations, self.generation_stats['avg_makespan'], 
                     'r--', linewidth=1, label='Population Average')
        axes[0].set_xlabel('Generation')
        axes[0].set_ylabel('Makespan')
        axes[0].set_title('Makespan Convergence')
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)
        
        # Plot Total Waiting Time
        axes[1].plot(generations, self.generation_stats['best_waiting_time'], 
                     'b-', linewidth=2, label='Best in Pareto Front')
        axes[1].plot(generations, self.generation_stats['avg_waiting_time'], 
                     'r--', linewidth=1, label='Population Average')
        axes[1].set_xlabel('Generation')
        axes[1].set_ylabel('Total Waiting Time')
        axes[1].set_title('Waiting Time Convergence')
        axes[1].legend()
        axes[1].grid(True, alpha=0.3)
        
        # Plot Total Weighted Completion Time
        axes[2].plot(generations, self.generation_stats['best_weighted_time'], 
                     'b-', linewidth=2, label='Best in Pareto Front')
        axes[2].plot(generations, self.generation_stats['avg_weighted_time'], 
                     'r--', linewidth=1, label='Population Average')
        axes[2].set_xlabel('Generation')
        axes[2].set_ylabel('Total Weighted Completion Time')
        axes[2].set_title('Weighted Completion Time Convergence')
        axes[2].legend()
        axes[2].grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"\nConvergence plot saved to: {save_path}")
        
        plt.show()
    
    def plot_pareto_front(self, save_path: str = None):
        """Plot 3D Pareto front"""
        pareto_front = self.get_pareto_front()
        
        if not pareto_front:
            print("No Pareto front solutions found!")
            return
        
        makespans = [sol.makespan for sol in pareto_front]
        waiting_times = [sol.total_waiting_time for sol in pareto_front]
        weighted_times = [sol.total_weighted_completion_time for sol in pareto_front]
        
        fig = plt.figure(figsize=(12, 9))
        ax = fig.add_subplot(111, projection='3d')
        
        scatter = ax.scatter(makespans, waiting_times, weighted_times,
                           c=range(len(pareto_front)), cmap='viridis',
                           s=100, alpha=0.7, edgecolors='black', linewidth=0.5)
        
        ax.set_xlabel('Makespan', fontsize=12)
        ax.set_ylabel('Total Waiting Time', fontsize=12)
        ax.set_zlabel('Total Weighted Time', fontsize=12)
        ax.set_title(f'Pareto Front (3D) - {len(pareto_front)} Solutions', fontsize=14)
        
        plt.colorbar(scatter, ax=ax, label='Solution Index')
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Pareto front plot saved to: {save_path}")
        
        plt.show()
    
    def print_pareto_solutions(self, max_solutions: int = 10):
        """Print details of Pareto optimal solutions"""
        pareto_front = self.get_pareto_front()
        
        print("\n" + "=" * 80)
        print(f"PARETO OPTIMAL SOLUTIONS (Total: {len(pareto_front)})")
        print("=" * 80)
        
        # Sort by makespan
        pareto_front.sort(key=lambda x: x.makespan)
        
        print(f"\n{'#':<4} {'Makespan':<12} {'Waiting Time':<15} {'Weighted Time':<18} {'Crowding Dist':<15}")
        print("-" * 80)
        
        for idx, sol in enumerate(pareto_front[:max_solutions]):
            print(f"{idx+1:<4} {sol.makespan:<12.2f} {sol.total_waiting_time:<15.2f} "
                  f"{sol.total_weighted_completion_time:<18.2f} {sol.crowding_distance:<15.4f}")
        
        if len(pareto_front) > max_solutions:
            print(f"... and {len(pareto_front) - max_solutions} more solutions")
        
        print("=" * 80)


def create_sample_fjsp_instance(num_jobs: int = 5, num_machines: int = 4,
                                 max_operations_per_job: int = 4) -> List[Job]:
    """
    Create a sample FJSP instance for testing
    
    Args:
        num_jobs: Number of jobs
        num_machines: Number of machines
        max_operations_per_job: Maximum operations per job
        
    Returns:
        List of Job objects
    """
    jobs = []
    
    for job_id in range(num_jobs):
        num_operations = random.randint(2, max_operations_per_job)
        operations = []
        
        for op_id in range(num_operations):
            # Each operation can be processed on 1-3 machines (randomly)
            num_capable_machines = random.randint(1, min(3, num_machines))
            capable_machines = random.sample(range(num_machines), num_capable_machines)
            
            # Generate random processing times
            machine_times = [
                (machine_id, random.uniform(5.0, 20.0))
                for machine_id in capable_machines
            ]
            
            operations.append((op_id, machine_times))
        
        job = Job(job_id, operations)
        job.weight = random.uniform(0.5, 2.0)  # Random weight
        jobs.append(job)
    
    return jobs


if __name__ == "__main__":
    # Set random seed for reproducibility
    random.seed(42)
    np.random.seed(42)
    
    print("=" * 80)
    print("NSGA-II FOR FLEXIBLE JOB SHOP SCHEDULING PROBLEM (FJSP)")
    print("Based on: 'An effective multi-objective metaheuristic for SVM with FS'")
    print("=" * 80)
    
    # Create sample FJSP instance
    print("\nCreating sample FJSP instance...")
    num_jobs = 8
    num_machines = 5
    jobs = create_sample_fjsp_instance(num_jobs, num_machines, max_operations_per_job=5)
    
    print(f"Instance created: {num_jobs} jobs, {num_machines} machines")
    total_ops = sum(len(job.operations) for job in jobs)
    print(f"Total operations: {total_ops}")
    
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
    nsga2.print_pareto_solutions(max_solutions=15)
    
    # Plot convergence
    print("\nGenerating convergence plots...")
    nsga2.plot_convergence(save_path='d:\\nithya\\convergence_plot.png')
    
    # Plot Pareto front
    print("\nGenerating Pareto front plot...")
    nsga2.plot_pareto_front(save_path='d:\\nithya\\pareto_front.png')
    
    print("\n" + "=" * 80)
    print("EXECUTION COMPLETED SUCCESSFULLY!")
    print("=" * 80)
