"""
Benchmark FJSP Instances and Test Cases
Contains standard FJSP instances from literature for testing
"""

from .fjsp_nsga2 import Job


def load_kacem_instance() -> tuple:
    """
    Load Kacem 4x5 instance (4 jobs, 5 machines)
    A small but challenging FJSP instance from literature
    """
    # Job 0: 2 operations
    job0_ops = [
        (0, [(0, 1), (1, 2), (2, 3)]),  # Op 0 can be done on M0(1), M1(2), M2(3)
        (1, [(0, 2), (1, 1), (3, 4)])   # Op 1 can be done on M0(2), M1(1), M3(4)
    ]
    
    # Job 1: 2 operations
    job1_ops = [
        (0, [(0, 2), (2, 2), (4, 1)]),  # Op 0
        (1, [(1, 2), (2, 3), (3, 2)])   # Op 1
    ]
    
    # Job 2: 2 operations
    job2_ops = [
        (0, [(0, 3), (1, 2), (3, 2)]),  # Op 0
        (1, [(1, 1), (2, 2), (4, 3)])   # Op 1
    ]
    
    # Job 3: 3 operations
    job3_ops = [
        (0, [(0, 2), (2, 1), (4, 2)]),  # Op 0
        (1, [(1, 3), (3, 2)]),          # Op 1
        (2, [(2, 2), (3, 1), (4, 3)])   # Op 2
    ]
    
    jobs = [
        Job(0, job0_ops),
        Job(1, job1_ops),
        Job(2, job2_ops),
        Job(3, job3_ops)
    ]
    
    # Set weights
    for i, job in enumerate(jobs):
        job.weight = 1.0 + i * 0.2  # Increasing weights
    
    return jobs, 5  # 5 machines


def load_medium_instance() -> tuple:
    """
    Medium-sized instance: 8 jobs, 6 machines
    """
    jobs = []
    
    # Job 0: 4 operations
    jobs.append(Job(0, [
        (0, [(0, 10), (1, 12), (2, 8)]),
        (1, [(1, 8), (2, 10), (3, 9)]),
        (2, [(0, 7), (3, 11), (4, 9)]),
        (3, [(2, 6), (4, 8), (5, 10)])
    ]))
    
    # Job 1: 3 operations
    jobs.append(Job(1, [
        (0, [(0, 9), (2, 11), (4, 7)]),
        (1, [(1, 10), (3, 8), (5, 12)]),
        (2, [(0, 8), (2, 9), (4, 11)])
    ]))
    
    # Job 2: 5 operations
    jobs.append(Job(2, [
        (0, [(0, 12), (1, 10), (3, 14)]),
        (1, [(1, 9), (2, 11)]),
        (2, [(2, 8), (3, 10), (4, 7)]),
        (3, [(3, 11), (4, 9)]),
        (4, [(4, 8), (5, 10)])
    ]))
    
    # Job 3: 4 operations
    jobs.append(Job(3, [
        (0, [(0, 11), (2, 9), (5, 13)]),
        (1, [(1, 10), (3, 12)]),
        (2, [(2, 9), (4, 11), (5, 8)]),
        (3, [(0, 7), (3, 10)])
    ]))
    
    # Job 4: 3 operations
    jobs.append(Job(4, [
        (0, [(0, 8), (1, 10), (4, 9)]),
        (1, [(2, 11), (3, 9), (5, 12)]),
        (2, [(1, 7), (4, 10)])
    ]))
    
    # Job 5: 4 operations
    jobs.append(Job(5, [
        (0, [(0, 10), (2, 12), (3, 8)]),
        (1, [(1, 9), (4, 11)]),
        (2, [(2, 10), (3, 8), (5, 12)]),
        (3, [(0, 9), (4, 7)])
    ]))
    
    # Job 6: 3 operations
    jobs.append(Job(6, [
        (0, [(1, 11), (3, 9), (5, 10)]),
        (1, [(0, 8), (2, 12)]),
        (2, [(3, 10), (4, 9), (5, 11)])
    ]))
    
    # Job 7: 4 operations
    jobs.append(Job(7, [
        (0, [(0, 9), (1, 11), (2, 10)]),
        (1, [(2, 8), (3, 10), (4, 12)]),
        (2, [(1, 9), (4, 11)]),
        (3, [(3, 8), (5, 10)])
    ]))
    
    # Set weights (vary by job priority)
    weights = [1.5, 1.0, 2.0, 1.2, 1.8, 1.0, 1.3, 1.6]
    for i, job in enumerate(jobs):
        job.weight = weights[i]
    
    return jobs, 6  # 6 machines


def load_large_instance() -> tuple:
    """
    Large instance: 15 jobs, 8 machines
    """
    import random
    random.seed(123)  # For reproducibility
    
    jobs = []
    
    for job_id in range(15):
        num_operations = random.randint(3, 6)
        operations = []
        
        for op_id in range(num_operations):
            # Each operation can be processed on 2-4 machines
            num_capable = random.randint(2, min(4, 8))
            capable_machines = random.sample(range(8), num_capable)
            
            # Processing times between 5 and 20
            machine_times = [
                (m, random.uniform(5.0, 20.0))
                for m in capable_machines
            ]
            
            operations.append((op_id, machine_times))
        
        job = Job(job_id, operations)
        job.weight = random.uniform(0.8, 2.2)
        jobs.append(job)
    
    return jobs, 8  # 8 machines


def print_instance_info(jobs, num_machines, instance_name):
    """Print information about an FJSP instance"""
    print("\n" + "=" * 80)
    print(f"INSTANCE: {instance_name}")
    print("=" * 80)
    print(f"Number of jobs: {len(jobs)}")
    print(f"Number of machines: {num_machines}")
    
    total_ops = sum(len(job.operations) for job in jobs)
    print(f"Total operations: {total_ops}")
    
    print(f"\nJob Details:")
    print(f"{'Job':<8} {'Ops':<8} {'Weight':<10} {'Machine Flexibility'}")
    print("-" * 60)
    
    for job in jobs:
        avg_flexibility = sum(
            len(machine_times) for _, machine_times in job.operations
        ) / len(job.operations)
        print(f"{job.job_id:<8} {len(job.operations):<8} {job.weight:<10.2f} "
              f"{avg_flexibility:.1f} machines/op")
    
    print("=" * 80)


if __name__ == "__main__":
    print("FJSP Benchmark Instances")
    
    # Kacem instance
    jobs_kacem, machines_kacem = load_kacem_instance()
    print_instance_info(jobs_kacem, machines_kacem, "Kacem 4x5 (Small)")
    
    # Medium instance
    jobs_medium, machines_medium = load_medium_instance()
    print_instance_info(jobs_medium, machines_medium, "Medium 8x6")
    
    # Large instance
    jobs_large, machines_large = load_large_instance()
    print_instance_info(jobs_large, machines_large, "Large 15x8")
