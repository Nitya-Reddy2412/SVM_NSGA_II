"""
FJSP NSGA-II Package
Multi-objective optimization for Flexible Job Shop Scheduling Problem
"""

from .fjsp_nsga2 import FJSP_NSGA2, Schedule, Job, Operation
from .fjsp_instances import (
    load_kacem_instance,
    load_medium_instance,
    load_large_instance,
    print_instance_info
)

__version__ = '1.0.0'
__author__ = 'Sanjay Cheekati'

__all__ = [
    'FJSP_NSGA2',
    'Schedule',
    'Job',
    'Operation',
    'load_kacem_instance',
    'load_medium_instance',
    'load_large_instance',
    'print_instance_info'
]
