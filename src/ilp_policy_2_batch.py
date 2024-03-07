import math
import time
import collections
from ortools.sat.python import cp_model
import logging
import random

from policy import Policy, GreedyParallelMachinesSchedulingPolicy

from ilp_policy_2 import UnrelatedParallelMachinesSchedulingPolicy2

"""
Implements k-batching from Zeng and Zhao
"""
class UnrelatedParallelMachinesSchedulingBatchPolicy2(Policy):
    def __init__(self, alpha, beta, gamma, delta, selection_strategy, batch_size):
        self.alpha = alpha     # time
        self.beta  = beta      # occupation
        self.gamma = gamma     # fairness
        self.delta = delta     # non-allocation cost factor
        self.selection_strategy = selection_strategy
        self.batch_size = batch_size

        self.resources_queues = collections.defaultdict(list)

        self.num_postponed = 0
        self.num_allocated = 0
        self.logging = False
        self.optimal, self.feasible, self.no_solution = (0, 0, 0)

        self.policy = UnrelatedParallelMachinesSchedulingPolicy2(alpha, beta, gamma, delta, selection_strategy)

    def get_real_unassigned_tasks(self, unassigned_tasks):
        real_unassigned_tasks = unassigned_tasks.copy()
        for resource, tasks in self.resources_queues.items():
            for task in tasks:
                if task in real_unassigned_tasks:
                    real_unassigned_tasks.remove(task)
        return real_unassigned_tasks

    def allocate(self, unassigned_tasks, available_resources, resource_pool, trd,
                 occupations, fairness, task_costs, working_resources, current_time):
        
        allocations = []
        allocated_resources = []
        # allocate according to resource queues
        for available_resource in available_resources:
            if len(self.resources_queues[available_resource]):
                next_task = self.resources_queues[available_resource].pop(0)
                allocations.append((next_task, available_resource))
                allocated_resources.append(available_resource)
                unassigned_tasks.remove(next_task)

        # remove unavailable resources' queues
        for resource in self.resources_queues.keys():
            if resource not in available_resources and resource not in working_resources:
                self.resources_queues[resource] = []

        # obtain tasks that need to be allocated to resource queues
        real_unassigned_tasks = self.get_real_unassigned_tasks(unassigned_tasks)

        # allocate tasks when batch size has been reached
        if len(real_unassigned_tasks) >= self.batch_size:
            schedule = self.policy.allocate_all(real_unassigned_tasks, available_resources, resource_pool, trd,
                                        occupations, fairness, task_costs, working_resources, current_time
                                        )
            for resource, tasks in schedule.items():
                if self.selection_strategy == 'fastest':
                    tasks_sorted = list(zip(*sorted(tasks, key=lambda t : t[1])))[0]
                self.resources_queues[resource] += tasks_sorted

        # if still available resources that now have gotten new tasks, allocate them
        for available_resource in available_resources:
            if available_resource not in allocated_resources:
                if len(self.resources_queues[available_resource]):
                    next_task = self.resources_queues[available_resource].pop(0)
                    allocations.append((next_task, available_resource))

        return allocations