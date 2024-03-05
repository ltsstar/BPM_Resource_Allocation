import math
import time
import collections
from ortools.sat.python import cp_model
import logging
import random

from policy import Policy, GreedyParallelMachinesSchedulingPolicy

from ilp_policy_2 import UnrelatedParallelMachinesSchedulingPolicy2


class UnrelatedParallelMachinesSchedulingBatchPolicy2(Policy):
    def __init__(self, alpha, beta, gamma, delta, selection_strategy, batch_size):
        self.alpha = alpha     # time
        self.beta  = beta      # occupation
        self.gamma = gamma     # fairness
        self.delta = delta     # non-allocation cost factor
        self.selection_strategy = selection_strategy
        self.batch_size = batch_size

        self.num_postponed = 0
        self.num_allocated = 0
        self.logging = False
        self.optimal, self.feasible, self.no_solution = (0, 0, 0)

        self.policy = UnrelatedParallelMachinesSchedulingPolicy2(alpha, beta, gamma, delta, selection_strategy)

    def allocate(self, unassigned_tasks, available_resources, resource_pool, trd,
                 occupations, fairness, task_costs, working_resources, current_time):
        if len(unassigned_tasks) >= self.batch_size:
            return self.policy.allocate(unassigned_tasks, available_resources, resource_pool, trd,
                                        occupations, fairness, task_costs, working_resources, current_time
                                        )
        else:
            return []