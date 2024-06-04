import collections
from itertools import cycle
import random

from policy import Policy

class RoundRobinPolicy(Policy):
    def __init__(self):
        self.num_allocated = 0
        self.num_postponed = 0

        self.resource_enum = None
        self.max_iter = 0
        self.resources_queues = collections.defaultdict(list)
        
    def get_real_unassigned_tasks(self, unassigned_tasks):
        real_unassigned_tasks = unassigned_tasks.copy()
        for resource, tasks in self.resources_queues.items():
            for task in tasks:
                if task in real_unassigned_tasks:
                    real_unassigned_tasks.remove(task)
        return real_unassigned_tasks



    def allocate(self, unassigned_tasks, available_resources, resource_pool, trd,
                occupations, fairness, task_costs, working_resources, current_time):
        if not self.resource_enum:
            self.max_iter = len(set(sum(resource_pool.values(), [])))
            self.resource_enum = cycle(set(sum(resource_pool.values(), [])))

        relevant_resources = set(available_resources) | set(working_resources.keys())

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

        # allocate tasks
        for unassigned_task in real_unassigned_tasks:
            for i in range(self.max_iter):
                resource = next(self.resource_enum)
                if  resource in relevant_resources and \
                    resource in resource_pool[unassigned_task.task_type]:
                    self.resources_queues[resource].append(unassigned_task)
                    break

        # if still available resources that now have gotten new tasks, allocate them
        for available_resource in available_resources:
            if available_resource not in allocated_resources:
                if len(self.resources_queues[available_resource]):
                    next_task = self.resources_queues[available_resource].pop(0)
                    allocations.append((next_task, available_resource))

        return allocations


class RandomPolicy(Policy):
    def __init__(self):
        self.num_allocated = 0
        self.num_postponed = 0

    def allocate(self, unassigned_tasks, available_resources, resource_pool, trd=None,
                 occupations=None, fairness=None, task_costs=None, working_resources=None,
                 current_time=None):
        random.shuffle(unassigned_tasks)
        it_resources = list(available_resources)
        random.shuffle(it_resources)
        assignments = []
        # assign the first unassigned task to the first available resource, the second task to the second resource, etc.
        for task in unassigned_tasks:
            for resource in it_resources:
                if resource in resource_pool[task.task_type]:
                    it_resources.remove(resource)
                    assignments.append((task, resource))
                    break

        self.num_allocated += len(assignments)
        return assignments
    

class ShortestQueuePolicy(Policy):
    def __init__(self):
        self.num_allocated = 0
        self.num_postponed = 0

        self.resources_queues = collections.defaultdict(list)
        
    def get_real_unassigned_tasks(self, unassigned_tasks):
        real_unassigned_tasks = unassigned_tasks.copy()
        for resource, tasks in self.resources_queues.items():
            for task in tasks:
                if task in real_unassigned_tasks:
                    real_unassigned_tasks.remove(task)
        return real_unassigned_tasks

    def allocate(self, unassigned_tasks, available_resources, resource_pool, trd,
                 occupations, fairness, task_costs, working_resources, current_time):
        relevant_resources = set(available_resources) | set(working_resources.keys())

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

        # allocate tasks
        for unassigned_task in real_unassigned_tasks:
            # sort resources by queue length
            sorted_res = list(sorted(self.resources_queues.items(),
                                     key=lambda k: len(k[1])))
            # apply to assign the task to the first resource
            # that has the authorization
            for (res, queue) in sorted_res:
                if res in resource_pool[unassigned_task.task_type]:
                    self.resources_queues[res].append(unassigned_task)
                    break

        # if still available resources that now have gotten new tasks, allocate them
        for available_resource in available_resources:
            if available_resource not in allocated_resources:
                if len(self.resources_queues[available_resource]):
                    next_task = self.resources_queues[available_resource].pop(0)
                    allocations.append((next_task, available_resource))

        return allocations