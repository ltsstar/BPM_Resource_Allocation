import collections
from itertools import cycle

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
     