import numpy as np
import math
import collections
import random
import itertools

class Policy:
    def get_task_data_from_trd(self, trd, factor=3600):
        resources_dict, task_dict = dict(), dict()
        task_data = []
        for ((task, resource), duration) in trd.items():
            if task not in task_dict:
                task_dict[task] = len(task_dict)
            if resource not in resources_dict:
                resources_dict[resource] = len(resources_dict)
            task_data.append(
                (task_dict[task], resources_dict[resource], int(duration*factor))
            )
        return (task_data, task_dict, resources_dict)
    
    def factor_task_costs(self, task_costs, factor=3600):
        for task in task_costs:
            task_costs[task] = int(task_costs[task] * factor)
        return task_costs

    def allocate(self, unassigned_tasks, available_resources, resource_pool, trd,
                 occupations, fairness, task_costs, working_resources, current_time):
        pass

    def prune_invalid_assignments(self, theoretical_assignments, available_resources, resource_pool, unassigned_tasks):
        ar = available_resources.copy()
        ut = unassigned_tasks.copy()
        assignments = []
        for task, resource in theoretical_assignments:
            if resource in ar and task in ut and \
                resource in resource_pool[task.task_type]:
                assignments.append((task, resource))
                ar.remove(resource)
                ut.remove(task)
        return assignments

    def prune_trd(self, trd, tasks, resources):
        pruned_trd = dict()
        for task, resource in itertools.product(tasks, resources):
            if (task, resource) in trd:
                pruned_trd[(task, resource)] = trd[(task, resource)]
        return pruned_trd


class RandomPolicy(Policy):
    def __init__(self):
        self.num_allocated = 0
        self.num_postponed = 0

    def allocate(self, unassigned_tasks, available_resources, resource_pool, trd,
                 occupations, fairness, task_costs, working_resources, current_time):
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

class FastestTaskFirst(Policy):
    def allocate(self, unassigned_tasks, available_resources, resource_pool, trd,
                 occupations, fairness, task_costs, working_resources, current_time):
        task_runtimes = collections.defaultdict(list)
        for ((task, resource), duration) in trd.items():
            if resource in resource_pool[task.task_type] and \
                resource in available_resources and \
                task in unassigned_tasks:
                task_runtimes[task].append((resource, duration))
        
        sorted_task_runtimes = collections.defaultdict(list)
        for task, task_runtime in task_runtimes.items():
            sorted_task_runtimes[task] = sorted(task_runtime, key=lambda x : x[1])

        assignments = []
        assigned_resources = []
        for task, sorted_task_runtime in sorted_task_runtimes.items():
            for resource, duration in sorted_task_runtime:
                if resource not in assigned_resources:
                    assignments.append((task, resource))
                    assigned_resources.append(resource)
                    break
        return assignments
        

class FastestResourceFirst(Policy):
    def allocate(self, unassigned_tasks, available_resources, resource_pool, trd,
                 occupations, fairness, task_costs, working_resources, current_time):
        resource_runtimes = collections.defaultdict(list)
        for ((task, resource), duration) in trd.items():
            if resource in resource_pool[task.task_type] and \
                resource in available_resources and \
                task in unassigned_tasks:
                resource_runtimes[resource].append((task, duration))
        
        sorted_resource_runtimes = collections.defaultdict(list)
        for resource, resource_runtime in resource_runtimes.items():
            sorted_resource_runtimes[resource] = sorted(resource_runtime, key=lambda x : x[1])

        assignments = []
        assigned_tasks = []
        for resource, sorted_resource_runtime in sorted_resource_runtimes.items():
            for task, duration in sorted_resource_runtime:
                if task not in assigned_tasks:
                    assignments.append((task, resource))
                    assigned_tasks.append(task)
                    break
        return assignments

class GreedyParallelMachinesSchedulingPolicy(Policy):
    def greedy_policy(self, task_data, machines_start):
        max_task = max([task[0] for task in task_data])
        max_resources = max([task[1] for task in task_data])

        resource_times = machines_start.copy()
        schedule = collections.defaultdict(list)

        for task in range(max_task+1):
            min_max_time = math.inf
            selected_resource = None
            
            feasible_resources = list(filter(lambda t : t is not None, [task if task[0]==i else None for task in task_data]))
            min_resource_time, min_resource = math.inf, None
            for j, feasible_resource in enumerate(feasible_resources):
                rt = resource_times[feasible_resource[1]] + feasible_resource[2]
                if rt < min_resource_time:
                    #pre allocate task to resource
                    min_resource_time, min_resource = rt, feasible_resource[1]
                    schedule[min_resource] += [(resource_times[min_resource], feasible_resource[0])]

            #print(i, min_resource, min_resource_time)
            resource_times[min_resource] = min_resource_time

        greedy_resource_time = max(resource_times.values())
        return schedule

    
    def allocate(self, unassigned_tasks, available_resources, resource_pool, trd,
                 occupations, fairness, task_costs, working_resources, current_time):
        task_data, task_encoding, resource_encoding = self.get_task_data_from_trd(trd)

        # get encoded machines start
        machines_start = {}
        for resource, resource_enc in resource_encoding.items():
            if resource in working_resources:
                start_time = max(0, working_resources[resource][0] - current_time + working_resources[resource][1])
                machines_start[resource_enc] = int(start_time * 3600)
            else:
                machines_start[resource_enc] = 0

        schedule = self.greedy_policy(task_data, machines_start)
        swaped_tasks_dict = {v : k for k, v in task_encoding.items()}
        swaped_resources_dict = {v : k for k, v in resource_encoding.items()}

        selected = []
        for resource, tasks in schedule.items():
            task = swaped_tasks_dict[tasks[0][0]]
            resource = swaped_resources_dict[resource]
            selected.append([task, resource])

        return self.prune_invalid_assignments(selected, available_resources, resource_pool, unassigned_tasks)