import numpy as np
import scipy
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

    def allocate(self, unassigned_tasks, available_resources, resource_pool, trd, occupations, fairness, task_costs):
        pass

    def prune_invalid_assignments(self, theoretical_assignments, available_resources, resource_pool, unassigned_tasks):
        assignments = []
        for task, resource in theoretical_assignments:
            if resource in available_resources and task in unassigned_tasks and \
                resource in resource_pool[task.task_type]:
                assignments.append((task, resource))
                available_resources.remove(resource)
                unassigned_tasks.remove(task)
        return assignments

    def prune_trd(self, trd, tasks, resources):
        pruned_trd = dict()
        for task, resource in itertools.product(tasks, resources):
            if (task, resource) in trd:
                pruned_trd[(task, resource)] = trd[(task, resource)]
        return pruned_trd


class RandomPolicy(Policy):
    def allocate(self, unassigned_tasks, available_resources, resource_pool, trd, occupations, fairness):
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
        return assignments

class FastestTaskFirst(Policy):
    def allocate(self, unassigned_tasks, available_resources, resource_pool, trd, occupations, fairness):
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
    def allocate(self, unassigned_tasks, available_resources, resource_pool, trd, occupations, fairness):
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

class HungarianPolicy(Policy):
    def allocate(self, unassigned_tasks, available_resources, resource_pool, trd, occupations, fairness):
        #trd = self.prune_trd(trd, resource_pool)
        task_data, task_encoding, resource_encoding = self.get_task_data_from_trd(trd)
        swaped_tasks_dict = {v : k for k, v in task_encoding.items()}
        swaped_resources_dict = {v : k for k, v in resource_encoding.items()}

        task_np = np.zeros((len(swaped_tasks_dict), len(swaped_resources_dict)))
        for x, y, v in task_data:
            task_np[x,y] = v

        task_ind, resource_ind = scipy.optimize.linear_sum_assignment(task_np)
        selected = []
        for task_i, resource_i in zip(task_ind, resource_ind):
            selected.append((swaped_tasks_dict[task_i], swaped_resources_dict[resource_i]))

        return self.prune_invalid_assignments(selected, available_resources, resource_pool, unassigned_tasks)

class HungarianMultiObjectivePolicy(Policy):
    def __init__(self, delta):
        self.alpha = 1     # time
        self.beta  = 0     # occupation
        self.gamma = 0     # fairness
        self.delta = delta

        self.num_postponed = 0
        self.num_allocated = 0

    def allocate(self, unassigned_tasks, available_resources, resource_pool, trd, occupations, fairness, task_costs):
        trd = self.prune_trd(trd, unassigned_tasks, available_resources)
        task_data, task_encoding, resource_encoding = self.get_task_data_from_trd(trd)

        task_costs = self.factor_task_costs(task_costs)
        swaped_tasks_dict = {v : k for k, v in task_encoding.items()}
        swaped_resources_dict = {v : k for k, v in resource_encoding.items()}

        # tasks x (resources + dummy resources)
        task_np = np.full((len(swaped_tasks_dict), len(swaped_resources_dict)+len(swaped_tasks_dict)),
                          np.inf,
                          dtype=np.double)

        # replace matrix with normal resource costs
        for x, y, v in task_data:
            # check if assignment is valid
            resource = swaped_resources_dict[y]
            task = swaped_tasks_dict[x]
            if resource not in resource_pool[task.task_type]:
                continue
            cost_1 = self.alpha*v
            cost_2 = self.beta*occupations[resource] if resource in occupations else 0
            cost_3 = self.gamma*fairness[resource] if resource in fairness else 0
            cost = cost_1+cost_2+cost_3
            task_np[x,y] = cost

        for y in range(len(swaped_resources_dict), len(swaped_resources_dict)+len(swaped_tasks_dict)):
            x = y - len(swaped_resources_dict)
            task_id = swaped_tasks_dict[x]
            task_np[x, y] = self.delta * task_costs[task_id]
        
        #print(task_np)
        task_ind, resource_ind = scipy.optimize.linear_sum_assignment(task_np)
        selected = []
        for task_i, resource_i in zip(task_ind, resource_ind):
            if resource_i in swaped_resources_dict:
                selected.append((swaped_tasks_dict[task_i], swaped_resources_dict[resource_i]))
            else:
                # selected dummy resource
                self.num_postponed += 1
        
        self.num_allocated += len(selected)
        return selected


class GreedyParallelMachinesSchedulingPolicy(Policy):
    def greedy_policy(self, task_data):
        max_task = max([task[0] for task in task_data])
        max_resources = max([task[1] for task in task_data])

        resource_times = {key: 0 for key in range(max_resources+1)}
        schedule = collections.defaultdict(list)

        for i in range(max_task+1):
            feasible_resources = list(filter(lambda t : t is not None, [task if task[0]==i else None for task in task_data]))
            min_resource_time, min_resource = math.inf, None
            for j, feasible_resource in enumerate(feasible_resources):
                rt = resource_times[feasible_resource[1]] + feasible_resource[2]
                if rt < min_resource_time:
                    #allocate task to resource
                    min_resource_time, min_resource = rt, feasible_resource[1]
                    schedule[min_resource] += [(resource_times[min_resource], feasible_resource[0])]

            #print(i, min_resource, min_resource_time)
            resource_times[min_resource] = min_resource_time

        greedy_resource_time = max(resource_times.values())
        return schedule

    
    def allocate(self, unassigned_tasks, available_resources, resource_pool, trd, occupations, fairness):
        task_data, task_encoding, resource_encoding = self.get_task_data_from_trd(trd)
        schedule = self.greedy_policy(task_data)
        swaped_tasks_dict = {v : k for k, v in task_encoding.items()}
        swaped_resources_dict = {v : k for k, v in resource_encoding.items()}

        selected = []
        for resource, tasks in schedule.items():
            task = swaped_tasks_dict[tasks[0][0]]
            resource = swaped_resources_dict[resource]
            selected.append([task, resource])

        return self.prune_invalid_assignments(selected, available_resources, resource_pool, unassigned_tasks)