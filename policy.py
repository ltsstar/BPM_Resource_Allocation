import numpy as np
import scipy
import math
import collections
import random

class Policy:
    def allocate(self, trd):
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


class RandomPolicy(Policy):
    def allocate(self, unassigned_tasks, available_resources, resource_pool, trd):
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
    

class HungarianPolicy(Policy):
    def allocate(self, unassigned_tasks, available_resources, resource_pool, trd):
        i, j, k = (0, 0, 0)
        task_indices, resource_indices = {}, {}
        index_taskresources = {}
        for task, resources in trd.items():
            task_indices[task] = i
            for resource, duration in resources.items():
                if resource not in resource_indices:
                    resource_indices[resource] = j
                    j += 1
                index_taskresources[(resource_indices[resource],i)] = (task, resource)
            i += 1

        mat = np.full((j,i), np.inf, dtype=np.dtype("float32"))
        for task, resources in trd.items():
            for resource, duration in resources.items():
                mat[resource_indices[resource]][task_indices[task]] = duration

        row_ind, col_ind = scipy.optimize.linear_sum_assignment(mat)
        selected = []
        for row, col in zip(row_ind, col_ind):
            if mat[row][col] == np.inf:
                if mat[row].argmin() != np.inf:
                    tr = index_taskresources[(row, mat[row].argmin())]
                else:
                    continue
            else:
                tr = index_taskresources[(row, col)]
            selected.append(tr)
        
        return self.prune_invalid_assignments(selected, available_resources, resource_pool, unassigned_tasks)
    

class GreedyParallelMachinesSchedulingPolicy(Policy):
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

    
    def allocate(self, unassigned_tasks, available_resources, resource_pool, trd):
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