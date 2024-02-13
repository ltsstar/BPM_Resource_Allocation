import numpy as np
import scipy

from policy import Policy

class HungarianPolicy(Policy):
    def allocate(self, unassigned_tasks, available_resources, resource_pool, trd,
                 occupations, fairness, task_costs, working_resources, current_time):
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
    def __init__(self, alpha, beta, gamma, delta):
        self.alpha = alpha     # time
        self.beta  = beta      # occupation
        self.gamma = gamma     # fairness
        self.delta = delta     # non-allocation cost factor

        self.num_postponed = 0
        self.num_allocated = 0

    def allocate(self, unassigned_tasks, available_resources, resource_pool, trd,
                 occupations, fairness, task_costs, working_resources, current_time):
        relevant_resources = set(available_resources) | set(working_resources)
        trd = self.prune_trd(trd, unassigned_tasks, relevant_resources)
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
            if resource in working_resources:
                start_time = max(0, working_resources[resource][0] - current_time + working_resources[resource][1])
            else:
                start_time = 0
            cost_1 = self.alpha*(v + start_time)
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
        return self.prune_invalid_assignments(selected, available_resources, resource_pool, unassigned_tasks)
        #return selected
