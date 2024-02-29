import numpy as np
import scipy
import copy

from policy import Policy

class ParkPolicy(Policy):
    def __init__(self, next_task_distribution, predictor, task_type_occurrences):
        self.next_task_distribution = next_task_distribution
        self.predictor = predictor
        self.task_type_occurrences = task_type_occurrences

        self.num_postponed = 0
        self.num_allocated = 0

    def get_next_tasks(self, unassigned_tasks, task_costs):
        next_tasks = []
        next_task_penalties = dict()
        next_task_type_occurrences = copy.deepcopy(self.task_type_occurrences)
        get_next_task_type = lambda task : sorted(self.next_task_distribution[task.task_type],
                                                  key=lambda e : e[0],
                                                  reverse=True)[0][1]
        for unassigned_task in unassigned_tasks:
            next_task_type = get_next_task_type(unassigned_task)
            if next_task_type != None:
                next_task = copy.copy(unassigned_task)
                next_task_type_occurrences[unassigned_task.case_id][unassigned_task.task_type] += 1
                next_task.task_type = next_task_type
                next_tasks.append(next_task)
                next_task_penalties[next_task] = task_costs[unassigned_task]
        return next_tasks, next_task_penalties, next_task_type_occurrences

    def estimate_next_task_durations(self, next_tasks, resources, resource_pool):
        task_durations = dict()
        self.predictor.predict(next_tasks, resource_pool)

    def allocate(self, unassigned_tasks, available_resources, resource_pool, trd,
                 occupations, fairness, task_costs, working_resources, current_time):
        relevant_resources = set(available_resources) | set(working_resources.keys())

        next_tasks, next_task_penalties, next_task_type_occurrences = self.get_next_tasks(unassigned_tasks, task_costs)
        next_task_rd = self.predictor.model.predict_multiple_filtered(next_tasks, relevant_resources,
                                                                      resource_pool, next_task_type_occurrences)

        #next_task_data, next_task_encoding, next_resource_encoding = self.get_task_data_from_trd(next_task_rd)

        all_trd = trd | next_task_rd
        # filter trd resources by available + working resources
        relevant_trd = dict()
        for (task, resource), duration in trd.items():
            if resource in available_resources:
                relevant_trd[(task, resource)] = duration
            elif resource in working_resources:
                start_time = max(0, working_resources[resource][0] - current_time + working_resources[resource][1])
                relevant_trd[(task, resource)] = start_time + duration

        for (task, resource), duration in next_task_rd.items():
            if resource in available_resources:
                relevant_trd[(task, resource)] = duration + next_task_penalties[task]
            elif resource in working_resources:
                start_time = max(0, working_resources[resource][0] - current_time + working_resources[resource][1])
                relevant_trd[(task, resource)] = duration + next_task_penalties[task] + start_time

        relevant_task_data, relevant_task_encoding, relevant_resources_encoding = \
            self.get_task_data_from_trd(relevant_trd)


        swaped_tasks_dict = {v : k for k, v in relevant_task_encoding.items()}
        swaped_resources_dict = {v : k for k, v in relevant_resources_encoding.items()}

        # tasks x (resources + dummy resources)
        task_np = np.full((len(swaped_tasks_dict), len(swaped_resources_dict)),
                          np.inf,
                          dtype=np.double)
        
        for x, y, v in relevant_task_data:
            task_np[x,y] = v

        task_ind, resource_ind = scipy.optimize.linear_sum_assignment(task_np)
        selected = []
        for task_i, resource_i in zip(task_ind, resource_ind):
            selected.append((swaped_tasks_dict[task_i], swaped_resources_dict[resource_i]))

        selected_size = len(selected)
        task_assignment =  self.prune_invalid_assignments(selected, available_resources, resource_pool, unassigned_tasks)
        assignment_size = len(task_assignment)
        self.num_allocated += assignment_size
        self.num_postponed += selected_size - assignment_size

        return task_assignment
