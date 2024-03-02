import numpy as np
import scipy
from ortools.graph.python import min_cost_flow
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

    def do_matching(self, relevant_task_data, num_tasks, num_resources, reduction = 0):
        start_nodes = [0 for i in range(num_resources)]
        end_nodes = [i+1 for i in range(num_resources)]
        capacities = [1 for i in range(num_resources)]
        costs = [0 for i in range(num_resources)]
        for task_ix, resource_ix, d in relevant_task_data:
            start_nodes += [resource_ix+1]
            end_nodes += [num_resources+1+task_ix]
            capacities += [1]
            costs += [d]

        start_nodes += [num_resources+1+task_ix for task_ix in range(num_tasks)]
        end_nodes += [num_tasks+num_resources+1 for task_ix in range(num_tasks)]
        capacities += [1 for i in range(num_tasks)]
        costs += [0 for i in range(num_tasks)]

        supplies = [min(num_tasks, num_resources) - reduction] +\
                    [0 for i in range(num_tasks + num_resources - 1)] +\
                    [-min(num_tasks, num_resources) + reduction]

        smcf = min_cost_flow.SimpleMinCostFlow()

        # Add each arc.
        for i in range(len(start_nodes)):
            smcf.add_arc_with_capacity_and_unit_cost(
                start_nodes[i], end_nodes[i], capacities[i], costs[i]
            )
        # Add node supplies.
        for i in range(len(supplies)):
            smcf.set_node_supply(i, supplies[i])

        status = smcf.solve()
        return smcf, status

    def allocate(self, unassigned_tasks, available_resources, resource_pool, trd,
                 occupations, fairness, task_costs, working_resources, current_time):
        relevant_resources = set(available_resources) | set(working_resources.keys())

        next_tasks, next_task_penalties, next_task_type_occurrences = self.get_next_tasks(unassigned_tasks, task_costs)
        next_task_rd = self.predictor.model.predict_multiple_filtered(next_tasks, relevant_resources,
                                                                      resource_pool, next_task_type_occurrences)

        #next_task_data, next_task_encoding, next_resource_encoding = self.get_task_data_from_trd(next_task_rd)

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

        num_resources = len(swaped_resources_dict)
        num_tasks = len(swaped_tasks_dict)

        smcf, status = self.do_matching(relevant_task_data, num_tasks, num_resources)
        # Sometimes there exists no matching for all resources
        # e.g. because all tasks can only be conduced by one resource
        # Therefore reduce number of supplies iteratively
        r = 1
        while status != smcf.OPTIMAL:
            smcf, status = self.do_matching(relevant_task_data, num_tasks, num_resources, r)
            r += 1
        selected = []

        for arc in range(smcf.num_arcs()):
            if smcf.tail(arc) != 0 and smcf.head(arc) != num_tasks+num_resources+1:
                if smcf.flow(arc) > 0:
                    selected.append((swaped_tasks_dict[smcf.head(arc) - num_resources - 1],
                                        swaped_resources_dict[smcf.tail(arc) - 1]))


        selected_size = len(selected)
        task_assignment =  self.prune_invalid_assignments(selected, available_resources, resource_pool, unassigned_tasks)
        assignment_size = len(task_assignment)
        self.num_allocated += selected_size
        self.num_postponed += selected_size - assignment_size

        return task_assignment
