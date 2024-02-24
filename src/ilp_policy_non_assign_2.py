import math
import time
import collections
from ortools.sat.python import cp_model
import logging
import random

from policy import Policy, GreedyParallelMachinesSchedulingPolicy
from hungarian_policy import HungarianMultiObjectivePolicy

from ilp_policy_non_assign import *

class UnrelatedMachinesSchedulingNonAssign2:
    def __init__(self, task_data, non_assign_cost, 
                 machines_start, delta, max_value=None):
        self.task_data = task_data
        self.non_assign_cost = non_assign_cost
        self.machines_start = machines_start
        self.delta = delta
        self.__define_model(max_value)
        self.__define_constraints()
        self.__define_objective()

    def __define_model(self, max_value, greedy_max=True):
        self.model = cp_model.CpModel()
        # Named tuple to store information about created variables.
        self.assignment = collections.namedtuple("assignment", "task machine assigned_var duration")
        # Creates job intervals and add to the corresponding machine lists.
        self.all_tasks = {}
        self.task_assigned_machines = collections.defaultdict(list)
        self.task_non_assign = {}
        self.machine_durations = {}
        self.non_assign_cost_variables = []
        self.assignments = collections.defaultdict(list)
        self.machines = collections.defaultdict(list)
        self.intervals = collections.defaultdict(list)
        self.goal_variables = collections.defaultdict(list)
        if max_value:
            self.horizon = 0, max_value
        elif greedy_max:
            self.horizon = 0, self.__get_greedy_max()
        else:
            self.horizon = self.__get_horizon()
        #print(self.horizon)
            
    def __get_greedy_max(self):
        task_resource_durations = collections.defaultdict(list)
        for task, resource, duration in self.task_data:
            task_resource_durations[task].append((resource, duration))

        resource_times = self.machines_start.copy()
        for task, resource_durations in task_resource_durations.items():
            min_max_time = math.inf
            selected_resource = None
            for resource, duration in resource_durations:
                if resource_times[resource] + duration < min_max_time:
                    selected_resource = resource
                    min_max_time = resource_times[resource] + duration
            resource_times[selected_resource] = min_max_time
        
        return max(resource_times.values())

    def __get_horizon(self):
        max_task = max([task[0] for task in self.task_data])
        #max_resource = max([task[1] for task in self.task_data])
        min_task_duration, max_task_duration = dict(), dict()
        for t in range(max_task+1):
            min_task_duration[t] = min([task[2] if task[0]==t else math.inf for task in self.task_data])
            #max_task_duration[t] = max([task[2] if task[0]==t else 0 for task in self.task_data])
        min_horizon = 0
        horizon = min_horizon, sum(min_task_duration.values())+1
        return horizon
    
    
    def __define_constraints(self):
        #Define decision variable
        for task, machine, processing_time in self.task_data:
            suffix = f"_{task}_{machine}"
            assigned_var = self.model.NewBoolVar('assigned' + suffix)
            self.task_assigned_machines[task].append(assigned_var)
            duration_var = self.model.NewIntVar(processing_time, processing_time, 'duration' + suffix)
            goal_variable = self.model.NewIntVar(0, processing_time, 'goal' + suffix)
            self.model.AddMultiplicationEquality(goal_variable, assigned_var, duration_var)
            self.goal_variables[task, machine] = goal_variable

            #print(interval_var, processing_time)
            self.machines[machine].append(goal_variable)

            assignment = self.assignment(
                task = task,
                machine = machine,
                assigned_var = assigned_var,
                duration = duration_var
            )
            self.assignments[machine].append(assignment)

        #Machines processing times
        for machine, machine_goal_variables in self.machines.items():
            machine_duration = self.model.NewIntVar(0, self.horizon[1], 'machine_duration_' + str(machine))
            self.model.Add(self.machines_start[machine] + cp_model.LinearExpr.Sum(machine_goal_variables) == machine_duration)
            self.machine_durations[machine] = machine_duration



        #Add non assign
        for task, task_data in self.task_assigned_machines.items():
            non_assign_variable = self.model.NewBoolVar('non_assigned_' + str(task))
            self.task_non_assign[task] = non_assign_variable
            non_assign_cost_variable = self.model.NewIntVar(0, self.non_assign_cost[task], 'non_assigned_cost_' + str(task))
            self.model.AddMultiplicationEquality(non_assign_cost_variable, [non_assign_variable, self.non_assign_cost[task]])
            self.non_assign_cost_variables.append(non_assign_cost_variable)
            #print(non_assign_cost_variable, self.non_assign_cost[task])


        #Constraint 2: All tasks must be assigned to exactly one machine
        #               or to -non assign-
        tasks_count = 1+ max(task[0] for task in self.task_data)
        all_tasks = range(tasks_count)
        for task in all_tasks:
            #sum of assignment variable of task must be one
            self.model.Add(cp_model.LinearExpr.Sum(self.task_assigned_machines[task]) +
                           self.task_non_assign[task] == 1)

    
    def __define_objective(self):
        # Makespan objective:
        self.duration_var = self.model.NewIntVar(self.horizon[0], self.horizon[1], "makespan")
        self.model.AddMaxEquality(
            self.duration_var,
            [machine_duration_var for machine, machine_duration_var in self.machine_durations.items()] +
            [max(self.machines_start.values())]
        )
        # Non Assign objective:
        self.non_assign_sum = self.model.NewIntVar(0, sum(self.non_assign_cost.values()), 'non_assign_sum')
        self.model.Add(cp_model.LinearExpr.Sum(self.non_assign_cost_variables) == self.non_assign_sum)

        # Deviation from makespan objective:
        self.makespan_deviations = []
        for machine, machine_duration_var in self.machine_durations.items():
            makespan_deviation = self.model.NewIntVar(0, self.horizon[1], 'makespan_deviation_' + str(machine))
            self.model.Add(makespan_deviation == self.duration_var - machine_duration_var)
            self.makespan_deviations.append(makespan_deviation)
        self.deviation_var = self.model.NewIntVar(0, self.horizon[1]*len(self.machines), 'makespan_deviation')
        self.model.Add(cp_model.LinearExpr.Sum(self.makespan_deviations) == self.deviation_var)

        obj_var = self.model.NewIntVar(self.horizon[0], self.horizon[1] * len(self.machines) + self.horizon[1] * (len(self.machines) - 1), 'obj')
        self.model.Add((self.duration_var + self.non_assign_sum) * len(self.machines) + self.deviation_var == obj_var)
        self.model.Minimize(obj_var)

    def solve(self, solver=cp_model.CpSolver()):
        status = solver.Solve(self.model)
        return (solver, status)


class UnrelatedParallelMachinesSchedulingNonAssignPolicy2(Policy):
    def __init__(self, alpha, beta, gamma, delta, selection_strategy):
        self.alpha = alpha     # time
        self.beta  = beta      # occupation
        self.gamma = gamma     # fairness
        self.delta = delta     # non-allocation cost factor
        self.selection_strategy = selection_strategy

        self.num_postponed = 0
        self.num_allocated = 0
        self.logging = False
        self.optimal, self.feasible, self.no_solution = (0, 0, 0)

        self.back_up_policy = HungarianMultiObjectivePolicy(alpha, beta, gamma, delta)
        self.policy_2 =  UnrelatedParallelMachinesSchedulingNonAssignPolicy(alpha, beta, gamma, delta, selection_strategy)

    def allocate(self, unassigned_tasks, available_resources, resource_pool, trd,
                 occupations, fairness, task_costs, working_resources, current_time):
        relevant_resources = set(available_resources) | set(working_resources.keys())
        trd = self.prune_trd(trd, unassigned_tasks, relevant_resources)
        if not trd:
            return []
        task_data, task_encoding, resource_encoding = self.get_task_data_from_trd(trd)
        swaped_tasks_dict = {v : k for k, v in task_encoding.items()}
        swaped_resources_dict = {v : k for k, v in resource_encoding.items()}

        task_costs = self.factor_task_costs(task_costs.copy(), factor=3600*self.delta)
        encoded_task_costs = dict()
        for task, cost in task_costs.items():
            if task in task_encoding:
                encoded_task_costs[task_encoding[task]] = cost

        # get encoded machines start
        machines_start = {}
        for resource, resource_enc in resource_encoding.items():
            if resource in working_resources:
                start_time = max(0, working_resources[resource][0] - current_time + working_resources[resource][1])
                machines_start[resource_enc] = int(start_time * 3600)
            else:
                machines_start[resource_enc] = 0
        #print(machines_start)

        # Creates the solver and solve.
        model = UnrelatedMachinesSchedulingNonAssign2(task_data, encoded_task_costs,
                                                     machines_start, self.delta)
        start_time = time.time()

        solver = cp_model.CpSolver()
        if self.logging:
            logging.basicConfig(level=logging.INFO, filename="log.txt", filemode="w")
            solver.parameters.log_search_progress = True
            solver.log_callback = logging.info

        # Sets a time limit of 10 seconds.
        solver.parameters.max_time_in_seconds = 2.0

        #model.model.ExportToFile('model.pd.txt')
        status = solver.Solve(model.model)
        end_time = time.time()
        duration = end_time - start_time

        if status != cp_model.OPTIMAL:
            if status == cp_model.FEASIBLE:
                self.feasible += 1
                print('2 Feasible', round(duration, 2), len(relevant_resources), len(unassigned_tasks), len(trd),
                      solver.ObjectiveValue(), model.horizon)
            else:
                self.no_solution += 1
                print('2 No solution', round(duration, 2), len(relevant_resources), len(unassigned_tasks), len(trd),
                      model.horizon)
                return self.back_up_policy.allocate(unassigned_tasks, available_resources, resource_pool, trd,
                        occupations, fairness, task_costs, working_resources, current_time)
        else:
            self.optimal += 1
            print('2 Optimal', round(duration, 2), len(relevant_resources), len(unassigned_tasks), len(trd),
                      solver.ObjectiveValue(), model.horizon)
        print(solver.Value(model.duration_var)*len(relevant_resources),
              solver.Value(model.non_assign_sum)*len(relevant_resources),
              solver.Value(model.deviation_var),
              solver.ObjectiveValue())

        #postponed_vars = []
        for task, postponed_var in model.task_non_assign.items():
            if solver.Value(postponed_var):
                #postponed_vars.append(postponed_var)
                #print('Postponed:', postponed_var)
                self.num_postponed += 1
            else:
                pass
                #print('Not postponed:', postponed_var)
        #print('Postponed vars:', postponed_vars)

        machine_tasks = collections.defaultdict(list)
        for machine, machine_assignments in model.assignments.items():
            for machine_assignment in machine_assignments:
                if solver.Value(machine_assignment.assigned_var):
                    machine_tasks[machine].append(machine_assignment)
        #print('Assignments', machine_tasks)
        selected = []
        
        # select first task (for every resource)
        for machine, tasks in machine_tasks.items():
            decoded_resource = swaped_resources_dict[machine]
            if self.selection_strategy == 'first':
                selected_task = tasks[0].task
            elif self.selection_strategy == 'fastest':
                selected_task = sorted(tasks, key=lambda task: solver.Value(task.duration))[0].task
            elif self.selection_strategy == 'slowest':
                selected_task = sorted(tasks, key=lambda task: solver.Value(task.duration))[-1].task
            elif self.selection_strategy == 'random':
                selected_task = random.choice(tasks).task
            decoded_task = swaped_tasks_dict[selected_task]
            selected.append((decoded_task, decoded_resource))
            self.num_allocated += 1

        res = self.prune_invalid_assignments(selected, available_resources, resource_pool, unassigned_tasks)
        return res
        #return selected