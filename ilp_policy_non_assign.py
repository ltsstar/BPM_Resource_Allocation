import math
import time
import collections
from ortools.sat.python import cp_model
import logging

from policy import Policy, GreedyParallelMachinesSchedulingPolicy

class UnrelatedMachinesSchedulingNonAssign:
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
        self.task_type = collections.namedtuple("task_type", "start end interval assigned")
        # Creates job intervals and add to the corresponding machine lists.
        self.all_tasks = {}
        self.task_assigned_machines = collections.defaultdict(list)
        self.task_non_assign = {}
        self.non_assign_cost_variables = []
        self.machines = collections.defaultdict(list)
        self.intervals = collections.defaultdict(list)
        self.goal_variables = collections.defaultdict(list)
        if max_value:
            self.horizon = self.__get_horizon()[0], max_value
        elif greedy_max:
            self.horizon = self.__get_horizon()[0], self.__get_greedy_max()
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
        max_resource = max([task[1] for task in self.task_data])
        max_resource_duration = dict()
        min_resource_duration = dict()
        min_task_duration, max_task_duration = dict(), dict()
        for t in range(max_task+1):
            #min_resource_duration[r] = min([task[2] if task[1]==r else math.inf for task in self.task_data])
            #max_resource_duration[r] = max([task[2] if task[1]==r else 0 for task in self.task_data])
            min_task_duration[t] = min([task[2] if task[0]==t else math.inf for task in self.task_data])
            max_task_duration[t] = max([task[2] if task[0]==t else 0 for task in self.task_data])
        #horizon = max(min_task_duration.values()), sum(min_task_duration.values())+1#, sum(max_task_duration.values())
        #min_horizon = min(int(sum(min_task_duration.values())/(max_resource+1)),
        #                  sum(self.non_assign_cost.values()))
        min_horizon = 0
        horizon = min_horizon, sum(min_task_duration.values())+1
        return horizon
    
    
    def __define_constraints(self):
        #Define decision variable
        for task, machine, processing_time in self.task_data:
            suffix = f"_{task}_{machine}"
            assigned_var = self.model.NewBoolVar('assigned' + suffix)
            self.task_assigned_machines[task].append(assigned_var)
            start_var = self.model.NewIntVar(self.machines_start[machine], self.horizon[1], 'start' + suffix)
            end_var = self.model.NewIntVar(self.machines_start[machine], self.horizon[1], 'end' + suffix)
            interval_var = self.model.NewOptionalIntervalVar(
                start    = start_var,
                end      = end_var,
                size = processing_time,
                is_present = assigned_var,
                name = 'interval'+suffix
            )
            goal_variable = self.model.NewIntVar(0, self.horizon[1], 'goal' + suffix)
            #self.model.Add(goal_variable == end_var).OnlyEnforceIf(assigned_var)
            #self.model.Add(goal_variable == 0).OnlyEnforceIf(assigned_var.Not())
            self.model.AddMultiplicationEquality(goal_variable, assigned_var, end_var)
            self.goal_variables[task, machine] = goal_variable
            t = self.task_type(
                start = start_var,
                end = end_var,
                assigned = assigned_var,
                interval = interval_var
            )
            self.intervals[task, machine] = t 
            self.machines[machine].append(t)

        #Add non assign
        for task, task_data in self.task_assigned_machines.items():
            non_assign_variable = self.model.NewBoolVar('non_assigned_' + str(task))
            self.task_non_assign[task] = non_assign_variable
            non_assign_cost_variable = self.model.NewIntVar(0, self.non_assign_cost[task], 'non_assigned_cost_' + str(task))
            self.model.AddMultiplicationEquality(non_assign_cost_variable, [non_assign_variable, self.non_assign_cost[task]])
            self.non_assign_cost_variables.append(non_assign_cost_variable)

        #Constraint 1: No machine may work on more than one task simultaneously
        machines_count = 1 + max(task[1] for task in self.task_data)
        all_machines = range(machines_count)
        for machine in all_machines:
            #no overlap between intervals of one machine
            self.model.AddNoOverlap([t.interval for t in self.machines[machine]])

        #Constraint 2: All tasks must be assigned to exactly one machine
        #               or to -non assign-
        tasks_count = 1+ max(task[0] for task in self.task_data)
        all_tasks = range(tasks_count)
        for task in all_tasks:
            #sum of assignment variable of task must be one
            self.model.Add(cp_model.LinearExpr.Sum(self.task_assigned_machines[task]) +
                           self.task_non_assign[task] == 1)

        #Optimization Constraint 3: No gaps between tasks on one machine
        """
        for i, machine_tasks in enumerate(self.machines.values()):
            print(i)
            for t1 in machine_tasks:
                iv = self.model.NewBoolVar(str(t1)+"0")
                self.model.Add(t1.start == 0).OnlyEnforceIf(iv)
                literals = [iv]
                for t2 in machine_tasks:
                    if t1 != t2:
                        iv = self.model.NewBoolVar(str(t1)+str(t2))
                        self.model.Add(t1.start == t2.end).OnlyEnforceIf(iv)
                        literals.append(iv)
                self.model.AddBoolXOr(literals)
        """
    
    def __define_objective(self):
        # Makespan objective.
        self.duration_var = self.model.NewIntVar(self.horizon[0], self.horizon[1], "makespan")
        self.model.AddMaxEquality(
            self.duration_var,
            [self.goal_variables[task, machine] for task, machine, _ in self.task_data],
        )
        self.non_assign_sum = self.model.NewIntVar(0, sum(self.non_assign_cost.values()), 'non_assign_sum')
        self.model.Add(cp_model.LinearExpr.Sum(self.non_assign_cost_variables) == self.non_assign_sum)

        obj_var = self.model.NewIntVar(self.horizon[0], self.horizon[1], 'obj')
        self.model.Add(self.duration_var + self.non_assign_sum == obj_var)
        self.model.Minimize(obj_var)

    def solve(self, solver=cp_model.CpSolver()):
        status = solver.Solve(self.model)
        return (solver, status)


class UnrelatedParallelMachinesSchedulingNonAssignPolicy(Policy):
    def __init__(self, alpha, beta, gamma, delta):
        self.alpha = alpha     # time
        self.beta  = beta      # occupation
        self.gamma = gamma     # fairness
        self.delta = delta     # non-allocation cost factor

        self.num_postponed = 0
        self.num_allocated = 0
        self.logging = False
        self.optimal, self.feasible, self.no_solution = (0, 0, 0)

    def allocate(self, unassigned_tasks, available_resources, resource_pool, trd,
                 occupations, fairness, task_costs, working_resources, current_time):
        relevant_resources = set(available_resources) | set(working_resources)
        trd = self.prune_trd(trd, unassigned_tasks, relevant_resources)
        if not trd:
            return []
        task_data, task_encoding, resource_encoding = self.get_task_data_from_trd(trd)
        swaped_tasks_dict = {v : k for k, v in task_encoding.items()}
        swaped_resources_dict = {v : k for k, v in resource_encoding.items()}

        task_costs = self.factor_task_costs(task_costs, factor=3600*self.delta)
        encoded_task_costs = dict()
        for task, cost in task_costs.items():
            if task in task_encoding:
                encoded_task_costs[task_encoding[task]] = cost

        # get encoded machines start
        machines_start = {}
        for resource, resource_enc in resource_encoding.items():
            if resource in working_resources:
                machines_start[resource_enc] = 0
            else:
                machines_start[resource_enc] = 0

        # Creates the solver and solve.
        model = UnrelatedMachinesSchedulingNonAssign(task_data, encoded_task_costs,
                                                     machines_start, self.delta)
        start_time = time.time()

        solver = cp_model.CpSolver()
        if self.logging:
            logging.basicConfig(level=logging.INFO, filename="log.txt", filemode="w")
            solver.parameters.log_search_progress = True
            solver.log_callback = logging.info

        # Sets a time limit of 10 seconds.
        solver.parameters.max_time_in_seconds = 1.0

        status = solver.Solve(model.model)
        end_time = time.time()

        if end_time - start_time > 60:
            print(int(end_time - start_time), len(unassigned_tasks), int(len(trd)/len(unassigned_tasks)),
                      solver.ObjectiveValue(), model.horizon)

        if status != cp_model.OPTIMAL:
            if status == cp_model.FEASIBLE:
                self.feasible += 1
                print('Feasible', int(end_time - start_time), len(unassigned_tasks), len(trd),
                      solver.ObjectiveValue(), model.horizon)
            else:
                self.no_solution += 1
                print('No solution', int(end_time - start_time), len(unassigned_tasks), len(trd),
                      model.horizon)
                return GreedyParallelMachinesSchedulingPolicy().allocate(unassigned_tasks, available_resources, resource_pool, trd, occupations, fairness, task_costs)
        self.optimal += 1
        #print(solver.Value(model.duration_var), solver.Value(model.non_assign_sum))

        for task, postponed_var in model.task_non_assign.items():
            if solver.Value(postponed_var):
                self.num_postponed += 1
        
        selected = []
        schedule = collections.defaultdict(list)
        for (task, resource), interval in model.intervals.items():
            if solver.Value(interval.assigned):
                schedule[resource].append((task, solver.Value(interval.start)))
        
        for resource, resource_schedule in schedule.items():
            decoded_resource = swaped_resources_dict[resource]
            first_task = sorted(resource_schedule, key=lambda s: s[1])[0][0]
            decoded_task = swaped_tasks_dict[first_task]
            selected.append((decoded_task, decoded_resource))
            self.num_allocated += 1
        
        return self.prune_invalid_assignments(selected, available_resources, resource_pool, unassigned_tasks)
        #return selected