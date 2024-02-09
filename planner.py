from simulator.simulator import EventType
import pandas
import datetime
import task_execution_time
import time
import collections
import numpy as np
import math
#from pympler.tracker import SummaryTracker

#TRACKER = SummaryTracker()

class Planner:
    activity_names = ['W_Complete application', 'W_Call after offers', 'W_Validate application', 'W_Call incomplete files', 'W_Handle leads', 'W_Assess potential fraud', 'W_Shortened completion']
    initial_time = datetime.datetime(2020, 1, 1)
    time_format = "%Y-%m-%d %H:%M:%S.%f"

    def __init__(self, prediction_model,
                 warm_up_policy, warm_up_time,
                 policy,
                 predict_multiple = False,
                 hour_timeout = math.inf,
                 debug = False):
        self.debug = debug
        self.stop = False # Tell simulator to stop
        self.hour_timeout = hour_timeout
        self.prediction_model = prediction_model
        self.task_started = dict()
        self.task_type_occurrences = dict()
        self.task_resource_duration = dict()
        self.current_time, self.warm_up_time = 0, warm_up_time
        self.is_warm_up = True
        self.warm_up_policy, self.policy = warm_up_policy, policy
        self.resources = None
        self.predictor = task_execution_time.TaskExecutionPrediction(self.prediction_model,
                                                                     predict_multiple_enabled=predict_multiple)
        self.working_resources = {}
        self.task_queue = dict()
        self.last_time = time.time()
        self.resource_occupation = collections.defaultdict(float)
        self.resources_last_active = collections.defaultdict(float)
        self.resource_active_time = collections.defaultdict(float)
        self.last_available_resources = set()
        self.total_cycle_time = 0
        self.case_start = dict()
        self.cases_completed = 0

        if self.warm_up_time == 0:
            self.is_warm_up = False

    def current_time_str(self):
        return (self.initial_time + datetime.timedelta(hours=self.current_time)).strftime(self.time_format)

    def plan(self, available_resources, unassigned_tasks, resource_pool):
        self.resource_update(available_resources, unassigned_tasks, resource_pool)

        if not self.resources:
            self.resources = list(set(sum(resource_pool.values(), [])))

        if self.is_warm_up:
            assignments = self.warm_up_policy.allocate(unassigned_tasks,
                                               available_resources,
                                               resource_pool,
                                               None,
                                               None,
                                               None,
                                               None)
        else:
            # Predict task x resource durations
            trds, task_costs = self.predictor.predict(unassigned_tasks,
                                          resource_pool,
                                          self.task_type_occurrences)
            
            # Get resource occupations
            occupations = self.get_resource_occupations()

            # Get resource fairnesses
            fairness = self.get_resource_fairness(occupations)

            # Make allocation decision
            assignments = self.policy.allocate(unassigned_tasks,
                                               available_resources,
                                               resource_pool,
                                               trds,
                                               occupations,
                                               fairness,
                                               task_costs)
        return assignments

    def report(self, event):
        if int(event.timestamp) > int(self.current_time):
            time_diff = time.time() - self.last_time
            if self.debug:
                print(self.current_time_str(), time_diff,len(self.task_queue), len(self.last_available_resources))
            if time_diff > self.hour_timeout:
                self.stop = True
            self.last_time = time.time()
            #TRACKER.print_diff()
        self.current_time = event.timestamp

        if self.is_warm_up and self.current_time > self.warm_up_time:
            self.predictor.train(self.resources, self.task_resource_duration, self.task_type_occurrences)
            self.is_warm_up = False

        if event.lifecycle_state == EventType.CASE_ARRIVAL:
            self.case_arival(event)
            self.task_type_occurrences[event.case_id] = dict.fromkeys(self.activity_names, 0)

        elif event.lifecycle_state == EventType.TASK_ACTIVATE:
            self.task_activate(event)
            self.task_type_occurrences[event.case_id][event.task.task_type] += 1
            self.task_queue[event.task] = None

        elif event.lifecycle_state == EventType.START_TASK:
            self.task_started[event.task] = event.timestamp
            if self.is_warm_up:
                predicted_duration = 0
            else:
                predicted_duration = self.prediction_model.predict(event.task, event.resource, self.task_type_occurrences[event.case_id])
            self.working_resources[event.resource] = (self.current_time, predicted_duration)
            self.start_task(event)

        elif event.lifecycle_state == EventType.COMPLETE_TASK:
            self.complete_task(event)
            duration = event.timestamp - self.task_started[event.task]
            self.task_started.pop(event.task)
            if self.is_warm_up:
                self.task_resource_duration[(event.task, event.resource)] = duration
            del self.working_resources[event.resource]
            self.task_queue.pop(event.task)

        elif event.lifecycle_state == EventType.COMPLETE_CASE:
            self.complete_case(event)
            if not self.is_warm_up:
                self.task_type_occurrences.pop(event.case_id)
                self.prediction_model.delete_case_from_cache(event.case_id)

    def resource_update(self, available_resources, unassigned_tasks, resource_pool):
        available_resources = available_resources | set(self.working_resources.keys())
        resources_gone = self.last_available_resources.copy()
        for available_resource in available_resources:
            if available_resource in resources_gone:
                resources_gone.remove(available_resource)

        resources_added = available_resources.copy()
        for available_resource in self.last_available_resources:
            if available_resource in resources_added:
                resources_added.remove(available_resource)
        
        for resource_added in resources_added:
            self.resources_last_active[resource_added] = self.current_time

        for resource_gone in resources_gone:
            self.resource_active_time[resource_gone] += self.current_time - self.resources_last_active[resource_gone]

        self.last_available_resources = available_resources | set(self.working_resources.keys())

    def get_resource_occupations(self):
        res = dict()
        for resource, availability in self.resource_active_time.items():
            occupation = self.resource_occupation[resource]
            res[resource] = occupation / availability if availability else 0
        return res
    
    def get_resource_fairness(self, resource_occupations):
        v = np.array(list(resource_occupations.values()))
        average_occupation = np.mean(v) if len(v) else {}
        res = dict()
        for resource, occupation in resource_occupations.items():
            res[resource] = (occupation - average_occupation)**2
        return res
    
    def get_current_loss(self):
        time_loss = self.total_cycle_time
        for case_start_time in self.case_start.values():
            time_loss += self.current_time - case_start_time
        
        occupations = self.get_resource_occupations()
        fairness = self.get_resource_fairness(occupations)
        return (time_loss / self.cases_completed if self.cases_completed else time_loss,
              sum(occupations.values()) / len(occupations) if len(occupations) else sum(occupations.values()),
              sum(fairness.values()) / len(fairness) if len(fairness) else sum(fairness.values())
        )

    def case_arival(self, event):
        self.case_start[event.case_id] = self.current_time

    def task_activate(self, event):
        pass

    def start_task(self, event):
        self.resource_occupation[event.resource] += self.working_resources[event.resource][1]

    def complete_task(self, event):
        # resource occupation update
        self.resource_occupation[event.resource] += (event.timestamp - self.working_resources[event.resource][0]) - self.working_resources[event.resource][1]

    def complete_case(self, event):
        self.total_cycle_time += self.current_time - self.case_start[event.case_id]
        del self.case_start[event.case_id]
        self.cases_completed += 1
