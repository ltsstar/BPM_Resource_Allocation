from simulator.simulator import EventType
import pandas
import datetime
import task_execution_time
import time
from pympler.tracker import SummaryTracker

TRACKER = SummaryTracker()

class Planner:
    activity_names = ['W_Complete application', 'W_Call after offers', 'W_Validate application', 'W_Call incomplete files', 'W_Handle leads', 'W_Assess potential fraud', 'W_Shortened completion']
    initial_time = datetime.datetime(2020, 1, 1)
    time_format = "%Y-%m-%d %H:%M:%S.%f"

    def __init__(self, prediction_model,
                 warm_up_policy, warm_up_time,
                 policy,
                 predict_multiple = False):
        self.task_started = dict()
        self.task_type_occurrences = dict()
        self.task_resource_duration = dict()
        self.current_time, self.warm_up_time = 0, warm_up_time
        self.is_warm_up = True
        self.warm_up_policy, self.policy = warm_up_policy, policy
        self.resources = None
        self.predictor = task_execution_time.TaskExecutionPrediction(prediction_model,
                                                                     predict_multiple_enabled=predict_multiple)
        self.working_resources = {}
        self.task_queue = dict()
        self.last_time = time.time()

        if self.warm_up_time == 0:
            self.is_warm_up = False

    def current_time_str(self):
        return (self.initial_time + datetime.timedelta(hours=self.current_time)).strftime(self.time_format)

    def plan(self, available_resources, unassigned_tasks, resource_pool):
        if not self.resources:
            self.resources = list(set(sum(resource_pool.values(), [])))

        if self.is_warm_up:
            assignments = self.warm_up_policy.allocate(unassigned_tasks,
                                               available_resources,
                                               resource_pool,
                                               None)
        else:
            # Predict task x resource durations
            trds = self.predictor.predict(self.working_resources,
                                          available_resources,
                                          unassigned_tasks,
                                          resource_pool,
                                          self.task_type_occurrences)
            # Make allocation decision
            assignments = self.policy.allocate(unassigned_tasks,
                                               available_resources,
                                               resource_pool,
                                               trds)
        return assignments

    def report(self, event):
        if int(event.timestamp) > int(self.current_time):
            print(self.current_time_str(), time.time() - self.last_time, len(self.task_queue), len(self.working_resources))
            self.last_time = time.time()
            TRACKER.print_diff()
        self.current_time = event.timestamp

        if self.is_warm_up and self.current_time > self.warm_up_time:
            self.predictor.train(self.resources, self.task_resource_duration, self.task_type_occurrences)
            self.is_warm_up = False

        if event.lifecycle_state == EventType.CASE_ARRIVAL:
            self.task_type_occurrences[event.case_id] = dict.fromkeys(self.activity_names, 0)
            self.case_arival(event)

        elif event.lifecycle_state == EventType.TASK_ACTIVATE:
            self.task_type_occurrences[event.case_id][event.task.task_type] += 1
            self.task_queue[event.task] = None
            self.task_activate(event)

        elif event.lifecycle_state == EventType.START_TASK:
            self.task_started[event.task] = event.timestamp
            predicted_finish = 0
            self.working_resources[event.resource] = (self.current_time, predicted_finish)
            self.start_task(event)

        elif event.lifecycle_state == EventType.COMPLETE_TASK:
            duration = event.timestamp - self.task_started[event.task]
            if self.is_warm_up:
                self.task_resource_duration[(event.task, event.resource)] = duration
            del self.working_resources[event.resource]
            self.task_queue.pop(event.task)
            self.complete_task(event)

    def case_arival(self, event):
        pass

    def task_activate(self, event):
        pass

    def start_task(self, event):
        pass

    def complete_task(self, event):
        pass

