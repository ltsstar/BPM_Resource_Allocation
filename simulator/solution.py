from simulator import Simulator
from simulator import EventType
from datetime import timedelta, datetime
import random
import pandas
import sklearn
import sklearn.preprocessing
import sklearn.neural_network
import numpy as np
import itertools
from scipy.optimize import linear_sum_assignment

class RunTimePredicator:
    def __init__(self, resources):
        self._onehot_columns = ['Activity', 'Resource', 'ApplicationType', 'LoanGoal']
        self._rest_columns = ['W_Complete application', 'W_Call after offers', 'W_Validate application', 'W_Call incomplete files', 'W_Handle leads', 'W_Assess potential fraud', 'W_Shortened completion']
        self._standardization_columns = ['RequestedAmount']
        self._standardizer = sklearn.preprocessing.StandardScaler()
        self._normalizer = sklearn.preprocessing.Normalizer()
        self._encoder = sklearn.preprocessing.OneHotEncoder(sparse=False, handle_unknown='ignore')

        self._encoder = self._encoder.fit(pandas.DataFrame({"Resource" : resources}))

        self._regressor = sklearn.neural_network.MLPRegressor(hidden_layer_sizes=(150, 100, 25))
        self.trained = False
        self.predict_cache = dict()

    def _encode_df(self, df):
        standardized_data = self._standardizer.transform(df[self._standardization_columns])
        normalized_data = self._normalizer.transform(df[self._rest_columns])
        onehot_data = self._encoder.transform(df[self._onehot_columns])

        x = np.concatenate([standardized_data, normalized_data, onehot_data], axis=1)
        y = df["y"].to_numpy()
        return x,y

    def _hash_data(self, task_type, task_data, resource, number_task_type_occurrences):
        return hash(task_type) + hash(frozenset(task_data.items())) \
            + hash(resource) + hash(frozenset(task_data.items()))

    def train(self, df):
        self._normalizer = self._normalizer.fit(df[self._rest_columns])
        self._standarizer = self._standardizer.fit(df[self._standardization_columns])
        self._encoder = self._encoder.fit(df[self._onehot_columns])

        x,y = self._encode_df(df)
        self._regressor = self._regressor.fit(x, y)
        self.trained = True
        self.predict_cache = dict()

    def partial_train(self, df):
        x,y = self._encode_df(df)
        self._regressor = self._regressor.partial_fit(x,y)


    def predict(self, task_type, task_data, resource, number_task_type_occurrences):
        hashed_data = self._hash_data(task_type, task_data, resource, number_task_type_occurrences)
        if hashed_data in self.predict_cache:
            return self.predict_cache[hashed_data]
        
        features = {**number_task_type_occurrences, 'Activity': task_type, 'Resource': resource, **task_data}
        data = pandas.DataFrame(features, index=[1])
        standardized_data = self._standardizer.transform(data[self._standardization_columns])
        normalized_data = self._normalizer.transform(data[self._rest_columns])
        onehot_data = self._encoder.transform(data[self._onehot_columns])
        x = np.concatenate([standardized_data, normalized_data, onehot_data], axis=1)
        pred = self._regressor.predict(x)[0]
        res = max(0, pred)

        self.predict_cache[hashed_data] = res
        return res

class TRDLinearAssignment:
    def compute(trd):
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

        mat = np.full((j,i), 99999, dtype=np.dtype("float32"))
        for task, resources in trd.items():
            for resource, duration in resources.items():
                mat[resource_indices[resource]][task_indices[task]] = duration

        row_ind, col_ind = linear_sum_assignment(mat)
        selected = []
        for row, col in zip(row_ind, col_ind):
            if mat[row][col] == 99999:
                if mat[row].argmin() != 99999:
                    tr = index_taskresources[(row, mat[row].argmin())]
                else:
                    continue
            else:
                tr = index_taskresources[(row, col)]
            selected.append(tr)
        
        return selected
        

class MyPlanner:
    activity_names = ['W_Complete application', 'W_Call after offers', 'W_Validate application', 'W_Call incomplete files', 'W_Handle leads', 'W_Assess potential fraud', 'W_Shortened completion']
    initial_time = datetime(2020, 1, 1)
    time_format = "%Y-%m-%d %H:%M:%S.%f"

    def __init__(self):
        self.task_started = dict()
        self.task_resource_duration = dict()
        self.no_assignment_possible = dict()
        self.task_type_occurrences = dict()
        self.current_time = None
        self.current_time_str = ""
        self.steps = 0
        self.max_case_id = 0
        self.resources = None
        self.predictor = None #RunTimePredicator()
        self.working_resources = {}

    def time_str(self, time):
        return (self.initial_time + timedelta(hours=time)).strftime(self.time_format)

    def generate_train_df(self):
        feature_list = []
        for task_resource, value in self.task_resource_duration.items():
            task, resource = task_resource
            number_task_type_occurrences = self.task_type_occurrences[task.case_id]
            features = {**number_task_type_occurrences, 'Activity': task.task_type, 'Resource': resource, **task.data}
            features["y"] = value
            feature_list.append(features)

        return pandas.DataFrame(feature_list)

    def generate_partial_df(self, task, resource):
        number_task_type_occurrences = self.task_type_occurrences[task.case_id]
        features = {**number_task_type_occurrences, 'Activity': task.task_type, 'Resource': resource, **task.data}
        features["y"] = self.task_resource_duration[task, resource]
        return pandas.DataFrame([features])

    def plan(self, available_resources, unassigned_tasks, resource_pool):
        if not self.resources:
            self.resources = list(set(sum(resource_pool.values(), [])))

        if self.predictor:
            trds = dict()
            all_resources = list(self.working_resources.keys()) + list(available_resources)
            for trd in itertools.product(unassigned_tasks, all_resources):
                #check permission
                if trd[1] in resource_pool[trd[0].task_type]:
                    duration = self.predictor.predict(
                                    trd[0].task_type, trd[0].data, trd[1],
                                            self.task_type_occurrences[trd[0].case_id]
                                        )
                    if trd[0] not in trds:
                        trds[trd[0]] = dict()
                    trds[trd[0]][trd[1]] = duration


            #matching = ungarn.algorithm.find_matching(trds, matching_type = 'min', return_type='list')
            theoretical_assignments = TRDLinearAssignment.compute(trds)

            assignments = []
            for task, resource in theoretical_assignments:
                if resource in available_resources and task in unassigned_tasks:
                    assignments.append((task, resource))
                    available_resources.remove(resource)
                    unassigned_tasks.remove(task)
            
            resource_starts = {}
            for ar in available_resources:
                resource_starts[ar] = 0
            for wr, at in self.working_resources.items():
                resource_starts[wr] = max(0, at[1] - self.current_time)
        else:
            assignments = []
            # assign the first unassigned task to the first available resource, the second task to the second resource, etc.
            for task in unassigned_tasks:
                for resource in available_resources:
                    if resource in resource_pool[task.task_type]:
                        available_resources.remove(resource)
                        assignments.append((task, resource))
                        break
        
        print(self.current_time_str, len(unassigned_tasks), len(available_resources), len(self.working_resources))
        return assignments

    def report(self, event):
        self.current_time = event.timestamp
        #
        #print(event.lifecycle_state)
        if event.lifecycle_state == EventType.CASE_ARRIVAL:
            self.task_type_occurrences[event.case_id] = dict.fromkeys(self.activity_names, 0)

        elif event.lifecycle_state == EventType.TASK_ACTIVATE:
            self.task_type_occurrences[event.case_id][event.task.task_type] += 1
            self.current_time_str = self.time_str(self.current_time)

        elif event.lifecycle_state == EventType.START_TASK:
            self.task_started[event.task] = event.timestamp
            predicted_finish = 0
            #if self.predictor:
            #    predicted_finish = self.current_time +\
            #                        self.predictor.predict(event.task.task_type, event.task.data,
            #                            event.resource, self.task_type_occurrences[event.task.case_id]
            #                        )
            self.working_resources[event.resource] = (self.current_time, predicted_finish)

        elif event.lifecycle_state == EventType.COMPLETE_TASK:
            duration = event.timestamp - self.task_started[event.task]
            self.task_resource_duration[(event.task, event.resource)] = duration
            del self.working_resources[event.resource]

            #if self.predictor:
            #    df = self.generate_partial_df(event.task, event.resource)
            #    self.predictor.partial_train(df)
            #if self.predictor:
            #    y = self.predictor.predict(event.task.task_type, event.task.data, event.resource, self.task_type_occurrences[event.case_id])
            #    print(abs(y - duration)*60, abs(y - duration) / (y + duration))

        if not (len(self.task_resource_duration)+1) % 1000:
            df = self.generate_train_df()
            self.predictor = RunTimePredicator(self.resources)
            self.predictor.train(df)



my_planner = MyPlanner()
simulator = Simulator(my_planner)
result = simulator.run(24*365)
print(result)
