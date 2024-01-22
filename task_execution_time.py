import sklearn.preprocessing
import sklearn.neural_network
import pandas
import numpy as np
import itertools
import collections


class ExecutionTimeModel:
    def __init__(self):
        self._onehot_columns = ['Activity', 'Resource', 'ApplicationType', 'LoanGoal']
        self._rest_columns = ['W_Complete application', 'W_Call after offers', 'W_Validate application', 'W_Call incomplete files', 'W_Handle leads', 'W_Assess potential fraud', 'W_Shortened completion']
        self._standardization_columns = ['RequestedAmount']
        self._standardizer = sklearn.preprocessing.StandardScaler()
        self._normalizer = sklearn.preprocessing.Normalizer()
        self._encoder = sklearn.preprocessing.OneHotEncoder(sparse=False, handle_unknown='ignore')

        self._regressor = sklearn.neural_network.MLPRegressor(hidden_layer_sizes=(150, 100, 25))
        self.trained = False
        self.predict_cache = dict()

    def __generate_train_df(self, task_resource_durations, task_type_occurrences):
        feature_list = []
        for task_resource, value in task_resource_durations.items():
            task, resource = task_resource
            number_task_type_occurrences = task_type_occurrences[task.case_id]
            features = {**number_task_type_occurrences,
                        'Activity': task.task_type,
                        'Resource': resource,
                        **task.data,
                        'y' : value}
            feature_list.append(features)

        return pandas.DataFrame(feature_list)

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

    def train(self, resources, task_resource_durations, task_type_occurrences):
        self._encoder = self._encoder.fit(pandas.DataFrame({"Resource" : resources}))
        train_df = self.__generate_train_df(task_resource_durations, task_type_occurrences)

        self._normalizer = self._normalizer.fit(train_df[self._rest_columns])
        self._standarizer = self._standardizer.fit(train_df[self._standardization_columns])
        self._encoder = self._encoder.fit(train_df[self._onehot_columns])

        x,y = self._encode_df(train_df)
        self._regressor = self._regressor.fit(x, y)
        self.trained = True
        self.predict_cache = dict()

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
    

class TaskExecutionPrediction:
    def __init__(self, model) -> None:
        self.model = model

    def train(self, resources, task_resource_durations, task_type_occurrences):
        self.model.train(resources, task_resource_durations, task_type_occurrences)

    def predict(self, working_resources, available_resources,
                unassigned_tasks, resource_pool, task_type_occurrences):
        trds = collections.defaultdict(dict)
        all_resources = list(working_resources.keys()) + list(available_resources)
        for task, resource in itertools.product(unassigned_tasks, all_resources):
            #check permission
            if resource in resource_pool[task.task_type]:
                duration = self.model.predict(task.task_type, task.data, resource, task_type_occurrences[task.case_id])
                trds[task][resource] = duration
        return trds