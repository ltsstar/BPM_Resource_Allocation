from keras.models import Sequential
from keras.layers import Dense
from sklearn.preprocessing import StandardScaler, Normalizer, OneHotEncoder
from keras.callbacks import Callback
from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np
import itertools
import collections

class TrainingLogger(Callback):
    pass

class ExecutionTimeModel:
    def __init__(self):
        self._onehot_columns = ['Activity', 'Resource', 'ApplicationType', 'LoanGoal']
        self._rest_columns = ['W_Complete application', 'W_Call after offers', 'W_Validate application', 'W_Call incomplete files', 'W_Handle leads', 'W_Assess potential fraud', 'W_Shortened completion']
        self._standardization_columns = ['RequestedAmount']

        self._model = None
        self.trained = False
        self.predict_cache = collections.defaultdict(dict)

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

        return pd.DataFrame(feature_list)

    def _transform_data(self, df):
        # Transform the data
        normalized_data = self._normalizer.transform(df[self._rest_columns])
        standardized_data = self._standarizer .transform(df[self._standardization_columns])
        onehot_data = self._encoder.transform(df[self._onehot_columns])
        return (normalized_data, standardized_data, onehot_data)
    
    def _encode_df(self, resources, df):
        self._encoder = OneHotEncoder(sparse=False, handle_unknown='ignore')
        self._encoder.fit(df[self._onehot_columns])

        # Normalizer
        self._normalizer = Normalizer()
        self._normalizer.fit(df[self._rest_columns])

        # StandardScaler
        self._standarizer  = StandardScaler()
        self._standarizer.fit(df[self._standardization_columns])

        # Combine the features
        x = np.concatenate(self._transform_data(df), axis=1)
        y = df["y"].to_numpy()

        return x,y

    def _hash_data(self, task, resource, number_task_type_occurrences):
        return hash(task.task_type) + hash(frozenset(task.data.items())) \
            + hash(resource) + hash(frozenset(number_task_type_occurrences.items()))

    def train(self, resources, task_resource_durations, task_type_occurrences):
        train_df = self.__generate_train_df(task_resource_durations, task_type_occurrences)

        x,y = self._encode_df(resources, train_df)
        # Split the data into train and validation sets
        x_train, x_val, y_train, y_val = train_test_split(x, y, test_size=0.2, random_state=42)

        # Build the Keras model
        self._model = Sequential()
        self._model.add(Dense(150, input_dim=x_train.shape[1], activation='relu'))
        self._model.add(Dense(100, activation='relu'))
        self._model.add(Dense(25, activation='relu'))
        self._model.add(Dense(1))  # Assuming you have a single output in regression problems

        # Compile the model
        self._model.compile(optimizer='adam', loss='mean_squared_error')

        # Train the model
        self._model.fit(x_train, y_train, epochs=300, batch_size=256, validation_data=(x_val, y_val),
                        verbose=1)
        self.trained = True
        self.predict_cache.clear()

    def predict(self, task, resource, number_task_type_occurrences):
        hashed_data = self._hash_data(task, resource, number_task_type_occurrences)
        if hashed_data in self.predict_cache[task.case_id]:
            return self.predict_cache[task.case_id][hashed_data]

        features = {**number_task_type_occurrences, 'Activity': task.task_type, 'Resource': resource, **task.data}
        data = pd.DataFrame(features, index=[1])

        x = np.concatenate(self._transform_data(data), axis=1)
        pred = self._model(x, training=False)
        res = max(0, pred)
        res = float(res)

        self.predict_cache[task.case_id][hashed_data] = res
        return res
    
    def predict_multiple(self, unassigned_tasks, resource_pool, task_type_occurrences):
        results = dict()
        to_predict = []
        to_normalize_data = []
        to_standardize_data = []
        to_onehot_data = []

        for task in unassigned_tasks:
            for resource in resource_pool[task.task_type]:
                hashed_data = self._hash_data(task, resource, task_type_occurrences[task.case_id])
                if hashed_data in self.predict_cache[task.case_id]:
                    results[(task, resource)] = self.predict_cache[task.case_id][hashed_data]
                else:
                    #to_normalize_data.append(list(task_type_occurrences[task.case_id].values()))
                    to_normalize_data.append(list(task_type_occurrences[task.case_id].values()))
                    to_standardize_data.append([task.data['RequestedAmount']])
                    to_onehot_data.append([task.task_type, resource, task.data['ApplicationType'], task.data['LoanGoal']])
                    to_predict.append((task, resource))

        if to_predict:
            normalize_data_df = pd.DataFrame(to_normalize_data, columns=self._rest_columns)
            normalized_data = self._normalizer.transform(normalize_data_df)
            standardized_data_df = pd.DataFrame(to_standardize_data, columns=self._standardization_columns)
            standardized_data = self._standarizer.transform(standardized_data_df)
            onehot_data_df = pd.DataFrame(to_onehot_data, columns=self._onehot_columns)
            onehot_data = self._encoder.transform(onehot_data_df)
            x = np.concatenate((normalized_data, standardized_data, onehot_data), axis=1)
            y = self._model(x, training=False)
            for i, idx in enumerate(to_predict):
                res = float(y[i][0])
                results[idx] = res
                case_id = idx[0].case_id
                hash_value = self._hash_data(idx[0], idx[1], task_type_occurrences[case_id])
                self.predict_cache[case_id][hash_value] = res

        return results
    
    def delete_case_from_cache(self, case_id):
        self.predict_cache.pop(case_id)


class TaskExecutionPrediction:
    def __init__(self, model, predict_multiple_enabled = False) -> None:
        self.model = model
        self.predict_multiple_enabled = predict_multiple_enabled

    def train(self, resources, task_resource_durations, task_type_occurrences):
        self.model.train(resources, task_resource_durations, task_type_occurrences)

    def predict(self, unassigned_tasks, resource_pool, task_type_occurrences):
        trds = dict()
        mean_task_durations = dict()
        if self.predict_multiple_enabled:
            trds = self.model.predict_multiple(unassigned_tasks, resource_pool, task_type_occurrences)
            for task in unassigned_tasks:
                durations = []
                for resource in resource_pool[task.task_type]:
                    durations.append(trds[(task, resource)])
                mean_task_durations[task] = np.mean(durations)
        else:
            for task in unassigned_tasks:
                durations = []
                for resource in resource_pool[task.task_type]:
                    duration = self.model.predict(task, resource, task_type_occurrences[task.case_id])
                    trds[(task, resource)] = duration
                    durations.append(duration)
                mean_task_durations[task] = np.mean(durations)
        return trds, mean_task_durations