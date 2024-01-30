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

    def _hash_data(self, task_type, task_data, resource, number_task_type_occurrences):
        return hash(task_type) + hash(frozenset(task_data.items())) \
            + hash(resource) + hash(frozenset(task_data.items()))

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
        self.predict_cache = dict()

    def predict(self, task_type, task_data, resource, number_task_type_occurrences):
        hashed_data = self._hash_data(task_type, task_data, resource, number_task_type_occurrences)
        if hashed_data in self.predict_cache:
            return self.predict_cache[hashed_data]
        
        features = {**number_task_type_occurrences, 'Activity': task_type, 'Resource': resource, **task_data}
        data = pd.DataFrame(features, index=[1])

        x = np.concatenate(self._transform_data(data), axis=1)
        pred = self._model.predict(x, verbose=0)[0]
        res = max(0, pred)

        self.predict_cache[hashed_data] = res
        return res
    
    def predict_multiple(self, data):
        x = np.concatenate(self._transform_data(data.loc[:, ~data.columns.isin(['T', 'R'])]), axis=1)
        pred = self._model.predict(x, verbose=0)
        pred = np.maximum(pred, np.zeros_like(pred))
        data['y'] = pred


class TaskExecutionPrediction:
    def __init__(self, model, predict_multiple_enabled = False) -> None:
        self.model = model
        self.predict_multiple_enabled = predict_multiple_enabled

    def train(self, resources, task_resource_durations, task_type_occurrences):
        self.model.train(resources, task_resource_durations, task_type_occurrences)

    def predict(self, working_resources, available_resources,
                unassigned_tasks, resource_pool, task_type_occurrences):
        trds = collections.defaultdict(dict)
        all_resources = list(working_resources.keys()) + list(available_resources)

        if self.predict_multiple_enabled:
            df = pd.DataFrame(columns = ['T', 'R', *self.model._rest_columns, *unassigned_tasks[0].data, 
                                        'Activity', 'Resource', 'y'])
            df = df.set_index(['T', 'R'])
            for task, resource in itertools.product(unassigned_tasks, all_resources):
                df.loc[(task, resource),:] = {
                            **task_type_occurrences[task.case_id], 
                            'Activity' : task.task_type,
                            'Resource' : resource,
                            **task.data,
                            'y' : 0.0
                }

            self.model.predict_multiple(df)
            trds = df['y'].to_dict()
        else:
            trds = dict()
            for task, resource in itertools.product(unassigned_tasks, all_resources):
                trds[(task, resource)] = self.model.predict(task.task_type, task.task_data,
                                                            resource, task_type_occurrences[task])
        return trds