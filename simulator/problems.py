import random
import pickle
from math import factorial
from abc import ABC, abstractmethod


class Task:
    def __init__(self, task_id, case_id, task_type):
        self.id = task_id
        self.case_id = case_id
        self.task_type = task_type
        self.data = dict()

    def __str__(self):
        return self.task_type + "(" + str(self.case_id) + ")_" + str(self.id) + (str(self.data) if len(self.data) > 0 else "")


class Problem(ABC):
    @property
    @abstractmethod
    def resources(self):
        raise NotImplementedError

    @property
    def resource_weights(self):
        return self._resource_weights

    @resource_weights.setter
    def resource_weights(self, value):
        self._resource_weights = value

    @property
    def schedule(self):
        return self._schedule

    @schedule.setter
    def schedule(self, value):
        self._schedule = value

    @property
    @abstractmethod
    def task_types(self):
        raise NotImplementedError

    def is_event(self, task_type):
        return False

    @abstractmethod
    def sample_initial_task_type(self):
        raise NotImplementedError

    def resource_pool(self, task_type):
        return self.resources

    def __init__(self):
        self._resource_weights = [1]*len(self.resources)
        self._schedule = [len(self.resources)]
        self.next_case_id = 0
        self.previous_case_arrival_time = 0
        self.next_task_id = 0
        self.history = dict()
        self.restart()

    @classmethod
    def from_file(cls, filename):
        with open(filename, 'rb') as handle:
            instance = pickle.load(handle)
        return instance

    def save(self, filename):
        with open(filename, 'wb') as handle:
            pickle.dump(self, handle, protocol=pickle.HIGHEST_PROTOCOL)

    @abstractmethod
    def processing_time_sample(self, resource, task):
        raise NotImplementedError

    @abstractmethod
    def interarrival_time_sample(self):
        raise NotImplementedError

    def data_sample(self, task):
        return dict()

    def next_task_types_sample(self, task):
        return []

    def restart(self):
        self.next_case_id = 0
        self.previous_case_arrival_time = 0
        self.next_task_id = 0
        self.history = dict()

    def next_case(self):
        arrival_time = self.previous_case_arrival_time + self.interarrival_time_sample()
        initial_task_type = self.sample_initial_task_type()
        case_id = self.next_case_id
        initial_task = Task(self.next_task_id, case_id, initial_task_type)
        initial_task.data = self.data_sample(initial_task)
        self.next_case_id += 1
        self.next_task_id += 1
        self.previous_case_arrival_time = arrival_time
        self.history[case_id] = []
        return arrival_time, initial_task

    def nr_cases_generated(self):
        return self.next_case_id

    def complete_task(self, task):
        self.history[task.case_id].append(task)
        next_tasks = []
        for tt in self.next_task_types_sample(task):
            new_task = Task(self.next_task_id, task.case_id, tt)
            new_task.data = self.data_sample(new_task)
            self.next_task_id += 1
            next_tasks.append(new_task)
        return next_tasks


class MinedProblem(Problem):

    resources = []
    task_types = []

    def __init__(self):
        super().__init__()
        self.initial_task_distribution = []
        self.next_task_distribution = dict()
        self.interarrival_time = 0
        self.resource_pools = dict()
        self.data_types = dict()
        self.__case_data = dict()
        self.processing_times = dict()
        self.__number_task_type_occurrences = dict()

    def sample_initial_task_type(self):
        rd = random.random()
        rs = 0
        for (p, tt) in self.initial_task_distribution:
            rs += p
            if rd < rs:
                return tt
        print("WARNING: the probabilities of initial tasks do not add up to 1.0")
        return self.initial_task_distribution[0]

    def resource_pool(self, task_type):
        return self.resource_pools[task_type]

    def interarrival_time_sample(self):
        return self.interarrival_time.sample()

    def next_task_types_sample(self, task):
        rd = random.random()
        rs = 0
        for (p, tt) in self.next_task_distribution[task.task_type]:
            rs += p
            if rd < rs:
                if tt is None:
                    return []
                else:
                    return [tt]
        print("WARNING: the probabilities of next tasks do not add up to 1.0")
        if self.next_task_distribution[0][1] is None:
            return []
        else:
            return [self.next_task_distribution[0][1]]

    def processing_time_sample(self, resource, task):
        features = {**self.__number_task_type_occurrences[task.case_id], 'Activity': task.task_type, 'Resource': resource, **task.data}
        return self.processing_times.sample(features)

    def data_sample(self, task):
        if task.case_id not in self.__case_data:
            self.__case_data[task.case_id] = dict()
            for dt in self.data_types:
                self.__case_data[task.case_id][dt] = self.data_types[dt].sample()
        return self.__case_data[task.case_id]

    def restart(self):
        super().restart()
        self.__case_data = dict()
        self.__number_task_type_occurrences = dict()

    def next_case(self):
        arrival_time, initial_task = super().next_case()
        self.__number_task_type_occurrences[initial_task.case_id] = dict()
        for tt in self.task_types:
            self.__number_task_type_occurrences[initial_task.case_id][tt] = 0
        return arrival_time, initial_task

    def complete_task(self, task):
        next_tasks = super().complete_task(task)
        self.__number_task_type_occurrences[task.case_id][task.task_type] += 1
        return next_tasks
