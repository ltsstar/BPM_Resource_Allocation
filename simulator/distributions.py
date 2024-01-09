import pandas
import random
import scipy
import numpy as np
from enum import Enum, auto
from sklearn.preprocessing import StandardScaler, MinMaxScaler, OneHotEncoder
from sklearn.neural_network import MLPRegressor


class DistributionType(Enum):
    CATEGORICAL = auto()
    GAMMA = auto()
    NORMAL = auto()
    BETA = auto()
    ERLANG = auto()
    UNIFORM = auto()


class CategoricalDistribution:

    def __init__(self):
        self._values = []
        self._weights = []

    def learn(self, values, counts):
        self._values = values
        self._weights = counts

    def sample(self):
        return random.choices(self._values, weights=self._weights)[0]


class UniformDistribution:

    def __init__(self, minimum=0, maximum=0):
        self.minimum = minimum
        self.maximum = maximum

    def learn(self, values):
        self.minimum = min(values)
        self.maximum = max(values)

    def sample(self):
        return random.uniform(self.minimum, self.maximum)


class GammaDistribution:

    def __init__(self):
        self._alpha = 0
        self._loc = 0
        self._scale = 0

    def learn(self, values):
        fit_alpha, fit_loc, fit_scale = scipy.stats.gamma.fit(values)
        self._alpha = fit_alpha
        self._loc = fit_loc
        self._scale = fit_scale

    def sample(self):
        return scipy.stats.gamma.rvs(self._alpha, loc=self._loc, scale=self._scale)


class ErlangDistribution:

    def __init__(self):
        self._shape = 0
        self._rate = 0

    def __init__(self, shape, scale):
        self._shape = shape
        self._rate = scale

    def learn(self, values):
        shape, loc, scale = scipy.stats.erlang.fit(values)
        self._shape = shape
        self._rate = scale

    def sample(self):
        return scipy.stats.erlang.rvs(self._shape, scale=self._rate)

    def mean(self):
        return scipy.stats.erlang.mean(self._shape, scale=self._rate)

    def std(self):
        return scipy.stats.erlang.std(self._shape, scale=self._rate)

    def var(self):
        return scipy.stats.erlang.var(self._shape, scale=self._rate)


class NormalDistribution:

    def __init__(self, mu = 0, std = 0):
        self.mu = mu
        self.std = std

    def learn(self, values):
        fit_mu, fit_std = scipy.stats.norm.fit(values)
        self.mu = fit_mu
        self.std = fit_std

    def sample(self):
        return scipy.stats.norm.rvs(self.mu, self.std)


class BetaDistribution:

    def __init__(self):
        self._a = 0
        self._b = 0
        self._loc = 0
        self._scale = 0

    def learn(self, values):
        fit_a, fit_b, fit_loc, fit_scale = scipy.stats.beta.fit(values)
        self._a = fit_a
        self._b = fit_b
        self._loc = fit_loc
        self._scale = fit_scale

    def sample(self):
        return scipy.stats.beta.rvs(self._a, self._b, self._loc, self._scale)


class StratifiedNumericDistribution:

    def __init__(self):
        self._target_column = ''
        self._feature_columns = []
        self._onehot_columns = []
        self._standardization_columns = []
        self._rest_columns = []
        self._normalizer = None
        self._standardizer = None
        self._encoder = None
        self._regressor = None
        self._stratifier = ''
        self._stratified_errors = dict()
        self._overall_mean = 0

    def sample(self, features):
        data = pandas.DataFrame(features, index=[1])
        standardized_data = self._standardizer.transform(data[self._standardization_columns])
        normalized_data = self._normalizer.transform(data[self._rest_columns])
        onehot_data = self._encoder.transform(data[self._onehot_columns])
        x = np.concatenate([standardized_data, normalized_data, onehot_data], axis=1)
        processing_time = self._regressor.predict(x)[0]
        if processing_time <= 0:
            processing_time = self._overall_mean
        error = self._stratified_errors[features[self._stratifier]].sample()
        max_retries = 10
        retry = 0
        while retry < max_retries and processing_time + error <= 0:
            error = self._stratified_errors[features[self._stratifier]].sample()
            retry += 1
        if processing_time + error > 0:
            return processing_time + error
        else:
            return processing_time
