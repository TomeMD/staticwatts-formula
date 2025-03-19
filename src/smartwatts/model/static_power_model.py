# BSD 3-Clause License
#
# Copyright (c) 2022, Inria
# Copyright (c) 2022, University of Lille
# All rights reserved.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
# * Redistributions of source code must retain the above copyright notice, this
#   list of conditions and the following disclaimer.
#
# * Redistributions in binary form must reproduce the above copyright notice,
#   this list of conditions and the following disclaimer in the documentation
#   and/or other materials provided with the distribution.
#
# * Neither the name of the copyright holder nor the names of its
#   contributors may be used to endorse or promote products derived from
#   this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
# FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
# DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
# SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
# OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

import warnings
from hashlib import sha1
from pickle import dumps

from sklearn.linear_model import SGDRegressor
from sklearn.preprocessing import StandardScaler

from .sample_history import ReportHistory


class StaticPowerModel:
    """
    This Power model compute the power estimations based on all the historical data since the beginning of the execution.
    """

    @staticmethod
    def __get_sgd_regressor() -> SGDRegressor:
        """
        Helper function to create a SGDRegressor object with the default parameters.
        """
        return SGDRegressor(
            eta0=0.01,
            learning_rate='constant',
            alpha=0.0001,
            penalty='elasticnet',
            max_iter=10000,
            fit_intercept=True,
            random_state=42
        )

    def __init__(self, frequency: int, min_samples: int):
        """
        Initialize a new power model.
        :param frequency: Frequency of the power model (in MHz)
        :param min_samples: Minimum amount of samples required before trying to learn a power model
        """
        self.frequency = frequency
        self.min_samples = min_samples
        self.scaler = StandardScaler()
        self.clf = self.__get_sgd_regressor()
        self.hash = 'uninitialized'
        self.cumulative_events = 0
        self.id = -1

    def learn_power_model(self, samples_history: ReportHistory, min_intercept: float, max_intercept: float) -> None:
        """
        Learn a new power model using the stored reports and update the formula id/hash.
        :param samples_history: History of the reports used to learn the model
        :param min_intercept: Minimum value allowed for the intercept of the model
        :param max_intercept: Maximum value allowed for the intercept of the model
        """

        # Model already fitted, do nothing
        if self.hash != 'uninitialized':
            return

        # If data is not enough we don't initialise the model yet
        if len(samples_history) < self.min_samples:
            return

        # Fit the scaler with the initial data
        self.scaler.partial_fit(samples_history.events_values)

        # Scale the data, as the SGDRegressor is sensitive to the scale of the input data
        events_values_scaled = self.scaler.transform(samples_history.events_values)

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            self.clf.fit(events_values_scaled, samples_history.power_values)

        # Discard the new model when the intercept is not in specified range
        if not min_intercept <= self.clf.intercept_ < max_intercept:
            self.clf = self.__get_sgd_regressor()
            return

        self.cumulative_events = len(samples_history)
        self.hash = sha1(dumps(self.clf)).hexdigest()

    def learn_from_new_event(self, power_reference: float, events_value: list[float]) -> None:
        """
        Update the power model with new samples.
        :param power_reference: Power reference (RAPL) of the machine
        :param events_value: Events value (Hardware Performance Counters) of the target
        """
        if self.hash != 'uninitialized':
            self.scaler.partial_fit([events_value])
            self.clf.partial_fit(self.scaler.transform([events_value]), [power_reference])
            self.hash = sha1(dumps(self.clf)).hexdigest()
            self.cumulative_events += 1

    def predict_power_consumption(self, events: list[float]) -> float | None:
        """
        Compute a power estimation from the events value using the power model.
        :param events: Events value
        :raise: NotFittedError when the model haven't been fitted
        :return: Power estimation for the given events value
        """
        return self.clf.predict(self.scaler.transform([events]))[0]

    def cap_power_estimation(self, raw_target_power: float, raw_global_power: float) -> (float, float):
        """
        Cap target's power estimation to the global power estimation.
        :param raw_target_power: Target power estimation from the power model (in Watt)
        :param raw_global_power: Global power estimation from the power model (in Watt)
        :return: Capped power estimation (in Watt) with its ratio over global power consumption
        """
        target_power = raw_target_power - self.clf.intercept_[0]
        global_power = raw_global_power - self.clf.intercept_[0]

        if global_power <= 0.0 or target_power <= 0.0:
            return 0.0, 0.0

        target_ratio = target_power / global_power
        target_intercept_share = target_ratio * self.clf.intercept_[0]

        return target_power + target_intercept_share, target_ratio
