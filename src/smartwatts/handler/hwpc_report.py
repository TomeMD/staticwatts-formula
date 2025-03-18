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
import datetime
import itertools
import logging
from collections import OrderedDict, defaultdict
from math import ldexp, fabs
from typing import Any

from powerapi.handler import Handler
from powerapi.report import PowerReport, HWPCReport, FormulaReport
from sklearn.exceptions import NotFittedError

from smartwatts.model import FrequencyLayer


class HwPCReportHandler(Handler):
    """
    HwPC reports handler.
    """

    def __init__(self, state):
        Handler.__init__(self, state)
        self.layers = self._generate_frequency_layers()
        self.ticks: OrderedDict[datetime.datetime, dict[str, HWPCReport]] = OrderedDict()

    def _generate_frequency_layers(self) -> OrderedDict[int, FrequencyLayer]:
        """
        Generate and returns a layered container to store per-frequency power models
        :return: Initialized Ordered dict containing a power model for each frequency layer
        """
        return OrderedDict(
            (freq, FrequencyLayer(freq, self.state.config.min_samples_required, self.state.config.history_window_size, self.state.config.error_window_size))
            for freq in self.state.config.cpu_topology.get_supported_frequencies()
        )

    def _get_nearest_frequency_layer(self, frequency: int) -> FrequencyLayer:
        """
        Find and returns the nearest frequency layer for the given frequency.
        :param frequency: CPU frequency
        :return: The nearest frequency layer for the given frequency
        """
        return self.layers.get(max(freq for freq in self.layers.keys() if freq <= frequency))

    def _compute_avg_pkg_frequency(self, system_msr: dict[str, float]) -> int:
        """
        Compute the average package frequency.
        :param system_msr: MSR events group of System target
        :return: Average frequency (in MHz) of the Package
        """
        return int((self.state.config.cpu_topology.get_base_frequency() * system_msr['APERF']) / system_msr['MPERF'])

    def handle(self, msg: HWPCReport) -> None:
        """
        Process a HWPC report and send the result(s) to a pusher actor.
        :param msg: Received HWPC report
        """
        logging.debug('received message: %s', msg)
        self.ticks.setdefault(msg.timestamp, {}).update({msg.target: msg})

        # Start to process the oldest tick only after receiving at least 5 ticks.
        # We wait before processing the ticks in order to mitigate the possible delay between the sensor/database.
        if len(self.ticks) > 5:
            power_reports, formula_reports = self._process_oldest_tick()
            for report in itertools.chain(power_reports, formula_reports):
                for name, pusher in self.state.pushers.items():
                    if isinstance(report, pusher.state.report_model):
                        pusher.send_data(report)
                        logging.debug('sent report: %s to %s', report, name)

    def _process_oldest_tick(self) -> tuple[list[PowerReport], list[FormulaReport]]:
        """
        Process the oldest tick stored in the stack and generate power reports for the running target(s).
        :return: Power reports of the running target(s)
        """
        timestamp, hwpc_reports = self.ticks.popitem(last=False)
        power_reports = []
        formula_reports = []

        try:
            global_report = hwpc_reports.pop('all')
        except KeyError:
            # cannot process this tick without the reference measurements
            logging.error('Failed to process tick %s: missing global report', timestamp)
            return power_reports, formula_reports

        # Don't continue if there is no reports available.
        # Can happen when reports are dropped by a pre-processor.
        if len(hwpc_reports) == 0:
            return power_reports, formula_reports

        rapl = self._gen_rapl_events_group(global_report)
        avg_msr = self._gen_msr_events_group(global_report)
        global_core = self._gen_agg_core_report_from_running_targets(hwpc_reports)
        rapl_power = rapl[self.state.config.rapl_event]
        power_reports.append(self._gen_power_report(timestamp, 'rapl', self.state.config.rapl_event, rapl_power, 1.0, global_report.metadata))

        try:
            pkg_frequency = self._compute_avg_pkg_frequency(avg_msr)
        except ZeroDivisionError:
            logging.error('Failed to process tick %s: PKG frequency is invalid', timestamp)
            return power_reports, formula_reports

        layer = self._get_nearest_frequency_layer(pkg_frequency)

        # try to get power estimations from both models
        power_estimations = {}
        for label in ['static', 'dynamic']:
            try:
                value = layer.model[label].predict_power_consumption(self._extract_events_value(global_core))
                power_estimations[label] = {'value': value, 'error': fabs(rapl_power - value)}
                layer.store_error_in_history(power_estimations[label]['error'], label)
                power_estimations[label]['window_error'] = layer.error_history[label].compute_error(self.state.config.error_window_method)
            except NotFittedError:
                layer.update_power_model(0.0, self.state.config.cpu_topology.tdp, label)

        if not power_estimations:
            layer.store_sample_in_history(rapl_power, self._extract_events_value(global_core))
            return power_reports, formula_reports

        # choose the model with the lowest error during the whole error window
        best_model = min(power_estimations, key=lambda x: power_estimations[x]["window_error"])
        raw_global_power = power_estimations[best_model]['value']
        # alternative: choose the model with the lowest error for this tick
        # best_model = min(power_estimations, key=lambda x: power_estimations[x]['error'])

        # append global power report with the estimation of the best model
        power_reports.append(self._gen_power_report(timestamp, 'global', layer.model[best_model].hash, raw_global_power, 1.0, global_report.metadata))

        # use the best model to compute per-target power report
        for target_name, target_report in hwpc_reports.items():
            target_core = self._gen_core_events_group(target_report)
            raw_target_power = layer.model[best_model].predict_power_consumption(self._extract_events_value(target_core))
            target_power, target_ratio = layer.model[best_model].cap_power_estimation(raw_target_power, raw_global_power)
            power_reports.append(self._gen_power_report(timestamp, target_name, layer.model[best_model].hash, target_power, target_ratio, target_report.metadata))

        layer.store_sample_in_history(rapl_power, self._extract_events_value(global_core))

        # always train the static power model
        layer.model["static"].learn_from_new_event(rapl_power, self._extract_events_value(global_core))

        # learn new dynamic power model if error exceeds the error threshold
        if 'dynamic' in power_estimations and power_estimations['dynamic']['window_error'] > self.state.config.error_threshold:
            layer.update_power_model(0.0, self.state.config.cpu_topology.tdp, 'dynamic')

        # store information about the power model used for this tick
        formula_reports.append(self._gen_formula_report(timestamp, pkg_frequency, layer, power_estimations[best_model]['error'], best_model))
        return power_reports, formula_reports

    def _gen_formula_report(self, timestamp: datetime, pkg_frequency: int, layer: FrequencyLayer, error: float, model_label: str) -> FormulaReport:
        """
        Generate a formula report using the given parameters.
        :param timestamp: Timestamp of the measurements
        :param pkg_frequency: Package average frequency
        :param error: Error rate of the model
        :param model_label: Label to identify the model (static or dynamic)
        :return: Formula report filled with the given parameters
        """
        metadata = {
            'scope': self.state.config.scope.value,
            'socket': self.state.socket,
            'layer_frequency': layer.model[model_label].frequency,
            'pkg_frequency': pkg_frequency,
            'samples': len(layer.samples_history),
            'id': layer.model[model_label].id,
            'error': error,
            'intercept': layer.model[model_label].clf.intercept_,
            'coef': str(layer.model[model_label].clf.coef_)
        }
        return FormulaReport(timestamp, self.state.sensor, layer.model[model_label].hash, metadata)

    def _gen_power_report(self, timestamp: datetime, target: str, formula: str, power: float, ratio: float, metadata: dict[str, Any]) -> PowerReport:
        """
        Generate a power report using the given parameters.
        :param timestamp: Timestamp of the measurements
        :param target: Target name
        :param formula: Formula identifier
        :param power: Power estimation
        :return: Power report filled with the given parameters
        """
        report_metadata = metadata | {
            'scope': self.state.config.scope.value,
            'socket': self.state.socket,
            'formula': formula,
            'ratio': ratio,
        }
        return PowerReport(timestamp, self.state.sensor, target, power, report_metadata)

    def _gen_rapl_events_group(self, system_report) -> dict[str, float]:
        """
        Generate an events group with the RAPL reference event converted in Watts for the current socket.
        :param system_report: The HWPC report of the System target
        :return: A dictionary containing the RAPL reference event with its value converted in Watts
        """
        cpu_events = next(iter(system_report.groups['rapl'][str(self.state.socket)].values()))
        energy = ldexp(cpu_events[self.state.config.rapl_event], -32) / (self.state.config.reports_frequency / 1000)
        return {self.state.config.rapl_event: energy}

    def _gen_msr_events_group(self, system_report) -> dict[str, float]:
        """
        Generate an events group with the average of the MSR counters for the current socket.
        :param system_report: The HWPC report of the System target
        :return: A dictionary containing the average of the MSR counters
        """
        msr_events_group = defaultdict(int)
        msr_events_count = defaultdict(int)
        for _, cpu_events in system_report.groups['msr'][str(self.state.socket)].items():
            for event_name, event_value in {k: v for k, v in cpu_events.items() if not k.startswith('time_')}.items():
                msr_events_group[event_name] += event_value
                msr_events_count[event_name] += 1
        return {k: (v / msr_events_count[k]) for k, v in msr_events_group.items()}

    def _gen_core_events_group(self, report) -> dict[str, float]:
        """
        Generate an events group with Core events for the current socket.
        The events value are the sum of the value for each CPU.
        :param report: The HWPC report of any target
        :return: A dictionary containing the Core events of the current socket
        """
        core_events_group = defaultdict(int)
        for _, cpu_events in report.groups['core'][str(self.state.socket)].items():
            for event_name, event_value in {k: v for k, v in cpu_events.items() if not k.startswith('time_')}.items():
                core_events_group[event_name] += event_value

        return core_events_group

    def _gen_agg_core_report_from_running_targets(self, targets_report) -> dict[str, float]:
        """
        Generate an aggregate Core events group of the running targets for the current socket.
        :param targets_report: List of Core events group of the running targets
        :return: A dictionary containing an aggregate of the Core events for the running targets of the current socket
        """
        agg_core_events_group = defaultdict(int)
        for _, target_report in targets_report.items():
            for event_name, event_value in self._gen_core_events_group(target_report).items():
                agg_core_events_group[event_name] += event_value

        return agg_core_events_group

    @staticmethod
    def _extract_events_value(events: dict[str, float]) -> list[float]:
        """
        Creates and return a list of events value from the events group.
        :param events: Events group
        :return: List containing the events value sorted by event name
        """
        return [value for _, value in sorted(events.items())]
