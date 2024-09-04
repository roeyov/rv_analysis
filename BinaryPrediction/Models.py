import torch
import torch.nn as nn
import numpy as np
from scipy.stats import binned_statistic
from scipy.signal import lombscargle

from Scripts.utils.constants import *
from Scripts.utils.periodagram import ls, plotls, pdc

columns_to_select = [PERIOD, RADIAL_VELS, TIME_STAMPS, ERRORS]


def lomb_scargle(row):
    pmin = 2.0
    pmax = 1000.0
    best_period1, fap1, fal1, frequency1, power1 = ls(row[TIME_STAMPS], row[RADIAL_VELS], row[ERRORS],
                                                      probabilities=[0.5, 0.01, 0.001], pmin=pmin, pmax=pmax,
                                                      norm="model", ls_method="fast", fa_method="baluev",
                                                      samples_per_peak=1000)
    # plotls(frequency1, power1, fal1, pmin=1.0, pmax=10000)
    pdc_best_period, pdc_fap, pdc_freq, pdc_power_reg = pdc(row[TIME_STAMPS], row[RADIAL_VELS], data_err=row[ERRORS], pmin=pmin, pmax=pmax)
    # print("1 ")
    return best_period1, fap1, fal1, np.max(power1), pdc_best_period, pdc_fap, np.max(pdc_power_reg)


def significance(row):
    a = row[RADIAL_VELS]
    b = row[ERRORS]
    n = len(a)
    max_value = 0
    for i in range(n):
        for j in range(i + 1, n):  # Start j from i+1 to avoid redundant pairs
            numerator = abs(a[i] - a[j])
            denominator = np.sqrt(b[i] ** 2 + b[j] ** 2)
            value = numerator / denominator
            if value > max_value:
                max_value = value

    return max_value


class PyTorchModel:
    def __init__(self, model_class, model_path):
        # Initialize the model and load state_dict from the provided path
        self.model = model_class()
        self.model.load_state_dict(torch.load(model_path))
        self.model.eval()  # Set model to evaluation mode

    def predict(self, features):
        # Make a prediction and return period and binary decision
        with torch.no_grad():
            features_tensor = torch.tensor(features, dtype=torch.float32)
            output = self.model(features_tensor)
            period = 1.0 / torch.abs(output).item()  # Example: inverse of the model output
            binary_decision = (output > 0.5).item()  # Example binary decision
        return period, binary_decision


    def model_type(self):
        # Return a string of the class' type
        return str(type(self))


class ClassicModel:
    def __init__(self, rv_diff, sample_significance):
        # Initialize the model with parameters
        self.rv_diff = rv_diff
        self.sample_significance = sample_significance

    def predict(self, features):
        # Features should include speeds and other necessary data
        speeds = features[RADIAL_VELS]
        time = features[TIME_STAMPS]

        # Step a: Check conditions
        features[MAX_MIN_DIFF] = features[RADIAL_VELS].apply(lambda x: np.max(x) - np.min(x))
        features[SIGNIFICANCE] = features.apply(significance, axis=1)  # Placeholder for the second condition

        # Step b: Compute Lomb-Scargle periodogram
        features[BEST_PERIOD], \
            features[FALSE_ALARM_PROBABILITY], \
            features[FALSE_ALARM_LEVELS], \
            features[BEST_PERIOD_POWER], \
            features[PDC_BEST_PERIOD], \
            features[PDC_FALSE_ALARM_PROBABILITY], \
            features[PDC_BEST_PERIOD_POWER] = zip(*features.apply(lomb_scargle, axis=1))

    def prediction_distribution(self, df):
        # Assuming the first five parameters define each feature vector
        params = df[columns_to_select]
        binary_decisions = []
        periods = []

        # Loop through each feature vector and make predictions
        for features in df:
            period, decision = self.predict({RVS: features['speeds'], 'time': features['time']})
            binary_decisions.append(decision)
            periods.append(period)

        # Calculate binned accuracy
        accuracy_per_param = {}
        for c in columns_to_select:
            param_values = params[c]
            bins = np.linspace(param_values.min(), param_values.max(), num=10)
            bin_means, bin_edges, _ = binned_statistic(param_values, binary_decisions, statistic='mean', bins=bins)
            accuracy_per_param[c] = (bin_edges, bin_means)

        return binary_decisions, periods, accuracy_per_param

    def model_type(self):
        # Return the type of the class as a string
        return str(type(self))
