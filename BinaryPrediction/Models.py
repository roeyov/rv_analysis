import torch
import torch.nn as nn
import numpy as np
from scipy.stats import binned_statistic
from scipy.signal import lombscargle
import joblib
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
    pdc_best_period, pdc_fap, pdc_freq, pdc_power_reg = pdc(row[TIME_STAMPS], row[RADIAL_VELS], data_err=row[ERRORS],
                                                            pmin=pmin, pmax=pmax)
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
    def __init__(self, model_path, scaler_path):
        # Initialize the model and load state_dict from the provided path
        self.model = torch.load(model_path)
        self.model.eval()  # Set the model to evaluation mode

        # Load the scaler
        self.scaler = joblib.load(scaler_path)

    def predict(self, features):
        features[FEATURES] = features.apply(lambda row: np.concatenate([row[RADIAL_VELS],
                                                                        row[TIME_STAMPS],
                                                                        np.ediff1d(row[TIME_STAMPS]),
                                                                        [3.0,
                                                                         np.max(row[RADIAL_VELS]) - np.min(row[RADIAL_VELS]),
                                                                         np.mean(row[RADIAL_VELS]),
                                                                         np.std(row[RADIAL_VELS])]]), axis=1)
        # Make a prediction and return period and binary decision
        X = np.vstack(features[FEATURES].values)

        X = self.scaler.transform(X)
        with torch.no_grad():
            features_tensor = torch.tensor(X, dtype=torch.float32)
            bin_outputs, flt_outputs = self.model(features_tensor)
            period = 10 ** (flt_outputs * 3).float()  # Example: inverse of the model output
            binary_decision = bin_outputs.float()  # Example binary decision
            x = self.model.relu(self.model.layer1(features_tensor))
            x = self.model.relu(self.model.layer2(x))
            x = self.model.relu(self.model.layer3(x))
            # Get pre-sigmoid output for the binary prediction
            binary_output_before_sigmoid =  self.model.output_is_bianry(x)

        features[NN_PERIOD], features[NN_DECISION], features[NN_DECISION_PRE] = period, binary_decision, binary_output_before_sigmoid

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


class MultiOutputNN(nn.Module):
    def __init__(self):
        super(MultiOutputNN, self).__init__()
        self.layer1 = nn.Linear(75, 256)
        self.relu = nn.ReLU()
        self.layer3 = nn.Linear(256, 64)
        self.output_is_bianry = nn.Linear(64, 1)
        self.output_num_of_ps = nn.Linear(64, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.relu(self.layer1(x))
        # x = self.relu(self.layer2(x))
        x = self.relu(self.layer3(x))
        float_output = self.output_num_of_ps(x)
        # Compute the binary output and apply sigmoid and threshold
        binary_output = self.sigmoid(self.output_is_bianry(x))
        return binary_output, float_output
