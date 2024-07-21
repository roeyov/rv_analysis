import torch
import torch.nn as nn
import numpy as np
from scipy.stats import binned_statistic
from scipy.signal import lombscargle

columns_to_select = "p", "", "", "", "", ""


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

    def prediction_distribution(self, df):
        # Calculate the binned accuracy for five parameters that define each feature vector
        # Assuming the first five parameters define each feature vector

        params = df[columns_to_select]

        feature_tensors = torch.tensor(df.features, dtype=torch.float32)

        # Make predictions
        with torch.no_grad():
            outputs = self.model(feature_tensors)
            decisions = (outputs > 0.5).numpy()

        # Calculate accuracy in bins
        accuracy_per_param = {}
        for c in columns_to_select:
            param_values = params[c]
            bins = np.linspace(param_values.min(), param_values.max(), num=10)
            bin_means, bin_edges, _ = binned_statistic(param_values, decisions, statistic='mean', bins=bins)
            accuracy_per_param[c] = (bin_edges, bin_means)

        return accuracy_per_param

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
        speeds = features['speeds']
        time = features['time']

        # Step a: Check conditions
        max_min_diff = np.max(speeds) - np.min(speeds)
        second_condition = self.placeholder_condition(features)  # Placeholder for the second condition
        binary_decision = False

        if max_min_diff > self.rv_diff and second_condition:
            binary_decision = True

        # Step b: Compute Lomb-Scargle periodogram
        freq = np.linspace(0.01, 10, len(time))
        power = lombscargle(time, speeds, freq)
        max_power_index = np.argmax(power)
        max_power_period = 1 / freq[max_power_index]

        fal = 0.1  # Placeholder for False Alarm Level
        if power[max_power_index] > fal:
            binary_decision = True
        elif not binary_decision:
            binary_decision = False
            max_power_period = np.inf

        if not binary_decision:
            return np.inf, False
        else:
            return max_power_period, binary_decision

    def placeholder_condition(self, features):
        a = features['speeds']
        b = features['errs']

        # Define the calculation function
        def calc(a_i, a_j, b_i, b_j):
            return np.abs(a_i - a_j ) / np.sqrt(b_i*b_i + b_j*b_j)

        # Create the meshgrid of indices
        i_indices, j_indices = np.meshgrid(np.arange(len(a)), np.arange(len(a)), indexing='ij')
        # Apply the calculation to each pair of indices using vectorized operations
        result_matrix = calc(a[i_indices], a[j_indices], b[i_indices], b[j_indices])
        # Flatten the result matrix to find the maximum value
        max_result = np.max(result_matrix)
        return max_result > self.sample_significance

    def prediction_distribution(self, df):
        # Assuming the first five parameters define each feature vector
        params = df[columns_to_select]
        binary_decisions = []
        periods = []

        # Loop through each feature vector and make predictions
        for features in df:
            period, decision = self.predict({'speeds': features['speeds'], 'time': features['time']})
            binary_decisions.append(decision)

        # Calculate binned accuracy
        accuracy_per_param = {}
        for c in columns_to_select:
            param_values = params[c]
            bins = np.linspace(param_values.min(), param_values.max(), num=10)
            bin_means, bin_edges, _ = binned_statistic(param_values, binary_decisions, statistic='mean', bins=bins)
            accuracy_per_param[c] = (bin_edges, bin_means)
        out_dict = {
            PRED_IS_BINARY: binary_decisions
            PRED_PERIOD: 
        }
        return binary_decisions,periods, accuracy_per_param

    def model_type(self):
        # Return the type of the class as a string
        return str(type(self))


