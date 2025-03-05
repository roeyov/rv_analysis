import os

import pandas as pd
from torch.utils.data import Dataset, DataLoader
import numpy as np
import torch
import torch.nn as nn

class ParquetLoader(Dataset):
    def __init__(self, true_dir, false_dir):
        self.data = []
        self.labels = []
        self.periods = []

        # Load true data
        for filename in os.listdir(true_dir):
            file_path = os.path.join(true_dir, filename)
            if file_path.endswith('.parquet'):
                df = pd.read_parquet(file_path)
                self.data.append(df.features)  # Assuming df is structured correctly
                self.labels.append(df.labels)  # True label
                self.periods.append(df.NumOfPeriods)


        # Load false data
        for filename in os.listdir(false_dir):
            file_path = os.path.join(false_dir, filename)
            if file_path.endswith('.parquet'):
                df = pd.read_parquet(file_path)
                self.data.append(df.features)  # Assuming df is structured correctly
                self.labels.append(df.labels)  # True label
                self.periods.append(df.NumOfPeriods)

        # Convert to numpy arrays for processing
        # print(self.data)

        self.data = np.stack(np.array(pd.concat(self.data, axis=0)), axis = 0)
        self.labels = np.stack(np.array(pd.concat(self.labels, axis=0)), axis=0).reshape(-1,1)
        self.periods = np.stack(np.array(pd.concat(self.periods, axis=0)), axis=0).reshape(-1,1)
        # print("\n\n" , self.data)

        # Normalize data
        scaler = StandardScaler()
        self.data = scaler.fit_transform(self.data.reshape(-1, self.data.shape[-1])).reshape(self.data.shape)

        # Convert to PyTorch tensors
        self.data = torch.tensor(self.data, dtype=torch.float32)
        self.labels = torch.tensor(self.labels, dtype=torch.float32)
        self.periods = torch.tensor(self.periods, dtype=torch.float32)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return self.data[idx], self.labels[idx], self.periods[idx]

# Example usage
# true_dir = '/content/drive/My Drive/PythonScripts/scriptsOut/RVDataGen/Trues/'
# false_dir = '/content/drive/My Drive/PythonScripts/scriptsOut/RVDataGen/Falses/'
# dataset = CSVDataset(true_dir, false_dir)
# dataloader = DataLoader(dataset, batch_size=10, shuffle=True)

# Iterating through the DataLoader
# for features, labels in dataloader:
    # print(features, labels)  # features are the input tensors, labels are the target labels