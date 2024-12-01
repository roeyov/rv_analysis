import numpy as np
import pandas as pd
import os
import pickle
import joblib

import torch
import torch.optim as optim
from sklearn.preprocessing import StandardScaler
from torch.utils.data import Dataset, DataLoader,TensorDataset
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from Scripts.BinaryPrediction.ModelsNN import MultiOutputNN


class CSVDataset(Dataset):
    def __init__(self, true_dir, false_dir):
        self.data = []
        self.labels = []
        self.periods = []
        self.main_df = []

        # Load true data
        for filename in os.listdir(true_dir):
            file_path = os.path.join(true_dir, filename)
            if True:
                df = pd.read_parquet(file_path)
                bool_arr =   df.labels > -1
                # bool_arr =  df.apply(lambda row: (np.max(row.rvs) - np.min(row.rvs)) < 20, axis=1)
                self.main_df.append(df[bool_arr])
                self.data.append(df.features[bool_arr])  # Assuming df is structured correctly
                self.labels.append(df.labels[bool_arr])  # True label
                self.periods.append(np.log10(df.Period[bool_arr])/3)


        # Load false data
        for filename in os.listdir(false_dir):
            file_path = os.path.join(false_dir, filename)
            if True:
                df = pd.read_parquet(file_path)
                bool_arr =   df.labels > -1
                # bool_arr =  df.apply(lambda row: (np.max(row.rvs) - np.min(row.rvs)) < 20, axis=1)
                self.main_df.append(df[bool_arr])
                self.data.append(df.features[bool_arr])  # Assuming df is structured correctly
                self.labels.append(df.labels[bool_arr])  # True label
                self.periods.append(np.log10(df.Period[bool_arr])/3)

        # Convert to numpy arrays for processing
        # print(self.data)
        self.main_df = pd.concat(self.main_df, axis=0)
        self.data = np.stack(np.array(pd.concat(self.data, axis=0)), axis = 0)
        self.labels = np.stack(np.array(pd.concat(self.labels, axis=0)), axis=0).reshape(-1,1)
        self.periods = np.stack(np.array(pd.concat(self.periods, axis=0)), axis=0).reshape(-1,1)
        # print("\n\n" , self.data)

        # Normalize data
        # scaler = StandardScaler()
        # self.data = scaler.fit_transform(self.data.reshape(-1, self.data.shape[-1])).reshape(self.data.shape)

        # Convert to PyTorch tensors
        self.data = torch.tensor(self.data, dtype=torch.float32)
        self.labels = torch.tensor(self.labels, dtype=torch.float32)
        self.periods = torch.tensor(self.periods, dtype=torch.float32)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return self.data[idx], self.labels[idx], self.periods[idx]

def train_model(model, train_loader, test_loader, bin_criterion, flt_criterion,
                optimizer, num_epochs=25):
    test_logs = {}
    train_logs = {}
    for epoch in range(num_epochs):
        model.train()

        running_loss = 0.0

        for inputs, binary_labels, float_labels in train_loader:
            # inputs, binary_labels, float_labels = inputs.to(device), binary_labels.to(device), float_labels.to(device)

            optimizer.zero_grad()
            bin_outputs, flt_outputs = model(inputs)

            loss_binary = bin_criterion(bin_outputs, binary_labels)

            # Create a mask for examples where binary_true is False
            mask = (binary_labels == 1)

            # Apply the mask to float predictions and true labels
            masked_flt_outputs = flt_outputs[mask]
            masked_flt_labels = float_labels[mask]

            # Compute MSE loss only for masked examples
            if len(masked_flt_outputs) > 0:  # To avoid calculating loss if no masked elements
                loss_float = flt_criterion(masked_flt_outputs, masked_flt_labels)
            else:
                loss_float = torch.tensor(0.0)  # Set MSE loss to zero if no masked examples
            train_logs[epoch] = [loss_binary.item(), loss_float.item()]
            loss = loss_binary + 5*loss_float

            loss.backward()
            optimizer.step()
            running_loss += loss.item()
        # scheduler.step()

        print(f'Epoch {epoch + 1}, Train Loss: {running_loss / len(train_loader)}')
        # current_lr = scheduler.get_last_lr()[0]
        # print(f"Epoch {epoch + 1}, Learning Rate: {current_lr}")
        model.eval()
        all_bin_preds = []
        all_flt_preds = []
        all_bin_labels = []
        all_flt_labels = []
        all_bin_probs = []

        with torch.no_grad():
            for inputs, binary_labels, float_labels  in test_loader:
                # inputs, binary_labels, float_labels = inputs.to(device), binary_labels.to(device), float_labels.to(device)
                bin_outputs, flt_outputs = model(inputs)
                bin_preds = (bin_outputs > 0.5).float()
                all_bin_preds.extend(bin_preds.cpu().numpy())
                all_bin_probs.extend(bin_outputs.cpu().numpy())

                flt_preds = flt_outputs.float()
                all_flt_preds.extend(flt_preds.cpu().numpy())

                all_bin_labels.extend(binary_labels.cpu().numpy())
                all_flt_labels.extend(float_labels.cpu().numpy())

        accuracy = accuracy_score(all_bin_labels, all_bin_preds)
        loss_binary = bin_criterion(torch.tensor(np.array(all_bin_probs)), torch.tensor(np.array(all_bin_labels)))
        print(f'Epoch {epoch + 1}, Test Binary Accuracy: {accuracy:.4f}')
        mask = (all_bin_labels == 1)

        # Apply the mask to float predictions and true labels
        masked_flt_outputs = all_flt_preds[mask]
        masked_flt_labels = all_flt_labels[mask]

        # Compute MSE loss only for masked examples
        if len(masked_flt_outputs) > 0:  # To avoid calculating loss if no masked elements
            loss_float = flt_criterion(torch.tensor(masked_flt_outputs), torch.tensor(masked_flt_labels))
        else:
            loss_float = torch.tensor(0.0)  # Set MSE loss to zero if no masked examples
        test_logs[epoch] = [accuracy, loss_float.item()]
        print(f'Epoch {epoch + 1}, Test Period Loss: {loss_float:.4f}')
        print(f'Epoch {epoch + 1}, Test Loss: {(loss_float+loss_binary):.4f}')

    print('Finished Training')
    return train_logs, test_logs


def main():
    # Example usage
    data_name = "train_data_2_2000000"
    BASE_OUT_DIR = r"C:\Users\roeyo\Documents\Roey's\Masters\Reasearch\scriptsOut\torchModelsLocal"
    true_dir = r"C:\Users\roeyo\Documents\Roey's\Masters\Reasearch\scriptsOut\RVDataGen\{}_Trues".format(data_name)
    false_dir = r"C:\Users\roeyo\Documents\Roey's\Masters\Reasearch\scriptsOut\RVDataGen\{}_Falses".format(data_name)
    model_to_train =  MultiOutputNN
    dataset = CSVDataset(true_dir, false_dir)
    identifier = "5PeriodLoss"
    out_dir = os.path.join(BASE_OUT_DIR,  data_name + "_" + identifier)
    os.makedirs(out_dir)
    X = dataset.data
    y_binary = dataset.labels
    y_float = dataset.periods
    # Normalize the data
    scaler = StandardScaler()
    X = scaler.fit_transform(X)

    X = torch.tensor(X, dtype=torch.float32)
    y_float = torch.tensor(y_float, dtype=torch.float32)
    y_binary = torch.tensor(y_binary, dtype=torch.float32)
    # Split the data
    train_idx, test_idx = train_test_split(np.arange(len(y_binary)), test_size=0.2)
    train_mask = np.zeros(len(X), dtype=bool)
    test_mask = np.zeros(len(X), dtype=bool)

    train_mask[train_idx] = True
    test_mask[test_idx] = True

    x_train = X[train_mask]
    x_test = X[test_mask]
    y_binary_train = y_binary[train_mask]
    y_binary_test = y_binary[test_mask]
    y_float_train = y_float[train_mask]
    y_float_test = y_float[test_mask]

    # Ensure y_train and y_test are the correct shape
    print(f"X_train shape: {x_train.shape}, y_train shape: {y_binary_train.shape}")
    print(f"X_test shape: {x_test.shape}, y_test shape: {y_binary_test.shape}")
    # Create DataLoader
    train_dataset = TensorDataset(x_train, y_binary_train, y_float_train)
    test_dataset = TensorDataset(x_test, y_binary_test, y_float_test)

    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)
    # Initialize the model, loss functions, and optimizer
    train_zeros = torch.sum(y_binary_train == 0).item()
    train_ones = torch.sum(y_binary_train == 1).item()

    # Count number of 0s and 1s in y_binary_test
    test_zeros = torch.sum(y_binary_test == 0).item()
    test_ones = torch.sum(y_binary_test == 1).item()

    # Print counts
    print(f"y_train: 0s = {train_zeros}, 1s = {train_ones}")
    print(f"y_test: 0s = {test_zeros}, 1s = {test_ones}")
    model = model_to_train()
    pos_weight = torch.tensor([5.0])  # Weigh True labels 2x more than False labels

    # Using BCEWithLogitsLoss
    # criterion_binary = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    criterion_binary = torch.nn.BCELoss()  # Binary Cross-Entropy Loss for binary output
    criterion_float = torch.nn.MSELoss()  # Mean Squared Error Loss for float output
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    # scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=4, gamma=0.1)  # Decay LR by 0.1 every 4 epochs

    # Train the model
    train_l, test_l = train_model(model, train_loader, test_loader, criterion_binary,
                                  criterion_float, optimizer, num_epochs=20)
    model_path = os.path.join(out_dir, "model.pth")
    test_log_path =  os.path.join(out_dir, "test_log.pickle")
    train_log_path =  os.path.join(out_dir, "train_log.pickle")
    scal  =  os.path.join(out_dir, "scaler.pickle")

    with open(test_log_path, 'wb') as f:
        pickle.dump(test_l, f)
    with open(train_log_path, 'wb') as f:
        pickle.dump(train_l, f)

    joblib.dump(scaler, scal)

    torch.save(model, model_path)

if __name__ == '__main__':
    main()



