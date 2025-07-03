import os
import glob
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from utils.constants import *
from torch.utils.tensorboard import SummaryWriter
from sklearn.metrics import confusion_matrix, roc_curve, auc
import matplotlib.pyplot as plt
import shutil

# TensorBoard writer

# --- Plot helpers ---
def plot_confusion(cm):
    fig, ax = plt.subplots()
    im = ax.imshow(cm, cmap='Blues')
    ax.set_xlabel('Predicted')
    ax.set_ylabel('True')
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, cm[i,j], ha='center', va='center', color='white')
    return fig


def plot_roc(fpr, tpr, roc_auc):
    fig, ax = plt.subplots()
    ax.plot(fpr, tpr, label=f"AUC = {roc_auc:.3f}")
    ax.plot([0,1],[0,1],'--', color='gray')
    ax.set_xlabel('FPR')
    ax.set_ylabel('TPR')
    ax.set_title('ROC Curve')
    ax.legend(loc='lower right')
    return fig

# --- Data loading function ---
def load_data(sample_dir, results_dir,input_pattern, debug=False, debug_limit=5):
    frames = []
    pattern = os.path.join(results_dir, input_pattern , '*.parquet')
    all_res_files = sorted(glob.glob(pattern))
    if debug:
        all_res_files = all_res_files[:debug_limit]

    for res_path in all_res_files:
        noise_dir = os.path.basename(os.path.dirname(res_path))
        file_name = os.path.basename(res_path)
        sam_path = os.path.join(sample_dir, noise_dir, file_name)
        parts = noise_dir.split('_')
        try:
            amplitude = float(parts[2])
        except (IndexError, ValueError):
            amplitude = np.nan
        df_res = pd.read_parquet(
            res_path,
            columns=[PDC_BEST_PERIOD_POWER, BEST_PERIOD_POWER, MAX_MIN_DIFF, SIGNIFICANCE]
        )
        df_sam = pd.read_parquet(
            sam_path,
            columns=[PERIOD, LABELS, ECC, K1_STR,FEATURES, IS_TRAIN]
        )
        df = pd.concat(
            [df_res.reset_index(drop=True), df_sam.reset_index(drop=True)],
            axis=1
        )
        df.loc[df[SIGNIFICANCE] < 4, MAX_MIN_DIFF] = 2
        df['noise_dir'] = noise_dir
        df['noise_amplitude'] = amplitude
        frames.append(df)

    return pd.concat(frames, ignore_index=True)

# --- PyTorch Dataset ---
class RVDataset(Dataset):
    def __init__(self, df):
        scalar_feats = df[[MAX_MIN_DIFF,
                            PDC_BEST_PERIOD_POWER,
                            BEST_PERIOD_POWER,
                            SIGNIFICANCE]].values
        array_feats = np.stack(df[FEATURES].values, axis=0)
        X_all = np.hstack([scalar_feats, array_feats])
        self.X = torch.from_numpy(X_all).float()
        self.y = torch.tensor(df[LABELS].values, dtype=torch.float32).unsqueeze(1)

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

# --- Simple DNN Model ---
class SimpleDNN(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 32),
            nn.ReLU(),
            nn.Linear(32, 16),
            nn.ReLU(),
            nn.Linear(16, 1),
            nn.Sigmoid()
        )
    def forward(self, x):
        return self.net(x)

# --- Training loop ---
def train_model(model, dataloader, criterion, optimizer, device):
    model.train()
    total_loss = 0.0
    for X_batch, y_batch in dataloader:
        X_batch, y_batch = X_batch.to(device), y_batch.to(device)
        optimizer.zero_grad()
        outputs = model(X_batch)
        loss = criterion(outputs, y_batch)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * X_batch.size(0)
    return total_loss / len(dataloader.dataset)

# --- Evaluation ---
def evaluate_model(model, dataloader, device):
    model.eval()
    preds, trues = [], []
    with torch.no_grad():
        for X_batch, y_batch in dataloader:
            X_batch = X_batch.to(device)
            outputs = model(X_batch)
            preds.append(outputs.cpu())
            trues.append(y_batch)
    return torch.cat(preds), torch.cat(trues)

# --- Main script ---
def main():
    device = torch.device("mps") if torch.backends.mps.is_available() else torch.device("cpu")
    print(f"Using device: {device}")

    samples_dir = "/Users/roeyovadia/Documents/Data/simulatedData/noise_full/samples"
    results_dir = "/Users/roeyovadia/Documents/Data/simulatedData/noise_full/periodogram"
    out_dir = "/Users/roeyovadia/Roey/Masters/Reasearch/scriptsOut/trainedModels/experiment_2/"
    writer = SummaryWriter(log_dir=out_dir)
    script_path = os.path.abspath(__file__)
    os.makedirs(writer.log_dir, exist_ok=True)
    script_dest = os.path.join(writer.log_dir, os.path.basename(script_path))
    shutil.copy(script_path, script_dest)
    print(f"Saved training script to {script_dest}")

    df = load_data(samples_dir, results_dir, input_pattern="noise_analysis_*", debug=False)

    train_df = df[df[IS_TRAIN] == True].reset_index(drop=True)
    test_df  = df[df[IS_TRAIN] == False].reset_index(drop=True)
    train_loader = DataLoader(RVDataset(train_df), batch_size=64, shuffle=True)
    test_loader  = DataLoader(RVDataset(test_df),  batch_size=64)

    model = SimpleDNN(input_dim=train_loader.dataset.X.shape[1]).to(device)
    criterion = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    # learning rate scheduler: decrease LR by factor 0.5 every 20 epochs
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=50, gamma=0.5)

    # save every M epochs
    save_every = 50
    epochs = 1000
    for epoch in range(1, epochs+1):
        train_loss = train_model(model, train_loader, criterion, optimizer, device)

        # evaluation
        test_loss = 0.0
        all_preds, all_trues = [], []
        model.eval()
        with torch.no_grad():
            for Xb, yb in test_loader:
                Xb, yb = Xb.to(device), yb.to(device)
                out = model(Xb)
                test_loss += criterion(out, yb).item() * Xb.size(0)
                all_preds.append(out.cpu())
                all_trues.append(yb.cpu())
        test_loss /= len(test_loader.dataset)
        preds = torch.cat(all_preds).numpy().flatten()
        trues = torch.cat(all_trues).numpy().flatten()
        preds_binary = (preds >= 0.5).astype(int)
        cm = confusion_matrix(trues, preds_binary)
        fpr, tpr, _ = roc_curve(trues, preds)
        roc_auc = auc(fpr, tpr)

        # log metrics
        writer.add_scalar('Loss/Train', train_loss, epoch)
        writer.add_scalar('Loss/Test', test_loss, epoch)
        writer.add_scalar('ROC/AUC', roc_auc, epoch)
        writer.add_figure('Confusion Matrix', plot_confusion(cm), epoch)
        writer.add_figure('ROC Curve', plot_roc(fpr, tpr, roc_auc), epoch)

        print(f"Epoch {epoch}/{epochs}  train_loss={train_loss:.4f}  test_loss={test_loss:.4f}  AUC={roc_auc:.3f}")

        # save model checkpoint
        if epoch % save_every == 0:
            ckpt_path = f"{out_dir}model_epoch_{epoch}.pth"
            torch.save(model.state_dict(), ckpt_path)
            print(f"Saved checkpoint: {ckpt_path}")

    # final save
    torch.save(model.state_dict(), f"{out_dir}model_final.pth")
    print(f"Saved final model: {out_dir}model_final.pth")
    writer.close()

if __name__ == '__main__':
    main()
