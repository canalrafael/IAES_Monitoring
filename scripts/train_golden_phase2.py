import pandas as pd
import numpy as np
import os
import glob
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import time

# --- Configuration (Phase 2 Parity) ---
DATA_DIR    = '/home/canal/github/IAES_Monitoring/data/online validation data'
EXPORT_PATH = '/home/canal/github/IAES_Monitoring/deploy/online_validation/model_weights_golden.h'

BEST_W       = 10
BEST_LAYERS  = 2
BEST_UNITS   = 16
BEST_LR      = 2e-3
BEST_PW_MULT = 2.0
MAX_EPOCHS   = 1000  # High-intensity training
RATIO_SIGNALS = ['IPC', 'MPKI', 'L2_PRESSURE', 'BRANCH_MISS_RATE']

# --- Feature Engineering (Phase 2 Parity) ---
def engineer_features(df, w):
    df = df.copy()
    eps = 1e-9
    
    # Calculate ratios from raw counters
    df['IPC']              = df['INSTRUCTIONS']    / (df['CPU_CYCLES']      + eps)
    df['MPKI']             = (df['CACHE_MISSES'] * 1000) / (df['INSTRUCTIONS'] + eps)
    df['L2_PRESSURE']      = df['L2_CACHE_ACCESS'] / (df['CPU_CYCLES']      + eps)
    df['BRANCH_MISS_RATE'] = df['BRANCH_MISSES']   / (df['INSTRUCTIONS']    + eps)
    
    feat_cols = []
    for col in RATIO_SIGNALS:
        r = df[col].rolling(window=w)
        df[f'{col}_mean']  = r.mean()
        df[f'{col}_std']   = r.std().fillna(0)
        df[f'{col}_delta'] = (df[col] - df[col].shift(w - 1)).fillna(0)
        feat_cols += [f'{col}_mean', f'{col}_std', f'{col}_delta']

    df_c = df.dropna()
    # Use ALL labels for context
    # Target: Label 2 (Secure Core suffering interference)
    # Background: Labels 0, 1, 3
    X = df_c[feat_cols].values.astype(np.float32)
    y = (df_c['LABEL'] == 2).values.astype(np.float32)
    return X, y, feat_cols

# --- Model (Phase 2 Parity) ---
class MLP(nn.Module):
    def __init__(self, in_dim, layers, units):
        super().__init__()
        seq, last = [], in_dim
        for _ in range(layers):
            seq += [nn.Linear(last, units), nn.ReLU()]
            last = units
        seq.append(nn.Linear(last, 1))
        self.net = nn.Sequential(*seq)

    def forward(self, x):
        return self.net(x)

def main():
    print(">>> Starting Golden Phase 2 Pipeline...")
    all_files = sorted(glob.glob(os.path.join(DATA_DIR, '*.csv')))
    
    X_list, y_list = [], []
    for f in all_files:
        df = pd.read_csv(f)
        X, y, _ = engineer_features(df, BEST_W)
        X_list.append(X)
        y_list.append(y)
    
    X_all = np.vstack(X_list)
    y_all = np.concatenate(y_list)
    
    print(f"Total Samples: {len(X_all):,}")
    print(f"Attack Samples (Label 3): {int(y_all.sum()):,}")
    
    # Standard Scaler
    scaler = StandardScaler()
    X_s = scaler.fit_transform(X_all)
    
    # Train/Val Split
    X_train, X_val, y_train, y_val = train_test_split(X_s, y_all, test_size=0.2, stratify=y_all, random_state=42)
    
    tr_ldr = DataLoader(TensorDataset(torch.from_numpy(X_train), torch.from_numpy(y_train).view(-1,1)), batch_size=2048, shuffle=True)
    va_ldr = DataLoader(TensorDataset(torch.from_numpy(X_val), torch.from_numpy(y_val).view(-1,1)), batch_size=4096)

    # Class Weights
    class_ratio = (y_train == 0).sum() / max((y_train == 1).sum(), 1)
    pw = torch.tensor([BEST_PW_MULT * class_ratio])
    
    model = MLP(12, BEST_LAYERS, BEST_UNITS)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pw)
    optimizer = optim.Adam(model.parameters(), lr=BEST_LR)
    
    best_loss = float('inf')
    for epoch in range(MAX_EPOCHS + 1):
        model.train()
        for bX, by in tr_ldr:
            optimizer.zero_grad()
            loss = criterion(model(bX), by)
            loss.backward()
            optimizer.step()
        
        if epoch % 50 == 0:
            model.eval()
            val_loss = 0
            with torch.no_grad():
                for bX, by in va_ldr:
                    val_loss += criterion(model(bX), by).item()
            val_loss /= len(va_ldr)
            print(f"Epoch {epoch} | Val Loss: {val_loss:.4f}")
            if val_loss < best_loss:
                best_loss = val_loss
                torch.save(model.state_dict(), 'best_golden.pth')

    # Export Logic
    model.load_state_dict(torch.load('best_golden.pth'))
    params = list(model.parameters())
    w1, b1 = params[0].detach().numpy(), params[1].detach().numpy()
    w2, b2 = params[2].detach().numpy(), params[3].detach().numpy()
    w3, b3 = params[4].detach().numpy(), params[5].detach().numpy()
    
    with open(EXPORT_PATH, 'w') as f:
        f.write("/* GOLDEN PHASE 2 MODEL */\n#ifndef MODEL_WEIGHTS_H\n#define MODEL_WEIGHTS_H\n\n")
        f.write("#define MDL_N_FEATURES 12\n#define MDL_N_H1 16\n#define MDL_N_H2 16\n#define MDL_N_OUT 1\n")
        f.write("#define MDL_WINDOW_SIZE 10\n#define MDL_TEMPERATURE 1.0f\n#define MDL_THRESHOLD 0.7f\n\n")
        f.write(f"static const float MDL_FEAT_MEAN[12] = {{{', '.join([f'{x}f' for x in scaler.mean_])}}};\n")
        f.write(f"static const float MDL_FEAT_STD[12] = {{{', '.join([f'{x}f' for x in scaler.scale_])}}};\n\n")
        def wm(n, m):
            f.write(f"static const float {n}[{m.shape[0]}][{m.shape[1]}] = {{\n")
            for r in m: f.write(f"    {{{', '.join([f'{x}f' for x in r])}}},\n")
            f.write("};\n\n")
        def wv(n, v): f.write(f"static const float {n}[{len(v)}] = {{{', '.join([f'{x}f' for x in v])}}};\n\n")
        wm("MDL_W1", w1); wv("MDL_B1", b1)
        wm("MDL_W2", w2); wv("MDL_B2", b2)
        wm("MDL_W3", w3); wv("MDL_B3", b3)
        f.write("#endif\n")
    print(f">>> Golden Model Exported to {EXPORT_PATH}")

if __name__ == '__main__':
    main()
