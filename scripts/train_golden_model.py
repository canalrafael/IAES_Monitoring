import os
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import glob

# --- Configuration ---
DATA_DIR = "/home/canal/github/IAES_Monitoring/data/online validation data"
EXPORT_PATH = "/home/canal/github/IAES_Monitoring/deploy/online_validation/model_weights_golden.h"

# 1. Load All Data
files = glob.glob(os.path.join(DATA_DIR, "*.csv"))
print(f"Loading {len(files)} files for Golden Training...")

dfs = []
for f in files:
    df = pd.read_csv(f)
    dfs.append(df)

full_df = pd.concat(dfs, ignore_index=True)

# 2. Preprocessing
# Features: IPC, MPKI, L2_PRESSURE, BRANCH_MISS_RATE (and their rolling mean/std/delta)
# The current dataset should already have these or the raw counters
features = [
    'IPC', 'MPKI', 'L2_PRESSURE', 'BRANCH_MISS_RATE',
    'IPC_mean', 'IPC_std', 'IPC_delta',
    'MPKI_mean', 'MPKI_std', 'MPKI_delta',
    'L2_PRESSURE_mean', 'L2_PRESSURE_std'
]

# Ensure features exist, if not, calculate them (Simulating detector.c logic)
def augment_features(df):
    for feat in ['IPC', 'MPKI', 'L2_PRESSURE', 'BRANCH_MISS_RATE']:
        df[f'{feat}_mean'] = df[feat].rolling(window=10, min_periods=1).mean()
        df[f'{feat}_std'] = df[feat].rolling(window=10, min_periods=1).std().fillna(0)
        df[f'{feat}_delta'] = df[feat].diff().fillna(0)
    return df

# We need exactly 12 features for the C detector structure
target_features = [
    'IPC', 'MPKI', 'L2_PRESSURE', 'BRANCH_MISS_RATE',
    'IPC_mean', 'IPC_std', 'IPC_delta',
    'MPKI_mean', 'MPKI_std', 'L2_PRESSURE_mean', 'L2_PRESSURE_std', 'BRANCH_MISS_RATE_mean'
]

full_df = augment_features(full_df)
X = full_df[target_features].values
y = (full_df['LABEL'] == 3).astype(float).values # Binary: Attack Core (3) vs everything else

# 3. Train/Test Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# 4. Model Definition (Must match detector.c architecture)
class AnomalyMLP(nn.Module):
    def __init__(self):
        super(AnomalyMLP, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(12, 16),
            nn.ReLU(),
            nn.Linear(16, 16),
            nn.ReLU(),
            nn.Linear(16, 1)
        )
    def forward(self, x):
        return self.net(x)

model = AnomalyMLP()
criterion = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([5.0])) # Heavily weight the attacks
optimizer = optim.Adam(model.parameters(), lr=0.001)

# 5. Training Loop
print("Training Golden Model...")
X_t = torch.FloatTensor(X_train)
y_t = torch.FloatTensor(y_train).view(-1, 1)

for epoch in range(5001):
    optimizer.zero_grad()
    outputs = model(X_t)
    loss = criterion(outputs, y_t)
    loss.backward()
    optimizer.step()
    if epoch % 500 == 0:
        print(f"Epoch {epoch}, Loss: {loss.item():.4f}")

# 6. Export to C Header
def export_to_c(model, scaler, path):
    params = list(model.parameters())
    w1, b1 = params[0].detach().numpy(), params[1].detach().numpy()
    w2, b2 = params[2].detach().numpy(), params[3].detach().numpy()
    w3, b3 = params[4].detach().numpy(), params[5].detach().numpy()
    
    with open(path, 'w') as f:
        f.write("/* GOLDEN MODEL WEIGHTS */\n#ifndef MODEL_WEIGHTS_H\n#define MODEL_WEIGHTS_H\n\n")
        f.write("#define MDL_N_FEATURES 12\n#define MDL_N_H1 16\n#define MDL_N_H2 16\n#define MDL_N_OUT 1\n")
        f.write("#define MDL_WINDOW_SIZE 10\n#define MDL_TEMPERATURE 1.0f\n#define MDL_THRESHOLD 0.7f\n\n")
        
        # Means and STDs
        f.write(f"static const float MDL_FEAT_MEAN[12] = {{{', '.join([f'{x}f' for x in scaler.mean_])}}};\n")
        f.write(f"static const float MDL_FEAT_STD[12] = {{{', '.join([f'{x}f' for x in scaler.scale_])}}};\n\n")
        
        # Weights
        def write_mat(name, mat):
            f.write(f"static const float {name}[{mat.shape[0]}][{mat.shape[1]}] = {{\n")
            for row in mat:
                f.write(f"    {{{', '.join([f'{x}f' for x in row])}}},\n")
            f.write("};\n\n")
            
        def write_vec(name, vec):
            f.write(f"static const float {name}[{len(vec)}] = {{{', '.join([f'{x}f' for x in vec])}}};\n\n")
            
        write_mat("MDL_W1", w1)
        write_vec("MDL_B1", b1)
        write_mat("MDL_W2", w2)
        write_vec("MDL_B2", b2)
        write_mat("MDL_W3", w3)
        write_vec("MDL_B3", b3)
        f.write("#endif\n")

export_to_c(model, scaler, EXPORT_PATH)
print(f"Golden weights exported to {EXPORT_PATH}")
