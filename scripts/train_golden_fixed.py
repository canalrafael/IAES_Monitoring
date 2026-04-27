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

# Paths
DATA_DIR = "/home/canal/github/IAES_Monitoring/data/online validation data"
EXPORT_PATH = "/home/canal/github/IAES_Monitoring/deploy/online_validation/model_weights_golden.h"

WINDOW_SIZE = 10
FEATURES = ['IPC', 'MPKI', 'L2_PRESSURE', 'BRANCH_MISS_RATE']

def engineer_features(df):
    eps = 1e-9
    df = df.copy()
    
    # 1. Ratios
    df['IPC'] = df['INSTRUCTIONS'] / (df['CPU_CYCLES'] + eps)
    df['MPKI'] = (df['CACHE_MISSES'] * 1000) / (df['INSTRUCTIONS'] + eps)
    df['L2_PRESSURE'] = df['L2_CACHE_ACCESS'] / (df['CPU_CYCLES'] + eps)
    df['BRANCH_MISS_RATE'] = df['BRANCH_MISSES'] / (df['INSTRUCTIONS'] + eps)
    
    # 2. Rolling Stats per Core
    engineered_list = []
    for core_id in range(3):
        core_df = df.iloc[core_id::3].copy()
        
        for f in FEATURES:
            core_df[f'{f}_mean'] = core_df[f].rolling(window=WINDOW_SIZE).mean()
            core_df[f'{f}_std'] = core_df[f].rolling(window=WINDOW_SIZE).std()
            core_df[f'{f}_delta'] = core_df[f].diff(periods=WINDOW_SIZE-1)
            
        engineered_list.append(core_df.dropna())
        
    return pd.concat(engineered_list).sort_index()

# 1. Load Data (Everything except 20 and 21)
all_files = sorted(glob.glob(os.path.join(DATA_DIR, "data_new*.csv")))
files = [f for f in all_files if "20" not in f and "21" not in f]

print(f"Loading and engineering features for datasets: {[os.path.basename(f) for f in files]}...", flush=True)
dfs = []
for f in files:
    df_raw = pd.read_csv(f)
    # Filter dead samples
    df_active = df_raw[df_raw['CPU_CYCLES'] > 100000].copy()
    dfs.append(engineer_features(df_active))

df = pd.concat(dfs, ignore_index=True)
print(f"Loaded {len(df):,} samples after engineering.", flush=True)

# 2. Extract X and y
feat_cols = [f'{sig}_{stat}' for sig in FEATURES for stat in ['mean', 'std', 'delta']]
X = df[feat_cols].values.astype(np.float32)
y = df['LABEL'].isin([1, 2, 3]).values.astype(np.float32)

print(f"Class Balance: {int(y.sum()):,} Attack vs {len(y) - int(y.sum()):,} Benign", flush=True)

# 3. Train/Test Split
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_val = scaler.transform(X_val)

# 4. Model Definition
class MLP(nn.Module):
    def __init__(self):
        super(MLP, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(12, 16), nn.ReLU(),
            nn.Linear(16, 16), nn.ReLU(),
            nn.Linear(16, 1)
        )
    def forward(self, x): return self.net(x)

model = MLP()

# pos_weight = 0.1 to suppress False Positives (gives Benign 10x more weight)
criterion = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([0.1]))
optimizer = optim.Adam(model.parameters(), lr=0.001)

# 5. Training Loop
batch_size = 4096
train_data = TensorDataset(torch.FloatTensor(X_train), torch.FloatTensor(y_train).view(-1, 1))
train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)

epochs = 200
print(f"Training for {epochs} epochs...", flush=True)
for epoch in range(epochs):
    model.train()
    running_loss = 0.0
    for inputs, labels in train_loader:
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item() * inputs.size(0)
    
    if epoch % 10 == 0 or epoch == epochs - 1:
        epoch_loss = running_loss / len(X_train)
        print(f"Epoch {epoch}/{epochs}, Loss: {epoch_loss:.4f}", flush=True)

# 6. Export to C Header
def export_weights(model, scaler, path):
    state = model.state_dict()
    w1 = state['net.0.weight'].numpy()
    b1 = state['net.0.bias'].numpy()
    w2 = state['net.2.weight'].numpy()
    b2 = state['net.2.bias'].numpy()
    w3 = state['net.4.weight'].numpy()
    b3 = state['net.4.bias'].numpy()
    
    with open(path, 'w') as f:
        f.write("/* GOLDEN FIXED MODEL */\n#ifndef MODEL_WEIGHTS_H\n#define MODEL_WEIGHTS_H\n\n")
        f.write("#define MDL_N_FEATURES 12\n#define MDL_N_H1 16\n#define MDL_N_H2 16\n#define MDL_N_OUT 1\n")
        f.write(f"#define MDL_WINDOW_SIZE {WINDOW_SIZE}\n#define MDL_TEMPERATURE 1.0f\n#define MDL_THRESHOLD 0.5f\n\n")
        
        def write_arr(name, arr):
            f.write(f"static const float {name}[{len(arr)}] = {{{', '.join([f'{x}f' for x in arr])}}};\n")
            
        write_arr("MDL_FEAT_MEAN", scaler.mean_)
        write_arr("MDL_FEAT_STD", scaler.scale_)
        
        f.write(f"\nstatic const float MDL_W1[16][12] = {{\n")
        for row in w1: f.write(f"    {{{', '.join([f'{x}f' for x in row])}}},\n")
        f.write("};\n")
        
        write_arr("MDL_B1", b1)
        
        f.write(f"\nstatic const float MDL_W2[16][16] = {{\n")
        for row in w2: f.write(f"    {{{', '.join([f'{x}f' for x in row])}}},\n")
        f.write("};\n")
        
        write_arr("MDL_B2", b2)
        
        f.write(f"\nstatic const float MDL_W3[1][16] = {{\n")
        for row in w3: f.write(f"    {{{', '.join([f'{x}f' for x in row])}}},\n")
        f.write("};\n")
        
        write_arr("MDL_B3", b3)
        f.write("\n#endif\n")

export_weights(model, scaler, EXPORT_PATH)
print(f"Exported to {EXPORT_PATH}", flush=True)
