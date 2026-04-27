import pandas as pd
import numpy as np
import os
import glob
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import torch
import torch.nn as nn
import torch.optim as optim

# Paths
DATA_DIR = "/home/canal/github/IAES_Monitoring/data/online validation data"
EXPORT_PATH = "/home/canal/github/IAES_Monitoring/deploy/online_validation/model_weights_golden.h"

# 1. Load Data (22, 23, 24, 25)
files = [
    os.path.join(DATA_DIR, "data_new22_clean.csv"),
    os.path.join(DATA_DIR, "data_new23_clean.csv"),
    os.path.join(DATA_DIR, "data_new24_clean.csv"),
    os.path.join(DATA_DIR, "data_new25_clean.csv"),
    os.path.join(DATA_DIR, "data_new26_clean.csv"),
    os.path.join(DATA_DIR, "data_new27_clean.csv")
]

print("Loading datasets...", flush=True)
dfs = []
for f in files:
    if os.path.exists(f):
        dfs.append(pd.read_csv(f))
    else:
        print(f"Warning: File not found: {f}", flush=True)

df = pd.concat(dfs)
print(f"Loaded {len(df)} samples total.", flush=True)

# 2. Feature Engineering (12 features)
def engineer_features(df):
    df['IPC'] = df['INSTRUCTIONS'] / df['CPU_CYCLES'].replace(0, 1)
    df['MPKI'] = (df['CACHE_MISSES'] * 1000) / df['INSTRUCTIONS'].replace(0, 1)
    df['L2_PRESSURE'] = df['L2_CACHE_ACCESS'] / df['INSTRUCTIONS'].replace(0, 1)
    df['BRANCH_MISS_RATE'] = df['BRANCH_MISSES'] / df['INSTRUCTIONS'].replace(0, 1)
    
    features = ['IPC', 'MPKI', 'L2_PRESSURE', 'BRANCH_MISS_RATE']
    all_feats = []
    for f in features:
        all_feats.append(df[f])
        all_feats.append(df[f].rolling(window=10).mean().fillna(0))
        all_feats.append(df[f].rolling(window=10).std().fillna(0))
    
    return np.column_stack(all_feats), (df['LABEL'] > 0).astype(int).values

X, y = engineer_features(df)

# 3. Train/Test Split (80/20)
print("Splitting data...", flush=True)
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# Normalize
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_val = scaler.transform(X_val)

# 4. Model Definition
class MLP(nn.Module):
    def __init__(self):
        super(MLP, self).__init__()
        self.layers = nn.Sequential(
            nn.Linear(12, 16), nn.ReLU(),
            nn.Linear(16, 16), nn.ReLU(),
            nn.Linear(16, 1)
        )
    def forward(self, x): return self.layers(x)

model = MLP()
criterion = nn.BCEWithLogitsLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# 5. Training Loop (Mini-batch for performance on large dataset)
print("Training with mini-batches...", flush=True)
batch_size = 4096
train_data = torch.utils.data.TensorDataset(torch.FloatTensor(X_train), torch.FloatTensor(y_train))
train_loader = torch.utils.data.DataLoader(train_data, batch_size=batch_size, shuffle=True)

epochs = 2000
for epoch in range(epochs):
    model.train()
    running_loss = 0.0
    for inputs, labels in train_loader:
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs.squeeze(), labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item() * inputs.size(0)
    
    epoch_loss = running_loss / len(X_train)
    if epoch % 10 == 0:
        print(f"Epoch {epoch}/{epochs}, Loss: {epoch_loss:.4f}", flush=True)

# 6. Export to C Header
def export_weights(model, scaler, path):
    w1 = model.layers[0].weight.detach().numpy()
    b1 = model.layers[0].bias.detach().numpy()
    w2 = model.layers[2].weight.detach().numpy()
    b2 = model.layers[2].bias.detach().numpy()
    w3 = model.layers[4].weight.detach().numpy()
    b3 = model.layers[4].bias.detach().numpy()
    
    with open(path, 'w') as f:
        f.write("#ifndef MODEL_WEIGHTS_H\n#define MODEL_WEIGHTS_H\n\n")
        f.write("#define MDL_N_FEATURES 12\n#define MDL_N_H1 16\n#define MDL_N_H2 16\n#define MDL_N_OUT 1\n")
        f.write("#define MDL_WINDOW_SIZE 10\n#define MDL_TEMPERATURE 1.0f\n#define MDL_THRESHOLD 0.5f\n\n")
        
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
