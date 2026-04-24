import pandas as pd
import numpy as np
import os
import glob
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

# --- Configuration ---
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
BASE_DIR = os.path.dirname(SCRIPT_DIR)
DATA_DIR = os.path.join(BASE_DIR, 'data', 'online validation data')
DEPLOY_DIR = os.path.join(BASE_DIR, 'deploy', 'online_validation')
os.makedirs(DEPLOY_DIR, exist_ok=True)

WINDOW_SIZE = 10
FEATURES = ['IPC', 'MPKI', 'L2_PRESSURE', 'BRANCH_MISS_RATE']

# --- Feature Engineering (Matches detector.c) ---
def engineer_features(df):
    eps = 1e-9
    df = df.copy()
    # 1. Ratios
    df['IPC'] = df['INSTRUCTIONS'] / (df['CPU_CYCLES'] + eps)
    df['MPKI'] = (df['CACHE_MISSES'] * 1000) / (df['INSTRUCTIONS'] + eps)
    df['L2_PRESSURE'] = df['L2_CACHE_ACCESS'] / (df['CPU_CYCLES'] + eps)
    df['BRANCH_MISS_RATE'] = df['BRANCH_MISSES'] / (df['INSTRUCTIONS'] + eps)
    
    # 2. Rolling Stats (per file to avoid cross-boundary leakage)
    engineered_list = []
    
    # Group by Core (since each timestamp has 3 cores, we assume they are interleaved)
    # To be safe, we split by file and core index
    for core_id in range(3):
        core_df = df.iloc[core_id::3].copy()
        
        feat_cols = []
        for f in FEATURES:
            m = core_df[f].rolling(window=WINDOW_SIZE).mean()
            s = core_df[f].rolling(window=WINDOW_SIZE).std()
            d = core_df[f].diff(periods=WINDOW_SIZE-1) # Matches signal[t] - signal[t-W] logic roughly
            
            core_df[f'{f}_mean'] = m
            core_df[f'{f}_std'] = s
            core_df[f'{f}_delta'] = d
            feat_cols.extend([f'{f}_mean', f'{f}_std', f'{f}_delta'])
            
        engineered_list.append(core_df.dropna())
        
    return pd.concat(engineered_list).sort_index()

# 1. Load and Prepare Data (L0 vs L3)
print("Loading data for Bao-Specific training...")
csv_files = [
    os.path.join(DATA_DIR, 'data_new18_clean.csv'),
    os.path.join(DATA_DIR, 'data_new19_clean.csv')
]
dfs = []
for f in csv_files:
    df_raw = pd.read_csv(f)
    # Filter for Benign (0) and Attacker (3)
    # We include all data for feature engineering first to maintain rolling state
    df_eng = engineer_features(df_raw)
    # Then filter
    dfs.append(df_eng[df_eng['LABEL'].isin([0, 3])])

full_df = pd.concat(dfs, ignore_index=True)

X_raw = full_df[[f'{sig}_{stat}' for sig in FEATURES for stat in ['mean', 'std', 'delta']]]
y = (full_df['LABEL'] == 3).astype(int).values

print(f"Dataset size: {len(X_raw)} samples ({sum(y==0)} Benign, {sum(y==1)} Attacker)")

# 2. Normalization
scaler = StandardScaler()
X = scaler.fit_transform(X_raw)

# 3. Training a simple MLP (12 -> 16 -> 16 -> 1)
class MLP(nn.Module):
    def __init__(self):
        super(MLP, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(12, 16),
            nn.ReLU(),
            nn.Linear(16, 16),
            nn.ReLU(),
            nn.Linear(16, 1)
        )
    def forward(self, x):
        return self.net(x)

model = MLP()
criterion = nn.BCEWithLogitsLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=42)

X_train_t = torch.FloatTensor(X_train)
y_train_t = torch.FloatTensor(y_train).view(-1, 1)

print("Training...")
for epoch in range(100):
    optimizer.zero_grad()
    outputs = model(X_train_t)
    loss = criterion(outputs, y_train_t)
    loss.backward()
    optimizer.step()
    if epoch % 20 == 0:
        print(f"Epoch {epoch}, Loss: {loss.item():.4f}")

model.eval()
with torch.no_grad():
    y_pred = torch.sigmoid(model(torch.FloatTensor(X_test))).numpy() > 0.5
    acc = (y_pred.flatten() == y_test).mean()
    print(f"Validation Accuracy: {acc:.2%}")

# 4. Export to model_weights.h
def export_to_c(model, scaler, out_path):
    state = model.state_dict()
    with open(out_path, 'w') as f:
        f.write("/* AUTO-GENERATED for Bao Online Validation */\n")
        f.write("#ifndef MODEL_WEIGHTS_H\n#define MODEL_WEIGHTS_H\n\n")
        f.write("#define MDL_N_FEATURES 12\n#define MDL_N_H1 16\n#define MDL_N_H2 16\n#define MDL_N_OUT 1\n")
        f.write(f"#define MDL_WINDOW_SIZE {WINDOW_SIZE}\n")
        f.write("#define MDL_TEMPERATURE 1.0f\n#define MDL_THRESHOLD 0.5f\n\n")
        
        # Means
        f.write("static const float MDL_FEAT_MEAN[12] = {")
        f.write(", ".join([f"{x}f" for x in scaler.mean_]))
        f.write("};\n")
        
        # Stds
        f.write("static const float MDL_FEAT_STD[12] = {")
        f.write(", ".join([f"{x}f" for x in scaler.scale_]))
        f.write("};\n\n")
        
        # Layer 1
        w1 = state['net.0.weight'].numpy()
        b1 = state['net.0.bias'].numpy()
        f.write("static const float MDL_W1[16][12] = {\n")
        for i in range(16):
            f.write("    {" + ", ".join([f"{x}f" for x in w1[i]]) + "}" + ("," if i<15 else "") + "\n")
        f.write("};\n")
        f.write("static const float MDL_B1[16] = {" + ", ".join([f"{x}f" for x in b1]) + "};\n\n")
        
        # Layer 2
        w2 = state['net.2.weight'].numpy()
        b2 = state['net.2.bias'].numpy()
        f.write("static const float MDL_W2[16][16] = {\n")
        for i in range(16):
            f.write("    {" + ", ".join([f"{x}f" for x in w2[i]]) + "}" + ("," if i<15 else "") + "\n")
        f.write("};\n")
        f.write("static const float MDL_B2[16] = {" + ", ".join([f"{x}f" for x in b2]) + "};\n\n")
        
        # Layer 3
        w3 = state['net.4.weight'].numpy()
        b3 = state['net.4.bias'].numpy()
        f.write("static const float MDL_W3[1][16] = {\n")
        f.write("    {" + ", ".join([f"{x}f" for x in w3[0]]) + "}\n")
        f.write("};\n")
        f.write("static const float MDL_B3[1] = {" + ", ".join([f"{x}f" for x in b3]) + "};\n\n")
        
        f.write("#endif\n")

print(f"Exporting weights to {DEPLOY_DIR}/model_weights.h")
export_to_c(model, scaler, os.path.join(DEPLOY_DIR, "model_weights.h"))

# 5. Done
print("Done! Online Validation model is ready in deploy/online_validation/")
