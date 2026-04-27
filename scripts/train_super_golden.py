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
TRAIN_DIR = "/home/canal/github/IAES_Monitoring/data/train data"
VALID_DIR = "/home/canal/github/IAES_Monitoring/data/online validation data"
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
    unique_cores = df['CORE_ID'].unique() if 'CORE_ID' in df.columns else range(3)
    for core_id in unique_cores:
        if 'CORE_ID' in df.columns:
            core_df = df[df['CORE_ID'] == core_id].copy()
        else:
            core_df = df.iloc[core_id::3].copy()

        
        for f in FEATURES:
            core_df[f'{f}_mean'] = core_df[f].rolling(window=WINDOW_SIZE).mean()
            core_df[f'{f}_std'] = core_df[f].rolling(window=WINDOW_SIZE).std()
            core_df[f'{f}_delta'] = core_df[f].diff(periods=WINDOW_SIZE-1)
            
        engineered_list.append(core_df.dropna())
        
    return pd.concat(engineered_list).sort_index()

# 1. Load Data from BOTH directories
train_files = glob.glob(os.path.join(TRAIN_DIR, "*.csv"))
valid_files = glob.glob(os.path.join(VALID_DIR, "*.csv"))
all_files = sorted(train_files + valid_files)

files = [f for f in all_files if "29" not in f and "30" not in f and "31" not in f and "32" not in f]

print(f"Loading and engineering features for {len(files)} datasets...", flush=True)
dfs = []
for f in files:
    try:
        df_raw = pd.read_csv(f)
        df_active = df_raw[df_raw['CPU_CYCLES'] > 100000].copy()
        dfs.append(engineer_features(df_active))
    except Exception as e:
        print(f"Error loading {os.path.basename(f)}: {e}", flush=True)

df = pd.concat(dfs, ignore_index=True)
print(f"Total Combined Samples after engineering: {len(df):,}", flush=True)

# Balance Classes by downsampling majority class
df_attack = df[df['LABEL'].isin([1, 2, 3])]
df_benign = df[~df['LABEL'].isin([1, 2, 3])]

n_samples = min(500000, len(df_attack), len(df_benign))
print(f"Balancing dataset down to {n_samples:,} samples per class...", flush=True)


df_attack_down = df_attack.sample(n=n_samples, random_state=42)
df_benign_down = df_benign.sample(n=n_samples, random_state=42)

df = pd.concat([df_attack_down, df_benign_down]).sample(frac=1, random_state=42)

# 2. Extract X and y
feat_cols = [f'{sig}_{stat}' for sig in FEATURES for stat in ['mean', 'std', 'delta']]
X = df[feat_cols].values.astype(np.float32)
y = df['LABEL'].isin([1, 2, 3]).values.astype(np.float32)

print(f"Final Class Balance: {int(y.sum()):,} Attack vs {len(y) - int(y.sum()):,} Benign", flush=True)

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

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}", flush=True)

model = MLP().to(device)

# pos_weight = 0.1 to suppress False Positives
criterion = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([0.1]).to(device))
optimizer = optim.Adam(model.parameters(), lr=0.001)

# 5. Training Loop
batch_size = 65536
epochs = 50

scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-5)

train_data = TensorDataset(torch.FloatTensor(X_train).to(device), torch.FloatTensor(y_train).view(-1, 1).to(device))
train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=False)

val_data = TensorDataset(torch.FloatTensor(X_val).to(device), torch.FloatTensor(y_val).view(-1, 1).to(device))
val_loader = DataLoader(val_data, batch_size=batch_size, shuffle=False)


train_losses = []
val_losses = []
train_accs = []
val_accs = []
lrs = []

print(f"Training for {epochs} epochs...", flush=True)
for epoch in range(epochs):
    # Train pass
    model.train()
    running_loss = torch.tensor(0.0, device=device)
    train_correct = 0
    train_total = 0
    for inputs, labels in train_loader:
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.detach() * inputs.size(0)
        
        preds = (torch.sigmoid(outputs) > 0.5).float()
        train_correct += (preds == labels).sum().item()
        train_total += labels.size(0)
        
    epoch_train_loss = running_loss.item() / len(X_train)
    epoch_train_acc = train_correct / train_total
    train_losses.append(epoch_train_loss)
    train_accs.append(epoch_train_acc)
    scheduler.step()

    # Val pass
    model.eval()
    running_val_loss = torch.tensor(0.0, device=device)
    val_correct = 0
    val_total = 0
    with torch.no_grad():
        for inputs, labels in val_loader:
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            running_val_loss += loss.detach() * inputs.size(0)
            
            preds = (torch.sigmoid(outputs) > 0.5).float()
            val_correct += (preds == labels).sum().item()
            val_total += labels.size(0)
            
    epoch_val_loss = running_val_loss.item() / len(X_val)
    epoch_val_acc = val_correct / val_total

    val_losses.append(epoch_val_loss)
    val_accs.append(epoch_val_acc)
    
    lr = optimizer.param_groups[0]['lr']
    lrs.append(lr)
    print(f"Epoch {epoch}/{epochs} | Train Loss: {epoch_train_loss:.4f} | Train Acc: {epoch_train_acc:.4f} | Val Loss: {epoch_val_loss:.4f} | Val Acc: {epoch_val_acc:.4f} | LR: {lr:.6f}", flush=True)


# Plot Loss Curves and Learning Rate
try:
    import matplotlib.pyplot as plt
    os.makedirs("/home/canal/github/IAES_Monitoring/results/online_validation", exist_ok=True)
    
    # 1. Plot Loss and LR Curve
    fig, ax1 = plt.subplots(figsize=(10, 5))
    ax1.plot(train_losses, label="Train Loss", color="dodgerblue")
    ax1.plot(val_losses, label="Validation Loss", color="crimson")
    ax1.set_xlabel("Epochs")
    ax1.set_ylabel("BCE Loss")
    ax1.grid(True, alpha=0.3)
    
    ax2 = ax1.twinx()
    ax2.plot(lrs, label="Learning Rate", color="orange", linestyle="--")
    ax2.set_ylabel("Learning Rate", color="orange")
    ax2.tick_params(axis='y', labelcolor='orange')
    
    lines, labels = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines + lines2, labels + labels2, loc="upper right")
    plt.title("Convergence Profile (Loss)")
    loss_plot_path = "/home/canal/github/IAES_Monitoring/results/online_validation/loss_curve.png"
    plt.savefig(loss_plot_path)
    plt.close()
    
    # 2. Plot Accuracy Curve
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(train_accs, label="Train Accuracy", color="mediumseagreen")
    ax.plot(val_accs, label="Validation Accuracy", color="mediumorchid")
    ax.set_xlabel("Epochs")
    ax.set_ylabel("Accuracy")
    ax.grid(True, alpha=0.3)
    ax.legend(loc="lower right")
    plt.title("Accuracy Profile")
    acc_plot_path = "/home/canal/github/IAES_Monitoring/results/online_validation/accuracy_curve.png"
    plt.savefig(acc_plot_path)
    plt.close()
    
    print(f"Loss curves graphed at: {loss_plot_path}", flush=True)
    print(f"Accuracy curves graphed at: {acc_plot_path}", flush=True)
except Exception as e:
    print(f"Graphing failed: {e}", flush=True)



# 6. Export to C Header
def export_weights(model, scaler, path):
    state = model.state_dict()
    w1 = state['net.0.weight'].cpu().numpy()
    b1 = state['net.0.bias'].cpu().numpy()
    w2 = state['net.2.weight'].cpu().numpy()
    b2 = state['net.2.bias'].cpu().numpy()
    w3 = state['net.4.weight'].cpu().numpy()
    b3 = state['net.4.bias'].cpu().numpy()
    
    with open(path, 'w') as f:
        f.write("/* SUPER GOLDEN MODEL */\n#ifndef MODEL_WEIGHTS_H\n#define MODEL_WEIGHTS_H\n\n")
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
