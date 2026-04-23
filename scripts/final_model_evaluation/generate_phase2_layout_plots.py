import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, roc_curve, auc

# Styling - SAME AS PHASE 2
sns.set_theme(style="whitegrid", palette="bright")
plt.rcParams['figure.figsize'] = (12, 10)

DATA_DIR    = 'data/train data/'
RESULTS_DIR = 'results/final_model_evaluation/'
MODELS_DIR  = 'models/simplified/'
os.makedirs(RESULTS_DIR, exist_ok=True)

# MLP V1 Constants
W = 10
LAYERS = 2
UNITS = 16
N_SMOOTH = 5
LR = 0.002
MAX_EPOCHS = 250
PATIENCE = 10

THRESHOLDS = {
    'High-Safety': 0.10,
    'Operational': 0.20,
    'Robust': 0.50
}

RATIO_SIGNALS = ['IPC', 'MPKI', 'L2_PRESSURE', 'BRANCH_MISS_RATE']

def engineer_features_v1(df, w=W):
    df = df[df['LABEL'].isin([0, 2])].copy()
    eps = 1e-9
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
    X = df_c[feat_cols].values.astype(np.float32)
    y = (df_c['LABEL'] == 2).values.astype(np.float32)
    return X, y, feat_cols, df_c

class MLP(nn.Module):
    def __init__(self, in_dim=12, layers=2, units=16, dropout=0.1):
        super().__init__()
        seq, last = [], in_dim
        for _ in range(layers):
            seq += [nn.Linear(last, units), nn.ReLU(), nn.Dropout(dropout)]
            last = units
        seq.append(nn.Linear(last, 1))
        self.net = nn.Sequential(*seq)
    def forward(self, x): return self.net(x)

def causal_smooth(probs, n):
    if n <= 1: return probs.copy()
    out = np.empty_like(probs)
    for i in range(len(probs)):
        out[i] = probs[max(0, i - n + 1): i + 1].mean()
    return out

def main():
    print("Loading V1 Model Weights for Phase 2 Layout Plots...")
    norm = torch.load(os.path.join(MODELS_DIR, 'normalization_params.pth'), weights_only=False)
    mu, sig, features, temp = norm['mean'], norm['std'], norm['features'], norm['temp']
    
    # Dataset split matching evaluate_v1_final.py
    benign_files = ['data0_clean.csv', 'data10_clean.csv', 'data12_clean.csv', 'data15_clean.csv', 'data18_clean.csv', 'data19_clean.csv', 'data21_clean.csv', 'data23_clean.csv', 'data24_clean.csv', 'data7_clean.csv']
    attack_files = ['data13_clean.csv', 'data14_clean.csv', 'data16_clean.csv', 'data17_clean.csv', 'data20_clean.csv', 'data22_clean.csv', 'data25_clean.csv', 'data26_clean.csv', 'data27_clean.csv', 'data3_clean.csv', 'data4_clean.csv', 'data5_clean.csv', 'data6_clean.csv']
    
    np.random.seed(42)
    b_te_names = list(np.random.choice(benign_files, size=int(len(benign_files)*0.3), replace=False))
    a_te_names = list(np.random.choice(attack_files, size=int(len(attack_files)*0.3), replace=False))
    test_files = b_te_names + a_te_names
    train_files = [f for f in benign_files + attack_files if f not in test_files]

    # ---------- 1. DIAGNOSTIC TRAINING FOR CURVES ----------
    print(f"Preparing {len(train_files)} training files for diagnostic curve generation...")
    X_tr_list, y_tr_list = [], []
    for f in train_files:
        df_raw = pd.read_csv(os.path.join(DATA_DIR, f))
        X_raw, y, _, _ = engineer_features_v1(df_raw)
        X_tr_list.append(X_raw)
        y_tr_list.append(y)
    
    X_train_raw = np.vstack(X_tr_list)
    y_train_all = np.concatenate(y_tr_list)
    X_train_s = (X_train_raw - mu) / sig

    # Validation split for curves
    X_t, X_v, y_t, y_v = train_test_split(X_train_s, y_train_all, test_size=0.1, random_state=42, stratify=y_train_all)
    tr_ds = TensorDataset(torch.from_numpy(X_t), torch.from_numpy(y_t).view(-1, 1))
    va_ds = TensorDataset(torch.from_numpy(X_v), torch.from_numpy(y_v).view(-1, 1))
    
    tr_loader = DataLoader(tr_ds, batch_size=256, shuffle=True)
    va_loader = DataLoader(va_ds, batch_size=1024)
    
    diag_model = MLP(in_dim=len(features), layers=LAYERS, units=UNITS, dropout=0.1)
    class_ratio = (y_train_all == 0).sum() / max((y_train_all == 1).sum(), 1)
    pw = torch.tensor([2.0 * class_ratio])
    criterion = nn.BCEWithLogitsLoss(pos_weight=pw)
    optimizer = optim.Adam(diag_model.parameters(), lr=LR, weight_decay=1e-4)

    history = {'train_loss': [], 'val_loss': [], 'train_acc': [], 'val_acc': []}
    best_val_loss = float('inf')
    counter = 0

    print("Training diagnostic model to extract Loss/Learning curves...")
    for epoch in range(MAX_EPOCHS):
        diag_model.train()
        t_loss, t_correct, t_total = 0, 0, 0
        for b_X, b_y in tr_loader:
            optimizer.zero_grad()
            out = diag_model(b_X)
            loss = criterion(out, b_y)
            loss.backward(); optimizer.step()
            t_loss += loss.item()
            t_correct += ((torch.sigmoid(out) > 0.5).float() == b_y).sum().item()
            t_total += b_y.size(0)
            
        diag_model.eval()
        v_loss, v_correct, v_total = 0, 0, 0
        with torch.no_grad():
            for b_X, b_y in va_loader:
                out = diag_model(b_X)
                v_loss += criterion(out, b_y).item()
                v_correct += ((torch.sigmoid(out) > 0.5).float() == b_y).sum().item()
                v_total += b_y.size(0)
                
        history['train_loss'].append(t_loss/len(tr_loader))
        history['val_loss'].append(v_loss/len(va_loader))
        history['train_acc'].append(t_correct/t_total)
        history['val_acc'].append(v_correct/v_total)
        
        current_val_loss = v_loss/len(va_loader)
        if current_val_loss < best_val_loss:
            best_val_loss = current_val_loss
            counter = 0
        else:
            counter += 1
            if counter >= PATIENCE:
                print(f"  Early stopping at epoch {epoch+1}")
                break

    # Replicate Phase 2 plot layout for curves - Dual graphics image
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    axes[0].plot(history['train_loss'], label='Training Loss', lw=2)
    axes[0].plot(history['val_loss'], label='Validation Loss', lw=2)
    axes[0].set_xlabel('Epochs', fontsize=12)
    axes[0].set_ylabel('Loss', fontsize=12)
    axes[0].set_title('Model Loss Curve', fontsize=14)
    axes[0].legend(fontsize=12)

    axes[1].plot(history['train_acc'], label='Training Accuracy', lw=2)
    axes[1].plot(history['val_acc'], label='Validation Accuracy', lw=2)
    axes[1].set_xlabel('Epochs', fontsize=12)
    axes[1].set_ylabel('Accuracy', fontsize=12)
    axes[1].set_title('Model Learning Curve', fontsize=14)
    axes[1].legend(fontsize=12)

    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_DIR, 'final_model_combined_curves.png'), dpi=300)
    plt.close()
    print("Generated Dual Learning/Loss Curve (combined).")

    # ---------- 2. DEPLOYED MODEL EVALUATION ----------
    print(f"\nApplying final deployed model to test files: {len(test_files)} files...")
    
    # Load actual deployed model for accurate test evaluations
    deployed_model = MLP(in_dim=len(features), layers=LAYERS, units=UNITS, dropout=0.0)
    deployed_model.load_state_dict(torch.load(os.path.join(MODELS_DIR, 'best_model.pth'), weights_only=True))
    deployed_model.eval()

    all_y_te, all_p_smooth = [], []
    for f in test_files:
        df_raw = pd.read_csv(os.path.join(DATA_DIR, f))
        X_raw, y, feat_cols, df_c = engineer_features_v1(df_raw)
        X_s = (X_raw - mu) / sig
        
        with torch.no_grad():
            logits = deployed_model(torch.from_numpy(X_s)).numpy().flatten()
        p_mlp = 1.0 / (1.0 + np.exp(-logits / temp))
        p_mlp_smooth = causal_smooth(p_mlp, N_SMOOTH)
        
        all_y_te.extend(y)
        all_p_smooth.extend(p_mlp_smooth)

    y_te = np.array(all_y_te)
    test_probs = np.array(all_p_smooth)

    # Generated Plot 3: Confusion Matrices for options
    for name, tau in THRESHOLDS.items():
        plt.figure()
        test_preds = (test_probs > tau).astype(int)
        cm = confusion_matrix(y_te, test_preds)
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                    xticklabels=['Benign', 'Interf'], yticklabels=['Benign', 'Interf'])
        plt.xlabel('Predicted'); plt.ylabel('Actual')
        plt.title(f'Confusion Matrix: {name} (τ={tau})')
        filename = f'final_model_confusion_matrix_{name.lower().replace("-", "_")}.png'
        plt.savefig(os.path.join(RESULTS_DIR, filename), dpi=300)
        plt.close()
        print(f"Generated Confusion Matrix for {name} operating point ({filename}).")

    # Generated Plot 4: ROC Curve
    plt.figure()
    fpr_vals, tpr_vals, _ = roc_curve(y_te, test_probs)
    plt.plot(fpr_vals, tpr_vals, color='darkorange', lw=2, label=f'ROC curve (AUC = {auc(fpr_vals, tpr_vals):.4f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlabel('False Positive Rate'); plt.ylabel('True Positive Rate'); plt.title('ROC Curve (Final Model)')
    plt.legend()
    plt.savefig(os.path.join(RESULTS_DIR, 'final_model_roc_curve.png'), dpi=300)
    plt.close()
    print("Generated ROC Curve.")

    # Generated Plot 5: Scatter Plot with Regression Line
    plt.figure()
    sample_indices = np.random.choice(len(y_te), min(500, len(y_te)), replace=False)
    plt.scatter(y_te[sample_indices], test_probs[sample_indices], alpha=0.3, label='Predictions')
    sns.regplot(x=y_te[sample_indices], y=test_probs[sample_indices], scatter=False, color='red', label='Trend Line')
    plt.xlabel('Actual Class (0=Benign, 1=Interf)'); plt.ylabel('Predicted Probability')
    plt.title('Prediction Probability vs Actual Class (Final Model)')
    plt.legend()
    plt.savefig(os.path.join(RESULTS_DIR, 'final_model_probability_reg_plot.png'), dpi=300)
    plt.close()
    print("Generated Probability Regression Plot.")
    
    print("\nPhase-2 layout plots successfully generated for the final model in results/final_model_evaluation/")

if __name__ == "__main__":
    main()
