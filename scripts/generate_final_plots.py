import pandas as pd
import numpy as np
import os
import glob
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, roc_curve, auc, mean_squared_error
from sklearn.model_selection import train_test_split

# Styling
sns.set_theme(style="whitegrid", palette="bright")
plt.rcParams.update({
    'font.size': 16,
    'axes.titlesize': 24,
    'axes.labelsize': 20,
    'legend.fontsize': 18,
    'xtick.labelsize': 16,
    'ytick.labelsize': 16,
    'axes.titleweight': 'bold'
})

DATA_DIR = 'data/train data/'
RESULTS_DIR = 'results/phase2/'
os.makedirs(RESULTS_DIR, exist_ok=True)

# Best Configuration and Threshold Options
BEST_CONFIG = {'layers': 2, 'units': 8, 'lr': 0.001, 'w': 19}
THRESHOLDS = {
    'High-Safety': 0.10,
    'Balanced': 0.72,
    'Low-FPR': 0.92
}
FEATURES = ['CPU_CYCLES', 'INSTRUCTIONS', 'CACHE_MISSES', 'BRANCH_MISSES', 'L2_CACHE_ACCESS']

def engineer_features(df, w):
    df = df.copy()
    eps = 1e-9
    df['IPC'] = df['INSTRUCTIONS'] / (df['CPU_CYCLES'] + eps)
    df['MPKI'] = (df['CACHE_MISSES'] * 1000) / (df['INSTRUCTIONS'] + eps)
    df['BRANCH_MISS_RATE'] = df['BRANCH_MISSES'] / (df['INSTRUCTIONS'] + eps)
    df['L2_PRESSURE'] = df['L2_CACHE_ACCESS'] / (df['CPU_CYCLES'] + eps)
    
    base_cols = FEATURES + ['IPC', 'MPKI', 'BRANCH_MISS_RATE', 'L2_PRESSURE']
    aug_feats = []
    for col in base_cols:
        rolling = df[col].rolling(window=w)
        df[f'{col}_mean'] = rolling.mean()
        df[f'{col}_std'] = rolling.std()
        df[f'{col}_min'] = rolling.min()
        df[f'{col}_max'] = rolling.max()
        df[f'{col}_delta'] = df[col] - df[col].shift(w-1)
        aug_feats.extend([f'{col}_mean', f'{col}_std', f'{col}_min', f'{col}_max', f'{col}_delta'])
    return df.dropna(), aug_feats

class BestMLP(nn.Module):
    def __init__(self, input_dim, hidden_layers, units_per_layer):
        super(BestMLP, self).__init__()
        layers = []
        last_dim = input_dim
        for i in range(hidden_layers):
            layers.append(nn.Linear(last_dim, units_per_layer))
            layers.append(nn.ReLU())
            last_dim = units_per_layer
        layers.append(nn.Linear(last_dim, 1))
        layers.append(nn.Sigmoid())
        self.model = nn.Sequential(*layers)
    def forward(self, x): return self.model(x)

def main():
    print("Loading data for final diagnostic run...")
    csv_files = sorted(glob.glob(os.path.join(DATA_DIR, "*.csv")))
    all_dfs = []
    for f in csv_files:
        temp = pd.read_csv(f)
        # Filter first, then check if any relevant data exists
        temp_filtered = temp[temp['LABEL'].isin([0, 2])]
        if not temp_filtered.empty:
            data, feats = engineer_features(temp_filtered, BEST_CONFIG['w'])
            if not data.empty:
                all_dfs.append(data)
    
    df = pd.concat(all_dfs, ignore_index=True)
    X = df[feats]
    y = (df['LABEL'] == 2).astype(int)
    
    # Representative Split
    X_train, X_remain, y_train, y_remain = train_test_split(X, y, test_size=0.3, random_state=42)
    X_val, X_test, y_val, y_test = train_test_split(X_remain, y_remain, test_size=0.5, random_state=42)
    
    # Scale (using train only)
    mean, std = X_train.mean(), X_train.std() + 1e-9
    X_train, X_val, X_test = (X_train - mean) / std, (X_val - mean) / std, (X_test - mean) / std
    
    # DataLoaders
    train_ds = TensorDataset(torch.FloatTensor(X_train.values), torch.FloatTensor(y_train.values).view(-1, 1))
    val_ds = TensorDataset(torch.FloatTensor(X_val.values), torch.FloatTensor(y_val.values).view(-1, 1))
    train_loader = DataLoader(train_ds, batch_size=256, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=1024)
    
    model = BestMLP(len(feats), BEST_CONFIG['layers'], BEST_CONFIG['units'])
    criterion = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=BEST_CONFIG['lr'])
    
    history = {'train_loss': [], 'val_loss': [], 'train_acc': [], 'val_acc': []}
    best_val_loss = float('inf')
    patience = 5
    counter = 0
    
    print("Training best candidate for diagnostics...")
    for epoch in range(100):
        model.train()
        t_loss, t_correct, t_total = 0, 0, 0
        for b_X, b_y in train_loader:
            optimizer.zero_grad()
            out = model(b_X)
            loss = criterion(out, b_y)
            loss.backward(); optimizer.step()
            t_loss += loss.item()
            t_correct += ((out > 0.5).float() == b_y).sum().item(); t_total += b_y.size(0)
        
        model.eval()
        v_loss, v_correct, v_total = 0, 0, 0
        with torch.no_grad():
            for b_X, b_y in val_loader:
                out = model(b_X)
                v_loss += criterion(out, b_y).item()
                v_correct += ((out > 0.5).float() == b_y).sum().item(); v_total += b_y.size(0)
        
        history['train_loss'].append(t_loss/len(train_loader))
        history['val_loss'].append(v_loss/len(val_loader))
        history['train_acc'].append(t_correct/t_total)
        history['val_acc'].append(v_correct/v_total)
        
        current_val_loss = v_loss/len(val_loader)
        if (epoch+1) % 10 == 0: 
            print(f"Epoch {epoch+1}/1000 | Val Loss: {current_val_loss:.4f} | Val Acc: {v_correct/v_total:.4f}")

        # Early Stopping
        if current_val_loss < best_val_loss:
            best_val_loss = current_val_loss
            counter = 0
            # Save temporary best state if needed, but here we just continue
        else:
            counter += 1
            if counter >= patience:
                print(f"Early stopping at epoch {epoch+1}")
                break

    # 1. Loss Curve
    plt.figure()
    plt.plot(history['train_loss'], label='Training Loss')
    plt.plot(history['val_loss'], label='Validation Loss')
    plt.xlabel('Epochs'); plt.ylabel('Loss'); plt.title('Loss Curve'); plt.legend()
    plt.savefig(os.path.join(RESULTS_DIR, 'loss_curve.png'))

    # 2. Learning Curve (Accuracy)
    plt.figure()
    plt.plot(history['train_acc'], label='Training Accuracy')
    plt.plot(history['val_acc'], label='Validation Accuracy')
    plt.xlabel('Epochs'); plt.ylabel('Accuracy'); plt.title('Learning Curve'); plt.legend()
    plt.savefig(os.path.join(RESULTS_DIR, 'learning_curve.png'))

    # Final Test Eval
    model.eval()
    with torch.no_grad():
        test_probs = model(torch.FloatTensor(X_test.values)).numpy().flatten()

    # 3. Confusion Matrices for all 3 options
    for name, tau in THRESHOLDS.items():
        plt.figure()
        test_preds = (test_probs > tau).astype(int)
        cm = confusion_matrix(y_test, test_preds)
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                    xticklabels=['Benign', 'Interf'], yticklabels=['Benign', 'Interf'])
        plt.xlabel('Predicted'); plt.ylabel('Actual')
        plt.title(f'Confusion Matrix: {name} (τ={tau})')
        filename = f'confusion_matrix_{name.lower().replace("-", "_")}.png'
        plt.savefig(os.path.join(RESULTS_DIR, filename))
        plt.close()
        print(f"Generated Confusion Matrix for {name} operating point.")

    # 4. ROC Curve
    plt.figure()
    fpr, tpr, _ = roc_curve(y_test, test_probs)
    plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (AUC = {auc(fpr, tpr):.4f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlabel('False Positive Rate'); plt.ylabel('True Positive Rate'); plt.title('ROC Curve'); plt.legend()
    plt.savefig(os.path.join(RESULTS_DIR, 'roc_curve.png'))

    # 5. Scatter Plot with Regression Line (Probability Regression visualization)
    plt.figure()
    # Sample subset for visibility
    sample_indices = np.random.choice(len(y_test), min(500, len(y_test)), replace=False)
    plt.scatter(y_test.iloc[sample_indices], test_probs[sample_indices], alpha=0.3, label='Predictions')
    # Perfect line (if it were a binary regression)
    sns.regplot(x=y_test.iloc[sample_indices], y=test_probs[sample_indices], scatter=False, color='red', label='Trend Line')
    plt.xlabel('Actual Class (0=Benign, 2=Interf)'); plt.ylabel('Predicted Probability')
    plt.title('Prediction Probability vs Actual Class'); plt.legend()
    plt.savefig(os.path.join(RESULTS_DIR, 'probability_reg_plot.png'))

    print("\nAll diagnostic plots generated in results/phase2/")

if __name__ == "__main__": main()
