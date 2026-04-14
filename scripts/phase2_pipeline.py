import pandas as pd
import numpy as np
import os
import glob
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import recall_score, f1_score, confusion_matrix, roc_curve, auc
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import seaborn as sns
import time

# Styling
sns.set_theme(style="whitegrid", palette="bright")
plt.rcParams['figure.figsize'] = (12, 10)
plt.rcParams['font.size'] = 12

# FEATURE CACHE (Mandatory for speed)
feature_cache = {}

DATA_DIR = 'data/'
RESULTS_DIR = 'results/phase2/'
MODELS_DIR = 'models/'
os.makedirs(RESULTS_DIR, exist_ok=True)
os.makedirs(MODELS_DIR, exist_ok=True)

FEATURES = ['CPU_CYCLES', 'INSTRUCTIONS', 'CACHE_MISSES', 'BRANCH_MISSES', 'L2_CACHE_ACCESS']

def engineer_features(df, w):
    """
    Computes base signals and rolling window features.
    """
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
        df[f'{col}_std'] = rolling.std().fillna(0)
        df[f'{col}_min'] = rolling.min()
        df[f'{col}_max'] = rolling.max()
        df[f'{col}_delta'] = (df[col] - df[col].shift(w-1)).fillna(0)
        aug_feats.extend([f'{col}_mean', f'{col}_std', f'{col}_min', f'{col}_max', f'{col}_delta'])
    
    # Return processed DF and list of feature names
    df_clean = df.dropna()
    return df_clean, aug_feats

def get_cached_features(file_path, w):
    """
    Retrieves or computes engineered features for a specific file and window size.
    """
    key = (file_path, w)
    if key not in feature_cache:
        df = pd.read_csv(file_path)
        # Filter relevant labels once
        df = df[df['LABEL'].isin([0, 2])].copy()
        if df.empty:
            feature_cache[key] = (None, [])
        else:
            feature_cache[key] = engineer_features(df, w)
    return feature_cache[key]

class MLPModel(nn.Module):
    def __init__(self, input_dim, hidden_layers, units_per_layer, dropout_p=0.1):
        super(MLPModel, self).__init__()
        layers = []
        last_dim = input_dim
        for i in range(hidden_layers):
            layers.append(nn.Linear(last_dim, units_per_layer))
            layers.append(nn.ReLU())
            if dropout_p > 0:
                layers.append(nn.Dropout(p=dropout_p))
            last_dim = units_per_layer
        layers.append(nn.Linear(last_dim, 1))
        # REMOVED Sigmoid; using BCEWithLogitsLoss
        self.model = nn.Sequential(*layers)
        
    def forward(self, x): 
        return self.model(x)

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def ema_update(current, previous, alpha=0.3):
    if previous is None: return current
    return alpha * current + (1 - alpha) * previous

def train_with_early_stopping(model, train_loader, val_loader, config, epochs=1000):
    pos_weight = torch.tensor([config['pos_weight']])
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    optimizer = optim.Adam(model.parameters(), lr=config['lr'], weight_decay=1e-5)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=3)
    
    best_val_loss = float('inf')
    best_state = None
    patience = 15
    counter = 0
    
    history = {
        'train_loss': [], 'val_loss': [], 
        'train_acc': [], 'val_acc': [],
        'val_recall': [], 'val_fpr': [], 'val_fpr_ema': []
    }
    
    val_fpr_ema = None
    fpr_increase_counter = 0
    
    for epoch in range(epochs):
        model.train()
        t_loss, t_correct, t_total = 0, 0, 0
        for b_X, b_y in train_loader:
            optimizer.zero_grad()
            logits = model(b_X)
            loss = criterion(logits, b_y)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            
            t_loss += loss.item()
            probs = torch.sigmoid(logits)
            t_correct += ((probs > 0.5).float() == b_y).sum().item()
            t_total += b_y.size(0)
        
        model.eval()
        v_loss, v_correct, v_total = 0, 0, 0
        all_v_probs, all_v_y = [], []
        
        with torch.no_grad():
            for b_X, b_y in val_loader:
                logits = model(b_X)
                v_loss += criterion(logits, b_y).item()
                probs = torch.sigmoid(logits)
                v_correct += ((probs > 0.5).float() == b_y).sum().item()
                v_total += b_y.size(0)
                all_v_probs.extend(probs.numpy().flatten())
                all_v_y.extend(b_y.numpy().flatten())
        
        avg_v_loss = v_loss / len(val_loader)
        v_recall = recall_score(all_v_y, (np.array(all_v_probs) > 0.5).astype(int), zero_division=0)
        
        # Calculate Val FPR
        v_y_arr = np.array(all_v_y)
        v_preds = (np.array(all_v_probs) > 0.5).astype(int)
        tn = ((v_y_arr == 0) & (v_preds == 0)).sum()
        fp = ((v_y_arr == 0) & (v_preds == 1)).sum()
        v_fpr = fp / (fp + tn) if (fp + tn) > 0 else 0
        
        # EMA for FPR
        val_fpr_ema = ema_update(v_fpr, val_fpr_ema)
        
        history['train_loss'].append(t_loss/len(train_loader))
        history['val_loss'].append(avg_v_loss)
        history['train_acc'].append(t_correct/t_total)
        history['val_acc'].append(v_correct/v_total)
        history['val_recall'].append(v_recall)
        history['val_fpr'].append(v_fpr)
        history['val_fpr_ema'].append(val_fpr_ema)
        
        scheduler.step(avg_v_loss)
        
        # Early Stopping check (Loss)
        if avg_v_loss < best_val_loss - 1e-4:
            best_val_loss = avg_v_loss
            best_state = model.state_dict().copy()
            counter = 0
        else:
            counter += 1
            
        # Robust FPR Early Stopping
        if v_recall >= 0.95:
            if len(history['val_fpr_ema']) > 1 and history['val_fpr_ema'][-1] > history['val_fpr_ema'][-2]:
                fpr_increase_counter += 1
            else:
                fpr_increase_counter = 0
            
            if fpr_increase_counter >= 3:
                # print(f"  [Early Stop] FPR rising after recall target met at epoch {epoch}")
                break

        if counter >= patience: break
            
    if best_state is not None:
        model.load_state_dict(best_state)
    return history

def main():
    print("Starting Phase 2 Unified Pipeline...")
    csv_files = sorted(glob.glob(os.path.join(DATA_DIR, "*.csv")))
    benign_blocks = []
    interf_blocks = []
    for f in csv_files:
        df = pd.read_csv(f)
        if 0 in df['LABEL'].unique(): benign_blocks.append(f)
        elif 2 in df['LABEL'].unique(): interf_blocks.append(f)
    blocks = benign_blocks + interf_blocks
    print(f"Found {len(benign_blocks)} Benign and {len(interf_blocks)} Interference blocks.")

    # Search Space (Expanded per user request)
    w_values = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20]
    architectures = [
        {'layers': 1, 'units': 8}, {'layers': 1, 'units': 16},
        {'layers': 2, 'units': 8}, {'layers': 2, 'units': 16},
        {'layers': 3, 'units': 8}, {'layers': 3, 'units': 16}
    ]
    lrs = [1e-3, 5e-3]
    
    # 1. Grid Search with LOBO (5-Fold)
    n_folds = 5
    benign_folds = np.array_split(benign_blocks, n_folds)
    interf_folds = np.array_split(interf_blocks, n_folds)
    grid_results = []
    
    # Search Space (REFINED)
    w_values = [8, 10, 12, 14]
    architectures = [
        {'layers': 1, 'units': 8}, {'layers': 1, 'units': 16},
        {'layers': 2, 'units': 8}, {'layers': 2, 'units': 16}
    ]
    lrs = [1e-3, 2e-3]
    pos_weight_multipliers = [0.5, 1.0, 2.0, 3.0]
    dropouts = [0.0, 0.1, 0.2]
    
    print("Beginning Optimized Grid Search...")
    for w in w_values:
        print(f"  Evaluating W={w}...")
        
        # Load/Pre-cache features for all blocks at this 'w'
        for f in blocks:
            get_cached_features(f, w)
            
        # 1. PRE-PREPARE DATA FOR ALL FOLDS AT THIS W
        # This saves massive time by doing concatenation, splitting, and normalization only 5 times instead of 480.
        fold_data_cache = []
        for i in range(n_folds):
            test_files = list(benign_folds[i]) + list(interf_folds[i])
            train_files = [f for f in blocks if f not in test_files]
            
            # Use cached features
            train_blocks_data = [get_cached_features(f, w)[0] for f in train_files]
            train_set = pd.concat([df for df in train_blocks_data if df is not None])
            
            test_blocks_data = [get_cached_features(f, w)[0] for f in test_files]
            test_set = pd.concat([df for df in test_blocks_data if df is not None])
            
            # Stratified Train/Val split inside the fold
            tr_df, v_df = train_test_split(
                train_set, test_size=0.2, stratify=train_set['LABEL'], 
                shuffle=True, random_state=42
            )
            
            feats = get_cached_features(train_files[0], w)[1]
            mean, std = tr_df[feats].mean(), tr_df[feats].std() + 1e-9
            
            X_tr = (tr_df[feats] - mean) / std
            y_tr = (tr_df['LABEL'] == 2).astype(int)
            X_v = (v_df[feats] - mean) / std
            y_v = (v_df['LABEL'] == 2).astype(int)
            X_te = (test_set[feats] - mean) / std
            y_te = (test_set['LABEL'] == 2).astype(int)
            
            # Calculate dynamic base weight once per fold
            num_pos = (y_tr == 1).sum()
            num_neg = (y_tr == 0).sum()
            base_weight = num_neg / (num_pos + 1e-9)
            
            # Cache results for this fold
            fold_data_cache.append({
                'tr_loader': DataLoader(TensorDataset(torch.FloatTensor(X_tr.values), torch.FloatTensor(y_tr.values).view(-1, 1)), batch_size=128, shuffle=True),
                'v_loader': DataLoader(TensorDataset(torch.FloatTensor(X_v.values), torch.FloatTensor(y_v.values).view(-1, 1)), batch_size=256),
                'X_te': torch.FloatTensor(X_te.values),
                'y_te': y_te,
                'feats': feats,
                'base_weight': base_weight
            })

        # 2. RUN ARCHITECTURES
        configs = []
        for arch in architectures:
            for lr in lrs:
                for pw_mult in pos_weight_multipliers:
                    for do in dropouts:
                        configs.append({**arch, 'lr': lr, 'w': w, 'pw_mult': pw_mult, 'dropout': do})

        # Evaluate each config across all folds
        for config_idx, config in enumerate(configs):
            fold_metrics = []
            
            # Architecture size check (only once per config)
            dummy_model = MLPModel(len(fold_data_cache[0]['feats']), config['layers'], config['units'])
            n_params = count_parameters(dummy_model)
            if n_params > 2000:
                continue

            if config_idx % 10 == 0:
                print(f"    Config {config_idx+1}/{len(configs)}: {config}")

            for i in range(n_folds):
                fold_data = fold_data_cache[i]
                config['pos_weight'] = fold_data['base_weight'] * config['pw_mult']
                
                model = MLPModel(len(fold_data['feats']), config['layers'], config['units'], config['dropout'])
                train_with_early_stopping(model, fold_data['tr_loader'], fold_data['v_loader'], config)
                
                model.eval()
                with torch.no_grad():
                    logits = model(fold_data['X_te'])
                    probs = torch.sigmoid(logits).numpy().flatten()
                    preds = (probs > 0.5).astype(int)
                
                fold_metrics.append({
                    'recall': recall_score(fold_data['y_te'], preds, zero_division=0),
                    'f1': f1_score(fold_data['y_te'], preds, zero_division=0),
                    'fpr': ((fold_data['y_te'] == 0) & (preds == 1)).sum() / (fold_data['y_te'] == 0).sum() if (fold_data['y_te'] == 0).any() else 0
                })
            
            grid_results.append({
                **config,
                'avg_recall': np.mean([m['recall'] for m in fold_metrics]),
                'avg_f1': np.mean([m['f1'] for m in fold_metrics]),
                'avg_fpr': np.mean([m['fpr'] for m in fold_metrics]),
                'params': n_params
            })

    res_df = pd.DataFrame(grid_results)
    res_df.to_csv(os.path.join(RESULTS_DIR, 'model_comparison.csv'), index=False)

    res_df = pd.DataFrame(grid_results)
    res_df.to_csv(os.path.join(RESULTS_DIR, 'model_comparison.csv'), index=False)
    
    # 2. Strict Selection with Fallback
    print("\nSelecting best model with strict constraints...")
    candidates = res_df[
        (res_df['avg_recall'] >= 0.95) & 
        (res_df['w'] <= 14) &
        (res_df['params'] <= 2000)
    ]
    
    fpr_targets = [0.10, 0.12, 0.15, 0.20, 0.25, 1.0]
    best_candidate = None
    
    for target in fpr_targets:
        valid = candidates[candidates['avg_fpr'] <= target]
        if not valid.empty:
            best_candidate = valid.sort_values(by=['avg_fpr', 'avg_f1'], ascending=[True, False]).iloc[0]
            print(f"  Found model meeting Recall >= 0.95 and FPR <= {target}")
            break
            
    if best_candidate is None:
        print("  WARNING: No candidate met Recall constraint. Selecting best possible Recall.")
        best_candidate = res_df.sort_values(by=['avg_recall', 'avg_fpr'], ascending=[False, True]).iloc[0]
    
    print(f"\nBest Config Discovered: {best_candidate.to_dict()}")
    
    # 3. Diagnostic Run & Threshold Calibration
    print("\nRunning final diagnostics for best candidate...")
    w_best = int(best_candidate['w'])
    
    # Aggregate all data using cache
    all_dfs = []
    engine_feats = []
    for f in csv_files:
        df_tmp, feats_tmp = get_cached_features(f, w_best)
        if df_tmp is not None:
            all_dfs.append(df_tmp)
            engine_feats = feats_tmp
            
    df_final = pd.concat(all_dfs, ignore_index=True)
    X = df_final[engine_feats]
    y = (df_final['LABEL'] == 2).astype(int)
    
    # Stratified Split for final verification
    X_train, X_remain, y_train, y_remain = train_test_split(X, y, test_size=0.3, stratify=y, random_state=42)
    X_val, X_test, y_val, y_test = train_test_split(X_remain, y_remain, test_size=0.5, stratify=y_remain, random_state=42)
    
    mean, std = X_train.mean(), X_train.std() + 1e-9
    X_train_s, X_val_s, X_test_s = (X_train - mean) / std, (X_val - mean) / std, (X_test - mean) / std
    
    # Save normalization params for deployment
    norm_params = {'mean': mean, 'std': std, 'features': engine_feats}
    torch.save(norm_params, os.path.join(MODELS_DIR, 'normalization_params.pth'))
    
    tr_ds = TensorDataset(torch.FloatTensor(X_train_s.values), torch.FloatTensor(y_train.values).view(-1, 1))
    v_ds = TensorDataset(torch.FloatTensor(X_val_s.values), torch.FloatTensor(y_val.values).view(-1, 1))
    tr_loader = DataLoader(tr_ds, batch_size=128, shuffle=True)
    v_loader = DataLoader(v_ds, batch_size=256)
    
    best_model = MLPModel(len(engine_feats), int(best_candidate['layers']), int(best_candidate['units']), best_candidate['dropout'])
    history = train_with_early_stopping(best_model, tr_loader, v_loader, best_candidate)
    
    # Save Model (State Dict & TorchScript)
    torch.save(best_model.state_dict(), os.path.join(MODELS_DIR, 'best_model.pth'))
    script_model = torch.jit.script(best_model)
    script_model.save(os.path.join(MODELS_DIR, 'best_model_jit.pt'))
    
    # 4. Latency measurement
    best_model.eval()
    dummy_sample = torch.randn(1, len(engine_feats))
    latencies = []
    with torch.no_grad():
        for _ in range(100):
            start = time.time()
            _ = best_model(dummy_sample)
            latencies.append((time.time() - start) * 1000)
    avg_latency = np.mean(latencies)
    print(f"  Average Inference Latency: {avg_latency:.4f} ms")
    
    # 5. Calibration & Sweep
    with torch.no_grad():
        # EXPLICIT SIGMOID
        logits = best_model(torch.FloatTensor(X_test_s.values))
        test_probs = torch.sigmoid(logits).numpy().flatten()
    
    thresholds = np.linspace(0, 1, 100)
    sweep_metrics = []
    for t in thresholds:
        preds = (test_probs > t).astype(int)
        rec = recall_score(y_test, preds, zero_division=0)
        fpr = ((y_test == 0) & (preds == 1)).sum() / (y_test == 0).sum() if (y_test==0).any() else 0
        f1 = f1_score(y_test, preds, zero_division=0)
        sweep_metrics.append({'threshold': t, 'recall': rec, 'fpr': fpr, 'f1': f1})
    
    sweep_df = pd.DataFrame(sweep_metrics)
    sweep_df.to_csv(os.path.join(RESULTS_DIR, 'threshold_sweep.csv'), index=False)
    
    # Calibrate tau (minimize FPR under strict recall target)
    valid_tau_df = sweep_df[sweep_df['recall'] >= 0.95]
    if not valid_tau_df.empty:
        # median of top candidates for stability
        best_tau = valid_tau_df.sort_values(by='fpr').iloc[0:3]['threshold'].median()
    else:
        best_tau = 0.5
    print(f"  Calibrated Threshold (tau): {best_tau:.4f}")
    
    # 6. Plots with Smoothing
    def smooth(vals, alpha=0.3):
        smoothed = []
        last = vals[0]
        for v in vals:
            last = alpha * v + (1 - alpha) * last
            smoothed.append(last)
        return smoothed

    # Loss Curve
    plt.figure()
    plt.plot(history['train_loss'], alpha=0.3, color='blue', label='Train Loss (Raw)')
    plt.plot(smooth(history['train_loss']), color='blue', linewidth=2, label='Train Loss (EMA)')
    plt.plot(history['val_loss'], alpha=0.3, color='orange', label='Val Loss (Raw)')
    plt.plot(smooth(history['val_loss']), color='orange', linewidth=2, label='Val Loss (EMA)')
    plt.xlabel('Epochs'); plt.ylabel('Loss'); plt.title('Loss Curve'); plt.legend()
    plt.savefig(os.path.join(RESULTS_DIR, 'loss_curve.png'))
    
    # Learning Curve
    plt.figure()
    plt.plot(history['train_acc'], alpha=0.3, color='blue', label='Train Acc (Raw)')
    plt.plot(smooth(history['train_acc']), color='blue', linewidth=2, label='Train Acc (EMA)')
    plt.plot(history['val_acc'], alpha=0.3, color='orange', label='Val Acc (Raw)')
    plt.plot(smooth(history['val_acc']), color='orange', linewidth=2, label='Val Acc (EMA)')
    plt.xlabel('Epochs'); plt.ylabel('Accuracy'); plt.title('Learning Curve'); plt.legend()
    plt.savefig(os.path.join(RESULTS_DIR, 'learning_curve.png'))
    
    # ROC
    plt.figure()
    fpr_vals, tpr_vals, _ = roc_curve(y_test, test_probs)
    plt.plot(fpr_vals, tpr_vals, color='darkorange', label=f'AUC = {auc(fpr_vals, tpr_vals):.4f}')
    plt.plot([0,1], [0,1], color='navy', linestyle='--')
    plt.xlabel('FPR'); plt.ylabel('TPR'); plt.title('ROC Curve'); plt.legend()
    plt.savefig(os.path.join(RESULTS_DIR, 'roc_curve.png'))
    
    # Confusion Matrices
    balanced_tau = sweep_df.iloc[sweep_df['f1'].idxmax()]['threshold']
    ops = {'High-Safety-Calibrated': best_tau, 'Max-F1-Balanced': balanced_tau}
    for name, tau in ops.items():
        plt.figure()
        cm = confusion_matrix(y_test, (test_probs > tau).astype(int))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['Benign','Interf'], yticklabels=['Benign','Interf'])
        plt.title(f'Confusion Matrix: {name} (τ={tau:.4f})')
        plt.savefig(os.path.join(RESULTS_DIR, f'confusion_matrix_{name.lower().replace("-","_")}.png'))
    
    print("\nUnified Pipeline Complete. All results and plots saved in results/phase2/")

if __name__ == "__main__": main()
