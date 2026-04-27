import os
import pandas as pd
import numpy as np
import glob
import re

# --- Configuration ---
DEPLOY_DIR = "/home/canal/github/IAES_Monitoring/deploy"
DATA_DIR = "/home/canal/github/IAES_Monitoring/data/online validation data"
WEIGHTS_H = os.path.join(DEPLOY_DIR, "model_weights.h")

def parse_header(h_path):
    with open(h_path, 'r') as f:
        content = f.read()
    
    def get_arr(name):
        match = re.search(f"static const float {name}\[\d+\] = \{{(.*?)\}};", content, re.DOTALL)
        if not match:
            # Try 2D array
            match = re.search(f"static const float {name}\[\d+\]\[\d+\] = \{{(.*?)\}};", content, re.DOTALL)
        if not match: return None
        vals = match.group(1).replace('{','').replace('}','').replace('f','').split(',')
        return np.array([float(v.strip()) for v in vals if v.strip()])

    mean = get_arr("MDL_FEAT_MEAN")
    std = get_arr("MDL_FEAT_STD")
    w1 = get_arr("MDL_W1").reshape(16, 12)
    b1 = get_arr("MDL_B1")
    w2 = get_arr("MDL_W2").reshape(16, 16)
    b2 = get_arr("MDL_B2")
    w3 = get_arr("MDL_W3").reshape(1, 16)
    b3 = get_arr("MDL_B3")
    
    return mean, std, w1, b1, w2, b2, w3, b3

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def run_sim(threshold, mean, std, w1, b1, w2, b2, w3, b3):
    csv_files = [f for f in sorted(glob.glob(os.path.join(DATA_DIR, '*.csv')))
                 if any(x in os.path.basename(f) for x in ["42", "43"])]
    
    tp, fp, tn, fn = 0, 0, 0, 0
    window_size = 10
    
    for f_path in csv_files:
        df = pd.read_csv(f_path)
        # Process per core
        for core_id in df['CORE_ID'].unique():
            cdf = df[df['CORE_ID'] == core_id].copy()
            eps = 1e-9
            
            # 1. Ratios
            cdf['IPC'] = cdf['INSTRUCTIONS'] / (cdf['CPU_CYCLES'] + eps)
            cdf['MPKI'] = (cdf['CACHE_MISSES'] * 1000) / (cdf['INSTRUCTIONS'] + eps)
            cdf['L2_PRESSURE'] = cdf['L2_CACHE_ACCESS'] / (cdf['CPU_CYCLES'] + eps)
            cdf['BRANCH_MISS_RATE'] = cdf['BRANCH_MISSES'] / (cdf['INSTRUCTIONS'] + eps)
            
            signals = ['IPC', 'MPKI', 'L2_PRESSURE', 'BRANCH_MISS_RATE']
            
            # 2. Rolling Stats
            feats = []
            for sig in signals:
                m = cdf[sig].rolling(window=window_size).mean()
                s = cdf[sig].rolling(window=window_size).std(ddof=1)
                d = cdf[sig].diff(periods=window_size-1)
                feats.extend([m, s, d])
            
            X = np.stack(feats, axis=1)
            # Remove warmup rows (first window_size-1 rows of the core)
            valid_mask = ~np.isnan(X).any(axis=1)
            X = X[valid_mask]
            labels = cdf['LABEL'].values[valid_mask]
            
            # 3. Normalization
            X = (X - mean) / (std + eps)
            
            # 4. MLP Inference
            h1 = np.maximum(0, X @ w1.T + b1)
            h2 = np.maximum(0, h1 @ w2.T + b2)
            logits = h2 @ w3.T + b3
            probs = sigmoid(logits).flatten()
            
            preds = (probs >= threshold)
            is_attack = np.isin(labels, [1, 2, 3])
            
            tp += np.sum(preds & is_attack)
            fp += np.sum(preds & ~is_attack)
            tn += np.sum(~preds & ~is_attack)
            fn += np.sum(~preds & is_attack)
            
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    fpr = fp / (fp + tn) if (fp + tn) > 0 else 0
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
    
    return recall, fpr, f1

def main():
    mean, std, w1, b1, w2, b2, w3, b3 = parse_header(WEIGHTS_H)
    
    print(f"{'Configuration':<20} | {'tau':<6} | {'Recall':<10} | {'FPR':<10} | {'F1-Score':<10}")
    print("-" * 65)
    
    points = [0.05, 0.50, 0.90]
    names = ["High-Sensitivity", "Operational", "Robust"]
    
    for name, tau in zip(names, points):
        recall, fpr, f1 = run_sim(tau, mean, std, w1, b1, w2, b2, w3, b3)
        print(f"{name:<20} | {tau:<6.2f} | {recall:<10.4f} | {fpr:<10.4f} | {f1:<10.4f}")

if __name__ == "__main__":
    main()
