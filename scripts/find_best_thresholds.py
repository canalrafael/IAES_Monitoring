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
        # Handle 1D and 2D arrays
        match = re.search(f"static const float {name}\[\d*\] = \{{(.*?)\}};", content, re.DOTALL)
        if not match:
            match = re.search(f"static const float {name}\[\d*\]\[\d*\] = \{{(.*?)\}};", content, re.DOTALL)
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

def get_all_probs(mean, std, w1, b1, w2, b2, w3, b3):
    csv_files = sorted(glob.glob(os.path.join(DATA_DIR, '*.csv')))
    
    all_probs = []
    all_labels = []
    window_size = 10
    
    print(f"Processing {len(csv_files)} files...")
    for f_path in csv_files:
        df = pd.read_csv(f_path)
        for core_id in df['CORE_ID'].unique():
            cdf = df[df['CORE_ID'] == core_id].copy()
            eps = 1e-9
            
            # Feature Engineering (Windowed Ratios)
            cdf['IPC'] = cdf['INSTRUCTIONS'] / (cdf['CPU_CYCLES'] + eps)
            cdf['MPKI'] = (cdf['CACHE_MISSES'] * 1000) / (cdf['INSTRUCTIONS'] + eps)
            cdf['L2_PRESSURE'] = cdf['L2_CACHE_ACCESS'] / (cdf['CPU_CYCLES'] + eps)
            cdf['BRANCH_MISS_RATE'] = cdf['BRANCH_MISSES'] / (cdf['INSTRUCTIONS'] + eps)
            
            signals = ['IPC', 'MPKI', 'L2_PRESSURE', 'BRANCH_MISS_RATE']
            feats = []
            for sig in signals:
                m = cdf[sig].rolling(window=window_size).mean()
                s = cdf[sig].rolling(window=window_size).std(ddof=1)
                d = cdf[sig].diff(periods=window_size-1)
                feats.extend([m, s, d])
            
            X = np.stack(feats, axis=1)
            valid_mask = ~np.isnan(X).any(axis=1)
            X = X[valid_mask]
            labels = cdf['LABEL'].values[valid_mask]
            
            # Normalization
            X = (X - mean) / (std + eps)
            
            # Inference
            h1 = np.maximum(0, X @ w1.T + b1)
            h2 = np.maximum(0, h1 @ w2.T + b2)
            logits = h2 @ w3.T + b3
            p = sigmoid(logits).flatten()
            
            all_probs.extend(p)
            all_labels.extend(labels)
            
    return np.array(all_probs), np.array(all_labels)

def main():
    mean, std, w1, b1, w2, b2, w3, b3 = parse_header(WEIGHTS_H)
    probs, labels = get_all_probs(mean, std, w1, b1, w2, b2, w3, b3)
    
    is_attack = (labels == 3) # Strict attack recall
    is_benign = np.isin(labels, [0, 1, 2])
    
    results = []
    thresholds = np.linspace(0.01, 0.99, 99)
    
    for tau in thresholds:
        preds = (probs >= tau)
        tp = np.sum(preds & is_attack)
        fn = np.sum(~preds & is_attack)
        fp = np.sum(preds & is_benign)
        tn = np.sum(~preds & is_benign)
        
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        fpr = fp / (fp + tn) if (fp + tn) > 0 else 0
        f1 = 2 * (tp) / (2 * tp + fp + fn) if (2 * tp + fp + fn) > 0 else 0
        
        results.append((tau, recall, fpr, f1))
    
    # 1. Operational (Best F1)
    operational = max(results, key=lambda x: x[3])
    
    # 2. High-Sensitivity (Target ~99.9% Recall)
    high_sens = min([r for r in results if r[1] >= 0.999], key=lambda x: x[2], default=results[0])
    
    # 3. Robust (Target < 3% FPR)
    robust = max([r for r in results if r[2] <= 0.03], key=lambda x: x[1], default=results[-1])

    print("\n--- Optimized Operating Points for C3 ---")
    print(f"{'Configuration':<20} | {'tau':<6} | {'Recall':<10} | {'FPR':<10} | {'F1-Score':<10}")
    print("-" * 65)
    for name, pt in zip(["High-Sensitivity", "Operational", "Robust"], [high_sens, operational, robust]):
        print(f"{name:<20} | {pt[0]:<6.2f} | {pt[1]:<10.4f} | {pt[2]:<10.4f} | {pt[3]:<10.4f}")

if __name__ == "__main__":
    main()
