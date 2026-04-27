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
        match = re.search(f"static const float {name}\[\d*\] = \{{(.*?)\}};", content, re.DOTALL)
        if not match:
            match = re.search(f"static const float {name}\[\d*\]\[\d*\] = \{{(.*?)\}};", content, re.DOTALL)
        if not match: return None
        vals = match.group(1).replace('{','').replace('}','').replace('f','').split(',')
        return np.array([float(v.strip()) for v in vals if v.strip()])
    return (get_arr("MDL_FEAT_MEAN"), get_arr("MDL_FEAT_STD"), 
            get_arr("MDL_W1").reshape(16, 12), get_arr("MDL_B1"), 
            get_arr("MDL_W2").reshape(16, 16), get_arr("MDL_B2"), 
            get_arr("MDL_W3").reshape(1, 16), get_arr("MDL_B3"))

def sigmoid(x): return 1 / (1 + np.exp(-x))

def main():
    mean, std, w1, b1, w2, b2, w3, b3 = parse_header(WEIGHTS_H)
    csv_files = sorted(glob.glob(os.path.join(DATA_DIR, '*.csv')))
    
    bench_probs = {} # bench_id -> list of probs
    bench_names = {0: "Idle", 1: "Bandwidth", 2: "Disparity", 3: "FFT", 4: "QSort", 5: "Dijkstra", 6: "SHA", 7: "Sorting", 10: "Spectre", 11: "Armageddon", 12: "Meltdown"}
    window_size = 10
    
    for f_path in csv_files:
        df = pd.read_csv(f_path)
        for core_id in df['CORE_ID'].unique():
            cdf = df[df['CORE_ID'] == core_id].copy()
            eps = 1e-9
            cdf['IPC'] = cdf['INSTRUCTIONS'] / (cdf['CPU_CYCLES'] + eps)
            cdf['MPKI'] = (cdf['CACHE_MISSES'] * 1000) / (cdf['INSTRUCTIONS'] + eps)
            cdf['L2_PRESSURE'] = cdf['L2_CACHE_ACCESS'] / (cdf['CPU_CYCLES'] + eps)
            cdf['BRANCH_MISS_RATE'] = cdf['BRANCH_MISSES'] / (cdf['INSTRUCTIONS'] + eps)
            
            signals = ['IPC', 'MPKI', 'L2_PRESSURE', 'BRANCH_MISS_RATE']
            feats = []
            for sig in signals:
                feats.extend([cdf[sig].rolling(10).mean(), cdf[sig].rolling(10).std(ddof=1), cdf[sig].diff(9)])
            
            X = np.stack(feats, axis=1)
            valid = ~np.isnan(X).any(axis=1)
            X = X[valid]
            b_ids = cdf['BENCH_ID'].values[valid]
            
            X = (X - mean) / (std + eps)
            h1 = np.maximum(0, X @ w1.T + b1)
            h2 = np.maximum(0, h1 @ w2.T + b2)
            p = sigmoid(h2 @ w3.T + b3).flatten()
            
            for b_id, prob in zip(b_ids, p):
                if b_id not in bench_probs: bench_probs[b_id] = []
                bench_probs[b_id].append(prob)
                
    print(f"{'Benchmark':<15} | {'Count':<8} | {'Mean Prob':<10} | {'Max Prob':<10} | {'Det@0.5':<10}")
    print("-" * 65)
    for b_id in sorted(bench_probs.keys()):
        name = bench_names.get(b_id, f"ID_{b_id}")
        probs = np.array(bench_probs[b_id])
        det = np.sum(probs >= 0.5) / len(probs)
        print(f"{name:<15} | {len(probs):<8} | {np.mean(probs):<10.4f} | {np.max(probs):<10.4f} | {det:<10.2%}")

if __name__ == "__main__":
    main()
