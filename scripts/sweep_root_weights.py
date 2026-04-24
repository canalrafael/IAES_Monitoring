import pandas as pd
import numpy as np
import os
import sys
sys.path.append(os.path.join(os.path.dirname(__file__)))
from parse_weights import parse_weights

H_PATH = "/home/canal/github/IAES_Monitoring/deploy/model_weights.h"
MDL_FEAT_MEAN, MDL_FEAT_STD, MDL_W1, MDL_B1, MDL_W2, MDL_B2, MDL_W3, MDL_B3 = parse_weights(H_PATH)

def sigmoid(x): return 1 / (1 + np.exp(-x))
def relu(x): return np.maximum(0, x)
def detector_inference(feat):
    h1 = relu(np.dot(MDL_W1, feat) + MDL_B1)
    h2 = relu(np.dot(MDL_W2, h1) + MDL_B2)
    logit = np.dot(MDL_W3, h2) + MDL_B3
    return sigmoid(logit[0])

def sweep_root():
    df = pd.read_csv("/home/canal/github/IAES_Monitoring/data/online validation data/data_new21_clean.csv")
    eps = 1e-9
    results = []
    for cid in sorted(df['CORE_ID'].unique()):
        core_df = df[df['CORE_ID'] == cid].copy()
        core_df['sig_ipc'] = core_df['INSTRUCTIONS'] / (core_df['CPU_CYCLES'] + eps)
        core_df['sig_mpki'] = (core_df['CACHE_MISSES'] * 1000) / (core_df['INSTRUCTIONS'] + eps)
        core_df['sig_l2p'] = core_df['L2_CACHE_ACCESS'] / (core_df['CPU_CYCLES'] + eps)
        core_df['sig_branch'] = core_df['BRANCH_MISSES'] / (core_df['INSTRUCTIONS'] + eps)
        window = []
        probs = []
        for _, row in core_df.iterrows():
            curr = np.array([row['sig_ipc'], row['sig_mpki'], row['sig_l2p'], row['sig_branch']])
            window.append(curr); 
            if len(window) > 10: window.pop(0)
            if len(window) < 10: probs.append(0.0); continue
            m = np.mean(window, axis=0); s = np.std(window, axis=0, ddof=1); d = curr - window[0]
            feat = []
            for i in range(4): feat.extend([m[i], s[i], d[i]])
            feat_norm = (np.array(feat) - MDL_FEAT_MEAN) / (MDL_FEAT_STD + eps)
            probs.append(detector_inference(feat_norm))
        core_df['PROB'] = probs
        results.append(core_df)
    
    full_df = pd.concat(results)
    ts_df = full_df.groupby('TIMESTAMP').agg({'LABEL': lambda x: any(l in [1, 2, 3] for l in x), 'PROB': 'max'})
    
    print("\nDeploy Root Weights - System-Wide Sweep:")
    print(f"{'TH':<5} | {'Recall (%)':<15} | {'FPR (%)':<10}")
    for th in [0.5, 0.6, 0.7, 0.8, 0.9, 0.95, 0.99]:
        rec = (ts_df[ts_df.LABEL].PROB >= th).mean() * 100
        fpr = (ts_df[~ts_df.LABEL].PROB >= th).mean() * 100
        print(f"{th:<5.2f} | {rec:<15.2f} | {fpr:<10.2f}")

if __name__ == "__main__": sweep_root()
