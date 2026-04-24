import pandas as pd
import numpy as np
import os
import sys

# Ensure we can import parse_weights
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from parse_weights import parse_weights

# --- Configuration ---
H_PATH = "/home/canal/github/IAES_Monitoring/deploy/online_validation/model_weights.h"
MDL_FEAT_MEAN, MDL_FEAT_STD, MDL_W1, MDL_B1, MDL_W2, MDL_B2, MDL_W3, MDL_B3 = parse_weights(H_PATH)

BENCH_NAMES = {
    0: "Idle", 1: "Bandwidth", 2: "Disparity", 3: "FFT", 4: "QSort", 
    5: "Dijkstra", 6: "SHA", 7: "Sorting", 10: "Spectre", 11: "Armageddon", 12: "Meltdown"
}

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def relu(x):
    return np.maximum(0, x)

def detector_inference(feat):
    h1 = relu(np.dot(MDL_W1, feat) + MDL_B1)
    h2 = relu(np.dot(MDL_W2, h1) + MDL_B2)
    logit = np.dot(MDL_W3, h2) + MDL_B3
    return sigmoid(logit[0])

def analyze_file(csv_path):
    print(f"Processing {os.path.basename(csv_path)}...")
    df = pd.read_csv(csv_path)
    eps = 1e-9
    
    results = []
    for cid in sorted(df['CORE_ID'].unique()):
        core_df = df[df['CORE_ID'] == cid].copy()
        
        # Signals
        core_df['sig_ipc'] = core_df['INSTRUCTIONS'] / (core_df['CPU_CYCLES'] + eps)
        core_df['sig_mpki'] = (core_df['CACHE_MISSES'] * 1000) / (core_df['INSTRUCTIONS'] + eps)
        core_df['sig_l2p'] = core_df['L2_CACHE_ACCESS'] / (core_df['CPU_CYCLES'] + eps)
        core_df['sig_branch'] = core_df['BRANCH_MISSES'] / (core_df['INSTRUCTIONS'] + eps)
        
        window = []
        sim_probs = []
        for _, row in core_df.iterrows():
            curr = np.array([row['sig_ipc'], row['sig_mpki'], row['sig_l2p'], row['sig_branch']])
            window.append(curr)
            if len(window) > 10: window.pop(0)
            
            if len(window) < 10:
                sim_probs.append(0.0)
                continue
                
            win_arr = np.array(window)
            m = np.mean(win_arr, axis=0)
            s = np.std(win_arr, axis=0, ddof=1)
            d = curr - window[0]
            
            feat = []
            for i in range(4): feat.extend([m[i], s[i], d[i]])
            feat_norm = (np.array(feat) - MDL_FEAT_MEAN) / (MDL_FEAT_STD + eps)
            
            sim_probs.append(detector_inference(feat_norm))
            
        core_df['SIM_PROB'] = sim_probs
        results.append(core_df)
        
    return pd.concat(results).sort_index()

def get_system_comparison(df, bao_col, sim_prob_col, threshold=0.5):
    # Group by timestamp for system-wide comparison (Bao Metric)
    ts_df = df.groupby('TIMESTAMP').agg({
        'LABEL': lambda x: any(l in [1, 2, 3] for l in x),
        bao_col: lambda x: any(s == 2 for s in x),
        sim_prob_col: lambda x: any(p >= threshold for p in x),
        'BENCH_ID': 'max'
    }).reset_index()
    
    report = []
    for bid in sorted(ts_df['BENCH_ID'].unique()):
        sub = ts_df[ts_df['BENCH_ID'] == bid]
        total_p = sub['LABEL'].sum()
        total_n = (~sub['LABEL']).sum()
        
        # Bao
        tp_bao = (sub[bao_col] & sub['LABEL']).sum()
        fp_bao = (sub[bao_col] & ~sub['LABEL']).sum()
        
        # Sim
        tp_sim = (sub[sim_prob_col] & sub['LABEL']).sum()
        fp_sim = (sub[sim_prob_col] & ~sub['LABEL']).sum()
        
        report.append({
            'Bench': BENCH_NAMES.get(bid, str(bid)),
            'Samples': len(sub),
            'Bao Rec (%)': (tp_bao/total_p*100) if total_p > 0 else np.nan,
            'Sim Rec (%)': (tp_sim/total_p*100) if total_p > 0 else np.nan,
            'Bao FPR (%)': (fp_bao/total_n*100) if total_n > 0 else np.nan,
            'Sim FPR (%)': (fp_sim/total_n*100) if total_n > 0 else np.nan
        })

    report.append({
        'Bench': '--- GENERAL SUM ---',
        'Samples': len(ts_df),
        'Bao Rec (%)': (ts_df[ts_df['LABEL'] == True][bao_col].sum() / ts_df['LABEL'].sum() * 100) if ts_df['LABEL'].any() else np.nan,
        'Sim Rec (%)': (ts_df[ts_df['LABEL'] == True][sim_prob_col].sum() / ts_df['LABEL'].sum() * 100) if ts_df['LABEL'].any() else np.nan,
        'Bao FPR (%)': (ts_df[ts_df['LABEL'] == False][bao_col].sum() / (~ts_df['LABEL']).sum() * 100) if (~ts_df['LABEL']).any() else np.nan,
        'Sim FPR (%)': (ts_df[ts_df['LABEL'] == False][sim_prob_col].sum() / (~ts_df['LABEL']).sum() * 100) if (~ts_df['LABEL']).any() else np.nan
    })
    
    return pd.DataFrame(report)

if __name__ == "__main__":
    DATA_DIR = "/home/canal/github/IAES_Monitoring/data/online validation data"
    FILES = ["data_new20_clean.csv", "data_new21_clean.csv"]
    
    for f in FILES:
        path = os.path.join(DATA_DIR, f)
        df_p = analyze_file(path)
        print(f"\n--- {f} System-Wide (Bao vs Sim) ---")
        comp = get_system_comparison(df_p, 'DET_STATUS', 'SIM_PROB')
        print(comp.to_string(index=False))
