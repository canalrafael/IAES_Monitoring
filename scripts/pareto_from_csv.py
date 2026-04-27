import os
import pandas as pd
import numpy as np
import glob

# --- Configuration ---
DATA_DIR = "/home/canal/github/IAES_Monitoring/data/online validation data"

def run_pareto_sweep():
    csv_files = [f for f in sorted(glob.glob(os.path.join(DATA_DIR, '*.csv')))
                 if any(x in os.path.basename(f) for x in ["42", "43"])]
    
    dfs = [pd.read_csv(f) for f in csv_files]
    df = pd.concat(dfs, ignore_index=True)
    
    # Filter out warmup (DET_STATUS == 0)
    df = df[df['DET_STATUS'] != 0].copy()
    
    # DET_PROB is in [0, 100]
    # Thresholds in [0, 1]
    
    thresholds = [0.05, 0.50, 0.90]
    names = ["High-Sensitivity", "Operational", "Robust"]
    
    print(f"{'Configuration':<20} | {'tau':<6} | {'Recall':<10} | {'FPR':<10} | {'F1-Score':<10}")
    print("-" * 65)
    
    # Attack recall focuses on Label 3 (Spectre, Armageddon, Meltdown)
    is_attack = (df['LABEL'] == 3)
    # FPR focuses on Label 0, 1, 2 (Benign, Untrusted, Interfered)
    is_benign = (df['LABEL'].isin([0, 1, 2]))
    
    probs = df['DET_PROB'].values / 100.0
    
    for name, tau in zip(names, thresholds):
        preds = (probs >= tau)
        
        tp = np.sum(preds & is_attack)
        fn = np.sum(~preds & is_attack)
        
        fp = np.sum(preds & is_benign)
        tn = np.sum(~preds & is_benign)
        
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        fpr = fp / (fp + tn) if (fp + tn) > 0 else 0
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
        
        print(f"{name:<20} | {tau:<6.2f} | {recall:<10.4f} | {fpr:<10.4f} | {f1:<10.4f}")

if __name__ == "__main__":
    run_pareto_sweep()
