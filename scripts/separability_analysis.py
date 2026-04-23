import os
import pandas as pd
import numpy as np
from io import StringIO

# --- Paths ---
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
BASE_DIR = os.path.dirname(SCRIPT_DIR)
DATA_DIR = os.path.join(BASE_DIR, "data")
NEW_DATA_DIR = os.path.join(DATA_DIR, "data_test", "data_detector")

# Baseline files (Historical)
OLD_BENIGN = os.path.join(DATA_DIR, "data0_clean.csv")
OLD_ATTACK = os.path.join(DATA_DIR, "data4_clean.csv")

# New Deployment files
NEW_FILES = ["data_detector3.txt", "data_detector4.txt"]

def cohen_d(x, y):
    nx = len(x)
    ny = len(y)
    dof = nx + ny - 2
    return (np.mean(x) - np.mean(y)) / np.sqrt(((nx-1)*np.std(x, ddof=1)**2 + (ny-1)*np.std(y, ddof=1)**2) / dof)

def parse_txt_pmu(filepath):
    with open(filepath, 'r') as f:
        lines = f.readlines()
    pmu_lines = []
    current_section = None
    if lines and "CORE_ID" in lines[0]: current_section = "PMU"
    for line in lines:
        line = line.strip()
        if not line: continue
        if line == "PMU_START": current_section = "PMU"; continue
        elif line == "PMU_END": current_section = None; continue
        if current_section == "PMU": pmu_lines.append(line)
    
    # Filter intermediate headers
    header = "CORE_ID,TIMESTAMP,CPU_CYCLES,INSTRUCTIONS,CACHE_MISSES,BRANCH_MISSES,L2_CACHE_ACCESS,LABEL"
    content = [pmu_lines[0]] + [l for l in pmu_lines[1:] if "CORE_ID" not in l]
    df = pd.read_csv(StringIO("\n".join(content)))
    
    # Feature Engineering
    eps = 1e-9
    df['IPC'] = df['INSTRUCTIONS'] / (df['CPU_CYCLES'] + eps)
    df['MPKI'] = (df['CACHE_MISSES'] * 1000) / (df['INSTRUCTIONS'] + eps)
    df['L2_PRESSURE'] = df['L2_CACHE_ACCESS'] / (df['CPU_CYCLES'] + eps)
    df['BRANCH_MISS_RATE'] = df['BRANCH_MISSES'] / (df['INSTRUCTIONS'] + eps)
    return df

def analyze_separability():
    print("--- Microarchitectural Separability Analysis ---")
    
    # 1. Load New Data
    new_dfs = []
    for f in NEW_FILES:
        path = os.path.join(NEW_DATA_DIR, f)
        if os.path.exists(path):
            new_dfs.append(parse_txt_pmu(path))
    
    if not new_dfs:
        print("No new data files found.")
        return
        
    df_new = pd.concat(new_dfs)
    new_benign = df_new[df_new['LABEL'] == 0]
    new_attack = df_new[df_new['LABEL'] == 2]
    
    # 2. Load Old Data (Baseline)
    old_benign = pd.read_csv(OLD_BENIGN)
    old_attack = pd.read_csv(OLD_ATTACK)
    
    # Add features to old data if missing
    for df_o in [old_benign, old_attack]:
        eps = 1e-9
        df_o['IPC'] = df_o['INSTRUCTIONS'] / (df_o['CPU_CYCLES'] + eps)
        df_o['MPKI'] = (df_o['CACHE_MISSES'] * 1000) / (df_o['INSTRUCTIONS'] + eps)
    
    features = ['IPC', 'MPKI']
    
    print("\n[V1] Comparison: New Benign vs New Attack")
    print("-" * 50)
    for feat in features:
        d = cohen_d(new_attack[feat], new_benign[feat])
        print(f"Feature: {feat}")
        print(f"  Atk Mean: {new_attack[feat].mean():.4f} | Ben Mean: {new_benign[feat].mean():.4f}")
        print(f"  Cohen's d: {d:.4f} (Separation)")
        
    print("\n[V2] Environment Shift: Old Benign vs New Benign")
    print("-" * 50)
    for feat in features:
        d = cohen_d(new_benign[feat], old_benign[feat])
        print(f"Feature: {feat}")
        print(f"  Old Ben: {old_benign[feat].mean():.4f} | New Ben: {new_benign[feat].mean():.4f}")
        print(f"  Shift (Cohen's d): {d:.4f}")

    print("\n--- Summary Verdict ---")
    ipc_sep = cohen_d(new_attack['IPC'], new_benign['IPC'])
    if abs(ipc_sep) > 1.5:
        print("RESULT: HIGH SEPARABILITY. A retrained model will likely solve the issue.")
    elif abs(ipc_sep) > 0.8:
        print("RESULT: MODERATE SEPARABILITY. Retraining + Online Calibration recommended.")
    else:
        print("RESULT: LOW SEPARABILITY. Consider adding new PMU counters (e.g. TLB misses).")

if __name__ == "__main__":
    analyze_separability()
