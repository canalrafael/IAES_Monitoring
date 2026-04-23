import os
import pandas as pd
import numpy as np
from io import StringIO
from scipy.stats import entropy

# --- Paths ---
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
BASE_DIR = os.path.dirname(SCRIPT_DIR)
DATA_DIR = os.path.join(BASE_DIR, "data/train data/")
NEW_DATA_DIR = os.path.join(DATA_DIR, "data_test", "data_detector")
NEW_FILES = ["data_detector3.txt", "data_detector4.txt"]

def calculate_js_divergence(p_data, q_data, bins=50):
    # Determine common range
    min_val = min(p_data.min(), q_data.min())
    max_val = max(p_data.max(), q_data.max())
    
    # Compute histograms
    p_hist, _ = np.histogram(p_data, bins=bins, range=(min_val, max_val), density=True)
    q_hist, _ = np.histogram(q_data, bins=bins, range=(min_val, max_val), density=True)
    
    # Normalize to probabilities (ensure sum=1 and no zeros for entropy)
    p = p_hist / (p_hist.sum() + 1e-10) + 1e-10
    q = q_hist / (q_hist.sum() + 1e-10) + 1e-10
    
    m = 0.5 * (p + q)
    return 0.5 * entropy(p, m) + 0.5 * entropy(q, m)

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
    
    header = "CORE_ID,TIMESTAMP,CPU_CYCLES,INSTRUCTIONS,CACHE_MISSES,BRANCH_MISSES,L2_CACHE_ACCESS,LABEL"
    content = [pmu_lines[0]] + [l for l in pmu_lines[1:] if "CORE_ID" not in l]
    df = pd.read_csv(StringIO("\n".join(content)))
    
    # Feature Engineering (Base + Temporal + TLB Proxy)
    eps = 1e-9
    df['IPC'] = df['INSTRUCTIONS'] / (df['CPU_CYCLES'] + eps)
    df['BRANCH_MISS_RATE'] = df['BRANCH_MISSES'] / (df['INSTRUCTIONS'] + eps)
    
    # TLB Proxy: Derived from L2 Access and Cache Misses
    # Higher combined cache activity per instruction approximates TLB pressure.
    df['TLB_Proxy'] = (df['L2_CACHE_ACCESS'] * 0.05 + df['CACHE_MISSES'] * 0.95) / (df['INSTRUCTIONS'] + eps)
    
    # Temporal features (Std Dev over W=10)
    for col in ['IPC', 'BRANCH_MISS_RATE', 'TLB_Proxy']:
        df[f'{col}_std'] = df[col].rolling(window=10).std().fillna(0)
    
    return df

def run_analysis():
    print("--- Advanced Separability Analysis (JS Divergence & TLB Proxy) ---")
    
    new_dfs = []
    for f in NEW_FILES:
        path = os.path.join(NEW_DATA_DIR, f)
        if os.path.exists(path):
            new_dfs.append(parse_txt_pmu(path))
    
    if not new_dfs:
        print("No new data files found.")
        return
        
    df = pd.concat(new_dfs)
    benign = df[df['LABEL'] == 0]
    attack = df[df['LABEL'] == 2]
    
    print(f"\nSamples Analyzed: Benign={len(benign)}, Attack={len(attack)}")
    
    features = ['IPC', 'BRANCH_MISS_RATE', 'TLB_Proxy', 'IPC_std', 'TLB_Proxy_std']
    
    print("\nMetrics (New Data: Label 0 vs Label 2)")
    print("-" * 65)
    print(f"{'Feature':<25} {'Mean_Ben':>10} {'Mean_Atk':>10} {'JS_Div':>10}")
    print("-" * 65)
    
    for feat in features:
        js_div = calculate_js_divergence(benign[feat], attack[feat])
        m_ben = benign[feat].mean()
        m_atk = attack[feat].mean()
        print(f"{feat:<25} {m_ben:>10.4f} {m_atk:>10.4f} {js_div:>10.4f}")

    print("\n[Comparison with Historical Baseline]")
    print("- Old IPC JS_Divergence: 0.4024")
    print("- Old Branch JS_Divergence: 0.3873")
    
    # Verdict
    new_ipc_js = calculate_js_divergence(benign['IPC'], attack['IPC'])
    if new_ipc_js > 0.25:
        print("\nVERDICT: HIGH JS DIVERGENCE SUSTAINED.")
        print("The distributions still have distinct shapes. Retraining the MLP will likely restore performance.")
    else:
        print("\nVERDICT: CRITICAL SIGNAL LOSS.")
        print("The distributions have converged. IPC/Branch Misses are no longer discriminative in this environment.")

if __name__ == "__main__":
    run_analysis()
