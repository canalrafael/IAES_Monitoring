import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import glob
import os
from scipy.stats import ks_2samp
from scipy.spatial.distance import jensenshannon

# --- Configuration ---
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
BASE_DIR = os.path.dirname(SCRIPT_DIR)
DATA_DIR = os.path.join(BASE_DIR, 'data', 'online validation data')
RESULTS_DIR = os.path.join(BASE_DIR, 'results', 'separability_analysis')
os.makedirs(RESULTS_DIR, exist_ok=True)

def calculate_derived(df):
    eps = 1e-9
    df = df.copy()
    df['IPC'] = df['INSTRUCTIONS'] / (df['CPU_CYCLES'] + eps)
    df['MPKI'] = (df['CACHE_MISSES'] * 1000) / (df['INSTRUCTIONS'] + eps)
    df['L2_PRESSURE'] = df['L2_CACHE_ACCESS'] / (df['CPU_CYCLES'] + eps)
    df['BRANCH_MISS_RATE'] = df['BRANCH_MISSES'] / (df['INSTRUCTIONS'] + eps)
    return df

def js_divergence(p, q, bins=100):
    p = p[~np.isnan(p)]; q = q[~np.isnan(q)]
    if len(p) == 0 or len(q) == 0: return 0
    mn = min(p.min(), q.min()); mx = max(p.max(), q.max())
    p_hist, _ = np.histogram(p, bins=bins, range=(mn, mx), density=True)
    q_hist, _ = np.histogram(q, bins=bins, range=(mn, mx), density=True)
    p_hist += 1e-10; q_hist += 1e-10
    return jensenshannon(p_hist, q_hist)

print("Loading Online Validation Data...")
csv_files = glob.glob(os.path.join(DATA_DIR, 'data_new*_clean.csv'))
df = pd.concat([pd.read_csv(f) for f in csv_files], ignore_index=True)
df = calculate_derived(df)

benign = df[df['LABEL'] == 0]
attacker = df[df['LABEL'] == 3]

print(f"Benign samples: {len(benign)}")
print(f"Attacker (L3) samples: {len(attacker)}")

features = ['IPC', 'MPKI', 'L2_PRESSURE', 'BRANCH_MISS_RATE']
metrics_results = []

for f in features:
    b_vals = benign[f].values
    a_vals = attacker[f].values
    b_mean, a_mean = np.mean(b_vals), np.mean(a_vals)
    b_std, a_std = np.std(b_vals), np.std(a_vals)
    d = (a_mean - b_mean) / np.sqrt((b_std**2 + a_std**2) / 2)
    js = js_divergence(b_vals, a_vals)
    
    metrics_results.append({
        "Feature": f,
        "Benign_Mean": b_mean,
        "Attacker_Mean": a_mean,
        "Cohen_d": d,
        "JS_Div": js
    })

metrics_df = pd.DataFrame(metrics_results).sort_values(by="JS_Div", ascending=False)
print("\n--- Feature Separability (L0 vs L3 - Attacker Core) ---")
print(metrics_df.to_string(index=False))

# Visualization
fig, axes = plt.subplots(2, 2, figsize=(16, 12))
axes = axes.flatten()
for i, f in enumerate(features):
    sns.kdeplot(benign[f], ax=axes[i], label='Benign (Alone)', fill=True, alpha=0.3, color='blue')
    sns.kdeplot(attacker[f], ax=axes[i], label='Attacker (L3)', fill=True, alpha=0.3, color='green')
    axes[i].set_title(f'{f} Separability (L0 vs L3)')
    axes[i].legend()
    q_high = max(benign[f].quantile(0.99), attacker[f].quantile(0.99))
    axes[i].set_xlim(0, q_high * 1.5)

plt.tight_layout()
plt.savefig(os.path.join(RESULTS_DIR, 'attacker_separability.png'))
print(f"\nAttacker analysis complete. Plot saved in: {RESULTS_DIR}/attacker_separability.png")
