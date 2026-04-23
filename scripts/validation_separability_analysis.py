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

sns.set_theme(style="whitegrid")

def calculate_derived(df):
    eps = 1e-9
    df = df.copy()
    df['IPC'] = df['INSTRUCTIONS'] / (df['CPU_CYCLES'] + eps)
    df['MPKI'] = (df['CACHE_MISSES'] * 1000) / (df['INSTRUCTIONS'] + eps)
    df['L2_PRESSURE'] = df['L2_CACHE_ACCESS'] / (df['CPU_CYCLES'] + eps)
    df['BRANCH_MISS_RATE'] = df['BRANCH_MISSES'] / (df['INSTRUCTIONS'] + eps)
    return df

def js_divergence(p, q, bins=100):
    """Calculate Jensen-Shannon divergence between two distributions."""
    # Create common bins
    p = p[~np.isnan(p)]
    q = q[~np.isnan(q)]
    if len(p) == 0 or len(q) == 0: return 0
    
    mn = min(p.min(), q.min())
    mx = max(p.max(), q.max())
    p_hist, _ = np.histogram(p, bins=bins, range=(mn, mx), density=True)
    q_hist, _ = np.histogram(q, bins=bins, range=(mn, mx), density=True)
    
    # Add small epsilon to avoid log(0)
    p_hist += 1e-10
    q_hist += 1e-10
    
    return jensenshannon(p_hist, q_hist)

# 1. Load data
print("Loading Online Validation Data...")
csv_files = glob.glob(os.path.join(DATA_DIR, 'data_new*_clean.csv'))
all_data = []
for f in csv_files:
    all_data.append(pd.read_csv(f))

df = pd.concat(all_data, ignore_index=True)
df = calculate_derived(df)

# 2. Focus on Label 0 (Benign) vs Label 2 (Interfered)
benign = df[df['LABEL'] == 0]
attack = df[df['LABEL'] == 2]

print(f"Benign samples: {len(benign)}")
print(f"Attack (L2) samples: {len(attack)}")

features = ['IPC', 'MPKI', 'L2_PRESSURE', 'BRANCH_MISS_RATE']
metrics_results = []

for f in features:
    b_vals = benign[f].values
    a_vals = attack[f].values
    
    # Cohen's d
    b_mean, a_mean = np.mean(b_vals), np.mean(a_vals)
    b_std, a_std = np.std(b_vals), np.std(a_vals)
    d = (a_mean - b_mean) / np.sqrt((b_std**2 + a_std**2) / 2)
    
    # KS Test
    ks_stat, _ = ks_2samp(b_vals, a_vals)
    
    # JS Divergence
    js = js_divergence(b_vals, a_vals)
    
    metrics_results.append({
        "Feature": f,
        "Benign_Mean": b_mean,
        "Attack_Mean": a_mean,
        "Cohen_d": d,
        "KS_Stat": ks_stat,
        "JS_Div": js
    })

metrics_df = pd.DataFrame(metrics_results)
metrics_df = metrics_df.sort_values(by="JS_Div", ascending=False)

print("\n--- Feature Separability (L0 vs L2) ---")
print(metrics_df.to_string(index=False))
metrics_df.to_csv(os.path.join(RESULTS_DIR, 'separability_metrics.csv'), index=False)

# 3. Visualization
# KDE Plots
fig, axes = plt.subplots(2, 2, figsize=(16, 12))
axes = axes.flatten()
for i, f in enumerate(features):
    sns.kdeplot(benign[f], ax=axes[i], label='Benign (Alone)', fill=True, alpha=0.3, color='blue')
    sns.kdeplot(attack[f], ax=axes[i], label='Interfered (L2)', fill=True, alpha=0.3, color='red')
    axes[i].set_title(f'{f} Separability (JS={metrics_df[metrics_df["Feature"]==f]["JS_Div"].values[0]:.3f})')
    axes[i].legend()
    # Limit x-axis to focus on main distribution
    q_high = max(benign[f].quantile(0.99), attack[f].quantile(0.99))
    axes[i].set_xlim(0, q_high * 1.5)

plt.tight_layout()
plt.savefig(os.path.join(RESULTS_DIR, 'distribution_separability.png'))

# Scatter plot for multi-dim separability
plt.figure(figsize=(12, 8))
# Sample data for plot to keep it responsive
sub_b = benign.sample(min(5000, len(benign)))
sub_a = attack.sample(min(5000, len(attack)))
plt.scatter(sub_b['IPC'], sub_b['MPKI'], alpha=0.2, label='Benign', color='blue', s=5)
plt.scatter(sub_a['IPC'], sub_a['MPKI'], alpha=0.2, label='Interfered', color='red', s=5)
plt.xlabel('IPC')
plt.ylabel('MPKI')
plt.title('Separability Map: IPC vs MPKI')
plt.legend()
plt.savefig(os.path.join(RESULTS_DIR, 'scatter_separability.png'))

print(f"\nAnalysis complete. Results saved in: {RESULTS_DIR}")
