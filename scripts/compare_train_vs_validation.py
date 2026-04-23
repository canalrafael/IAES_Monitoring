import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import glob
import os

# --- Configuration ---
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
BASE_DIR = os.path.dirname(SCRIPT_DIR)
TRAIN_DATA_DIR = os.path.join(BASE_DIR, 'data', 'train data')
VAL_DATA_DIR = os.path.join(BASE_DIR, 'data', 'online validation data')
RESULTS_DIR = os.path.join(BASE_DIR, 'results', 'data_comparison')
os.makedirs(RESULTS_DIR, exist_ok=True)

# Styling
sns.set_theme(style="whitegrid")
plt.rcParams.update({'font.size': 12, 'figure.figsize': (12, 8)})

def calculate_derived(df):
    eps = 1e-9
    df = df.copy()
    df['IPC'] = df['INSTRUCTIONS'] / (df['CPU_CYCLES'] + eps)
    df['MPKI'] = (df['CACHE_MISSES'] * 1000) / (df['INSTRUCTIONS'] + eps)
    df['L2_PRESSURE'] = df['L2_CACHE_ACCESS'] / (df['CPU_CYCLES'] + eps)
    df['BRANCH_MISS_RATE'] = df['BRANCH_MISSES'] / (df['INSTRUCTIONS'] + eps)
    return df[['IPC', 'MPKI', 'L2_PRESSURE', 'BRANCH_MISS_RATE']]

def load_benign_samples(directory, pattern='*.csv'):
    files = glob.glob(os.path.join(directory, pattern))
    all_samples = []
    for f in files:
        df = pd.read_csv(f)
        benign = df[df['LABEL'] == 0]
        if not benign.empty:
            derived = calculate_derived(benign)
            all_samples.append(derived)
    if not all_samples:
        return pd.DataFrame()
    return pd.concat(all_samples, ignore_index=True)

print("Loading Train Data (Label 0)...")
train_benign = load_benign_samples(TRAIN_DATA_DIR)

print("Loading New Validation Data (Label 0)...")
val_benign = load_benign_samples(VAL_DATA_DIR, 'data_new*_clean.csv')

print(f"Train samples: {len(train_benign)}")
print(f"Validation samples: {len(val_benign)}")

if train_benign.empty or val_benign.empty:
    print("Error: Missing data for comparison.")
    exit(1)

# --- Statistical Comparison ---
metrics = ['IPC', 'MPKI', 'L2_PRESSURE', 'BRANCH_MISS_RATE']
comparison = []

for m in metrics:
    t_mean, t_std = train_benign[m].mean(), train_benign[m].std()
    v_mean, v_std = val_benign[m].mean(), val_benign[m].std()
    
    # Cohen's d for effect size (distance)
    d = (v_mean - t_mean) / np.sqrt((t_std**2 + v_std**2) / 2)
    
    comparison.append({
        "Metric": m,
        "Train_Mean": t_mean,
        "Val_Mean": v_mean,
        "Diff_%": (v_mean - t_mean) / t_mean * 100,
        "Cohen_d": d
    })

comp_df = pd.DataFrame(comparison)
print("\n--- Statistical Drift Summary ---")
print(comp_df.to_string(index=False))
comp_df.to_csv(os.path.join(RESULTS_DIR, 'drift_stats.csv'), index=False)

# --- Visualization ---
fig, axes = plt.subplots(2, 2, figsize=(16, 12))
axes = axes.flatten()

for i, m in enumerate(metrics):
    sns.kdeplot(train_benign[m], ax=axes[i], label='Train (Original)', fill=True, color='blue', alpha=0.3)
    sns.kdeplot(val_benign[m], ax=axes[i], label='Validation (New)', fill=True, color='orange', alpha=0.3)
    axes[i].set_title(f'{m} Distribution Comparison')
    axes[i].legend()
    
    # Clip extreme outliers for better visibility in KDE
    q_low = train_benign[m].quantile(0.01)
    q_high = max(train_benign[m].quantile(0.99), val_benign[m].quantile(0.99))
    axes[i].set_xlim(q_low * 0.5, q_high * 1.2)

plt.tight_layout()
plt.savefig(os.path.join(RESULTS_DIR, 'distribution_drift.png'))
print(f"\nDrift plot saved to: {RESULTS_DIR}/distribution_drift.png")

# --- Boxplot for Scale check ---
plt.figure(figsize=(12, 8))
train_benign_melt = train_benign.copy()
train_benign_melt['Source'] = 'Train'
val_benign_melt = val_benign.copy()
val_benign_melt['Source'] = 'Validation'

combined = pd.concat([train_benign_melt, val_benign_melt])

# Normalize metrics for side-by-side boxplot
for m in metrics:
    combined[f'{m}_norm'] = (combined[m] - train_benign[m].mean()) / train_benign[m].std()

melted = combined.melt(id_vars=['Source'], value_vars=[f'{m}_norm' for m in metrics])

plt.figure(figsize=(14, 8))
sns.boxplot(data=melted, x='variable', y='value', hue='Source', palette='Set2')
plt.title('Normalized Feature Drift (Z-Score relative to Train)')
plt.ylabel('Standard Deviations from Train Mean')
plt.ylim(-5, 10) # focus on center
plt.savefig(os.path.join(RESULTS_DIR, 'boxplot_drift.png'))

print(f"Scale check plot saved to: {RESULTS_DIR}/boxplot_drift.png")
