import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

# --- Configuration ---
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
BASE_DIR = os.path.dirname(SCRIPT_DIR)
RESULTS_DIR = os.path.join(BASE_DIR, "results", "online_validation")
DATA_FILE = os.path.join(RESULTS_DIR, "results_data_new0_clean.csv")

if not os.path.exists(DATA_FILE):
    print(f"Error: {DATA_FILE} not found. Run evaluation script first.")
    exit(1)

df = pd.read_csv(DATA_FILE)

# Limit to a readable range
df_subset = df.iloc[500:2500].copy()

plt.figure(figsize=(15, 10))

# Subplot 1: Probability and Threshold
plt.subplot(3, 1, 1)
plt.plot(df_subset.index, df_subset['NEW_DET_PROB'], label='New Detector Prob', color='blue', alpha=0.8)
plt.plot(df_subset.index, df_subset['DET_PROB'], label='Original Prob', color='gray', linestyle='--', alpha=0.5)
plt.axhline(20, color='red', linestyle=':', label='Threshold (20%)')
plt.ylabel('Probability (%)')
plt.title('Detector Probability Comparison (data_new0)')
plt.legend(loc='upper right')

# Subplot 2: Label and Bench ID
plt.subplot(3, 1, 2)
# Create a simplified bench label
bench_map = {0: 'Idle', 1: 'BW', 2: 'Disp', 3: 'FFT', 4: 'QS', 5: 'Dijk', 6: 'SHA', 7: 'Sort', 10: 'Spectre', 11: 'Arma', 12: 'Melt'}
df_subset['BENCH_NAME'] = df_subset['BENCH_ID'].map(bench_map)

# Color by Label (2=Interference, 3=Attack Core)
sns.scatterplot(data=df_subset, x=df_subset.index, y='BENCH_NAME', hue='LABEL', palette='bright', s=10)
plt.ylabel('Workload')
plt.title('Workload and Label Sequence')

# Subplot 3: Status Comparison
plt.subplot(3, 1, 3)
plt.step(df_subset.index, df_subset['NEW_DET_STATUS'], label='New Status', color='blue', where='post')
plt.step(df_subset.index, df_subset['DET_STATUS'], label='Original Status', color='red', linestyle='--', alpha=0.6, where='post')
plt.yticks([0, 1, 2], ['Warmup', 'Benign', 'Attack'])
plt.ylabel('Status')
plt.xlabel('Sample Index')
plt.title('Detection Status Comparison')
plt.legend(loc='upper right')

plt.tight_layout()
plot_path = os.path.join(RESULTS_DIR, "validation_visualization.png")
plt.savefig(plot_path)
print(f"Visualization saved to {plot_path}")
