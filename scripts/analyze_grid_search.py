import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

# --- 1. Robust Path Handling ---
# This finds the directory the script is in (scripts/) and navigates from there
script_dir = Path(__file__).resolve().parent
project_root = script_dir.parent

# Point to the CSV file (Change 'model_comparison.csv' to 'threshold_sweep.csv' if needed)
csv_file_path = project_root / 'results' / 'phase2' / 'model_comparison.csv'
# Set where you want the plot to be saved
output_plot_path = project_root / 'results' / 'phase2' / 'grid_search_analysis.png'

print(f"Reading data from: {csv_file_path}")

# --- 2. Load the Data ---
try:
    df = pd.read_csv(csv_file_path)
    print(f"Successfully loaded {len(df)} configurations for analysis.\n")
except FileNotFoundError:
    print(f"Error: Could not find the file at {csv_file_path}")
    print("Please check the exact filename in your results/phase2/ directory.")
    exit()

# --- 3. Define a "Balance Score" ---
df['balance_score'] = (df['avg_recall'] + df['avg_f1']) - df['avg_fpr']

# --- 4. Find the Top Models ---
print("--- TOP 5 MODELS BY BEST OVERALL BALANCE ---")
top_balanced = df.sort_values(by='balance_score', ascending=False).head(5)
print(top_balanced[['layers', 'units', 'lr', 'w', 'avg_recall', 'avg_f1', 'avg_fpr', 'balance_score']].to_string(index=False))
print("\n")

print("--- TOP 5 MODELS BY F1-SCORE ---")
top_f1 = df.sort_values(by='avg_f1', ascending=False).head(5)
print(top_f1[['layers', 'units', 'lr', 'w', 'avg_f1', 'avg_recall', 'avg_fpr']].to_string(index=False))
print("\n")

# --- 5. Analyze Hyperparameter Impact ---
print("--- AVERAGE PERFORMANCE BY NUMBER OF LAYERS ---")
layer_impact = df.groupby('layers')[['avg_recall', 'avg_f1', 'avg_fpr']].mean().round(4)
print(layer_impact)
print("\n")

print("--- AVERAGE PERFORMANCE BY LEARNING RATE ---")
lr_impact = df.groupby('lr')[['avg_recall', 'avg_f1', 'avg_fpr']].mean().round(4)
print(lr_impact)
print("\n")

# --- 6. Visualizations ---
sns.set_theme(style="whitegrid")
fig, axes = plt.subplots(1, 3, figsize=(18, 5))

# Plot 1: Layers vs F1-Score
sns.boxplot(data=df, x='layers', y='avg_f1', ax=axes[0])
axes[0].set_title('Impact of Network Depth (Layers) on F1-Score')

# Plot 2: Learning Rate vs False Positive Rate
sns.boxplot(data=df, x='lr', y='avg_fpr', ax=axes[1])
axes[1].set_title('Impact of Learning Rate on FPR')

# Plot 3: Recall vs FPR (Scatter plot to visualize the trade-off)
sns.scatterplot(data=df, x='avg_fpr', y='avg_recall', hue='units', palette='viridis', alpha=0.7, ax=axes[2])
axes[2].set_title('Trade-off: Recall vs False Positive Rate')
axes[2].set_xlabel('Average FPR (Lower is better)')
axes[2].set_ylabel('Average Recall (Higher is better)')

plt.tight_layout()

# Save and Show
plt.savefig(output_plot_path)
print(f"Saved visualization plots to: {output_plot_path}")
plt.show()