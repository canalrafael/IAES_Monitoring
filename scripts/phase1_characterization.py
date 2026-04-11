import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import entropy
from scipy.spatial.distance import jensenshannon
import glob

# Set aesthetically pleasing style
sns.set_theme(style="whitegrid", palette="muted")
plt.rcParams['figure.figsize'] = (12, 8)
plt.rcParams['font.size'] = 12

DATA_DIR = 'data/'
RESULTS_DIR = 'results/phase1/'
os.makedirs(RESULTS_DIR, exist_ok=True)

def cohen_d(x, y):
    nx = len(x)
    ny = len(y)
    dof = nx + ny - 2
    return (np.mean(x) - np.mean(y)) / np.sqrt(((nx-1)*np.std(x, ddof=1)**2 + (ny-1)*np.std(y, ddof=1)**2) / dof)

def overlap_coefficient(x, y, bins=100):
    hist_x, bin_edges = np.histogram(x, bins=bins, density=True)
    hist_y, _ = np.histogram(y, bins=bin_edges, density=True)
    return np.sum(np.minimum(hist_x, hist_y) * np.diff(bin_edges))

def signal_to_noise_ratio(benign, attack):
    signal = np.abs(np.mean(benign) - np.mean(attack))
    noise = (np.std(benign) + np.std(attack)) / 2
    return signal / (noise + 1e-9)

def load_and_filter_data(data_dir):
    csv_files = glob.glob(os.path.join(data_dir, "*.csv"))
    all_dfs = []
    
    for f in csv_files:
        try:
            df = pd.read_csv(f)
            # Filter only Label 0 (Benign) and Label 2 (Interference)
            df = df[df['LABEL'].isin([0, 2])].copy()
            if not df.empty:
                df['source_file'] = os.path.basename(f)
                all_dfs.append(df)
        except Exception as e:
            print(f"Error loading {f}: {e}")
            
    return pd.concat(all_dfs, ignore_index=True)

def main():
    print("Loading data...")
    df = load_and_filter_data(DATA_DIR)
    
    features = ['CPU_CYCLES', 'INSTRUCTIONS', 'CACHE_MISSES', 'BRANCH_MISSES', 'L2_CACHE_ACCESS']
    
    # 1. Statistical Separability
    results = []
    for feat in features:
        benign = df[df['LABEL'] == 0][feat].values
        attack = df[df['LABEL'] == 2][feat].values
        
        if len(benign) == 0 or len(attack) == 0:
            continue
            
        d_f = np.abs(np.mean(benign) - np.mean(attack))
        d_cohen = cohen_d(benign, attack)
        snr = signal_to_noise_ratio(benign, attack)
        
        # Divergence (using histogram for distribution approximation)
        bx, be = np.histogram(benign, bins=100, density=True)
        ax, _ = np.histogram(attack, bins=be, density=True)
        # Avoid zero for JS divergence
        js_div = jensenshannon(bx + 1e-10, ax + 1e-10)
        
        overlap = overlap_coefficient(benign, attack)
        
        results.append({
            'feature': feat,
            'mean_diff': d_f,
            'cohen_d': d_cohen,
            'js_divergence': js_div,
            'snr': snr,
            'overlap': overlap
        })
        
        # Density Plots
        plt.figure()
        sns.kdeplot(data=df, x=feat, hue='LABEL', fill=True, common_norm=False, palette={0: 'blue', 2: 'red'})
        plt.title(f'Density Distribution: {feat} (Benign vs Interference)')
        plt.savefig(os.path.join(RESULTS_DIR, f'density_{feat.lower()}.png'))
        plt.close()

    # 2. Feature Ranking
    results_df = pd.DataFrame(results)
    # Normalize and combine metrics for ranking (Higher d_cohen, higher js_div, lower overlap = better)
    # We'll rank primarily by Cohen's d (effect size) as it's the standard for physical magnitude
    results_df = results_df.sort_values(by='cohen_d', ascending=False, key=abs)
    results_df.to_csv(os.path.join(RESULTS_DIR, 'feature_importance.csv'), index=False)
    print("Feature importance saved to feature_importance.csv")

    # 3. Correlation Matrix
    plt.figure(figsize=(10, 8))
    sns.heatmap(df[features + ['LABEL']].corr(), annot=True, cmap='coolwarm', fmt=".2f")
    plt.title('Feature Correlation Matrix')
    plt.savefig(os.path.join(RESULTS_DIR, 'correlation_matrix.png'))
    plt.close()

    # 4. Temporal Behavior (Example transition)
    # Pick a file that likely contains a transition (Label 0 -> 2)
    transition_found = False
    for group_name, group_df in df.groupby('source_file'):
        labels = group_df['LABEL'].unique()
        if 0 in labels and 2 in labels:
            # Sort by timestamp (approximate if TIMESTAMP is string)
            group_df = group_df.copy()
            # If TIMESTAMP is like "16:03:15:605", it might need parsing
            # For now, we use index as time proxy
            group_df = group_df.reset_index()
            
            plt.figure(figsize=(14, 10))
            for i, feat in enumerate(features, 1):
                plt.subplot(len(features), 1, i)
                sns.lineplot(data=group_df, x='index', y=feat, hue='LABEL', palette={0: 'blue', 2: 'red'}, legend=False)
                plt.ylabel(feat)
                if i == 1:
                    plt.title(f'Temporal Transition in {group_name}')
            
            plt.tight_layout()
            plt.savefig(os.path.join(RESULTS_DIR, f'temporal_transition_{group_name.split(".")[0]}.png'))
            plt.close()
            transition_found = True
            break
            
    if not transition_found:
        print("Warning: No single file with both Benign (0) and Interference (2) found for temporal plot.")

    print("Phase 1 analysis complete. Check /results/phase1/ for outputs.")

if __name__ == "__main__":
    main()
