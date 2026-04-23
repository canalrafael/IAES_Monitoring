import pandas as pd
import numpy as np
import os
import glob
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import ks_2samp
from scipy.spatial.distance import jensenshannon

# ── Style ──────────────────────────────────────────────────────────────────────
sns.set_theme(style='whitegrid', palette='muted')
plt.rcParams.update({'figure.figsize': (10, 6), 'font.size': 16})
plt.rcParams.update({
    'axes.titlesize': 24,
    'axes.labelsize': 20,
    'legend.fontsize': 20,
    'xtick.labelsize': 18,
    'ytick.labelsize': 18,
    'axes.titleweight': 'bold'
})
BENIGN_COLOR = '#2196F3' # blue
ATTACK_COLOR = '#F44336' # red

DATA_DIR = 'data/train data/'
RESULTS_DIR = 'results/replicated_analysis/'
os.makedirs(RESULTS_DIR, exist_ok=True)

def overlap_coefficient(x, y, bins=100):
    if len(x) == 0 or len(y) == 0: return 1.
    ax, ay = np.array(x), np.array(y)
    lo, hi = min(np.nanmin(ax), np.nanmin(ay)), max(np.nanmax(ax), np.nanmax(ay))
    if not (np.isfinite(lo) and np.isfinite(hi)) or hi - lo < 1e-9: return 1.
    hx, edges = np.histogram(ax, bins=bins, range=(lo, hi), density=True)
    hy, _ = np.histogram(ay, bins=edges, density=True)
    return float(np.sum(np.minimum(hx, hy) * np.diff(edges)))

def js_divergence(x, y, bins=100):
    if len(x) == 0 or len(y) == 0: return 0.
    ax, ay = np.array(x), np.array(y)
    lo, hi = min(np.nanmin(ax), np.nanmin(ay)), max(np.nanmax(ax), np.nanmax(ay))
    if not (np.isfinite(lo) and np.isfinite(hi)) or hi - lo < 1e-9: return 0.
    hx, edges = np.histogram(ax, bins=bins, range=(lo, hi), density=False)
    hy, _ = np.histogram(ay, bins=edges, density=False)
    # Convert to probabilities
    px = hx / (hx.sum() + 1e-9)
    py = hy / (hy.sum() + 1e-9)
    return jensenshannon(px, py)

def calculate_metrics(b_vals, a_vals):
    if len(b_vals) < 2 or len(a_vals) < 2:
        return {'cohen_d': 0, 'js_div': 0, 'snr': 0, 'overlap': 1}
    
    m0, m2 = np.mean(b_vals), np.mean(a_vals)
    s0, s2 = np.std(b_vals, ddof=1), np.std(a_vals, ddof=1)
    n0, n2 = len(b_vals), len(a_vals)
    
    # Cohen's d
    v0, v2 = s0**2, s2**2
    sp = np.sqrt(((n0 - 1) * v0 + (n2 - 1) * v2) / (n0 + n2 - 2))
    d = (m0 - m2) / sp if sp > 1e-9 else 0.
    
    # SNR
    snr = abs(m0 - m2) / (s0 + s2 + 1e-9)
    
    # Overlap & JS
    ov = overlap_coefficient(b_vals, a_vals)
    js = js_divergence(b_vals, a_vals)
    
    return {'cohen_d': d, 'js_div': js, 'snr': snr, 'overlap': ov}

def main():
    print("Loading data...")
    csv_files = sorted(glob.glob(os.path.join(DATA_DIR, '*.csv')))
    
    all_data = []
    for f in csv_files:
        try:
            # We want labels 0, 1, 2 for the two different tasks
            df = pd.read_csv(f, usecols=['LABEL', 'CPU_CYCLES', 'INSTRUCTIONS', 'CACHE_MISSES', 'BRANCH_MISSES', 'L2_CACHE_ACCESS'])
            df = df[df['LABEL'].isin([0, 1, 2])]
            if not df.empty:
                all_data.append(df)
        except Exception as e:
            pass

    if not all_data:
        print("No valid data found.")
        return
        
    df = pd.concat(all_data, ignore_index=True)
    
    # Feature engineering for the plots and metrics
    eps = 1e-9
    df['IPC'] = df['INSTRUCTIONS'] / (df['CPU_CYCLES'] + eps)
    df['MPKI'] = (df['CACHE_MISSES'] * 1000) / (df['INSTRUCTIONS'] + eps)
    df['BRANCH_MISS_RATE'] = df['BRANCH_MISSES'] / (df['INSTRUCTIONS'] + eps)
    
    # Using the standard phase 1 definition of waste ratio, modifying x label later
    df['WASTE_RATIO'] = df['CPU_CYCLES'] / (df['INSTRUCTIONS'] + eps)
    # Alternatively, if penalties / usable cycles was truly (cycles-instructions)/instructions:
    df['PIPELINE_WASTE'] = (df['CPU_CYCLES'] - df['INSTRUCTIONS']) / (df['INSTRUCTIONS'] + eps)
    # We will plot PIPELINE_WASTE as it better matches "Waste Ratio (Penalties/Usable Cycles)" meaning

    df_0 = df[df['LABEL'] == 0]
    df_1 = df[df['LABEL'] == 1]
    df_2 = df[df['LABEL'] == 2]

    print(f"Data loaded: Label 0: {len(df_0)}, Label 1: {len(df_1)}, Label 2: {len(df_2)}")

    # 1 & 2. Replicate KDE Plot and Scatterplot (Label 0 & 1) Combined
    print("Generating combined plots...")
    
    # 1. Standalone: Branch Misses Density Plot
    print("Generating standalone BRANCH_MISSES plot...")
    plt.figure(figsize=(10, 6))
    b_bm = df_0['BRANCH_MISSES'].dropna().values
    a_bm = df_2['BRANCH_MISSES'].dropna().values
    
    if len(b_bm) > 1:
        sns.kdeplot(b_bm, fill=True, color='blue', alpha=0.3, label='0')
    if len(a_bm) > 1:
        sns.kdeplot(a_bm, fill=True, color='red', alpha=0.3, label='2')
        
    plt.title('Density Distribution: BRANCH_MISSES (Benign vs Interference)')
    plt.xlabel('BRANCH_MISSES')
    plt.ylabel('Density')
    plt.xlim(-50000, 650000)
    plt.ylim(0, 1.05e-5)
    plt.legend(title='LABEL')
    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_DIR, 'branch_misses_distribution.png'), dpi=300)
    plt.close()

    # 2. Combined Analysis: 1x2 layout (Waste Ratio & Scatter)
    print("Generating combined 1x2 plot...")
    fig, axes = plt.subplots(1, 2, figsize=(18, 8))


    # Subplot 1: Execution Waste Isolation density plot
    b_vals = df_0['PIPELINE_WASTE'].dropna().values
    a_vals = df_1['PIPELINE_WASTE'].dropna().values
    
    lo = min(np.percentile(b_vals, 1) if len(b_vals) else 0, np.percentile(a_vals, 1) if len(a_vals) else 0)
    hi = max(np.percentile(b_vals, 99) if len(b_vals) else 1, np.percentile(a_vals, 99) if len(a_vals) else 1)
    b_plot = b_vals[(b_vals >= lo) & (b_vals <= hi)]
    a_plot = a_vals[(a_vals >= lo) & (a_vals <= hi)]
    
    if len(b_plot) > 1:
        sns.kdeplot(b_plot, fill=True, color=BENIGN_COLOR, alpha=0.3, label='Secure Domain (Benign)', ax=axes[0])
    if len(a_plot) > 1:
        sns.kdeplot(a_plot, fill=True, color=ATTACK_COLOR, alpha=0.3, label='Untrusted Domain (Attack)', ax=axes[0])
        
    axes[0].set_title('Execution Waste Isolation (Stall vs Compute Phase)')
    axes[0].set_xlabel('Pipeline Waste Ratio (Penalties / Usable Cycles)')
    axes[0].set_ylabel('Density')
    
    # Subplot 2: Scatterplot without marginals
    import warnings
    warnings.filterwarnings("ignore")
    df_joint = pd.concat([df_0, df_1])
    # Mapping label texts
    df_joint['Domain'] = df_joint['LABEL'].map({0: 'Secure Domain', 1: 'Untrusted Domain'})
    
    # Subsample if too large
    if len(df_joint) > 5000:
        df_joint = df_joint.sample(5000, random_state=42)
    
    sns.scatterplot(data=df_joint, x='IPC', y='BRANCH_MISS_RATE', hue='Domain', 
                    palette={'Secure Domain': BENIGN_COLOR, 'Untrusted Domain': ATTACK_COLOR},
                    s=15, alpha=0.5, ax=axes[1], legend=False)
                    
    axes[1].set_title('BRANCH_MISS_RATE vs IPC')
    axes[1].set_xlabel('IPC')
    axes[1].set_ylabel('BRANCH_MISS_RATE')
    
    # Global centered legend
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='upper center', bbox_to_anchor=(0.5, 1.05), ncol=2, frameon=True)

    
    # Adjust layout and save as a single image
    plt.tight_layout(rect=[0, 0, 1, 0.92])
    fig.savefig(os.path.join(RESULTS_DIR, 'combined_analysis.png'), dpi=300, bbox_inches='tight')
    plt.close()

    # 3. Calculate metrics for Label 0 and Label 2
    print("Computing metrics for Label 0 vs Label 2...")
    
    # Define features and their descriptions, placing raw counters first
    feature_meta = [
        ('CPU_CYCLES', 'Raw Counter: Total CPU cycles elapsed'),
        ('INSTRUCTIONS', 'Raw Counter: Total instructions retired'),
        ('CACHE_MISSES', 'Raw Counter: Last Level Cache (LLC) misses'),
        ('BRANCH_MISSES', 'Raw Counter: Branch mispredictions'),
        ('L2_CACHE_ACCESS', 'Raw Counter: Accesses to the L2 Cache'),
        ('IPC', 'Derived: Instructions per Cycle (INSTRUCTIONS / CPU_CYCLES)'),
        ('MPKI', 'Derived: Misses per 1000 Instructions ((CACHE_MISSES * 1000) / INSTRUCTIONS)'),
        ('BRANCH_MISS_RATE', 'Derived: Branch Misses per Instruction (BRANCH_MISSES / INSTRUCTIONS)'),
        ('WASTE_RATIO', 'Derived: CPU Cycles per Instruction (CPU_CYCLES / INSTRUCTIONS)'),
        ('PIPELINE_WASTE', 'Derived: Pipeline Waste Ratio ((CPU_CYCLES - INSTRUCTIONS) / INSTRUCTIONS)')
    ]
    
    metrics_list = []
    for feat, desc in feature_meta:
        b = df_0[feat].dropna().values
        a = df_2[feat].dropna().values
        m = calculate_metrics(b, a)
        metrics_list.append({
            'Feature': feat,
            'Description': desc,
            'Cohen_d': m['cohen_d'],
            'JS_Divergence': m['js_div'],
            'SNR': m['snr'],
            'Overlap': m['overlap']
        })
        
    metrics_df = pd.DataFrame(metrics_list)
    
    # Split, sort by JS_Divergence (descending) as the primary separability metric, and reconstitute
    raw_df = metrics_df[metrics_df['Description'].str.startswith('Raw Counter')].copy()
    der_df = metrics_df[metrics_df['Description'].str.startswith('Derived')].copy()
    
    raw_df = raw_df.sort_values('JS_Divergence', ascending=False)
    der_df = der_df.sort_values('JS_Divergence', ascending=False)
    
    final_df = pd.concat([raw_df, der_df])
    
    print("\nMetrics (Label 0 vs Label 2) [Sorted by JS Divergence]:")
    print(final_df.to_string(index=False))
    final_df.to_csv(os.path.join(RESULTS_DIR, 'recalculated_metrics_0_vs_2.csv'), index=False)
    print(f"\nAll results saved to {RESULTS_DIR}")

if __name__ == '__main__':
    main()
