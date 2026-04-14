import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
import seaborn as sns
import glob
import gc
from scipy.stats import ks_2samp

# Set aesthetically pleasing style
sns.set_theme(style="whitegrid", palette="muted")
plt.rcParams['figure.figsize'] = (12, 8)
plt.rcParams['font.size'] = 12

DATA_DIR = 'data/'
RESULTS_DIR = 'results/phase1/'
os.makedirs(RESULTS_DIR, exist_ok=True)

class SeparabilityAccumulator:
    """Accumulates sum, sum_sq, and N for weighted Cohen's d computation."""
    def __init__(self):
        self.stats = {} # feature -> label -> {sum, sum_sq, n}
        self.samples = {} # feature -> label -> list (subsampled for KS)
        self.max_global_samples = 50000
    
    def update(self, feat_name, label, values):
        # Filter NaNs and Infs to ensure numerical stability
        mask = np.isfinite(values)
        if not mask.any(): return
        clean_values = values[mask].astype(np.float64)
        
        if feat_name not in self.stats:
            self.stats[feat_name] = {0: {'s': 0.0, 'ss': 0.0, 'n': 0}, 2: {'s': 0.0, 'ss': 0.0, 'n': 0}}
            self.samples[feat_name] = {0: [], 2: []}
        
        self.stats[feat_name][label]['s'] += np.sum(clean_values)
        self.stats[feat_name][label]['ss'] += np.sum(np.square(clean_values))
        self.stats[feat_name][label]['n'] += len(clean_values)
        
        # Consistent sampling pool for KS-test
        if len(self.samples[feat_name][label]) < (self.max_global_samples // 2):
            # Take evenly spread samples
            step = max(1, len(clean_values) // 200)
            self.samples[feat_name][label].extend(clean_values[::step][:200])

    def get_cohen_d(self, feat_name):
        s0, ss0, n0 = self.stats[feat_name][0]['s'], self.stats[feat_name][0]['ss'], self.stats[feat_name][0]['n']
        s2, ss2, n2 = self.stats[feat_name][2]['s'], self.stats[feat_name][2]['ss'], self.stats[feat_name][2]['n']
        
        if n0 < 2 or n2 < 2: return 0.0
        
        m0, m2 = s0 / n0, s2 / n2
        v0 = max(0, (ss0 / n0) - (m0**2))
        v2 = max(0, (ss2 / n2) - (m2**2))
        
        std_pool = np.sqrt(((n0 - 1) * v0 + (n2 - 1) * v2) / (n0 + n2 - 2))
        if std_pool < 1e-9: return 0.0
        return (m0 - m2) / std_pool

def overlap_coefficient(x, y, bins=100):
    if len(x) == 0 or len(y) == 0: return 1.0
    arr_x = np.array(x); arr_y = np.array(y)
    xmin = min(np.nanmin(arr_x), np.nanmin(arr_y))
    xmax = max(np.nanmax(arr_x), np.nanmax(arr_y))
    if not np.isfinite(xmin) or not np.isfinite(xmax) or xmax - xmin < 1e-9: return 1.0
    hist_x, bin_edges = np.histogram(arr_x, bins=bins, range=(xmin, xmax), density=True)
    hist_y, _ = np.histogram(arr_y, bins=bin_edges, density=True)
    return np.sum(np.minimum(hist_x, hist_y) * np.diff(bin_edges))

def main():
    print("Starting Streaming Advanced Phase 1 Analysis...")
    csv_files = sorted(glob.glob(os.path.join(DATA_DIR, "*.csv")))
    
    accumulator = SeparabilityAccumulator()
    window_sizes = [8, 10, 12, 14]
    eps = 1e-9
    
    total_raw_samples = 0
    total_pure_samples = 0

    for f_idx, f_path in enumerate(csv_files):
        try:
            df = pd.read_csv(f_path)
            df = df[df['LABEL'].isin([0, 2])].reset_index(drop=True)
            if len(df) < 500: continue # Skip very short files
            
            raw_count = len(df)
            total_raw_samples += raw_count
            
            # Base Derived (Intermediate RAW for deltas)
            df['IPC'] = (df['INSTRUCTIONS'] / (df['CPU_CYCLES'] + eps)).astype(np.float32)
            df['MPKI'] = ((df['CACHE_MISSES'] * 1000) / (df['INSTRUCTIONS'] + eps)).astype(np.float32)
            df['WASTE'] = (df['CPU_CYCLES'] / (df['INSTRUCTIONS'] + eps)).astype(np.float32)
            df['BUS_P'] = (df['L2_CACHE_ACCESS'] / (df['CPU_CYCLES'] + eps)).astype(np.float32)
            
            base_cols = ['IPC', 'MPKI', 'WASTE', 'BUS_P']
            
            # Relative Deltas (On raw values)
            for col in ['IPC', 'MPKI']:
                df[f'd_{col}'] = df[col].pct_change().fillna(0).astype(np.float32)
            delta_cols = [f'd_{col}' for col in ['IPC', 'MPKI']]

            # Z-Normalization (After deltas)
            for col in base_cols:
                m, s = df[col].mean(), df[col].std()
                df[col] = ((df[col] - m) / (s + eps)).astype(np.float32)

            # Sequential Window Processing
            for w in window_sizes:
                # Purity Filter (90% threshold)
                rolling_label_sum = df['LABEL'].rolling(w).sum()
                # 0 for benign, 2 for interference. 
                # Pure benign (0*w) -> sum=0
                # Pure interference (2*w) -> sum=2*w
                # 90% purity: sum <= 0.1*(2*w) or sum >= 0.9*(2*w)
                p_benign = (rolling_label_sum <= (0.1 * 2 * w))
                p_interf = (rolling_label_sum >= (0.9 * 2 * w))
                pure_mask = p_benign | p_interf
                
                pure_indices = np.where(pure_mask)[0]
                if len(pure_indices) < 200: continue

                # Sampling consistent indices
                np.random.seed(42 + f_idx)
                if len(pure_indices) > 2500:
                    sample_idx = np.random.choice(pure_indices, size=2500, replace=False)
                else:
                    sample_idx = pure_indices
                
                labels = df['LABEL'].iloc[sample_idx].values
                total_pure_samples += len(sample_idx)

                # Feature Aggregation
                for col in base_cols:
                    # Mean metrics
                    feat_m = f'{col}_avg_w{w}'
                    vals_m = df[col].rolling(w).mean().iloc[sample_idx].values
                    for l in [0, 2]:
                        mask = (labels == l)
                        if mask.any(): accumulator.update(feat_m, l, vals_m[mask])
                    
                    # Std metrics
                    feat_s = f'{col}_std_w{w}'
                    vals_s = df[col].rolling(w).std().iloc[sample_idx].values
                    for l in [0, 2]:
                        mask = (labels == l)
                        if mask.any(): accumulator.update(feat_s, l, vals_s[mask])

                # Percentiles (IPC/MPKI)
                for col in ['IPC', 'MPKI']:
                    feat_p = f'{col}_p95p5_w{w}'
                    rolled = df[col].rolling(w)
                    vals_p = (rolled.quantile(0.95) - rolled.quantile(0.05)).iloc[sample_idx].values
                    for l in [0, 2]:
                        mask = (labels == l)
                        if mask.any(): accumulator.update(feat_p, l, vals_p[mask])
                
                # Deltas
                for col in delta_cols:
                    feat_d = f'{col}_avg_w{w}'
                    vals_d = df[col].rolling(w).mean().iloc[sample_idx].values
                    for l in [0, 2]:
                        mask = (labels == l)
                        if mask.any(): accumulator.update(feat_d, l, vals_d[mask])

                gc.collect()

            del df
            gc.collect()
            print(f"Processed {f_idx+1}/{len(csv_files)}: {os.path.basename(f_path)} (Pure Samples: {total_pure_samples})")

        except Exception as e:
            print(f"Error in {f_path}: {e}")

    # Finalize Statistics
    print(f"\nAggregation Complete. Total Pure Samples: {total_pure_samples}")
    stats_results = []
    
    for feat in accumulator.stats.keys():
        d = accumulator.get_cohen_d(feat)
        
        # Collect samples for KS/Overlap
        b = np.array(accumulator.samples[feat][0])
        i = np.array(accumulator.samples[feat][2])
        
        if len(b) < 20 or len(i) < 20: continue
        
        ks_stat, ks_pval = ks_2samp(b, i)
        overlap = overlap_coefficient(b, i)
        
        abs_d = abs(d)
        tier = "Negligible"
        if abs_d >= 0.8: tier = "Strong"
        elif abs_d >= 0.5: tier = "Moderate"
        elif abs_d >= 0.3: tier = "Weak"
        
        stats_results.append({
            'feature': feat,
            'weighted_cohen_d': d,
            'abs_d': abs_d,
            'ks_stat': ks_stat,
            'ks_pval': ks_pval,
            'overlap': overlap,
            'tier': tier,
            'n0': accumulator.stats[feat][0]['n'],
            'n2': accumulator.stats[feat][2]['n']
        })

    stats_df = pd.DataFrame(stats_results).sort_values(by='abs_d', ascending=False)
    
    # Correlation Pruning (Approximate using pooled samples)
    print("Pruning redundant features (corr > 0.95)...")
    final_output = []
    already_selected = []
    
    for _, row in stats_df.iterrows():
        f_name = row['feature']
        unique = True
        
        # Use full pooled samples for correlation check
        pool_f = np.concatenate([accumulator.samples[f_name][0], accumulator.samples[f_name][2]])
        
        for seen in already_selected:
            pool_s = np.concatenate([accumulator.samples[seen][0], accumulator.samples[seen][2]])
            # Length might differ slightly due to reservoir sampling, crop to min
            min_L = min(len(pool_f), len(pool_s))
            c = np.corrcoef(pool_f[:min_L], pool_s[:min_L])[0, 1]
            if abs(c) > 0.95:
                unique = False
                break
        
        if unique:
            final_output.append(row)
            already_selected.append(f_name)
    
    final_df = pd.DataFrame(final_output).drop(columns=['abs_d'])
    final_df.to_csv(os.path.join(RESULTS_DIR, 'phase1_feature_stats.csv'), index=False)
    print(f"Final Selection: {len(final_df)} features ranked and pruned.")
    
    # Visualization
    top_v = final_df[final_df['tier'].isin(['Strong', 'Moderate'])].head(5)
    for feat in top_v['feature'].values:
        plt.figure()
        sns.kdeplot(accumulator.samples[feat][0], fill=True, label='Benign', color='blue', alpha=0.5)
        sns.kdeplot(accumulator.samples[feat][2], fill=True, label='Interference', color='red', alpha=0.5)
        plt.title(f'Separator: {feat} (Tier: {top_v[top_v["feature"]==feat]["tier"].values[0]})')
        plt.legend()
        plt.savefig(os.path.join(RESULTS_DIR, f'top_separator_{feat}.png'))
        plt.close()

    print("Phase 1 streaming analysis complete. Check results/phase1/")

if __name__ == "__main__":
    main()
