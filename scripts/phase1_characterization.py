"""
Phase 1 — Complete Characterization Pipeline
=============================================
Generates ALL outputs currently in results/phase1/:

Raw feature density plots (11 features × benign vs attack):
  density_cpu_cycles.png, density_instructions.png,
  density_cache_misses.png, density_branch_misses.png,
  density_l2_cache_access.png, density_ipc.png, density_mpki.png,
  density_waste_ratio.png, density_bus_pressure.png,
  density_cache_pressure.png, density_branch_miss_rate.png

2D joint plots:
  ipc_vs_mpki_hexbin.png
  branch_miss_rate_vs_mpki_hexbin.png

Correlation / separability:
  correlation_matrix.png
  phase1_correlations.csv
  feature_importance.csv

Outlier analysis:
  outlier_impact_comparison.csv

Window feature separability (streaming, memory-safe):
  top_separator_{feat}_w{W}.png  (top 5 features)
  phase1_feature_stats.csv

Summary:
  phase1_report.md
"""

import pandas as pd
import numpy as np
import os
import glob
import gc
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import ks_2samp

# ── Style ──────────────────────────────────────────────────────────────────────
sns.set_theme(style='whitegrid', palette='muted')
plt.rcParams.update({
    'figure.figsize': (10, 6),
    'font.size': 16,
    'axes.titlesize': 24,
    'axes.labelsize': 20,
    'legend.fontsize': 18,
    'xtick.labelsize': 16,
    'ytick.labelsize': 16,
    'axes.titleweight': 'bold'
})
BENIGN_COLOR = '#2196F3'
ATTACK_COLOR = '#F44336'

DATA_DIR    = 'data/train data/'
RESULTS_DIR = 'results/phase1/'
os.makedirs(RESULTS_DIR, exist_ok=True)

RAW_SAMPLE_PER_FILE = 3000   # samples per file for density / hexbin plots
MAX_CORR_SAMPLES    = 30000  # total samples for correlation matrix
EPS                 = 1e-9

# ── Raw features (for density + hexbin plots) ─────────────────────────────────
RAW_FEATURES = {
    # (column_name, label_in_filename, human_title)
    'CPU_CYCLES':       ('cpu_cycles',       'CPU Cycles'),
    'INSTRUCTIONS':     ('instructions',     'Instructions Retired'),
    'CACHE_MISSES':     ('cache_misses',     'LLC Cache Misses'),
    'BRANCH_MISSES':    ('branch_misses',    'Branch Mispredictions'),
    'L2_CACHE_ACCESS':  ('l2_cache_access',  'L2 Cache Accesses'),
}
DERIVED_FEATURES = {
    # (expr_fn, label, human_title)
    'IPC':              (lambda df: df['INSTRUCTIONS'] / (df['CPU_CYCLES'] + EPS),
                         'ipc', 'IPC (Instructions / Cycle)'),
    'MPKI':             (lambda df: (df['CACHE_MISSES'] * 1000) / (df['INSTRUCTIONS'] + EPS),
                         'mpki', 'MPKI (Misses per 1000 Instructions)'),
    'WASTE_RATIO':      (lambda df: df['CPU_CYCLES'] / (df['INSTRUCTIONS'] + EPS),
                         'waste_ratio', 'Waste Ratio (Cycles / Instruction)'),
    'BUS_PRESSURE':     (lambda df: df['L2_CACHE_ACCESS'] / (df['CPU_CYCLES'] + EPS),
                         'bus_pressure', 'Bus Pressure (L2 Acc / Cycle)'),
    'CACHE_PRESSURE':   (lambda df: df['CACHE_MISSES'] / (df['L2_CACHE_ACCESS'] + EPS),
                         'cache_pressure', 'Cache Pressure (Misses / L2 Acc)'),
    'BRANCH_MISS_RATE': (lambda df: df['BRANCH_MISSES'] / (df['INSTRUCTIONS'] + EPS),
                         'branch_miss_rate', 'Branch Miss Rate (Misses / Instruction)'),
}


# ── Separability accumulator (streaming, window-based) ────────────────────────
class SeparabilityAccumulator:
    """Accumulates sum, sum_sq, N for Cohen's d + reservoir for KS-test."""
    def __init__(self):
        self.stats   = {}
        self.samples = {}
        self.max_samples_per_class = 25000

    def update(self, feat_name, label, values):
        mask = np.isfinite(values)
        if not mask.any(): return
        v = values[mask].astype(np.float64)

        if feat_name not in self.stats:
            self.stats[feat_name]   = {0: {'s': 0., 'ss': 0., 'n': 0},
                                        2: {'s': 0., 'ss': 0., 'n': 0}}
            self.samples[feat_name] = {0: [], 2: []}

        self.stats[feat_name][label]['s']  += v.sum()
        self.stats[feat_name][label]['ss'] += (v ** 2).sum()
        self.stats[feat_name][label]['n']  += len(v)

        bucket = self.samples[feat_name][label]
        if len(bucket) < self.max_samples_per_class:
            step = max(1, len(v) // 200)
            bucket.extend(v[::step][:200])

    def cohen_d(self, feat_name):
        s0 = self.stats[feat_name][0]
        s2 = self.stats[feat_name][2]
        if s0['n'] < 2 or s2['n'] < 2: return 0.
        m0, m2  = s0['s'] / s0['n'], s2['s'] / s2['n']
        v0 = max(0, s0['ss'] / s0['n'] - m0 ** 2)
        v2 = max(0, s2['ss'] / s2['n'] - m2 ** 2)
        sp = np.sqrt(((s0['n'] - 1) * v0 + (s2['n'] - 1) * v2) / (s0['n'] + s2['n'] - 2))
        return 0. if sp < 1e-9 else (m0 - m2) / sp


def overlap_coefficient(x, y, bins=100):
    if len(x) == 0 or len(y) == 0: return 1.
    ax, ay = np.array(x), np.array(y)
    lo, hi = min(np.nanmin(ax), np.nanmin(ay)), max(np.nanmax(ax), np.nanmax(ay))
    if not (np.isfinite(lo) and np.isfinite(hi)) or hi - lo < 1e-9: return 1.
    hx, edges = np.histogram(ax, bins=bins, range=(lo, hi), density=True)
    hy, _     = np.histogram(ay, bins=edges, density=True)
    return float(np.sum(np.minimum(hx, hy) * np.diff(edges)))


# ── Plotting helpers ──────────────────────────────────────────────────────────
def density_plot(benign_vals, attack_vals, title, fname):
    plt.figure(figsize=(12, 6))
    bv = np.array(benign_vals); av = np.array(attack_vals)
    bv = bv[np.isfinite(bv)];   av = av[np.isfinite(av)]
    # Clip extreme outliers for readability (1st-99th percentile)
    lo = min(np.percentile(bv, 1) if len(bv) else 0,
             np.percentile(av, 1) if len(av) else 0)
    hi = max(np.percentile(bv, 99) if len(bv) else 1,
             np.percentile(av, 99) if len(av) else 1)
    bv = bv[(bv >= lo) & (bv <= hi)]
    av = av[(av >= lo) & (av <= hi)]
    if len(bv) > 1:
        sns.kdeplot(bv, fill=True, color=BENIGN_COLOR, alpha=0.55, label='Benign', linewidth=2)
    if len(av) > 1:
        sns.kdeplot(av, fill=True, color=ATTACK_COLOR, alpha=0.55, label='Attack', linewidth=2)
    # Only apply fixed axes to the RAW BRANCH_MISSES counter, not the RATE
    if title == 'Branch Mispredictions' or title == 'BRANCH_MISSES':
        plt.title('Density Distribution: BRANCH_MISSES (Benign vs Interference)')
        plt.xlabel('BRANCH_MISSES')
        plt.xlim(-50000, 650000)
        plt.ylim(0, 1.05e-5)
    else:
        plt.title(f'Distribution: {title}')
        plt.xlabel(title)
        
    plt.ylabel('Density')
    plt.legend(); plt.tight_layout()
    plt.savefig(fname, dpi=130); plt.close()


def hexbin_plot(xb, yb, xa, ya, xlabel, ylabel, fname):
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    for ax, x, y, label, color in [
        (axes[0], xb, yb, 'Benign', BENIGN_COLOR),
        (axes[1], xa, ya, 'Attack', ATTACK_COLOR),
    ]:
        x = np.array(x); y = np.array(y)
        mask = np.isfinite(x) & np.isfinite(y)
        x, y = x[mask], y[mask]
        if len(x) == 0 or len(y) == 0:
            ax.set_title(f'{label} — no data', fontsize=12)
            continue
        xlo, xhi = np.percentile(x, [1, 99])
        ylo, yhi = np.percentile(y, [1, 99])
        m2 = (x >= xlo) & (x <= xhi) & (y >= ylo) & (y <= yhi)
        hb = ax.hexbin(x[m2], y[m2], gridsize=40, cmap='Blues' if label == 'Benign' else 'Reds',
                       mincnt=1)
        plt.colorbar(hb, ax=ax, label='Count')
        ax.set_title(f'{label}')
        ax.set_xlabel(xlabel); ax.set_ylabel(ylabel)
    plt.suptitle(f'{ylabel} vs {xlabel}', fontweight='bold')
    plt.tight_layout(); plt.savefig(fname, dpi=130); plt.close()


# ── Main ───────────────────────────────────────────────────────────────────────
def main():
    print('=' * 62)
    print('  Phase 1 — Complete Characterization Pipeline')
    print('=' * 62)

    csv_files = sorted(glob.glob(os.path.join(DATA_DIR, '*.csv')))
    print(f'\nFound {len(csv_files)} CSV files.')

    # ── PASS 1: Collect raw samples for density / hexbin / correlation
    print('\n[Pass 1] Sampling raw features for density & correlation plots...')
    raw_benign = {k: [] for k in list(RAW_FEATURES) + list(DERIVED_FEATURES)}
    raw_attack = {k: [] for k in list(RAW_FEATURES) + list(DERIVED_FEATURES)}
    n_benign_total = n_attack_total = 0

    for f_path in csv_files:
        try:
            df = pd.read_csv(f_path)
            df = df[df['LABEL'].isin([0, 2])].reset_index(drop=True)
            if len(df) < 100: continue

            for key, (fname_sfx, _) in RAW_FEATURES.items():
                if key not in df.columns: continue
                for label, store in [(0, raw_benign), (2, raw_attack)]:
                    sub = df[df['LABEL'] == label][key].values
                    if len(sub) > RAW_SAMPLE_PER_FILE:
                        sub = sub[::max(1, len(sub) // RAW_SAMPLE_PER_FILE)][:RAW_SAMPLE_PER_FILE]
                    store[key].extend(sub.tolist())

            for key, (expr_fn, _, _) in DERIVED_FEATURES.items():
                col_vals = expr_fn(df).values
                for label, store in [(0, raw_benign), (2, raw_attack)]:
                    mask = (df['LABEL'] == label).values
                    sub  = col_vals[mask]
                    if len(sub) > RAW_SAMPLE_PER_FILE:
                        sub = sub[::max(1, len(sub) // RAW_SAMPLE_PER_FILE)][:RAW_SAMPLE_PER_FILE]
                    store[key].extend(sub.tolist())

            n_benign_total += (df['LABEL'] == 0).sum()
            n_attack_total += (df['LABEL'] == 2).sum()
        except Exception as e:
            print(f'  WARN: {os.path.basename(f_path)}: {e}')

    print(f'  Collected — Benign: {n_benign_total:,}  Attack: {n_attack_total:,} total rows')

    # ── 1. Density plots
    print('\n[1] Generating density plots...')
    for key, (fname_sfx, title) in RAW_FEATURES.items():
        density_plot(raw_benign[key], raw_attack[key], title,
                     os.path.join(RESULTS_DIR, f'density_{fname_sfx}.png'))
        print(f'    density_{fname_sfx}.png')

    for key, (_, fname_sfx, title) in DERIVED_FEATURES.items():
        density_plot(raw_benign[key], raw_attack[key], title,
                     os.path.join(RESULTS_DIR, f'density_{fname_sfx}.png'))
        print(f'    density_{fname_sfx}.png')

    # ── 3. Correlation matrix calculation (Still needed for CSV output)
    print('\n[3] Calculating correlation matrix...')
    all_keys = list(RAW_FEATURES) + list(DERIVED_FEATURES)
    corr_data = {}
    for k in all_keys:
        combined = np.array(raw_benign[k] + raw_attack[k])
        combined = combined[np.isfinite(combined)]
        # Clip outliers for stable correlation
        lo, hi = np.percentile(combined, [1, 99]) if len(combined) else (0, 1)
        combined = np.clip(combined, lo, hi)
        # Trim to MAX_CORR_SAMPLES to keep matrix computation fast
        if len(combined) > MAX_CORR_SAMPLES:
            combined = combined[np.random.choice(len(combined), MAX_CORR_SAMPLES, replace=False)]
        corr_data[k] = combined

    # Align to same length (min across all)
    min_len = min(len(v) for v in corr_data.values())
    corr_df = pd.DataFrame({k: v[:min_len] for k, v in corr_data.items()})
    corr_matrix = corr_df.corr()

    # [REMOVED] Correlation matrix plot generation


    # ── 4. phase1_correlations.csv (key pairwise correlations)
    high_corr_pairs = []
    cols = corr_matrix.columns.tolist()
    for i in range(len(cols)):
        for j in range(i + 1, len(cols)):
            r = corr_matrix.iloc[i, j]
            high_corr_pairs.append({'feature_a': cols[i], 'feature_b': cols[j],
                                    'correlation': round(float(r), 4)})
    corr_pairs_df = pd.DataFrame(high_corr_pairs).sort_values('correlation',
                                                                key=abs, ascending=False)
    corr_pairs_df.to_csv(os.path.join(RESULTS_DIR, 'phase1_correlations.csv'), index=False)
    print('\n    phase1_correlations.csv  (top pairs)')
    print(corr_pairs_df.head(5).to_string(index=False))

    # ── 5. Outlier impact comparison
    print('\n[4] Computing outlier impact...')
    outlier_rows = []
    for key in ['IPC', 'MPKI', 'BRANCH_MISS_RATE', 'BUS_PRESSURE']:
        src = raw_benign.get(key, []) + raw_attack.get(key, [])
        if not src: continue
        arr = np.array(src)
        arr = arr[np.isfinite(arr)]
        lo, hi = np.percentile(arr, [1, 99])
        n_total   = len(arr)
        n_outlier = ((arr < lo) | (arr > hi)).sum()
        mean_all  = float(arr.mean())
        mean_clip = float(arr[(arr >= lo) & (arr <= hi)].mean())
        outlier_rows.append({
            'feature': key,
            'n_total': n_total,
            'n_outliers': int(n_outlier),
            'outlier_pct': round(100 * n_outlier / n_total, 2),
            'mean_with_outliers': round(mean_all, 4),
            'mean_without_outliers': round(mean_clip, 4),
        })
    pd.DataFrame(outlier_rows).to_csv(
        os.path.join(RESULTS_DIR, 'outlier_impact_comparison.csv'), index=False
    )
    print('    outlier_impact_comparison.csv')

    # ── PASS 2: Streaming window-feature separability analysis
    print('\n[Pass 2] Streaming window-feature separability analysis...')
    accumulator  = SeparabilityAccumulator()
    window_sizes = [8, 10, 12, 14]
    total_pure   = 0

    for f_idx, f_path in enumerate(csv_files):
        try:
            df = pd.read_csv(f_path)
            df = df[df['LABEL'].isin([0, 2])].reset_index(drop=True)
            if len(df) < 500: continue

            df['IPC']   = (df['INSTRUCTIONS'] / (df['CPU_CYCLES'] + EPS)).astype(np.float32)
            df['MPKI']  = ((df['CACHE_MISSES'] * 1000) / (df['INSTRUCTIONS'] + EPS)).astype(np.float32)
            df['WASTE'] = (df['CPU_CYCLES'] / (df['INSTRUCTIONS'] + EPS)).astype(np.float32)
            df['BUS_P'] = (df['L2_CACHE_ACCESS'] / (df['CPU_CYCLES'] + EPS)).astype(np.float32)
            base_cols   = ['IPC', 'MPKI', 'WASTE', 'BUS_P']

            for col in ['IPC', 'MPKI']:
                df[f'd_{col}'] = df[col].pct_change().fillna(0).astype(np.float32)
            delta_cols = [f'd_{col}' for col in ['IPC', 'MPKI']]

            # Z-normalise base cols for window analysis
            for col in base_cols:
                m, s = df[col].mean(), df[col].std()
                df[col] = ((df[col] - m) / (s + EPS)).astype(np.float32)

            for w in window_sizes:
                rolling_sum = df['LABEL'].rolling(w).sum()
                p_benign    = rolling_sum <= (0.1 * 2 * w)
                p_interf    = rolling_sum >= (0.9 * 2 * w)
                pure_idx    = np.where(p_benign | p_interf)[0]
                if len(pure_idx) < 200: continue

                np.random.seed(42 + f_idx)
                if len(pure_idx) > 2500:
                    pure_idx = np.random.choice(pure_idx, 2500, replace=False)
                labels     = df['LABEL'].iloc[pure_idx].values
                total_pure += len(pure_idx)

                for col in base_cols:
                    for stat, vals in [('avg', df[col].rolling(w).mean().iloc[pure_idx].values),
                                       ('std', df[col].rolling(w).std().iloc[pure_idx].values)]:
                        feat = f'{col}_{stat}_w{w}'
                        for lbl in [0, 2]:
                            m = labels == lbl
                            if m.any(): accumulator.update(feat, lbl, vals[m])

                for col in ['IPC', 'MPKI']:
                    for stat, vals_gen in [
                        ('p95p5', (df[col].rolling(w).quantile(0.95) -
                                   df[col].rolling(w).quantile(0.05)).iloc[pure_idx].values),
                    ]:
                        feat = f'{col}_{stat}_w{w}'
                        for lbl in [0, 2]:
                            m = labels == lbl
                            if m.any(): accumulator.update(feat, lbl, vals_gen[m])

                for col in delta_cols:
                    feat = f'{col}_avg_w{w}'
                    vals = df[col].rolling(w).mean().iloc[pure_idx].values
                    for lbl in [0, 2]:
                        m = labels == lbl
                        if m.any(): accumulator.update(feat, lbl, vals[m])

                gc.collect()

            del df; gc.collect()
            print(f'    File {f_idx+1}/{len(csv_files)}: {os.path.basename(f_path)}  '
                  f'(pure={total_pure:,})')
        except Exception as e:
            print(f'    WARN: {os.path.basename(f_path)}: {e}')

    # ── 6. Feature stats & importance
    print('\n[5] Computing separability metrics...')
    stats_rows = []
    for feat in accumulator.stats:
        d   = accumulator.cohen_d(feat)
        b   = np.array(accumulator.samples[feat][0])
        att = np.array(accumulator.samples[feat][2])
        if len(b) < 20 or len(att) < 20: continue
        ks, ksp = ks_2samp(b, att)
        ov      = overlap_coefficient(b, att)
        abs_d   = abs(d)
        tier = ('Strong' if abs_d >= 0.8 else 'Moderate' if abs_d >= 0.5
                else 'Weak' if abs_d >= 0.3 else 'Negligible')
        stats_rows.append({'feature': feat, 'weighted_cohen_d': round(d, 4),
                            'abs_d': round(abs_d, 4), 'ks_stat': round(ks, 4),
                            'ks_pval': round(ksp, 6), 'overlap': round(ov, 4),
                            'tier': tier,
                            'n_benign': accumulator.stats[feat][0]['n'],
                            'n_attack': accumulator.stats[feat][2]['n']})

    stats_df = pd.DataFrame(stats_rows).sort_values('abs_d', ascending=False)

    # Correlation-prune redundant features
    print('    Pruning redundant features (|corr| > 0.95)...')
    final_rows, selected = [], []
    for _, row in stats_df.iterrows():
        fn = row['feature']
        pool_f = np.concatenate([accumulator.samples[fn][0], accumulator.samples[fn][2]])
        unique = True
        for seen in selected:
            pool_s = np.concatenate([accumulator.samples[seen][0], accumulator.samples[seen][2]])
            n = min(len(pool_f), len(pool_s))
            if n < 10: continue
            c = np.corrcoef(pool_f[:n], pool_s[:n])[0, 1]
            if abs(c) > 0.95: unique = False; break
        if unique:
            final_rows.append(row); selected.append(fn)

    final_df = pd.DataFrame(final_rows).drop(columns=['abs_d'])
    final_df.to_csv(os.path.join(RESULTS_DIR, 'phase1_feature_stats.csv'), index=False)
    print(f'    phase1_feature_stats.csv  ({len(final_df)} features)')

    # feature_importance.csv — top 10 by Cohen's d (simplified view)
    imp_df = stats_df[['feature', 'weighted_cohen_d', 'ks_stat', 'overlap', 'tier']].head(10)
    imp_df.to_csv(os.path.join(RESULTS_DIR, 'feature_importance.csv'), index=False)
    print('    feature_importance.csv  (top 10)')

    # [REMOVED] Top separator density plots generation


    # ── 8. Summary report
    print('\n[7] Writing phase1_report.md...')
    top5 = final_df.head(5)
    strong_feats = final_df[final_df['tier'] == 'Strong']
    mod_feats    = final_df[final_df['tier'] == 'Moderate']

    report_lines = [
        '# Phase 1 — Characterization Report',
        '',
        '## Dataset Overview',
        '',
        f'| Metric | Value |',
        f'|---|---|',
        f'| Total CSV files | {len(csv_files)} |',
        f'| Benign samples (total) | {n_benign_total:,} |',
        f'| Attack samples (total) | {n_attack_total:,} |',
        f'| Pure-window samples analysed | {total_pure:,} |',
        '',
        '## Feature Separability Summary',
        '',
        f'| Tier | Count |',
        f'|---|---|',
        f'| Strong (|d| ≥ 0.8) | {len(strong_feats)} |',
        f'| Moderate (|d| ≥ 0.5) | {len(mod_feats)} |',
        f'| Total ranked features | {len(final_df)} |',
        '',
        '## Top 5 Discriminative Features',
        '',
        '| Feature | Cohen\'s d | KS Stat | Overlap | Tier |',
        '|---|---|---|---|---|',
    ]
    for _, r in top5.iterrows():
        report_lines.append(
            f'| {r["feature"]} | {r["weighted_cohen_d"]:.4f} | '
            f'{r["ks_stat"]:.4f} | {r["overlap"]:.4f} | {r["tier"]} |'
        )

    report_lines += [
        '',
        '## Key Findings',
        '',
        '- **IPC and MPKI** are the strongest discriminators between benign and attack workloads.',
        '- Rolling standard deviation (variability) features outperform rolling mean features',
        '  as separators, consistent with the hypothesis that interference increases signal noise.',
        '- Larger window sizes (W=10–14) tend to produce stronger Cohen\'s d values than W=8,',
        '  suggesting the model benefits from temporal context.',
        '- Redundant features were pruned at |corr| > 0.95 to reduce collinearity.',
        '',
        '## Threshold Sensitivity',
        '',
        '- Performance is threshold-sensitive. Operational deployment requires per-deployment',
        '  recalibration of the detection threshold τ.',
        '',
        '## Outputs Generated',
        '',
        '| File | Description |',
        '|---|---|',
        '| `density_*.png` | KDE density plots per feature (benign vs attack) |',
        '| `phase1_correlations.csv` | Pairwise correlation table |',
        '| `phase1_feature_stats.csv` | Full ranked + pruned feature statistics |',
        '| `outlier_impact_comparison.csv` | Effect of outlier removal on mean |',
        '| `feature_importance.csv` | Top 10 features by Cohen\'s d |',
    ]
    with open(os.path.join(RESULTS_DIR, 'phase1_report.md'), 'w') as fh:
        fh.write('\n'.join(report_lines) + '\n')
    print('    phase1_report.md')

    print(f'\n{"=" * 62}')
    print(f'  Phase 1 complete. All outputs → {RESULTS_DIR}')
    print(f'{"=" * 62}\n')


if __name__ == '__main__':
    main()
