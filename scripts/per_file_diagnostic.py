"""
Per-File FPR Diagnostic
========================
Loads the V3 model and evaluates each test file individually.

For each file reports:
  - File type (benign / attack)
  - Sample count after warmup
  - FPR  (benign files only)
  - Recall (attack files only)
  - Mean / std of predicted probabilities
  - Whether file is a "hard" outlier

Also generates:
  - per_file_fpr_bar.png      — FPR bar chart per benign test file
  - per_file_prob_box.png     — probability box plots per file
  - per_file_prob_trace.png   — probability time-series for each test file

Goal: identify whether the 14.5% FPR floor comes from 1-2 hard files
      (data distribution problem) or all files (fundamental overlap).
"""

import pandas as pd
import numpy as np
import os
import glob
import torch
import torch.nn as nn
from scipy.optimize import minimize_scalar
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns

# ── Config (must match V3) ─────────────────────────────────────────────────────
DATA_DIR    = 'data/train data/'
MODELS_DIR  = 'models/simplified_v3/'
RESULTS_DIR = 'results/phase2_simplified_v3/per_file_diagnostic/'
os.makedirs(RESULTS_DIR, exist_ok=True)

W_SHORT    = 10
W_LONG     = 50
EMA_ALPHA  = 0.2
WARMUP     = W_LONG
CAP        = 5000
TAU        = 0.5    # representative threshold for FPR/Recall reporting
                    # (separate bar chart will sweep τ for full picture)

BENIGN_FILES = {
    'data0_clean.csv', 'data1_clean.csv', 'data7_clean.csv',
    'data10_clean.csv', 'data12_clean.csv', 'data15_clean.csv',
    'data18_clean.csv', 'data19_clean.csv', 'data21_clean.csv',
    'data23_clean.csv', 'data24_clean.csv',
}
ATTACK_FILES = {
    'data3_clean.csv', 'data4_clean.csv', 'data5_clean.csv',
    'data6_clean.csv', 'data13_clean.csv', 'data14_clean.csv',
    'data16_clean.csv', 'data17_clean.csv', 'data20_clean.csv',
    'data22_clean.csv', 'data25_clean.csv', 'data26_clean.csv',
    'data27_clean.csv',
}
RATIO_SIGNALS = ['IPC', 'MPKI', 'L2_PRESSURE', 'BRANCH_MISS_RATE']

# Same train/test split as V3 (seed=42, 70/30)
# Determined by the permutation — recreate here to get the same test files
np.random.seed(42)
_benign_sorted = sorted(BENIGN_FILES)
_attack_sorted = sorted(ATTACK_FILES)
_bi = np.random.permutation(len(_benign_sorted))
_ai = np.random.permutation(len(_attack_sorted))
_bcut = int(len(_benign_sorted) * 0.70)
_acut = int(len(_attack_sorted) * 0.70)
TEST_BENIGN_FILES = {_benign_sorted[i] for i in _bi[_bcut:]}
TEST_ATTACK_FILES = {_attack_sorted[i] for i in _ai[_acut:]}

sns.set_theme(style='whitegrid', palette='bright')
plt.rcParams.update({'font.size': 11})


# ── Feature Engineering (identical to V3) ─────────────────────────────────────
def engineer_features_v3(df):
    df  = df.copy()
    eps = 1e-9
    df['IPC']              = df['INSTRUCTIONS']    / (df['CPU_CYCLES']       + eps)
    df['MPKI']             = (df['CACHE_MISSES'] * 1000) / (df['INSTRUCTIONS'] + eps)
    df['L2_PRESSURE']      = df['L2_CACHE_ACCESS'] / (df['CPU_CYCLES']       + eps)
    df['BRANCH_MISS_RATE'] = df['BRANCH_MISSES']   / (df['INSTRUCTIONS']     + eps)

    feat_cols = []
    for col in RATIO_SIGNALS:
        short_mean = df[col].rolling(W_SHORT).mean()
        long_mean  = df[col].rolling(W_LONG).mean()
        long_std   = df[col].rolling(W_LONG).std().fillna(eps)
        df[f'{col}_z']      = (short_mean - long_mean) / (long_std + eps)
        df[f'{col}_delta']  = (df[col] - df[col].shift(W_SHORT - 1)).fillna(0)
        df[f'{col}_accel']  = df[f'{col}_delta'].diff().fillna(0)
        df[f'{col}_energy'] = df[f'{col}_z'].abs().ewm(alpha=EMA_ALPHA, adjust=False).mean()
        feat_cols += [f'{col}_z', f'{col}_delta', f'{col}_accel', f'{col}_energy']

    df_c = df.dropna()
    X = df_c[feat_cols].values.astype(np.float32)
    y = (df_c['LABEL'] == 2).values.astype(np.float32)
    return X, y, feat_cols


def load_file(path):
    df = pd.read_csv(path)
    df = df[df['LABEL'].isin([0, 2])].copy()
    if len(df) < W_LONG + W_SHORT + 10:
        return None, None
    X, y, _ = engineer_features_v3(df)
    X, y = X[WARMUP:], y[WARMUP:]
    if len(X) == 0:
        return None, None
    if len(X) > CAP:
        X, y = X[:CAP], y[:CAP]
    return X, y


# ── Model (identical to V3) ────────────────────────────────────────────────────
class MLP(nn.Module):
    def __init__(self, in_dim, layers, units, dropout=0.1):
        super().__init__()
        seq, last = [], in_dim
        for _ in range(layers):
            seq += [nn.Linear(last, units), nn.ReLU(), nn.Dropout(dropout)]
            last = units
        seq.append(nn.Linear(last, 1))
        self.net = nn.Sequential(*seq)

    def forward(self, x):
        return self.net(x)


def safe_fpr(fp, tn):
    return float(np.clip(fp / (fp + tn + 1e-9), 0.0, 1.0))


# ── Main ───────────────────────────────────────────────────────────────────────
def main():
    # ── Load model
    params_path = os.path.join(MODELS_DIR, 'normalization_params.pth')
    model_path  = os.path.join(MODELS_DIR, 'best_model.pth')
    if not os.path.exists(params_path) or not os.path.exists(model_path):
        print(f'ERROR: Model not found in {MODELS_DIR}')
        print('  → Run phase2_simplified_v3.py first.')
        return

    params   = torch.load(params_path, weights_only=False)
    mean     = params['mean']
    std      = params['std']
    T        = params['temp']
    n_feats  = len(params['features'])

    model = MLP(n_feats, 2, 16, 0.1)
    model.load_state_dict(torch.load(model_path, weights_only=True))
    model.eval()

    print(f'Model loaded  ({n_feats} features, T={T:.4f})')
    print(f'Test benign : {sorted(TEST_BENIGN_FILES)}')
    print(f'Test attack : {sorted(TEST_ATTACK_FILES)}')
    print(f'Threshold τ  = {TAU}  (for per-file FPR/Recall summary)\n')

    all_files = sorted(glob.glob(os.path.join(DATA_DIR, '*.csv')))
    records   = []

    # ── Per-file inference
    for f in all_files:
        name = os.path.basename(f)
        if name not in TEST_BENIGN_FILES and name not in TEST_ATTACK_FILES:
            continue

        X, y = load_file(f)
        if X is None:
            print(f'  SKIP  {name}  (too short after warmup)')
            continue

        file_type = 'benign' if name in TEST_BENIGN_FILES else 'attack'
        X_s = (X - mean) / std

        with torch.no_grad():
            logits = model(torch.from_numpy(X_s)).numpy().flatten()
        probs = 1 / (1 + np.exp(-logits / T))

        # Metrics at τ
        preds = (probs > TAU).astype(int)
        n_benign = (y == 0).sum()
        n_attack = (y == 1).sum()

        if file_type == 'benign':
            fp    = ((y == 0) & (preds == 1)).sum()
            tn    = ((y == 0) & (preds == 0)).sum()
            fpr   = safe_fpr(fp, tn)
            rec   = float('nan')
        else:
            tp    = ((y == 1) & (preds == 1)).sum()
            fn    = ((y == 1) & (preds == 0)).sum()
            rec   = float(tp / (tp + fn + 1e-9))
            fpr   = float('nan')

        records.append({
            'file':     name,
            'type':     file_type,
            'n_samples': len(y),
            'n_benign': int(n_benign),
            'n_attack': int(n_attack),
            'fpr':      round(fpr, 4)  if not np.isnan(fpr) else float('nan'),
            'recall':   round(rec, 4)  if not np.isnan(rec) else float('nan'),
            'prob_mean': round(float(probs.mean()), 4),
            'prob_std':  round(float(probs.std()),  4),
            'prob_p10':  round(float(np.percentile(probs, 10)), 4),
            'prob_p50':  round(float(np.percentile(probs, 50)), 4),
            'prob_p90':  round(float(np.percentile(probs, 90)), 4),
            'probs':     probs,   # kept for plotting, removed before CSV
            'y':         y,
        })

    # ── Print summary table
    print('=' * 82)
    print(f'  {"File":30s}  {"Type":7s}  {"Samples":>8}  '
          f'{"FPR":>7}  {"Recall":>7}  {"P(mean)":>8}  {"P(p50)":>7}  {"P(p90)":>7}')
    print('  ' + '─' * 76)

    total_fp, total_tn = 0, 0
    for r in records:
        fpr_s = f'{r["fpr"]:.4f}' if not np.isnan(r['fpr']) else '    —  '
        rec_s = f'{r["recall"]:.4f}' if not np.isnan(r['recall']) else '    —  '
        flag  = ''
        if not np.isnan(r['fpr']):
            if r['fpr'] > 0.20:
                flag = '  ← 🔴 HIGH FPR'
            elif r['fpr'] > 0.10:
                flag = '  ← ⚠ ELEVATED'
            else:
                flag = '  ← ✓ OK'
        print(f'  {r["file"]:30s}  {r["type"]:7s}  {r["n_samples"]:>8,}  '
              f'{fpr_s:>7}  {rec_s:>7}  '
              f'{r["prob_mean"]:>8.4f}  {r["prob_p50"]:>7.4f}  {r["prob_p90"]:>7.4f}'
              f'{flag}')

    benign_records = [r for r in records if r['type'] == 'benign']
    attack_records = [r for r in records if r['type'] == 'attack']

    if benign_records:
        avg_fpr = np.nanmean([r['fpr'] for r in benign_records])
        max_fpr = np.nanmax([r['fpr'] for r in benign_records])
        min_fpr = np.nanmin([r['fpr'] for r in benign_records])
        print('  ' + '─' * 76)
        print(f'  {"BENIGN SUMMARY":30s}           '
              f'  avg={avg_fpr:.4f}  min={min_fpr:.4f}  max={max_fpr:.4f}')

    if attack_records:
        avg_rec = np.nanmean([r['recall'] for r in attack_records])
        print(f'  {"ATTACK SUMMARY":30s}                    avg_recall={avg_rec:.4f}')
    print('=' * 82)

    # Concentration diagnosis
    if benign_records and len(benign_records) > 1:
        fprs = [(r['file'], r['fpr']) for r in benign_records if not np.isnan(r['fpr'])]
        fprs_sorted = sorted(fprs, key=lambda x: x[1], reverse=True)
        top_file, top_fpr = fprs_sorted[0]
        # Estimate contribution
        top_n = next(r['n_benign'] for r in benign_records if r['file'] == top_file)
        total_n = sum(r['n_benign'] for r in benign_records)
        top_fp  = top_fpr * top_n
        all_fp  = sum(r['fpr'] * r['n_benign'] for r in benign_records if not np.isnan(r['fpr']))
        pct = 100 * top_fp / (all_fp + 1e-9)

        print(f'\n  🔍 FPR Concentration:')
        print(f'     Highest-FPR file : {top_file}  (FPR={top_fpr:.4f})')
        print(f'     Estimated share  : {pct:.1f}% of all false positives')
        if pct > 60:
            print(f'     → CONCENTRATED: FPR is driven by 1 file.')
            print(f'       Fix: Add similar benign workloads to training set.')
        else:
            print(f'     → DISTRIBUTED: FPR is spread across files.')
            print(f'       Fix: Fundamental overlap in feature space.')

    # ── Save CSV (remove raw probs/y arrays)
    csv_records = [{k: v for k, v in r.items() if k not in ('probs', 'y')}
                   for r in records]
    pd.DataFrame(csv_records).to_csv(
        os.path.join(RESULTS_DIR, 'per_file_summary.csv'), index=False
    )

    # ── Plot A — FPR bar chart (benign test files)
    if benign_records:
        fig, ax = plt.subplots(figsize=(10, 5))
        names = [r['file'].replace('_clean.csv', '') for r in benign_records]
        fprs  = [r['fpr'] for r in benign_records]
        colors = ['#d32f2f' if f > 0.20 else '#f57c00' if f > 0.10
                  else '#388e3c' for f in fprs]
        bars = ax.bar(names, fprs, color=colors, width=0.5, zorder=3)
        ax.axhline(0.10, color='red',    ls='--', lw=1.5, label='Target FPR = 0.10', zorder=4)
        ax.axhline(avg_fpr, color='grey', ls=':',  lw=1.5,
                   label=f'Average FPR = {avg_fpr:.4f}',    zorder=4)
        for bar, val in zip(bars, fprs):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.003,
                    f'{val:.3f}', ha='center', va='bottom', fontsize=10, fontweight='bold')
        ax.set_xlabel('Test Benign File', fontsize=12)
        ax.set_ylabel('False Positive Rate', fontsize=12)
        ax.set_title(f'Per-File FPR — Benign Test Files (τ={TAU})\n'
                     'Red = FPR > 0.20,  Orange = FPR > 0.10,  Green = OK', fontsize=12)
        ax.legend(fontsize=10); ax.set_ylim(0, min(1.0, max(fprs) * 1.35))
        plt.tight_layout()
        plt.savefig(os.path.join(RESULTS_DIR, 'per_file_fpr_bar.png'), dpi=150)
        plt.close()

    # ── Plot B — Probability box plots per file
    fig, ax = plt.subplots(figsize=(14, 6))
    all_probs_list = []
    all_labels     = []
    all_types      = []

    for r in records:
        label = r['file'].replace('_clean.csv', '') + f'\n({r["type"]})'
        all_probs_list.append(r['probs'])
        all_labels.append(label)
        all_types.append(r['type'])

    bp = ax.boxplot(all_probs_list, labels=all_labels, patch_artist=True,
                    medianprops={'color': 'black', 'lw': 2},
                    flierprops={'marker': '.', 'markersize': 2, 'alpha': 0.3})
    for patch, t in zip(bp['boxes'], all_types):
        patch.set_facecolor('#2196F3' if t == 'benign' else '#FF5722')
        patch.set_alpha(0.7)

    ax.axhline(TAU, color='red', ls='--', lw=1.5, label=f'Decision threshold τ={TAU}')
    ax.set_ylabel('Predicted Probability', fontsize=12)
    ax.set_title('Probability Distribution per Test File\n'
                 '(Blue = benign, Orange = attack — boxes above τ → FP or TP)', fontsize=12)
    ax.legend(fontsize=10)
    blue_patch   = mpatches.Patch(color='#2196F3', alpha=0.7, label='Benign file')
    orange_patch = mpatches.Patch(color='#FF5722', alpha=0.7, label='Attack file')
    ax.legend(handles=[blue_patch, orange_patch] +
              [mpatches.Patch(color='none', label=f'τ={TAU}')],
              fontsize=10)
    plt.xticks(rotation=20, ha='right')
    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_DIR, 'per_file_prob_box.png'), dpi=150)
    plt.close()

    # ── Plot C — Probability time-series traces per file
    n_files = len(records)
    fig, axes = plt.subplots(n_files, 1, figsize=(14, 3 * n_files), sharex=False)
    if n_files == 1:
        axes = [axes]

    for ax, r in zip(axes, records):
        probs = r['probs']
        y     = r['y']
        t     = np.arange(len(probs))
        color = '#2196F3' if r['type'] == 'benign' else '#FF5722'

        ax.fill_between(t, probs, alpha=0.35, color=color)
        ax.plot(t, probs, color=color, lw=0.8, alpha=0.8)
        ax.axhline(TAU, color='red', ls='--', lw=1, alpha=0.8)

        # Shade true attack regions (if any)
        if (y == 1).any():
            ax.fill_between(t, 0, 1,
                            where=(y == 1), alpha=0.10, color='red',
                            label='Ground truth attack')

        ax.set_ylim(-0.02, 1.05)
        ax.set_ylabel('P(attack)', fontsize=9)
        fpr_label = f'FPR={r["fpr"]:.3f}' if not np.isnan(r['fpr']) else f'Recall={r["recall"]:.3f}'
        ax.set_title(f'{r["file"]}  —  {r["type"].upper()}  '
                     f'({r["n_samples"]:,} samples,  {fpr_label})', fontsize=10)
        ax.set_xlabel('Sample index', fontsize=9)

    plt.suptitle(f'Probability Time-Series — Test Files  (τ={TAU})',
                 fontsize=13, fontweight='bold', y=1.01)
    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_DIR, 'per_file_prob_trace.png'),
                dpi=130, bbox_inches='tight')
    plt.close()

    print(f'\nAll diagnostic outputs saved to {RESULTS_DIR}')
    print('Key file: per_file_fpr_bar.png  (shows which benign file drives FPR)')


if __name__ == '__main__':
    main()
