"""
Phase 2 — 1000-Epoch Training Dynamics
=======================================
Same model and data as phase2_simplified.py but trained for exactly 1000 epochs
with NO early stopping, to reveal the full learning trajectory.

Generates:
  results/phase2_simplified/training_dynamics/
    loss_curve_1000ep.png      — Train vs Val loss (full run)
    accuracy_curve_1000ep.png  — Train vs Val accuracy (full run)
    roc_curve_1000ep.png       — Test-set ROC curve with AUC
    combined_curves_1000ep.png — Loss + Accuracy + ROC (publication-ready, 3-panel)
    training_log_1000ep.csv    — Epoch-by-epoch metrics

Configuration (identical to phase2_simplified.py):
  Features: 4 ratio signals × {mean, std, delta} = 12
  Model   : Linear(12→16) → ReLU → Dropout(0.1) → Linear(16→16) → ReLU → Dropout(0.1) → Linear(16→1)
  LR      : 2e-3, ReduceLROnPlateau(factor=0.5, patience=5, min_lr=1e-5)
  Loss    : BCEWithLogitsLoss(pos_weight = 2.0 × class_ratio)
  Epochs  : 1000 (no early stopping)
"""

import pandas as pd
import numpy as np
import os
import glob
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_curve, auc as sk_auc
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import matplotlib.patches as mpatches
import time

# ── Config ─────────────────────────────────────────────────────────────────────
DATA_DIR    = 'data/train data/'
OUT_DIR     = 'results/phase2_simplified/training_dynamics/'
os.makedirs(OUT_DIR, exist_ok=True)

W           = 10
N_EPOCHS    = 500
LR          = 2e-3
PW_MULT     = 2.0
DROPOUT     = 0.1
LAYERS      = 2
UNITS       = 16
BATCH_SIZE  = 128
CAP         = 5000
EARLY_STOP_PATIENCE = 10   # tracked but NOT used to stop — only annotated on plot

# File lists are auto-discovered from DATA_DIR at runtime (see discover_files()).
# No hardcoded names — compatible with any set of CSVs in data/.
RATIO_SIGNALS = ['IPC', 'MPKI', 'L2_PRESSURE', 'BRANCH_MISS_RATE']


# ── Feature Engineering (identical to phase2_simplified) ─────────────────────
def engineer_features(df):
    eps = 1e-9
    df  = df.copy()
    df['IPC']              = df['INSTRUCTIONS']    / (df['CPU_CYCLES']      + eps)
    df['MPKI']             = (df['CACHE_MISSES'] * 1000) / (df['INSTRUCTIONS'] + eps)
    df['L2_PRESSURE']      = df['L2_CACHE_ACCESS'] / (df['CPU_CYCLES']      + eps)
    df['BRANCH_MISS_RATE'] = df['BRANCH_MISSES']   / (df['INSTRUCTIONS']    + eps)

    feat_cols = []
    for col in RATIO_SIGNALS:
        df[f'{col}_mean']  = df[col].rolling(W).mean()
        df[f'{col}_std']   = df[col].rolling(W).std()
        df[f'{col}_delta'] = df[col].diff(W).fillna(0)
        feat_cols += [f'{col}_mean', f'{col}_std', f'{col}_delta']

    df_c = df.dropna()
    X = df_c[feat_cols].values.astype(np.float32)
    y = (df_c['LABEL'] == 2).values.astype(np.float32)
    return X, y


def load_file(path):
    df = pd.read_csv(path)
    df = df[df['LABEL'].isin([0, 2])].copy()
    if len(df) < W + 10:
        return None, None
    X, y = engineer_features(df)
    if len(X) > CAP:
        X, y = X[:CAP], y[:CAP]
    return X, y


def discover_files(data_dir):
    """
    Auto-detect benign and attack files from data_dir by reading dominant LABEL.
    Returns (benign_paths, attack_paths) sorted for reproducibility.
    """
    benign, attack = [], []
    for f in sorted(glob.glob(os.path.join(data_dir, '*.csv'))):
        try:
            labels = pd.read_csv(f, usecols=['LABEL'])['LABEL']
            labels = labels[labels.isin([0, 2])]
            if len(labels) < 100:
                continue
            (benign if (labels == 0).sum() >= (labels == 2).sum() else attack).append(f)
        except Exception:
            continue
    return benign, attack


# ── Data loading ───────────────────────────────────────────────────────────────
def load_all():
    """Auto-discovers files, returns (X_train, y_train, X_test, y_test, mean, std)."""
    benign_paths, attack_paths = discover_files(DATA_DIR)

    def _load_seqs(paths):
        seqs = []
        for f in paths:
            X, y = load_file(f)
            if X is not None and len(X) > 0:
                seqs.append((os.path.basename(f), X, y))
        return seqs

    benign_seqs = _load_seqs(benign_paths)
    attack_seqs = _load_seqs(attack_paths)

    def split_files(seqs, ratio=0.30, seed=42):
        np.random.seed(seed)
        idx = np.random.permutation(len(seqs))
        cut = int(len(seqs) * (1 - ratio))
        return [seqs[i] for i in idx[:cut]], [seqs[i] for i in idx[cut:]]

    b_tr, b_te = split_files(benign_seqs)
    a_tr, a_te = split_files(attack_seqs)
    train_seqs = b_tr + a_tr
    test_seqs  = b_te + a_te

    X_train = np.vstack([s[1] for s in train_seqs])
    y_train = np.concatenate([s[2] for s in train_seqs])

    mean = X_train.mean(axis=0)
    std  = X_train.std(axis=0) + 1e-9

    X_test = np.vstack([(s[1] - mean) / std for s in test_seqs])
    y_test = np.concatenate([s[2] for s in test_seqs])

    return X_train, y_train, X_test, y_test, mean, std


class MLP(nn.Module):
    def __init__(self, in_dim=12, layers=2, units=16, dropout=0.1):
        super().__init__()
        seq, last = [], in_dim
        for _ in range(layers):
            seq += [nn.Linear(last, units), nn.ReLU(), nn.Dropout(dropout)]
            last = units
        seq.append(nn.Linear(last, 1))
        self.net = nn.Sequential(*seq)

    def forward(self, x):
        return self.net(x)



# ── Plotting ───────────────────────────────────────────────────────────────────
def smooth_ema(vals, alpha=0.15):
    out, v = [], float(vals[0])
    for x in vals:
        v = alpha * float(x) + (1 - alpha) * v
        out.append(v)
    return out


def annotate_early_stop(ax, epoch, label='Early stop\n(patience=10)', color='red'):
    ax.axvline(epoch, color=color, ls='--', lw=1.5, alpha=0.7)
    ax.text(epoch + 5, ax.get_ylim()[1] * 0.97, label,
            color=color, fontsize=9, va='top')


def plot_curves(log_df, early_stop_epoch, out_dir):
    epochs    = log_df['epoch'].values
    tr_loss   = log_df['train_loss'].values
    va_loss   = log_df['val_loss'].values
    tr_acc    = log_df['train_acc'].values
    va_acc    = log_df['val_acc'].values
    lr_sched  = log_df['lr'].values

    BLUE  = '#1565C0'
    CORAL = '#D84315'
    LGREY = '#9E9E9E'

    # ── A — Loss curve
    fig, ax = plt.subplots(figsize=(13, 6))
    ax.plot(epochs, tr_loss, color=BLUE,  alpha=0.2, lw=0.8)
    ax.plot(epochs, smooth_ema(tr_loss), color=BLUE,  lw=2.2, label='Train Loss (EMA)')
    ax.plot(epochs, va_loss, color=CORAL, alpha=0.2, lw=0.8)
    ax.plot(epochs, smooth_ema(va_loss), color=CORAL, lw=2.2, label='Val Loss (EMA)')
    if early_stop_epoch:
        annotate_early_stop(ax, early_stop_epoch)
    best_ep = int(log_df.loc[log_df['val_loss'].idxmin(), 'epoch'])
    ax.axvline(best_ep, color='green', ls=':', lw=1.5, alpha=0.8)
    ax.text(best_ep + 5, ax.get_ylim()[1] * 0.90,
            f'Best val loss\n(epoch {best_ep})', color='green', fontsize=9, va='top')

    # LR drops on secondary axis
    ax2 = ax.twinx()
    ax2.plot(epochs, lr_sched, color=LGREY, lw=1.2, ls='dashdot', alpha=0.6, label='LR')
    ax2.set_ylabel('Learning Rate', color=LGREY, fontsize=11)
    ax2.tick_params(axis='y', colors=LGREY)
    ax2.set_yscale('log')

    ax.set_xlabel('Epoch', fontsize=13)
    ax.set_ylabel('BCEWithLogitsLoss', fontsize=13)
    ax.set_title('Training Dynamics — Loss Curve (1000 Epochs)\n'
                 'Phase 2 Simplified: W=10, 2×16 MLP, LR=2e-3', fontsize=13)
    lines1, labs1 = ax.get_legend_handles_labels()
    lines2, labs2 = ax2.get_legend_handles_labels()
    ax.legend(lines1 + lines2, labs1 + labs2, fontsize=10, loc='upper right')
    ax.set_xlim(0, N_EPOCHS)
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, 'loss_curve_1000ep.png'), dpi=150)
    plt.close()

    # ── B — Accuracy curve
    fig, ax = plt.subplots(figsize=(13, 6))
    ax.plot(epochs, tr_acc * 100, color=BLUE,  alpha=0.2, lw=0.8)
    ax.plot(epochs, [v * 100 for v in smooth_ema(tr_acc)],
            color=BLUE, lw=2.2, label='Train Accuracy (EMA)')
    ax.plot(epochs, va_acc * 100, color=CORAL, alpha=0.2, lw=0.8)
    ax.plot(epochs, [v * 100 for v in smooth_ema(va_acc)],
            color=CORAL, lw=2.2, label='Val Accuracy (EMA)')
    if early_stop_epoch:
        annotate_early_stop(ax, early_stop_epoch)
    ax.axvline(best_ep, color='green', ls=':', lw=1.5, alpha=0.8)

    ax.set_xlabel('Epoch', fontsize=13)
    ax.set_ylabel('Accuracy (%)', fontsize=13)
    ax.set_title('Training Dynamics — Accuracy Curve (1000 Epochs)\n'
                 'Phase 2 Simplified: W=10, 2×16 MLP, LR=2e-3', fontsize=13)
    ax.legend(fontsize=10, loc='lower right')
    ax.set_xlim(0, N_EPOCHS)
    ax.yaxis.set_major_formatter(ticker.FormatStrFormatter('%.1f'))
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, 'accuracy_curve_1000ep.png'), dpi=150)
    plt.close()

    # ── C — Combined publication-ready figure (2×1 grid)
    fig, axes = plt.subplots(2, 1, figsize=(13, 10), sharex=True)

    # Top: Loss
    ax = axes[0]
    ax.plot(epochs, tr_loss, color=BLUE,  alpha=0.15, lw=0.6)
    ax.plot(epochs, smooth_ema(tr_loss), color=BLUE,  lw=2.2, label='Train')
    ax.plot(epochs, va_loss, color=CORAL, alpha=0.15, lw=0.6)
    ax.plot(epochs, smooth_ema(va_loss), color=CORAL, lw=2.2, label='Validation')
    if early_stop_epoch:
        ax.axvline(early_stop_epoch, color='red', ls='--', lw=1.2, alpha=0.6,
                   label=f'Early stop threshold (ep {early_stop_epoch})')
    ax.axvline(best_ep, color='green', ls=':', lw=1.2, alpha=0.8,
               label=f'Best val loss (ep {best_ep})')
    ax.set_ylabel('BCE Loss', fontsize=12)
    ax.set_title('Phase 2 MLP Training Dynamics — 1000 Epochs\n'
                 '(W=10, 2×16, LR=2e-3, pos_weight=2×class_ratio)', fontsize=13)
    ax.legend(fontsize=10, loc='upper right')
    ax.set_xlim(0, N_EPOCHS)

    # Bottom: Accuracy
    ax = axes[1]
    ax.plot(epochs, [v * 100 for v in tr_acc], color=BLUE,  alpha=0.15, lw=0.6)
    ax.plot(epochs, [v * 100 for v in smooth_ema(tr_acc)],  color=BLUE,  lw=2.2, label='Train')
    ax.plot(epochs, [v * 100 for v in va_acc], color=CORAL, alpha=0.15, lw=0.6)
    ax.plot(epochs, [v * 100 for v in smooth_ema(va_acc)],  color=CORAL, lw=2.2, label='Validation')
    if early_stop_epoch:
        ax.axvline(early_stop_epoch, color='red', ls='--', lw=1.2, alpha=0.6)
    ax.axvline(best_ep, color='green', ls=':', lw=1.2, alpha=0.8)
    ax.set_xlabel('Epoch', fontsize=12)
    ax.set_ylabel('Accuracy (%)', fontsize=12)
    ax.legend(fontsize=10, loc='lower right')
    ax.yaxis.set_major_formatter(ticker.FormatStrFormatter('%.1f'))
    ax.set_xlim(0, N_EPOCHS)

    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, 'combined_curves_1000ep.png'), dpi=150)
    plt.close()

    print(f'\n  Best val loss : epoch {best_ep}  ({log_df.loc[log_df["val_loss"].idxmin(), "val_loss"]:.5f})')
    print(f'  Best val acc  : epoch {int(log_df.loc[log_df["val_acc"].idxmax(), "epoch"])}  '
          f'({log_df["val_acc"].max() * 100:.2f}%)')
    if early_stop_epoch:
        print(f'  Early stop    : would have triggered @ epoch {early_stop_epoch}')
    return best_ep


def plot_roc(y_true, probs, out_dir, n_smooth_vals=(1, 3, 5, 7)):
    """Full ROC curve on test set for multiple causal smoothing windows."""
    BLUE  = '#1565C0'
    CORAL = '#D84315'
    GREEN = '#2E7D32'
    PURP  = '#6A1B9A'
    COLORS = [BLUE, CORAL, GREEN, PURP]

    def causal_smooth(p, n):
        out = np.empty_like(p)
        for i in range(len(p)):
            out[i] = p[max(0, i - n + 1): i + 1].mean()
        return out

    fig, ax = plt.subplots(figsize=(9, 8))

    for n, color in zip(n_smooth_vals, COLORS):
        p = causal_smooth(probs, n) if n > 1 else probs
        fpr_arr, tpr_arr, _ = roc_curve(y_true, p)
        roc_auc = sk_auc(fpr_arr, tpr_arr)
        ax.plot(fpr_arr, tpr_arr, color=color, lw=2.2,
                label=f'N={n}  (AUC = {roc_auc:.4f})')
        # Mark operating point closest to (FPR=0.05, Recall=0.99)
        dists = np.sqrt((fpr_arr - 0.05) ** 2 + (tpr_arr - 0.99) ** 2)
        idx_op = int(np.argmin(dists))
        ax.scatter(fpr_arr[idx_op], tpr_arr[idx_op], color=color,
                   s=100, zorder=5, marker='*')

    # Target region box
    from matplotlib.patches import Rectangle
    ax.add_patch(Rectangle((0, 0.99), 0.10, 0.01,
                            edgecolor='red', facecolor='red', alpha=0.10, lw=1.5,
                            label='Target (Recall≥0.99, FPR≤0.10)'))
    ax.axhline(0.99, color='red', ls=':', lw=1.2, alpha=0.5)
    ax.axvline(0.10, color='red', ls='--', lw=1.2, alpha=0.5)

    ax.plot([0, 1], [0, 1], 'k--', lw=1, alpha=0.4, label='Random (AUC=0.5)')
    ax.set_xlabel('False Positive Rate', fontsize=13)
    ax.set_ylabel('Recall (True Positive Rate)', fontsize=13)
    ax.set_title('ROC Curve — Test Set\n'
                 'Phase 2 Simplified: W=10, 2×16 MLP (1000 epochs)\n'
                 '★ = operating point nearest (FPR=0.05, Recall=0.99)', fontsize=12)
    ax.legend(fontsize=10, loc='lower right')
    ax.set_xlim(-0.01, 1.01); ax.set_ylim(0.0, 1.02)
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, 'roc_curve_1000ep.png'), dpi=150)
    plt.close()
    print('    roc_curve_1000ep.png')
    return fig


def plot_combined_3panel(log_df, early_stop_epoch, y_true, probs, out_dir):
    """Publication-ready 3-panel: Loss | Accuracy | ROC."""
    BLUE  = '#1565C0'
    CORAL = '#D84315'
    epochs  = log_df['epoch'].values
    tr_loss = log_df['train_loss'].values
    va_loss = log_df['val_loss'].values
    tr_acc  = log_df['train_acc'].values
    va_acc  = log_df['val_acc'].values
    best_ep = int(log_df.loc[log_df['val_loss'].idxmin(), 'epoch'])

    fpr_arr, tpr_arr, _ = roc_curve(y_true, probs)
    roc_auc = sk_auc(fpr_arr, tpr_arr)

    fig = plt.figure(figsize=(18, 6))
    gs  = fig.add_gridspec(1, 3, wspace=0.32)

    # Panel 1 — Loss
    ax1 = fig.add_subplot(gs[0])
    ax1.plot(epochs, tr_loss, color=BLUE,  alpha=0.15, lw=0.6)
    ax1.plot(epochs, smooth_ema(tr_loss), color=BLUE,  lw=2.0, label='Train')
    ax1.plot(epochs, va_loss, color=CORAL, alpha=0.15, lw=0.6)
    ax1.plot(epochs, smooth_ema(va_loss), color=CORAL, lw=2.0, label='Validation')
    if early_stop_epoch:
        ax1.axvline(early_stop_epoch, color='red', ls='--', lw=1.2, alpha=0.6,
                    label=f'Early stop (ep {early_stop_epoch})')
    ax1.axvline(best_ep, color='green', ls=':', lw=1.2, alpha=0.8,
                label=f'Best (ep {best_ep})')
    ax1.set_xlabel('Epoch', fontsize=11); ax1.set_ylabel('BCE Loss', fontsize=11)
    ax1.set_title('(a) Loss Curve', fontsize=12, fontweight='bold')
    ax1.legend(fontsize=9); ax1.set_xlim(0, N_EPOCHS)

    # Panel 2 — Accuracy
    ax2 = fig.add_subplot(gs[1])
    ax2.plot(epochs, [v*100 for v in tr_acc], color=BLUE,  alpha=0.15, lw=0.6)
    ax2.plot(epochs, [v*100 for v in smooth_ema(tr_acc)], color=BLUE,  lw=2.0, label='Train')
    ax2.plot(epochs, [v*100 for v in va_acc], color=CORAL, alpha=0.15, lw=0.6)
    ax2.plot(epochs, [v*100 for v in smooth_ema(va_acc)], color=CORAL, lw=2.0, label='Validation')
    if early_stop_epoch:
        ax2.axvline(early_stop_epoch, color='red', ls='--', lw=1.2, alpha=0.6)
    ax2.axvline(best_ep, color='green', ls=':', lw=1.2, alpha=0.8)
    ax2.set_xlabel('Epoch', fontsize=11); ax2.set_ylabel('Accuracy (%)', fontsize=11)
    ax2.set_title('(b) Accuracy Curve', fontsize=12, fontweight='bold')
    ax2.legend(fontsize=9); ax2.set_xlim(0, N_EPOCHS)
    ax2.yaxis.set_major_formatter(ticker.FormatStrFormatter('%.1f'))

    # Panel 3 — ROC (N=1 and N=7)
    ax3 = fig.add_subplot(gs[2])
    for n, color, ls in [(1, BLUE, '-'), (7, CORAL, '-')]:
        p = probs.copy()
        if n > 1:
            out = np.empty_like(p)
            for i in range(len(p)):
                out[i] = p[max(0, i - n + 1): i + 1].mean()
            p = out
        fp, tp, _ = roc_curve(y_true, p)
        a = sk_auc(fp, tp)
        ax3.plot(fp, tp, color=color, lw=2.0, ls=ls, label=f'N={n}  AUC={a:.4f}')
    ax3.plot([0,1],[0,1],'k--',lw=1,alpha=0.4)
    ax3.axhline(0.99, color='red', ls=':', lw=1, alpha=0.5)
    ax3.axvline(0.10, color='red', ls='--', lw=1, alpha=0.5)
    from matplotlib.patches import Rectangle
    ax3.add_patch(Rectangle((0,0.99),0.10,0.01, edgecolor='red',
                             facecolor='red', alpha=0.10, lw=1.2,
                             label='Target region'))
    ax3.set_xlabel('False Positive Rate', fontsize=11)
    ax3.set_ylabel('Recall', fontsize=11)
    ax3.set_title('(c) ROC Curve — Test Set', fontsize=12, fontweight='bold')
    ax3.legend(fontsize=9, loc='lower right')
    ax3.set_xlim(-0.01, 1.01); ax3.set_ylim(0.0, 1.02)

    fig.suptitle('Phase 2 MLP Training Dynamics — W=10, 2×16, LR=2e-3 (1000 Epochs)',
                 fontsize=13, fontweight='bold', y=1.02)
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, 'combined_curves_1000ep.png'), dpi=150,
                bbox_inches='tight')
    plt.close()
    print('    combined_curves_1000ep.png  (3-panel publication figure)')


# ── Main ───────────────────────────────────────────────────────────────────────
def main():
    print('=' * 64)
    print('  Phase 2 — 1000-Epoch Training Dynamics')
    print('  Features : 4 signals × {mean, std, delta} = 12')
    print('  Model    : 2×16 MLP, LR=2e-3, no early stopping')
    print('  Epochs   : 1000')
    print('=' * 64)

    # ── 1. Load data (same split as phase2_simplified)
    print('\nLoading and preparing data...')
    X_all, y_all, X_test, y_test, mean, std = load_all()
    X_norm = (X_all - mean) / std

    class_ratio = (y_all == 0).sum() / max((y_all == 1).sum(), 1)
    print(f'  Train samples : {len(y_all):,}  '
          f'(benign {(y_all==0).sum():,} / attack {(y_all==1).sum():,})')
    print(f'  Test  samples : {len(y_test):,}  '
          f'(benign {(y_test==0).sum():,} / attack {(y_test==1).sum():,})')
    print(f'  Class ratio   : {class_ratio:.3f}  →  pos_weight = {PW_MULT * class_ratio:.3f}')

    X_fit, X_val, y_fit, y_val = train_test_split(
        X_norm, y_all, test_size=0.10, stratify=y_all, random_state=42
    )
    tr_ldr = DataLoader(
        TensorDataset(torch.from_numpy(X_fit), torch.from_numpy(y_fit).view(-1, 1)),
        batch_size=BATCH_SIZE, shuffle=True
    )
    va_ldr = DataLoader(
        TensorDataset(torch.from_numpy(X_val), torch.from_numpy(y_val).view(-1, 1)),
        batch_size=512
    )
    print(f'  Train {len(y_fit):,}  |  Val {len(y_val):,}')

    # ── 2. Model setup
    model = MLP(X_fit.shape[1], LAYERS, UNITS, DROPOUT)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f'  MLP params: {n_params:,}')

    pos_w     = torch.tensor([PW_MULT * class_ratio])
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_w)
    opt       = optim.Adam(model.parameters(), lr=LR, weight_decay=1e-4)
    sched     = optim.lr_scheduler.ReduceLROnPlateau(opt, factor=0.5, patience=5, min_lr=1e-5)

    # ── 3. 1000-epoch training loop
    print(f'\nTraining for {N_EPOCHS} epochs (no early stopping)...')
    log = []
    best_val_loss = float('inf')
    patience_ctr  = 0
    early_stop_ep = None   # epoch where early stopping WOULD have fired

    t0 = time.time()
    for epoch in range(1, N_EPOCHS + 1):
        # — Train
        model.train()
        tl, tc, tt = 0., 0, 0
        for bX, by in tr_ldr:
            opt.zero_grad()
            out  = model(bX)
            loss = criterion(out, by)
            loss.backward(); opt.step()
            tl += loss.item()
            tc += ((torch.sigmoid(out) > 0.5).float() == by).sum().item()
            tt += by.size(0)

        # — Validate
        model.eval()
        vl, vc, vt = 0., 0, 0
        with torch.no_grad():
            for bX, by in va_ldr:
                out  = model(bX)
                vl  += criterion(out, by).item()
                vc  += ((torch.sigmoid(out) > 0.5).float() == by).sum().item()
                vt  += by.size(0)

        avg_tr_loss = tl / len(tr_ldr)
        avg_va_loss = vl / len(va_ldr)
        tr_acc      = tc / tt
        va_acc      = vc / vt
        cur_lr      = opt.param_groups[0]['lr']

        sched.step(avg_va_loss)

        log.append({
            'epoch': epoch, 'train_loss': avg_tr_loss, 'val_loss': avg_va_loss,
            'train_acc': tr_acc, 'val_acc': va_acc, 'lr': cur_lr,
        })

        # Track early-stopping point (but don't stop)
        if avg_va_loss < best_val_loss - 1e-4:
            best_val_loss = avg_va_loss
            patience_ctr  = 0
        else:
            patience_ctr += 1
            if patience_ctr == EARLY_STOP_PATIENCE and early_stop_ep is None:
                early_stop_ep = epoch
                print(f'    [Note] Early stop WOULD fire @ epoch {epoch}  '
                      f'(val_loss={avg_va_loss:.5f}) — continuing anyway...')

        if epoch % 100 == 0 or epoch == 1:
            elapsed = time.time() - t0
            eta     = elapsed / epoch * (N_EPOCHS - epoch)
            print(f'  Ep {epoch:5d}/{N_EPOCHS}  '
                  f'loss={avg_tr_loss:.4f}/{avg_va_loss:.4f}  '
                  f'acc={tr_acc*100:.1f}%/{va_acc*100:.1f}%  '
                  f'lr={cur_lr:.2e}  '
                  f'ETA {eta/60:.1f}min')

    total_time = time.time() - t0
    print(f'\n  Done in {total_time:.1f}s  ({total_time/60:.1f} min)')

    # ── 4. Save log
    log_df = pd.DataFrame(log)
    csv_path = os.path.join(OUT_DIR, 'training_log_1000ep.csv')
    log_df.to_csv(csv_path, index=False)
    print(f'  Log saved → {csv_path}')

    # ── 5. Test-set inference for ROC
    print('\nRunning test-set inference...')
    model.eval()
    with torch.no_grad():
        te_logits = model(torch.from_numpy(X_test)).numpy().flatten()
    te_probs = 1 / (1 + np.exp(-te_logits))   # sigmoid (no temperature — use raw model)
    fpr_n1, tpr_n1, _ = roc_curve(y_test, te_probs)
    print(f'  Test AUC (N=1): {sk_auc(fpr_n1, tpr_n1):.4f}')

    # ── 6. Generate all plots
    print('\nGenerating plots...')
    best_ep = plot_curves(log_df, early_stop_ep, OUT_DIR)
    plot_roc(y_test, te_probs, OUT_DIR, n_smooth_vals=(1, 3, 5, 7))
    plot_combined_3panel(log_df, early_stop_ep, y_test, te_probs, OUT_DIR)

    print(f'\nAll outputs saved to {OUT_DIR}')
    print('  loss_curve_1000ep.png')
    print('  accuracy_curve_1000ep.png')
    print('  roc_curve_1000ep.png')
    print('  combined_curves_1000ep.png  (3-panel publication figure)')
    print('  training_log_1000ep.csv')
    print('Done.\n')


if __name__ == '__main__':
    main()
