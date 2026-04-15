"""
Phase 2 Simplified V2 — Anomaly-Based (Relative) Features
==========================================================
Same rigorous evaluation as phase2_simplified.py, but with
WORKLOAD-INVARIANT features based on deviation from running baseline.

Key insight from v1:
  The 12 absolute ratio features have ~30% FPR floor due to distribution
  shift: a heavy-but-benign workload produces similar IPC/MPKI/L2_PRESSURE
  as a light attack. Absolute features cannot distinguish these cases.

Solution — Two-timescale anomaly features:
  For each ratio signal, compute:
    z-score = (short_mean - long_mean) / (long_std + ε)
    delta   = current - value W_short samples ago  (rate of change)
    accel   = delta[t] - delta[t-1]                (acceleration)

  Short window W_short = 10  (captures current state)
  Long  window W_long  = 50  (tracks running workload baseline)

  z-score ≈ 0 for sustained benign workloads (any intensity)
  |z-score| >> 0 when interference suddenly shifts the ratio signal

Features: 4 signals × 3 anomaly stats = 12 inputs (same dimensionality)
Architecture: Input(12) → Linear(16) → ReLU → Dropout(0.1) → Linear(1)

Evaluation: Full ROC/Pareto sweep — same as phase2_simplified.py
"""

import pandas as pd
import numpy as np
import os
import glob
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import recall_score, f1_score, confusion_matrix, auc
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from scipy.optimize import minimize_scalar
import matplotlib.pyplot as plt
import seaborn as sns
import time

# ── Config ─────────────────────────────────────────────────────────────────────
DATA_DIR    = 'data/'
RESULTS_DIR = 'results/phase2_simplified_v2/'
MODELS_DIR  = 'models/simplified_v2/'
os.makedirs(RESULTS_DIR, exist_ok=True)
os.makedirs(MODELS_DIR,  exist_ok=True)

W_SHORT      = 10     # short window: current state
W_LONG       = 50     # long window: running baseline
BEST_LAYERS  = 2
BEST_UNITS   = 16
BEST_LR      = 2e-3
BEST_PW_MULT = 2.0
BEST_DROPOUT = 0.1
MAX_EPOCHS   = 250
PATIENCE     = 10
CAP          = 5000
SMOOTHING_WINS = [1, 3, 5, 7]
TARGET_RECALL  = 0.99
TARGET_FPR     = 0.01
N_TAU          = 200

# File lists are auto-discovered from DATA_DIR at runtime (see discover_files()).
# No hardcoded names — compatible with any set of CSVs in data/.
RATIO_SIGNALS = ['IPC', 'MPKI', 'L2_PRESSURE', 'BRANCH_MISS_RATE']

sns.set_theme(style='whitegrid', palette='bright')
plt.rcParams.update({'figure.figsize': (13, 8), 'font.size': 12})
PALETTE = ['#2196F3', '#FF5722', '#4CAF50', '#9C27B0']


# ── Feature Engineering V2 (Anomaly / Relative) ───────────────────────────────
def engineer_features_v2(df, w_short=W_SHORT, w_long=W_LONG):
    """
    Two-timescale anomaly features.

    For each ratio signal:
      z_score = (short_mean - long_mean) / (long_std + ε)
      delta   = signal[t] - signal[t - w_short + 1]
      accel   = delta[t] - delta[t-1]

    z-score is near 0 for any steady workload.
    It spikes when interference suddenly shifts the ratio signal.
    This makes features workload-intensity-invariant.
    """
    df  = df.copy()
    eps = 1e-9

    # Compute ratio signals
    df['IPC']              = df['INSTRUCTIONS']    / (df['CPU_CYCLES']      + eps)
    df['MPKI']             = (df['CACHE_MISSES'] * 1000) / (df['INSTRUCTIONS'] + eps)
    df['L2_PRESSURE']      = df['L2_CACHE_ACCESS'] / (df['CPU_CYCLES']      + eps)
    df['BRANCH_MISS_RATE'] = df['BRANCH_MISSES']   / (df['INSTRUCTIONS']    + eps)

    feat_cols = []
    for col in RATIO_SIGNALS:
        short_mean = df[col].rolling(w_short).mean()
        long_mean  = df[col].rolling(w_long).mean()
        long_std   = df[col].rolling(w_long).std().fillna(eps)

        # z-score: deviation from running baseline (key feature)
        df[f'{col}_z']     = (short_mean - long_mean) / (long_std + eps)
        # Delta: rate of change over short window
        df[f'{col}_delta'] = (df[col] - df[col].shift(w_short - 1)).fillna(0)
        # Acceleration: second derivative (captures onset sharpness)
        df[f'{col}_accel'] = df[f'{col}_delta'].diff().fillna(0)

        feat_cols += [f'{col}_z', f'{col}_delta', f'{col}_accel']

    # Require long-window warmup before first valid sample
    df_c = df.dropna()
    X    = df_c[feat_cols].values.astype(np.float32)
    y    = (df_c['LABEL'] == 2).values.astype(np.float32)
    return X, y, feat_cols


def load_file(path, cap=CAP):
    df = pd.read_csv(path)
    df = df[df['LABEL'].isin([0, 2])].copy()
    if df.empty or len(df) < W_LONG + W_SHORT:
        return None, None, []
    X, y, feats = engineer_features_v2(df)
    if len(X) > cap:
        X, y = X[:cap], y[:cap]
    return X, y, feats


def discover_files(data_dir):
    """
    Auto-detect benign and attack files from data_dir.
    Reads only the LABEL column from each CSV (fast).
    A file is classified as:
      benign  — dominant label is 0  (majority of rows with label 0 or 2)
      attack  — dominant label is 2
    Returns (benign_paths, attack_paths) sorted for reproducibility.
    """
    benign, attack = [], []
    for f in sorted(glob.glob(os.path.join(data_dir, '*.csv'))):
        try:
            labels = pd.read_csv(f, usecols=['LABEL'])['LABEL']
            labels = labels[labels.isin([0, 2])]
            if len(labels) < 100:
                continue          # file too short to be useful
            if (labels == 0).sum() >= (labels == 2).sum():
                benign.append(f)
            else:
                attack.append(f)
        except Exception:
            continue
    return benign, attack


# ── Model (identical to v1) ────────────────────────────────────────────────────
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

def count_params(m):
    return sum(p.numel() for p in m.parameters() if p.requires_grad)


# ── Training ───────────────────────────────────────────────────────────────────
def train_model(model, tr_ldr, va_ldr, lr, pw_mult, class_ratio):
    pw        = torch.tensor([pw_mult * class_ratio])
    criterion = nn.BCEWithLogitsLoss(pos_weight=pw)
    opt       = optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
    sched     = optim.lr_scheduler.ReduceLROnPlateau(opt, factor=0.5, patience=5, min_lr=1e-5)

    history = {'train_loss': [], 'val_loss': [], 'train_acc': [], 'val_acc': []}
    best_loss, best_state, ctr = float('inf'), None, 0

    for epoch in range(MAX_EPOCHS):
        model.train()
        tl, tc, tt = 0.0, 0, 0
        for bX, by in tr_ldr:
            opt.zero_grad()
            out  = model(bX)
            loss = criterion(out, by)
            loss.backward(); opt.step()
            tl += loss.item()
            tc += ((torch.sigmoid(out) > 0.5).float() == by).sum().item()
            tt += by.size(0)

        model.eval()
        vl, vc, vt = 0.0, 0, 0
        with torch.no_grad():
            for bX, by in va_ldr:
                out  = model(bX)
                vl  += criterion(out, by).item()
                vc  += ((torch.sigmoid(out) > 0.5).float() == by).sum().item()
                vt  += by.size(0)

        avg_vl = vl / len(va_ldr)
        sched.step(avg_vl)
        history['train_loss'].append(tl / len(tr_ldr))
        history['val_loss'].append(avg_vl)
        history['train_acc'].append(tc / tt)
        history['val_acc'].append(vc / vt)

        if avg_vl < best_loss - 1e-4:
            best_loss, ctr = avg_vl, 0
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
        else:
            ctr += 1
            if ctr >= PATIENCE:
                print(f'    Early stop @ epoch {epoch + 1}')
                break

    if best_state:
        model.load_state_dict(best_state)
    return history


# ── Calibration & Post-processing ─────────────────────────────────────────────
def calibrate_temperature(logits_np, labels_np):
    eps = 1e-15
    def obj(t):
        p = 1 / (1 + np.exp(-logits_np / t))
        return -np.mean(labels_np * np.log(p + eps) + (1 - labels_np) * np.log(1 - p + eps))
    return float(minimize_scalar(obj, bounds=(0.1, 5.0), method='bounded').x)


def causal_smooth(probs, n):
    out = np.empty_like(probs)
    for i in range(len(probs)):
        out[i] = probs[max(0, i - n + 1): i + 1].mean()
    return out

def safe_fpr(fp, tn):
    return float(np.clip(fp / (fp + tn + 1e-9), 0.0, 1.0))


# ── ROC Sweep ─────────────────────────────────────────────────────────────────
def roc_sweep(probs, y_true, n_tau=N_TAU):
    rows = []
    for tau in np.linspace(0.0, 1.0, n_tau):
        preds = (probs > tau).astype(int)
        rec   = recall_score(y_true, preds, zero_division=0)
        tn    = ((y_true == 0) & (preds == 0)).sum()
        fp    = ((y_true == 0) & (preds == 1)).sum()
        fpr   = safe_fpr(fp, tn)
        f1    = f1_score(y_true, preds, zero_division=0)
        rows.append({'tau': round(float(tau), 4), 'recall': round(float(rec), 4),
                     'fpr': round(fpr, 4), 'f1': round(float(f1), 4)})
    return rows

def best_in_region(rows, min_recall=TARGET_RECALL, max_fpr=TARGET_FPR):
    inside = [r for r in rows if r['recall'] >= min_recall and r['fpr'] <= max_fpr]
    if not inside:
        return None
    return min(inside, key=lambda r: r['fpr'])


# ── Helpers ────────────────────────────────────────────────────────────────────
def ema(vals, a=0.3):
    out, v = [], vals[0]
    for x in vals:
        v = a * x + (1 - a) * v; out.append(v)
    return out

def plot_curves(h, out_dir, title_suffix=''):
    for metric, label, fname in [('loss','Loss','loss_curve.png'),
                                  ('acc','Accuracy','learning_curve.png')]:
        plt.figure()
        plt.plot(h[f'train_{metric}'], alpha=0.25, color='steelblue')
        plt.plot(ema(h[f'train_{metric}']), color='steelblue', lw=2, label=f'Train {label}')
        plt.plot(h[f'val_{metric}'],   alpha=0.25, color='coral')
        plt.plot(ema(h[f'val_{metric}']),   color='coral', lw=2, label=f'Val {label}')
        plt.xlabel('Epoch'); plt.ylabel(label)
        plt.title(f'{label} Curve{title_suffix}')
        plt.legend(); plt.tight_layout()
        plt.savefig(os.path.join(out_dir, fname)); plt.close()


# ── Main ───────────────────────────────────────────────────────────────────────
def main():
    print('=' * 64)
    print('  Phase 2 Simplified V2 — Anomaly-Based Features')
    print(f'  Feature design: z-score + delta + accel (W_short={W_SHORT}, W_long={W_LONG})')
    print(f'  Signals: {RATIO_SIGNALS}')
    print(f'  Total  : {len(RATIO_SIGNALS)} × 3 = {len(RATIO_SIGNALS)*3} features')
    print(f'  Config : {BEST_LAYERS}×{BEST_UNITS}, LR={BEST_LR}')
    print(f'  Target : Recall ≥ {TARGET_RECALL}, FPR ≤ {TARGET_FPR}')
    print('=' * 64)

    # ── 1. Auto-discover + load files
    print('\nDiscovering files from data/ directory...')
    benign_paths, attack_paths = discover_files(DATA_DIR)
    print(f'  Benign : {len(benign_paths)} files  |  Attack : {len(attack_paths)} files')

    benign_seqs, attack_seqs, feats_ref = [], [], []
    skipped = []
    for f, role in ([(f, 'benign') for f in benign_paths] +
                    [(f, 'attack') for f in attack_paths]):
        name = os.path.basename(f)
        X, y, feats = load_file(f)
        if X is None or len(X) == 0:
            skipped.append(name); continue
        if not feats_ref:
            feats_ref = feats
        entry = (name, X, y)
        (benign_seqs if role == 'benign' else attack_seqs).append(entry)

    if skipped:
        print(f'  Skipped (too short / no valid rows): {skipped}')

    # ── 2. Stratified file-level split 70/30 (seed=42 for reproducibility)
    def split_files(seqs, ratio=0.30, seed=42):
        np.random.seed(seed)
        idx = np.random.permutation(len(seqs))
        cut = int(len(seqs) * (1 - ratio))
        return [seqs[i] for i in idx[:cut]], [seqs[i] for i in idx[cut:]]

    b_tr, b_te = split_files(benign_seqs)
    a_tr, a_te = split_files(attack_seqs)
    train_seqs = b_tr + a_tr
    test_seqs  = b_te + a_te

    print(f'  Train: {len(b_tr)} benign + {len(a_tr)} attack files')
    print(f'  Test : {len(b_te)} benign + {len(a_te)} attack files')

    X_train = np.vstack([s[1] for s in train_seqs])
    y_train = np.concatenate([s[2] for s in train_seqs])
    mean    = X_train.mean(axis=0)
    std     = X_train.std(axis=0) + 1e-9
    X_tr_s  = (X_train - mean) / std

    class_ratio = (y_train == 0).sum() / max((y_train == 1).sum(), 1)
    print(f'\n  Train samples: {len(y_train):,}  '
          f'(benign {(y_train==0).sum():,} / attack {(y_train==1).sum():,})')
    print(f'  Class ratio  : {class_ratio:.2f}:1')

    # ── 3. Calibration split from train
    X_fit_s, X_cal_s, y_fit, y_cal = train_test_split(
        X_tr_s, y_train, test_size=0.10, stratify=y_train, random_state=42
    )
    tr_ldr = DataLoader(
        TensorDataset(torch.from_numpy(X_fit_s), torch.from_numpy(y_fit).view(-1,1)),
        batch_size=128, shuffle=True)
    ca_ldr = DataLoader(
        TensorDataset(torch.from_numpy(X_cal_s), torch.from_numpy(y_cal).view(-1,1)),
        batch_size=512)

    # ── 4. Train
    model = MLP(X_fit_s.shape[1], BEST_LAYERS, BEST_UNITS, BEST_DROPOUT)
    print(f'\n  MLP params   : {count_params(model):,}')
    print('\nTraining MLP...')
    t0      = time.time()
    history = train_model(model, tr_ldr, ca_ldr, BEST_LR, BEST_PW_MULT, class_ratio)
    print(f'  Done in {time.time()-t0:.1f}s  ({len(history["train_loss"])} epochs)')
    plot_curves(history, RESULTS_DIR, title_suffix=' — Anomaly-Based Model (V2)')

    # ── 5. Temperature calibration
    model.eval()
    with torch.no_grad():
        cal_logits = model(torch.from_numpy(X_cal_s)).numpy().flatten()
    T = calibrate_temperature(cal_logits, y_cal.astype(np.float32))
    print(f'  Optimal T    : {T:.4f}')

    torch.save({'mean': mean, 'std': std, 'features': feats_ref, 'temp': T,
                'w_short': W_SHORT, 'w_long': W_LONG},
               os.path.join(MODELS_DIR, 'normalization_params.pth'))
    torch.save(model.state_dict(), os.path.join(MODELS_DIR, 'best_model.pth'))

    # ── 6. LR Baseline
    print('\nTraining Logistic Regression baseline...')
    lr_model = LogisticRegression(class_weight='balanced', max_iter=1000, random_state=42)
    lr_model.fit(X_fit_s, y_fit)

    # ── 7. Collect test probabilities per N
    print('\nCollecting test probabilities...')
    te_y_all = np.concatenate([s[2] for s in test_seqs])
    te_probs_per_n = {}
    for n in SMOOTHING_WINS:
        all_probs = []
        for _, te_X, te_y in test_seqs:
            te_X_s = (te_X - mean) / std
            model.eval()
            with torch.no_grad():
                logits = model(torch.from_numpy(te_X_s)).numpy().flatten()
            raw_p    = 1 / (1 + np.exp(-logits / T))
            smooth_p = causal_smooth(raw_p, n) if n > 1 else raw_p
            all_probs.extend(smooth_p)
        te_probs_per_n[n] = np.array(all_probs)

    te_X_all_s = np.vstack([(s[1] - mean) / std for s in test_seqs])
    lr_probs   = lr_model.predict_proba(te_X_all_s)[:, 1]

    # ── 8. Full threshold sweep
    print('Running threshold sweeps on test set...')
    all_rows = []
    sweep_per_n = {}
    for n in SMOOTHING_WINS:
        rows = roc_sweep(te_probs_per_n[n], te_y_all)
        for r in rows:
            r['n_smooth'] = n; r['model'] = 'MLP_v2'
        sweep_per_n[n] = rows
        all_rows.extend(rows)

    lr_rows = roc_sweep(lr_probs, te_y_all)
    for r in lr_rows:
        r['n_smooth'] = 0; r['model'] = 'LogisticRegression'
    all_rows.extend(lr_rows)

    pareto_df = pd.DataFrame(all_rows)
    pareto_df.to_csv(os.path.join(RESULTS_DIR, 'roc_sweep_full.csv'), index=False)

    # ── 9. Report
    print('\n' + '=' * 66)
    print('  V2 ROC Pareto — Anomaly Features — Threshold Sweep on Test Set')
    print(f'  Target region: Recall ≥ {TARGET_RECALL}, FPR ≤ {TARGET_FPR}')
    print('=' * 66)
    print(f'  {"Model":22s} {"N":>3}  {"Best τ":>7}  {"Recall":>7}  {"FPR":>7}  {"F1":>7}  In-Region?')
    print('  ' + '─' * 62)

    best_pts = {}
    for n in SMOOTHING_WINS:
        rows = sweep_per_n[n]
        pt   = best_in_region(rows, TARGET_RECALL, TARGET_FPR)
        flag = '✓' if pt else '✗'
        if pt is None:
            cands = [r for r in rows if r['recall'] >= TARGET_RECALL]
            pt = min(cands, key=lambda r: r['fpr']) if cands else rows[0]
        best_pts[n] = pt
        print(f'  {"MLP_v2":22s} N={n}  '
              f'{pt["tau"]:>7.4f}  {pt["recall"]:>7.4f}  '
              f'{pt["fpr"]:>7.4f}  {pt["f1"]:>7.4f}  {flag}')

    lr_pt   = best_in_region(lr_rows, TARGET_RECALL, TARGET_FPR)
    lr_flag = '✓' if lr_pt else '✗'
    if lr_pt is None:
        cands = [r for r in lr_rows if r['recall'] >= TARGET_RECALL]
        lr_pt = min(cands, key=lambda r: r['fpr']) if cands else lr_rows[0]
    print(f'  {"LogisticRegression":22s} N=—  '
          f'{lr_pt["tau"]:>7.4f}  {lr_pt["recall"]:>7.4f}  '
          f'{lr_pt["fpr"]:>7.4f}  {lr_pt["f1"]:>7.4f}  {lr_flag}')
    print('  ' + '─' * 62)

    print('\n  ROC AUC:')
    for n in SMOOTHING_WINS:
        rows_s  = sorted(sweep_per_n[n], key=lambda r: r['fpr'])
        roc_auc = auc([r['fpr'] for r in rows_s], [r['recall'] for r in rows_s])
        print(f'    MLP_v2 N={n}: AUC = {roc_auc:.4f}')
    rows_s = sorted(lr_rows, key=lambda r: r['fpr'])
    print(f'    LR     N=—: AUC = {auc([r["fpr"] for r in rows_s], [r["recall"] for r in rows_s]):.4f}')

    # ── 10. Plots
    from matplotlib.patches import Rectangle

    # Plot A — Full ROC Pareto
    fig, ax = plt.subplots()
    for i, n in enumerate(SMOOTHING_WINS):
        rows_s  = sorted(sweep_per_n[n], key=lambda r: r['fpr'])
        fprs    = [r['fpr']    for r in rows_s]
        recs    = [r['recall'] for r in rows_s]
        roc_auc = auc(fprs, recs)
        ax.plot(fprs, recs, color=PALETTE[i], lw=2.2,
                label=f'MLP_v2 N={n}  (AUC={roc_auc:.3f})')
        pt = best_pts[n]
        ax.scatter(pt['fpr'], pt['recall'], color=PALETTE[i], s=110, zorder=5, marker='*')

    rows_s  = sorted(lr_rows, key=lambda r: r['fpr'])
    lr_auc  = auc([r['fpr'] for r in rows_s], [r['recall'] for r in rows_s])
    ax.plot([r['fpr'] for r in rows_s], [r['recall'] for r in rows_s],
            color='grey', lw=1.8, ls='--', label=f'Logistic Reg (AUC={lr_auc:.3f})')

    ax.add_patch(Rectangle((0, TARGET_RECALL), TARGET_FPR, 1 - TARGET_RECALL,
                            linewidth=1.5, edgecolor='red', facecolor='red', alpha=0.08,
                            label=f'Target (R≥{TARGET_RECALL}, FPR≤{TARGET_FPR})'))
    ax.axvline(TARGET_FPR,    color='red', ls='--', alpha=0.4, lw=1)
    ax.axhline(TARGET_RECALL, color='red', ls=':',  alpha=0.4, lw=1)
    ax.set_xlabel('False Positive Rate', fontsize=13)
    ax.set_ylabel('Recall', fontsize=13)
    ax.set_title('ROC Pareto — Anomaly-Based Model V2\n(★ = best point in target region)', fontsize=13)
    ax.legend(fontsize=10, loc='lower right'); ax.set_xlim(-0.01, 1.01); ax.set_ylim(0.0, 1.02)
    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_DIR, 'roc_pareto.png'), dpi=150); plt.close()

    # Plot B — Zoom operational region
    fig, ax = plt.subplots()
    for i, n in enumerate(SMOOTHING_WINS):
        rows_s = sorted(sweep_per_n[n], key=lambda r: r['fpr'])
        fprs   = np.array([r['fpr']    for r in rows_s])
        recs   = np.array([r['recall'] for r in rows_s])
        mask   = fprs <= 0.40
        ax.plot(fprs[mask], recs[mask], color=PALETTE[i], lw=2.2, label=f'MLP_v2 N={n}')
        pt = best_pts[n]
        if pt['fpr'] <= 0.40:
            ax.scatter(pt['fpr'], pt['recall'], color=PALETTE[i], s=120, zorder=5, marker='*')

    rows_s = sorted(lr_rows, key=lambda r: r['fpr'])
    fprs   = np.array([r['fpr'] for r in rows_s])
    recs   = np.array([r['recall'] for r in rows_s])
    ax.plot(fprs[fprs<=0.40], recs[fprs<=0.40], color='grey', lw=1.8, ls='--', label='Logistic Reg')

    ax.add_patch(Rectangle((0, TARGET_RECALL), TARGET_FPR, 1 - TARGET_RECALL,
                            linewidth=1.5, edgecolor='red', facecolor='red', alpha=0.12,
                            label=f'Target (R≥{TARGET_RECALL}, FPR≤{TARGET_FPR})'))
    ax.axvline(TARGET_FPR,    color='red', ls='--', alpha=0.5, lw=1.2)
    ax.axhline(TARGET_RECALL, color='red', ls=':',  alpha=0.5, lw=1.2)
    ax.set_xlabel('False Positive Rate', fontsize=13)
    ax.set_ylabel('Recall', fontsize=13)
    ax.set_title('ROC Zoom — Operational Region (FPR ≤ 0.40)\nAnomaly-Based V2 vs V1', fontsize=13)
    ax.legend(fontsize=10); ax.set_xlim(-0.005, 0.41); ax.set_ylim(0.85, 1.005)
    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_DIR, 'roc_zoom_operational.png'), dpi=150); plt.close()

    # Plot C — Probability distribution
    te_p_n1 = te_probs_per_n[1]
    fig, ax = plt.subplots()
    ax.hist(te_p_n1[te_y_all == 0], bins=80, alpha=0.55, color='steelblue',
            density=True, label='Benign')
    ax.hist(te_p_n1[te_y_all == 1], bins=80, alpha=0.55, color='coral',
            density=True, label='Attack')
    ax.set_xlabel('Predicted Probability (N=1)', fontsize=13)
    ax.set_ylabel('Density', fontsize=13)
    ax.set_title('V2 Output Probability Distribution — Test Set\n'
                 '(Better separation expected from anomaly features)', fontsize=12)
    ax.legend(fontsize=11); plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_DIR, 'prob_distribution.png'), dpi=150); plt.close()

    # Plot D — Confusion matrix for best N
    best_n  = max(SMOOTHING_WINS, key=lambda n: best_pts[n]['recall'] - best_pts[n]['fpr'])
    best_pt = best_pts[best_n]
    preds   = (te_probs_per_n[best_n] > best_pt['tau']).astype(int)
    cm      = confusion_matrix(te_y_all, preds)
    fig, ax = plt.subplots(figsize=(6,5))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=['Benign','Attack'], yticklabels=['Benign','Attack'], ax=ax)
    ax.set_title(f'V2 Confusion Matrix — MLP N={best_n}\n'
                 f'τ={best_pt["tau"]:.3f}  Recall={best_pt["recall"]:.4f}  FPR={best_pt["fpr"]:.4f}')
    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_DIR, 'confusion_matrix_final.png'), dpi=130); plt.close()

    print(f'\nAll V2 outputs saved to {RESULTS_DIR}')
    print('Done.\n')


if __name__ == '__main__':
    main()
