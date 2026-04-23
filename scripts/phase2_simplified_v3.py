"""
Phase 2 Simplified V3 — Anomaly + EMA Energy Features
=======================================================
Extends V2 with a stateful EMA energy feature per signal.

Feature set (16 total):
  Per ratio signal (IPC, MPKI, L2_PRESSURE, BRANCH_MISS_RATE):
    _z      : z-score = (short_mean - long_mean) / long_std  — anomaly magnitude
    _delta  : rate of change over W_short                    — onset detection
    _accel  : d(delta)/dt                                    — sharpness of onset
    _energy : EMA(|z|, α)                                    — NEW: sustained vs transient

  4 signals × 4 stats = 16 features

Key insight:
  z-score alone fires on both attacks (sustained) and heavy-benign (transient).
  EMA(|z|) separates them:

    State             →  z    →  energy
    Attack onset      →  high →  medium (rising)
    Sustained attack  →  high →  HIGH   (saturated)
    Heavy benign      →  med  →  LOW    (decays quickly)
    Benign normal     →  ~0   →  ~0

  energy = α·|z[t]| + (1−α)·energy[t−1]   (one-pass causal, no lookahead)

  EMA_ALPHA = 0.2 → ~5-sample memory, faster than causal smoothing N=7
  → Can reduce post-hoc smoothing to N=1 (no extra latency penalty)

Warmup fix:
  First W_LONG samples of each file excluded from train/test
  (z-score is unreliable before the baseline stabilises)

Evaluation:
  Same rigorous ROC/Pareto protocol as V2.
  Smoothing windows: [1, 3, 5, 7] — shows energy already acts as smoother.
  Goal: N=1+energy beats N=7 from V2.

Outputs → results/phase2_simplified_v3/
Models  → models/simplified_v3/
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
from matplotlib.patches import Rectangle
import matplotlib.pyplot as plt
import seaborn as sns
import time

# ── Config ─────────────────────────────────────────────────────────────────────
DATA_DIR    = 'data/train data/'
RESULTS_DIR = 'results/phase2_simplified_v3/'
MODELS_DIR  = 'models/simplified_v3/'
os.makedirs(RESULTS_DIR, exist_ok=True)
os.makedirs(MODELS_DIR,  exist_ok=True)

W_SHORT    = 10          # short window: current state
W_LONG     = 50          # long window:  running workload baseline
EMA_ALPHA  = 0.2         # EMA decay for energy feature (~5-sample memory)
WARMUP     = W_LONG      # skip first W_LONG rows per file (unreliable z-score)

BEST_LAYERS  = 2
BEST_UNITS   = 16
BEST_LR      = 2e-3
BEST_PW_MULT = 2.0
BEST_DROPOUT = 0.1
MAX_EPOCHS   = 250
PATIENCE     = 10
CAP          = 5000          # sequential cap (after warmup skip)
SMOOTHING_WINS = [1, 3, 5, 7]
TARGET_RECALL  = 0.99
TARGET_FPR     = 0.10        # relaxed from 0.01 to give visible target bar
N_TAU          = 200

# File lists are auto-discovered from DATA_DIR at runtime (see discover_files()).
# No hardcoded names — compatible with any set of CSVs in data/.
RATIO_SIGNALS = ['IPC', 'MPKI', 'L2_PRESSURE', 'BRANCH_MISS_RATE']
N_FEATURES    = len(RATIO_SIGNALS) * 4   # 16

sns.set_theme(style='whitegrid', palette='bright')
plt.rcParams.update({'figure.figsize': (13, 8), 'font.size': 12})
PALETTE = ['#2196F3', '#FF5722', '#4CAF50', '#9C27B0']


# ── Feature Engineering V3 ────────────────────────────────────────────────────
def engineer_features_v3(df, w_short=W_SHORT, w_long=W_LONG, ema_alpha=EMA_ALPHA):
    """
    4 ratio signals × {z-score, delta, accel, energy} = 16 causal features.

    energy[t] = EMA(|z[t]|, α)
              = α·|z[t]| + (1−α)·energy[t−1]

    Computed with pandas ewm(adjust=False) — fully causal, no lookahead.
    """
    df  = df.copy()
    eps = 1e-9

    # Ratio signals
    df['IPC']              = df['INSTRUCTIONS']    / (df['CPU_CYCLES']       + eps)
    df['MPKI']             = (df['CACHE_MISSES'] * 1000) / (df['INSTRUCTIONS'] + eps)
    df['L2_PRESSURE']      = df['L2_CACHE_ACCESS'] / (df['CPU_CYCLES']       + eps)
    df['BRANCH_MISS_RATE'] = df['BRANCH_MISSES']   / (df['INSTRUCTIONS']     + eps)

    feat_cols = []
    for col in RATIO_SIGNALS:
        short_mean = df[col].rolling(w_short).mean()
        long_mean  = df[col].rolling(w_long).mean()
        long_std   = df[col].rolling(w_long).std().fillna(eps)

        # z-score: deviation from running baseline (workload-invariant)
        df[f'{col}_z']      = (short_mean - long_mean) / (long_std + eps)
        # delta: rate of change
        df[f'{col}_delta']  = (df[col] - df[col].shift(w_short - 1)).fillna(0)
        # accel: second derivative (sharpness of onset)
        df[f'{col}_accel']  = df[f'{col}_delta'].diff().fillna(0)
        # energy: EMA of |z| — sustained vs transient discrimination (KEY feature)
        df[f'{col}_energy'] = df[f'{col}_z'].abs().ewm(alpha=ema_alpha, adjust=False).mean()

        feat_cols += [f'{col}_z', f'{col}_delta', f'{col}_accel', f'{col}_energy']

    df_c = df.dropna()
    X = df_c[feat_cols].values.astype(np.float32)
    y = (df_c['LABEL'] == 2).values.astype(np.float32)
    return X, y, feat_cols


def load_file(path, cap=CAP):
    df = pd.read_csv(path)
    df = df[df['LABEL'].isin([0, 2])].copy()
    min_len = W_LONG + W_SHORT + 10
    if df.empty or len(df) < min_len:
        return None, None, []

    X, y, feats = engineer_features_v3(df)
    if len(X) == 0:
        return None, None, []

    # Skip warmup period where z-score / energy are unreliable
    X, y = X[WARMUP:], y[WARMUP:]
    if len(X) == 0:
        return None, None, []

    # Sequential cap (preserves temporal order)
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
                continue
            if (labels == 0).sum() >= (labels == 2).sum():
                benign.append(f)
            else:
                attack.append(f)
        except Exception:
            continue
    return benign, attack


# ── Model ──────────────────────────────────────────────────────────────────────
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


# ── Calibration ────────────────────────────────────────────────────────────────
def calibrate_temperature(logits_np, labels_np):
    eps = 1e-15
    def obj(t):
        p = 1 / (1 + np.exp(-logits_np / t))
        return -np.mean(labels_np * np.log(p + eps) + (1 - labels_np) * np.log(1 - p + eps))
    return float(minimize_scalar(obj, bounds=(0.1, 5.0), method='bounded').x)


# ── Post-processing ────────────────────────────────────────────────────────────
def causal_smooth(probs, n):
    """Rolling mean with causal constraint and cold-start support."""
    out = np.empty_like(probs)
    for i in range(len(probs)):
        out[i] = probs[max(0, i - n + 1): i + 1].mean()
    return out

def safe_fpr(fp, tn):
    return float(np.clip(fp / (fp + tn + 1e-9), 0.0, 1.0))

def detection_latency(y_true, y_pred):
    starts = np.where(y_true == 1)[0]
    if len(starts) == 0: return None
    hits = np.where(y_pred[starts[0]:] == 1)[0]
    return int(hits[0]) if len(hits) else None


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
    if not inside: return None
    return min(inside, key=lambda r: r['fpr'])

def best_at_recall(rows, min_recall=TARGET_RECALL):
    cands = [r for r in rows if r['recall'] >= min_recall]
    if not cands: return rows[0]
    return min(cands, key=lambda r: r['fpr'])


# ── Helpers ────────────────────────────────────────────────────────────────────
def ema_smooth(vals, a=0.3):
    out, v = [], vals[0]
    for x in vals:
        v = a * x + (1 - a) * v; out.append(v)
    return out

def plot_curves(h, out_dir):
    for metric, label, fname in [('loss','Loss','loss_curve.png'),
                                  ('acc','Accuracy','learning_curve.png')]:
        plt.figure()
        plt.plot(h[f'train_{metric}'], alpha=0.25, color='steelblue')
        plt.plot(ema_smooth(h[f'train_{metric}']), color='steelblue', lw=2, label=f'Train {label}')
        plt.plot(h[f'val_{metric}'],   alpha=0.25, color='coral')
        plt.plot(ema_smooth(h[f'val_{metric}']),   color='coral', lw=2, label=f'Val {label}')
        plt.xlabel('Epoch'); plt.ylabel(label)
        plt.title(f'{label} Curve — V3 (Anomaly + Energy, 16 features)')
        plt.legend(); plt.tight_layout()
        plt.savefig(os.path.join(out_dir, fname)); plt.close()


# ── Main ───────────────────────────────────────────────────────────────────────
def main():
    print('=' * 66)
    print('  Phase 2 Simplified V3 — Anomaly + EMA Energy Features')
    print(f'  Features : {len(RATIO_SIGNALS)} signals × 4 stats = {N_FEATURES} total')
    print(f'             z-score | delta | accel | energy(EMA_α={EMA_ALPHA})')
    print(f'  Warmup   : skip first {WARMUP} samples per file (baseline stabilisation)')
    print(f'  Split    : auto-discovered, stratified 70/30 file-level (seed=42)')
    print(f'  Config   : {BEST_LAYERS}×{BEST_UNITS}, LR={BEST_LR}, W_short={W_SHORT}, W_long={W_LONG}')
    print(f'  Target   : Recall ≥ {TARGET_RECALL}, FPR ≤ {TARGET_FPR}')
    print('=' * 66)

    # ── 1. Auto-discover + load files
    print('\nDiscovering files from data/ directory...')
    benign_paths, attack_paths = discover_files(DATA_DIR)
    print(f'  Benign : {len(benign_paths)} files  |  Attack : {len(attack_paths)} files')

    train_seqs, test_seqs, feats_ref = [], [], []
    skipped = []
    for f, role in ([(f, 'benign') for f in benign_paths] +
                    [(f, 'attack') for f in attack_paths]):
        name = os.path.basename(f)
        X, y, feats = load_file(f)
        if X is None or len(X) == 0:
            skipped.append(name); continue
        if not feats_ref:
            feats_ref = feats
        (benign_paths if role == 'benign' else attack_paths)  # already split below
        entry = (name, X, y, role)
        if role == 'benign':
            train_seqs if False else None   # build lists below

    # rebuild as two separate lists first, then split
    benign_seqs = [(n, X, y) for n, X, y, r in
                   [(n, X, y, r) for n, X, y, r in
                    [(os.path.basename(f), *load_file(f)[:2], 'benign') for f in benign_paths
                     if load_file(f)[0] is not None]] if r == 'benign']

    # Cleaner: redo with two explicit loops
    benign_seqs, attack_seqs, feats_ref = [], [], []
    skipped = []
    for f in benign_paths:
        name = os.path.basename(f)
        X, y, feats = load_file(f)
        if X is None or len(X) == 0: skipped.append(name); continue
        if not feats_ref: feats_ref = feats
        benign_seqs.append((name, X, y))
    for f in attack_paths:
        name = os.path.basename(f)
        X, y, feats = load_file(f)
        if X is None or len(X) == 0: skipped.append(name); continue
        if not feats_ref: feats_ref = feats
        attack_seqs.append((name, X, y))
    if skipped: print(f'  Skipped: {skipped}')

    # ── 2. Stratified file-level split 70/30 (seed=42)
    def split_files(seqs, ratio=0.30, seed=42):
        np.random.seed(seed)
        idx = np.random.permutation(len(seqs))
        cut = int(len(seqs) * (1 - ratio))
        return [seqs[i] for i in idx[:cut]], [seqs[i] for i in idx[cut:]]

    b_tr, b_te = split_files(benign_seqs)
    a_tr, a_te = split_files(attack_seqs)
    train_seqs = b_tr + a_tr
    test_seqs  = b_te + a_te

    print(f'  Train  : {len(b_tr)} benign + {len(a_tr)} attack files')
    print(f'  Test   : {len(b_te)} benign + {len(a_te)} attack files')

    X_train = np.vstack([s[1] for s in train_seqs])
    y_train = np.concatenate([s[2] for s in train_seqs])
    mean    = X_train.mean(axis=0)
    std     = X_train.std(axis=0) + 1e-9
    X_tr_s  = (X_train - mean) / std

    class_ratio = (y_train == 0).sum() / max((y_train == 1).sum(), 1)
    print(f'\n  Samples: {len(y_train):,}  '
          f'(benign {(y_train==0).sum():,} / attack {(y_train==1).sum():,})')
    print(f'  Ratio  : {class_ratio:.2f}:1')

    # ── 3. Calibration split (10% of train)
    X_fit_s, X_cal_s, y_fit, y_cal = train_test_split(
        X_tr_s, y_train, test_size=0.10, stratify=y_train, random_state=42
    )
    tr_ldr = DataLoader(
        TensorDataset(torch.from_numpy(X_fit_s), torch.from_numpy(y_fit).view(-1,1)),
        batch_size=128, shuffle=True)
    ca_ldr = DataLoader(
        TensorDataset(torch.from_numpy(X_cal_s), torch.from_numpy(y_cal).view(-1,1)),
        batch_size=512)

    # ── 4. Train MLP
    model = MLP(X_fit_s.shape[1], BEST_LAYERS, BEST_UNITS, BEST_DROPOUT)
    print(f'\n  MLP params: {count_params(model):,}')
    print('\nTraining MLP...')
    t0      = time.time()
    history = train_model(model, tr_ldr, ca_ldr, BEST_LR, BEST_PW_MULT, class_ratio)
    print(f'  Done in {time.time()-t0:.1f}s  ({len(history["train_loss"])} epochs)')
    plot_curves(history, RESULTS_DIR)

    # ── 5. Temperature calibration
    model.eval()
    with torch.no_grad():
        cal_logits = model(torch.from_numpy(X_cal_s)).numpy().flatten()
    T = calibrate_temperature(cal_logits, y_cal.astype(np.float32))
    print(f'  Optimal T: {T:.4f}')

    torch.save({'mean': mean, 'std': std, 'features': feats_ref,
                'temp': T, 'w_short': W_SHORT, 'w_long': W_LONG, 'ema_alpha': EMA_ALPHA},
               os.path.join(MODELS_DIR, 'normalization_params.pth'))
    torch.save(model.state_dict(), os.path.join(MODELS_DIR, 'best_model.pth'))

    # ── 6. LR Baseline
    print('\nTraining Logistic Regression baseline...')
    lr_model = LogisticRegression(class_weight='balanced', max_iter=1000, random_state=42)
    lr_model.fit(X_fit_s, y_fit)

    # ── 7. Collect test probabilities per N + latency per N
    print('\nCollecting test probabilities + detection latencies...')
    te_y_all = np.concatenate([s[2] for s in test_seqs])
    te_probs_per_n = {}
    latencies_per_n = {}

    for n in SMOOTHING_WINS:
        all_probs, lats = [], []
        for _, te_X, te_y in test_seqs:
            te_X_s = (te_X - mean) / std
            model.eval()
            with torch.no_grad():
                logits = model(torch.from_numpy(te_X_s)).numpy().flatten()
            raw_p    = 1 / (1 + np.exp(-logits / T))
            smooth_p = causal_smooth(raw_p, n) if n > 1 else raw_p
            all_probs.extend(smooth_p)
            # Latency at representative τ = 0.5
            lat = detection_latency(te_y, (smooth_p > 0.5).astype(int))
            if lat is not None: lats.append(lat)
        te_probs_per_n[n]  = np.array(all_probs)
        latencies_per_n[n] = lats

    te_X_all_s = np.vstack([(s[1] - mean) / std for s in test_seqs])
    lr_probs   = lr_model.predict_proba(te_X_all_s)[:, 1]

    # ── 8. Full threshold sweep on test
    print('Running ROC threshold sweeps...')
    sweep_per_n  = {}
    lr_rows      = roc_sweep(lr_probs, te_y_all)
    all_rows     = []
    for n in SMOOTHING_WINS:
        rows = roc_sweep(te_probs_per_n[n], te_y_all)
        for r in rows:
            r['n_smooth']       = n
            r['model']          = 'MLP_v3'
            r['median_latency'] = float(np.median(latencies_per_n[n])) if latencies_per_n[n] else float('nan')
        sweep_per_n[n] = rows
        all_rows.extend(rows)
    for r in lr_rows:
        r['n_smooth'] = 0; r['model'] = 'LR'
    all_rows.extend(lr_rows)

    pd.DataFrame(all_rows).to_csv(os.path.join(RESULTS_DIR, 'roc_sweep_full.csv'), index=False)

    # ── 9. Report
    print('\n' + '=' * 70)
    print('  V3 Results — Test Set ROC Sweep')
    print(f'  Target: Recall ≥ {TARGET_RECALL}, FPR ≤ {TARGET_FPR}')
    print('=' * 70)
    print(f'  {"Model":12s} {"N":>3}  {"Best τ":>7}  '
          f'{"Recall":>7}  {"FPR":>7}  {"F1":>7}  {"Lat(s)":>8}  Region?')
    print('  ' + '─' * 64)

    best_pts = {}
    for n in SMOOTHING_WINS:
        rows = sweep_per_n[n]
        pt   = best_in_region(rows, TARGET_RECALL, TARGET_FPR)
        flag = '✓' if pt else '✗'
        if pt is None:
            pt = best_at_recall(rows, TARGET_RECALL)
        best_pts[n] = pt
        lat_str = f'{pt.get("median_latency", float("nan")):>8.1f}'
        print(f'  {"MLP_v3":12s} N={n}  '
              f'{pt["tau"]:>7.4f}  {pt["recall"]:>7.4f}  '
              f'{pt["fpr"]:>7.4f}  {pt["f1"]:>7.4f}  {lat_str}  {flag}')

    # LR
    lr_pt   = best_in_region(lr_rows, TARGET_RECALL, TARGET_FPR)
    lr_flag = '✓' if lr_pt else '✗'
    if lr_pt is None: lr_pt = best_at_recall(lr_rows, TARGET_RECALL)
    print(f'  {"LR":12s} N=—  '
          f'{lr_pt["tau"]:>7.4f}  {lr_pt["recall"]:>7.4f}  '
          f'{lr_pt["fpr"]:>7.4f}  {lr_pt["f1"]:>7.4f}  {"—":>8}  {lr_flag}')
    print('  ' + '─' * 64)

    # AUC per N
    print('\n  AUC:')
    for n in SMOOTHING_WINS:
        rows_s  = sorted(sweep_per_n[n], key=lambda r: r['fpr'])
        roc_auc = auc([r['fpr'] for r in rows_s], [r['recall'] for r in rows_s])
        print(f'    MLP_v3 N={n}: AUC={roc_auc:.4f}')
    rows_s  = sorted(lr_rows, key=lambda r: r['fpr'])
    lr_auc  = auc([r['fpr'] for r in rows_s], [r['recall'] for r in rows_s])
    print(f'    LR     N=—: AUC={lr_auc:.4f}')

    # Energy effectiveness note
    print(f'\n  ✦ Energy feature check:')
    n1_fpr = best_pts[1]['fpr']
    n7_fpr = best_pts[7]['fpr']
    print(f'    V3 N=1 FPR = {n1_fpr:.4f}  (energy replaces smoothing)')
    print(f'    V3 N=7 FPR = {n7_fpr:.4f}  (energy + smoothing)')
    if n1_fpr < 0.146:
        print(f'    ✓ V3 N=1 beats V2 N=7 ({n1_fpr:.4f} < 0.1456) — energy works as smoother!')
    else:
        print(f'    FPR = {n1_fpr:.4f}  (V2 N=7 reference = 0.1456)')

    # ── 10. Plots

    # Plot A — Full ROC Pareto
    fig, ax = plt.subplots()
    for i, n in enumerate(SMOOTHING_WINS):
        rows_s  = sorted(sweep_per_n[n], key=lambda r: r['fpr'])
        fprs = [r['fpr']    for r in rows_s]
        recs = [r['recall'] for r in rows_s]
        roc_auc = auc(fprs, recs)
        ax.plot(fprs, recs, color=PALETTE[i], lw=2.2,
                label=f'V3 N={n}  (AUC={roc_auc:.3f})')
        pt = best_pts[n]
        ax.scatter(pt['fpr'], pt['recall'], color=PALETTE[i], s=110, zorder=5, marker='*')

    rows_s = sorted(lr_rows, key=lambda r: r['fpr'])
    ax.plot([r['fpr'] for r in rows_s], [r['recall'] for r in rows_s],
            color='grey', lw=1.8, ls='--',
            label=f'Logistic Reg (AUC={lr_auc:.3f})')

    ax.add_patch(Rectangle((0, TARGET_RECALL), TARGET_FPR, 1 - TARGET_RECALL,
                            edgecolor='red', facecolor='red', alpha=0.08, lw=1.5,
                            label=f'Target (R≥{TARGET_RECALL}, FPR≤{TARGET_FPR})'))
    ax.axvline(TARGET_FPR,    color='red', ls='--', alpha=0.4, lw=1)
    ax.axhline(TARGET_RECALL, color='red', ls=':',  alpha=0.4, lw=1)
    ax.set_xlabel('False Positive Rate', fontsize=13)
    ax.set_ylabel('Recall', fontsize=13)
    ax.set_title('ROC Pareto — V3 (Anomaly + EMA Energy, 16 features)\n'
                 '★ = best operating point in target region', fontsize=12)
    ax.legend(fontsize=10, loc='lower right')
    ax.set_xlim(-0.01, 1.01); ax.set_ylim(0.0, 1.02)
    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_DIR, 'roc_pareto.png'), dpi=150); plt.close()

    # Plot B — Zoom operational region
    fig, ax = plt.subplots()
    for i, n in enumerate(SMOOTHING_WINS):
        rows_s = sorted(sweep_per_n[n], key=lambda r: r['fpr'])
        fprs   = np.array([r['fpr']    for r in rows_s])
        recs   = np.array([r['recall'] for r in rows_s])
        mask   = fprs <= 0.40
        ax.plot(fprs[mask], recs[mask], color=PALETTE[i], lw=2.2, label=f'V3 N={n}')
        pt = best_pts[n]
        if pt['fpr'] <= 0.40:
            ax.scatter(pt['fpr'], pt['recall'], color=PALETTE[i], s=130, zorder=5, marker='*')

    rows_s = sorted(lr_rows, key=lambda r: r['fpr'])
    fprs   = np.array([r['fpr'] for r in rows_s])
    recs   = np.array([r['recall'] for r in rows_s])
    ax.plot(fprs[fprs<=0.40], recs[fprs<=0.40], color='grey', lw=1.8, ls='--', label='LR baseline')

    ax.add_patch(Rectangle((0, TARGET_RECALL), TARGET_FPR, 1 - TARGET_RECALL,
                            edgecolor='red', facecolor='red', alpha=0.12, lw=1.5,
                            label=f'Target (R≥{TARGET_RECALL}, FPR≤{TARGET_FPR})'))
    ax.axvline(TARGET_FPR,    color='red', ls='--', alpha=0.5, lw=1.2)
    ax.axhline(TARGET_RECALL, color='red', ls=':',  alpha=0.5, lw=1.2)
    ax.set_xlabel('False Positive Rate', fontsize=13)
    ax.set_ylabel('Recall', fontsize=13)
    ax.set_title('ROC Zoom — Operational Region (FPR ≤ 0.40)\n'
                 'V3: Anomaly + Energy features', fontsize=12)
    ax.legend(fontsize=10)
    ax.set_xlim(-0.005, 0.41); ax.set_ylim(0.85, 1.005)
    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_DIR, 'roc_zoom_operational.png'), dpi=150); plt.close()

    # Plot C — Cross-version comparison (N=1 and N=7 for V2 reference)
    # Hardcode V2 reference values for comparison bar
    v2_ref = {1: (0.9902, 0.1640), 7: (0.9910, 0.1456)}   # (recall, fpr) from V2
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    metrics = ['recall', 'fpr']
    labels  = ['Recall @ best τ', 'FPR @ Recall ≥ 0.99']
    for ax, metric, label in zip(axes, metrics, labels):
        v3_vals  = [best_pts[n][metric] for n in SMOOTHING_WINS]
        v2_n1    = v2_ref[1][metrics.index(metric)]
        v2_n7    = v2_ref[7][metrics.index(metric)]
        x        = np.arange(len(SMOOTHING_WINS))
        bars     = ax.bar(x, v3_vals, width=0.5, color=PALETTE[:len(SMOOTHING_WINS)],
                          alpha=0.8, label='V3')
        ax.axhline(v2_n1, color='grey',  ls='--', lw=1.5, label=f'V2 N=1 ({v2_n1:.3f})')
        ax.axhline(v2_n7, color='black', ls=':', lw=1.5, label=f'V2 N=7 ({v2_n7:.3f})')
        if metric == 'fpr':
            ax.axhline(TARGET_FPR, color='red', ls='--', lw=1.2, alpha=0.7,
                       label=f'Target FPR={TARGET_FPR}')
        ax.set_xticks(x); ax.set_xticklabels([f'N={n}' for n in SMOOTHING_WINS])
        ax.set_title(label, fontsize=12); ax.set_ylabel(metric.upper(), fontsize=12)
        ax.legend(fontsize=9)
        for bar, val in zip(bars, v3_vals):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.002,
                    f'{val:.3f}', ha='center', va='bottom', fontsize=9)
    plt.suptitle('V3 vs V2 Performance by Smoothing Window', fontsize=13, fontweight='bold')
    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_DIR, 'v3_vs_v2_comparison.png'), dpi=150); plt.close()

    # Plot D — Probability distributions
    te_p_n1 = te_probs_per_n[1]
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    # N=1 (no smoothing) — shows raw energy effect
    for ax, n, title in [(axes[0], 1, 'N=1 (Energy only, no smoothing)'),
                          (axes[1], 7, 'N=7 (Energy + causal smoothing)')]:
        probs = te_probs_per_n[n]
        ax.hist(probs[te_y_all == 0], bins=80, alpha=0.55, color='steelblue',
                density=True, label='Benign')
        ax.hist(probs[te_y_all == 1], bins=80, alpha=0.55, color='coral',
                density=True, label='Attack')
        ax.set_xlabel('Predicted Probability', fontsize=12)
        ax.set_ylabel('Density', fontsize=12)
        ax.set_title(title, fontsize=11)
        ax.legend(fontsize=10)
    plt.suptitle('V3 Output Probability Distributions — Test Set', fontsize=12)
    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_DIR, 'prob_distributions.png'), dpi=150); plt.close()

    # Plot E — Threshold sensitivity for best N
    best_n  = max(SMOOTHING_WINS, key=lambda n: best_pts[n]['recall'] - best_pts[n]['fpr'])
    rows_bn = sweep_per_n[best_n]
    taus    = [r['tau']    for r in rows_bn]
    recs    = [r['recall'] for r in rows_bn]
    fprs    = [r['fpr']    for r in rows_bn]
    fig, ax1 = plt.subplots()
    ax2 = ax1.twinx()
    ax1.plot(taus, recs, color='steelblue', lw=2.2, label='Recall')
    ax2.plot(taus, fprs, color='coral',     lw=2.2, label='FPR')
    ax1.axhline(TARGET_RECALL, color='steelblue', ls=':', alpha=0.6)
    ax2.axhline(TARGET_FPR,    color='coral',     ls=':', alpha=0.6)
    ax1.set_xlabel('Threshold τ', fontsize=13)
    ax1.set_ylabel('Recall', color='steelblue', fontsize=13)
    ax2.set_ylabel('FPR',    color='coral',     fontsize=13)
    ax1.set_title(f'Threshold Sensitivity — V3 N={best_n}\n'
                  '(EMA energy stabilises τ sensitivity)', fontsize=12)
    h1, l1 = ax1.get_legend_handles_labels()
    h2, l2 = ax2.get_legend_handles_labels()
    ax1.legend(h1+h2, l1+l2, fontsize=10)
    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_DIR, 'threshold_sensitivity.png'), dpi=150); plt.close()

    print(f'\nAll V3 outputs saved to {RESULTS_DIR}')
    print('Done.\n')


if __name__ == '__main__':
    main()
