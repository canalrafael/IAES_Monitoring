"""
Phase 2 Simplified — Rigorous ROC/Pareto Evaluation
=====================================================
12-feature model (4 ratio signals × 3 causal rolling stats).

Evaluation methodology (no fixed-τ primary metric):
  - Full threshold sweep on test set → Recall vs FPR curve per N
  - Pareto frontier reported for each smoothing window N ∈ {1,3,5,7}
  - Optimal operating region highlighted: Recall ≥ 0.99, FPR ≤ 0.01
  - Logistic Regression baseline evaluated with same ROC protocol
  - Threshold sensitivity note included in reports

Fixes from v1:
  1. No fixed τ as primary metric (calibration does not transfer)
  2. Full threshold sweep on test set (ROC curve per N)
  3. Pareto frontiers replace single-point tables
  4. LR baseline compared via same ROC evaluation
  5. Best operating region highlighted in plots/CSV

Outputs → results/phase2_simplified/
Models  → models/simplified/
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
RESULTS_DIR = 'results/phase2_simplified/'
MODELS_DIR  = 'models/simplified/'
os.makedirs(RESULTS_DIR, exist_ok=True)
os.makedirs(MODELS_DIR,  exist_ok=True)

BEST_W       = 10
BEST_LAYERS  = 2
BEST_UNITS   = 16
BEST_LR      = 2e-3
BEST_PW_MULT = 2.0
BEST_DROPOUT = 0.1
MAX_EPOCHS   = 250
PATIENCE     = 10
CAP          = 5000
SMOOTHING_WINS = [1, 3, 5, 7]
# Target operating region
TARGET_RECALL = 0.99
TARGET_FPR    = 0.01
N_TAU         = 200          # threshold sweep resolution

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

sns.set_theme(style='whitegrid', palette='bright')
plt.rcParams.update({'figure.figsize': (13, 8), 'font.size': 12})
PALETTE = ['#2196F3', '#FF5722', '#4CAF50', '#9C27B0']  # blue, orange, green, purple


# ── Feature Engineering ────────────────────────────────────────────────────────
def engineer_features(df, w):
    df = df.copy()
    eps = 1e-9
    df['IPC']              = df['INSTRUCTIONS']    / (df['CPU_CYCLES']      + eps)
    df['MPKI']             = (df['CACHE_MISSES'] * 1000) / (df['INSTRUCTIONS'] + eps)
    df['L2_PRESSURE']      = df['L2_CACHE_ACCESS'] / (df['CPU_CYCLES']      + eps)
    df['BRANCH_MISS_RATE'] = df['BRANCH_MISSES']   / (df['INSTRUCTIONS']    + eps)

    feat_cols = []
    for col in RATIO_SIGNALS:
        r = df[col].rolling(window=w)
        df[f'{col}_mean']  = r.mean()
        df[f'{col}_std']   = r.std().fillna(0)
        df[f'{col}_delta'] = (df[col] - df[col].shift(w - 1)).fillna(0)
        feat_cols += [f'{col}_mean', f'{col}_std', f'{col}_delta']

    df_c = df.dropna()
    X = df_c[feat_cols].values.astype(np.float32)
    y = (df_c['LABEL'] == 2).values.astype(np.float32)
    return X, y, feat_cols


def load_file(path, w):
    df = pd.read_csv(path)
    df = df[df['LABEL'].isin([0, 2])].copy()
    df = df.drop_duplicates(subset=['TIMESTAMP'], keep='first').copy()
    if df.empty:
        return None, None, []
    X, y, feats = engineer_features(df, w)
    return X, y, feats


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

    history   = {'train_loss': [], 'val_loss': [], 'train_acc': [], 'val_acc': []}
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
    """Full threshold sweep → list of (tau, recall, fpr, f1) dicts."""
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
    """Return the best operating point inside the target region."""
    inside = [r for r in rows if r['recall'] >= min_recall and r['fpr'] <= max_fpr]
    if not inside:
        return None
    # Best = lowest FPR inside region
    return min(inside, key=lambda r: r['fpr'])


# ── Helpers ────────────────────────────────────────────────────────────────────
def ema(vals, a=0.3):
    out, v = [], vals[0]
    for x in vals:
        v = a * x + (1 - a) * v; out.append(v)
    return out

def plot_curves(h, out_dir):
    for metric, label, fname in [('loss','Loss','loss_curve.png'),
                                  ('acc','Accuracy','learning_curve.png')]:
        plt.figure()
        plt.plot(h[f'train_{metric}'], alpha=0.25, color='steelblue')
        plt.plot(ema(h[f'train_{metric}']), color='steelblue', lw=2, label=f'Train {label}')
        plt.plot(h[f'val_{metric}'],   alpha=0.25, color='coral')
        plt.plot(ema(h[f'val_{metric}']),   color='coral',     lw=2, label=f'Val {label}')
        plt.xlabel('Epoch'); plt.ylabel(label)
        plt.title(f'{label} Curve — 12-Feature Simplified Model')
        plt.legend(); plt.tight_layout()
        plt.savefig(os.path.join(out_dir, fname)); plt.close()


# ── Main ───────────────────────────────────────────────────────────────────────
def main():
    print('=' * 64)
    print('  Phase 2 Simplified — ROC/Pareto Evaluation')
    print(f'  Features: {len(RATIO_SIGNALS)} ratio signals × 3 stats = {len(RATIO_SIGNALS)*3}')
    print(f'  Config  : W={BEST_W}, {BEST_LAYERS}×{BEST_UNITS}, LR={BEST_LR}')
    print(f'  Target  : Recall ≥ {TARGET_RECALL}, FPR ≤ {TARGET_FPR}')
    print('=' * 64)

    all_files = sorted(glob.glob(os.path.join(DATA_DIR, '*.csv')))

    # ── 1. Load files
    print('\nLoading files...')
    benign_seqs, attack_seqs, feats_ref = [], [], []
    for f in all_files:
        name = os.path.basename(f)
        if name not in BENIGN_FILES and name not in ATTACK_FILES:
            continue
        X, y, feats = load_file(f, BEST_W)
        if X is None or len(X) == 0:
            continue
        if not feats_ref:
            feats_ref = feats
        entry = (name, X, y)
        (benign_seqs if name in BENIGN_FILES else attack_seqs).append(entry)

    print(f'  Benign files: {len(benign_seqs)}  |  Attack files: {len(attack_seqs)}')

    # ── 2. Stratified file-level split 70/30
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

    # [NEW] Uniform steady-state sampling to preserve fast train/grid speed
    # without sacrificing the global dataset distribution (steady-state exposure).
    GLOBAL_CAP = 200000
    if len(X_train) > GLOBAL_CAP:
        np.random.seed(42)
        idx = np.random.choice(len(X_train), GLOBAL_CAP, replace=False)
        X_train, y_train = X_train[idx], y_train[idx]

    mean    = X_train.mean(axis=0)
    std     = X_train.std(axis=0) + 1e-9
    X_tr_s  = (X_train - mean) / std

    class_ratio = (y_train == 0).sum() / max((y_train == 1).sum(), 1)
    print(f'\n  Train samples: {len(y_train):,}  '
          f'(benign {(y_train==0).sum():,} / attack {(y_train==1).sum():,})')
    print(f'  Class ratio  : {class_ratio:.2f}:1')

    # ── 3. Cal split from train
    X_fit_s, X_cal_s, y_fit, y_cal = train_test_split(
        X_tr_s, y_train, test_size=0.10, stratify=y_train, random_state=42
    )
    tr_ldr = DataLoader(
        TensorDataset(torch.from_numpy(X_fit_s), torch.from_numpy(y_fit).view(-1,1)),
        batch_size=2048, shuffle=True)
    ca_ldr = DataLoader(
        TensorDataset(torch.from_numpy(X_cal_s), torch.from_numpy(y_cal).view(-1,1)),
        batch_size=4096)

    # ── 4. Train MLP
    model = MLP(X_fit_s.shape[1], BEST_LAYERS, BEST_UNITS, BEST_DROPOUT)
    print(f'\n  MLP params   : {count_params(model):,}')
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
    print(f'  Optimal T    : {T:.4f}')

    torch.save({'mean': mean, 'std': std, 'features': feats_ref, 'temp': T},
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

    # LR probabilities (no smoothing)
    te_X_all_s = np.vstack([(s[1] - mean) / std for s in test_seqs])
    lr_probs   = lr_model.predict_proba(te_X_all_s)[:, 1]

    # ── 8. Full threshold sweep — test set (ROC curves)
    print('Running threshold sweeps on test set...')
    all_rows = []
    sweep_per_n = {}
    for n in SMOOTHING_WINS:
        rows = roc_sweep(te_probs_per_n[n], te_y_all)
        for r in rows:
            r['n_smooth'] = n
            r['model']    = 'MLP'
        sweep_per_n[n] = rows
        all_rows.extend(rows)

    # LR sweep
    lr_rows = roc_sweep(lr_probs, te_y_all)
    for r in lr_rows:
        r['n_smooth'] = 0
        r['model']    = 'LogisticRegression'
    all_rows.extend(lr_rows)

    pareto_df = pd.DataFrame(all_rows)
    pareto_df.to_csv(os.path.join(RESULTS_DIR, 'roc_sweep_full.csv'), index=False)

    # ── 9. Best operating points per N
    print('\n' + '=' * 66)
    print('  ROC Pareto — Threshold Sweep on Test Set')
    print(f'  Target region: Recall ≥ {TARGET_RECALL}, FPR ≤ {TARGET_FPR}')
    print('=' * 66)
    print(f'  {"Model":22s} {"N":>3}  {"Best τ":>7}  '
          f'{"Recall":>7}  {"FPR":>7}  {"F1":>7}  In-Region?')
    print('  ' + '─' * 62)

    best_pts = {}
    for n in SMOOTHING_WINS:
        rows  = sweep_per_n[n]
        pt    = best_in_region(rows, TARGET_RECALL, TARGET_FPR)
        flag  = '✓' if pt else '✗'
        if pt is None:
            # Show best achievable at recall >= TARGET_RECALL
            candidates = [r for r in rows if r['recall'] >= TARGET_RECALL]
            pt = min(candidates, key=lambda r: r['fpr']) if candidates else rows[0]
        best_pts[n] = pt
        print(f'  {"MLP":22s} N={n}  '
              f'{pt["tau"]:>7.4f}  {pt["recall"]:>7.4f}  '
              f'{pt["fpr"]:>7.4f}  {pt["f1"]:>7.4f}  {flag}')

    # LR best
    lr_pt   = best_in_region(lr_rows, TARGET_RECALL, TARGET_FPR)
    lr_flag = '✓' if lr_pt else '✗'
    if lr_pt is None:
        cands = [r for r in lr_rows if r['recall'] >= TARGET_RECALL]
        lr_pt = min(cands, key=lambda r: r['fpr']) if cands else lr_rows[0]
    print(f'  {"LogisticRegression":22s} N=—  '
          f'{lr_pt["tau"]:>7.4f}  {lr_pt["recall"]:>7.4f}  '
          f'{lr_pt["fpr"]:>7.4f}  {lr_pt["f1"]:>7.4f}  {lr_flag}')
    print('  ' + '─' * 62)

    # AUC per N
    print('\n  ROC AUC:')
    for n in SMOOTHING_WINS:
        rows_sorted = sorted(sweep_per_n[n], key=lambda r: r['fpr'])
        fprs = [r['fpr'] for r in rows_sorted]
        recs = [r['recall'] for r in rows_sorted]
        roc_auc = auc(fprs, recs)
        print(f'    MLP N={n}: AUC = {roc_auc:.4f}')
    rows_sorted = sorted(lr_rows, key=lambda r: r['fpr'])
    lr_auc = auc([r['fpr'] for r in rows_sorted], [r['recall'] for r in rows_sorted])
    print(f'    LR  N=—: AUC = {lr_auc:.4f}')

    # ── 10. Threshold sensitivity note
    print('\n  ⚠  Threshold Sensitivity Note:')
    print('     Performance is strongly τ-dependent. The calibration-set τ')
    print('     does not transfer reliably to test files with different workloads.')
    print('     ROC curves above reflect the true model capability space.')
    print('     Operational deployment requires per-deployment τ recalibration.')

    # ── 11. Plots

    # Plot A — ROC Pareto by N (MLP only)
    fig, ax = plt.subplots()
    for i, n in enumerate(SMOOTHING_WINS):
        rows_sorted = sorted(sweep_per_n[n], key=lambda r: r['fpr'])
        fprs = [r['fpr'] for r in rows_sorted]
        recs = [r['recall'] for r in rows_sorted]
        roc_auc = auc(fprs, recs)
        ax.plot(fprs, recs, color=PALETTE[i], lw=2.2,
                label=f'MLP N={n}  (AUC={roc_auc:.3f})')
        # Mark best in-region point
        pt = best_pts[n]
        ax.scatter(pt['fpr'], pt['recall'], color=PALETTE[i], s=100,
                   zorder=5, marker='*')

    # LR baseline
    rows_s = sorted(lr_rows, key=lambda r: r['fpr'])
    ax.plot([r['fpr'] for r in rows_s], [r['recall'] for r in rows_s],
            color='grey', lw=1.8, ls='--',
            label=f'Logistic Reg (AUC={lr_auc:.3f})')

    # Target region rectangle
    from matplotlib.patches import Rectangle
    ax.add_patch(Rectangle((0, TARGET_RECALL), TARGET_FPR, 1 - TARGET_RECALL,
                            linewidth=1.5, edgecolor='red', facecolor='red', alpha=0.08,
                            label=f'Target region\n(Recall≥{TARGET_RECALL}, FPR≤{TARGET_FPR})'))
    ax.axvline(TARGET_FPR,    color='red',  ls='--', alpha=0.4, lw=1)
    ax.axhline(TARGET_RECALL, color='red',  ls=':',  alpha=0.4, lw=1)

    ax.set_xlabel('False Positive Rate', fontsize=13)
    ax.set_ylabel('Recall (True Positive Rate)', fontsize=13)
    ax.set_title('ROC Pareto — Simplified Model\n'
                 '(★ = best operating point in target region)', fontsize=13)
    ax.legend(fontsize=10, loc='lower right'); ax.set_xlim(-0.01, 1.01)
    ax.set_ylim(0.0, 1.02)
    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_DIR, 'roc_pareto.png'), dpi=150); plt.close()

    # Plot B — Zoom into bottom-left corner (operational region)
    fig, ax = plt.subplots()
    for i, n in enumerate(SMOOTHING_WINS):
        rows_s = sorted(sweep_per_n[n], key=lambda r: r['fpr'])
        fprs = np.array([r['fpr'] for r in rows_s])
        recs = np.array([r['recall'] for r in rows_s])
        mask = fprs <= 0.40
        ax.plot(fprs[mask], recs[mask], color=PALETTE[i], lw=2.2, label=f'MLP N={n}')
        pt = best_pts[n]
        if pt['fpr'] <= 0.40:
            ax.scatter(pt['fpr'], pt['recall'], color=PALETTE[i], s=120, zorder=5, marker='*')

    rows_s = sorted(lr_rows, key=lambda r: r['fpr'])
    fprs = np.array([r['fpr'] for r in rows_s])
    recs = np.array([r['recall'] for r in rows_s])
    mask = fprs <= 0.40
    ax.plot(fprs[mask], recs[mask], color='grey', lw=1.8, ls='--', label='Logistic Reg')

    ax.add_patch(Rectangle((0, TARGET_RECALL), TARGET_FPR, 1 - TARGET_RECALL,
                            linewidth=1.5, edgecolor='red', facecolor='red', alpha=0.12,
                            label=f'Target\n(R≥{TARGET_RECALL}, FPR≤{TARGET_FPR})'))
    ax.axvline(TARGET_FPR,    color='red', ls='--', alpha=0.5, lw=1.2)
    ax.axhline(TARGET_RECALL, color='red', ls=':',  alpha=0.5, lw=1.2)

    ax.set_xlabel('False Positive Rate', fontsize=13)
    ax.set_ylabel('Recall', fontsize=13)
    ax.set_title('ROC Zoom — Operational Region (FPR ≤ 0.40)', fontsize=13)
    ax.legend(fontsize=10); ax.set_xlim(-0.005, 0.41); ax.set_ylim(0.85, 1.005)
    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_DIR, 'roc_zoom_operational.png'), dpi=150); plt.close()

    # Plot C — Recall vs τ and FPR vs τ for best N
    best_n   = max(SMOOTHING_WINS, key=lambda n: best_pts[n]['recall'] - best_pts[n]['fpr'])
    rows_bn  = sweep_per_n[best_n]
    taus     = [r['tau']   for r in rows_bn]
    recs     = [r['recall'] for r in rows_bn]
    fprs     = [r['fpr']   for r in rows_bn]

    fig, ax1 = plt.subplots()
    ax2 = ax1.twinx()
    ax1.plot(taus, recs, color='steelblue', lw=2.2, label='Recall')
    ax2.plot(taus, fprs, color='coral',     lw=2.2, label='FPR')
    ax1.axhline(TARGET_RECALL, color='steelblue', ls=':', alpha=0.5)
    ax2.axhline(TARGET_FPR,    color='coral',     ls=':', alpha=0.5)
    ax1.set_xlabel('Threshold τ', fontsize=13)
    ax1.set_ylabel('Recall',              color='steelblue', fontsize=13)
    ax2.set_ylabel('False Positive Rate', color='coral',     fontsize=13)
    ax1.set_title(f'Threshold Sensitivity — MLP N={best_n}\n'
                  '(Note: calibrated τ may not transfer across datasets)', fontsize=12)
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, fontsize=10)
    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_DIR, 'threshold_sensitivity.png'), dpi=150); plt.close()

    # Plot D — Probability distribution
    te_p_n1 = te_probs_per_n[1]
    fig, ax = plt.subplots()
    ax.hist(te_p_n1[te_y_all == 0], bins=80, alpha=0.55, color='steelblue',
            density=True, label='Benign')
    ax.hist(te_p_n1[te_y_all == 1], bins=80, alpha=0.55, color='coral',
            density=True, label='Attack')
    ax.set_xlabel('Predicted Probability (N=1, no smoothing)', fontsize=13)
    ax.set_ylabel('Density', fontsize=13)
    ax.set_title('Output Probability Distribution — Test Set\n'
                 '(Overlap region = irreducible FPR floor for this feature set)', fontsize=12)
    ax.legend(fontsize=11); plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_DIR, 'prob_distribution.png'), dpi=150); plt.close()

    # ── 12. Best-point confusion matrices per N
    for n in SMOOTHING_WINS:
        pt    = best_pts[n]
        preds = (te_probs_per_n[n] > pt['tau']).astype(int)
        cm    = confusion_matrix(te_y_all, preds)
        fig, ax = plt.subplots(figsize=(6, 5))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                    xticklabels=['Benign', 'Attack'],
                    yticklabels=['Benign', 'Attack'], ax=ax)
        ax.set_title(f'Confusion Matrix — MLP N={n}\n'
                     f'τ={pt["tau"]:.3f}  Recall={pt["recall"]:.4f}  FPR={pt["fpr"]:.4f}')
        plt.tight_layout()
        plt.savefig(os.path.join(RESULTS_DIR, f'cm_N{n}.png'), dpi=130); plt.close()

    print(f'\nAll outputs saved to  {RESULTS_DIR}')
    print('Done.\n')


if __name__ == '__main__':
    main()
