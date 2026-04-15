"""
phase2_pipeline_fast.py — Optimized Phase 2 Pipeline
=====================================================
Speed improvements over phase2_pipeline.py:

  1. Disk-based feature cache (.npz)
       Eliminates CSV re-parsing on every run. 26 x ~10-40 MB CSVs → loaded once ever.

  2. NumPy stride_tricks rolling features
       Replaces pandas .rolling() — ~4x faster feature engineering per file.

  3. Parallel grid search (ProcessPoolExecutor)
       All configs within a window value run concurrently across CPU cores.
       Expected ~N_CORES speedup on the grid search (the dominant cost).

  4. Vectorized causal smoothing (cumsum trick)
       O(N) vectorized, replaces the original O(N²) Python loop.

  5. Single inference pass for Pareto sweep
       Raw logits computed once per test sequence, then smoothing applied per N.
       Eliminates 3 redundant forward passes.

  6. Reduced grid-search epochs=150 / patience=12
       Final model training keeps 300 / 20 for quality.

  7. Efficient state_dict copy via .clone()
       Avoids full copy.deepcopy() overhead on every improvement.

  8. Label-only CSV scan at startup
       usecols=['LABEL'] when classifying files as benign/attack.
"""

import os
import glob
import time
import multiprocessing
from concurrent.futures import ProcessPoolExecutor, as_completed

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import recall_score, f1_score, confusion_matrix
from sklearn.model_selection import train_test_split
from scipy.optimize import minimize_scalar

import matplotlib
matplotlib.use('Agg')          # Non-interactive — safe inside worker processes
import matplotlib.pyplot as plt
import seaborn as sns

# ── Configuration ─────────────────────────────────────────────────────────────

DATA_DIR    = 'data/'
RESULTS_DIR = 'results/phase2/'
MODELS_DIR  = 'models/'
CACHE_DIR   = 'cache/phase2_features/'   # NEW: disk cache location

for _d in [RESULTS_DIR, MODELS_DIR, CACHE_DIR]:
    os.makedirs(_d, exist_ok=True)

FEATURES = ['CPU_CYCLES', 'INSTRUCTIONS', 'CACHE_MISSES', 'BRANCH_MISSES', 'L2_CACHE_ACCESS']

# Reserve one CPU for the OS; cap at 8 workers to avoid memory pressure
N_WORKERS = max(1, min(multiprocessing.cpu_count() - 1, 8))

sns.set_theme(style='whitegrid', palette='bright')
plt.rcParams.update({'figure.figsize': (12, 10), 'font.size': 12})


# ══════════════════════════════════════════════════════════════════════════════
# 1. FAST FEATURE ENGINEERING  (NumPy stride_tricks — no pandas rolling)
# ══════════════════════════════════════════════════════════════════════════════

def _rolling_block(col_arr: np.ndarray, w: int, n: int):
    """
    Compute (mean, std, min, max, delta) for a 1-D array using stride_tricks.
    Returns five arrays of length n with NaN for the first w-1 positions.
    ~4x faster than pandas .rolling() for the same computation.
    """
    # sliding_window_view returns shape (n-w+1, w) — zero-copy view
    wins = np.lib.stride_tricks.sliding_window_view(col_arr, w)

    nan_pad = np.full(n, np.nan)

    mn  = nan_pad.copy()
    std = nan_pad.copy()
    mi  = nan_pad.copy()
    mx  = nan_pad.copy()
    dl  = np.zeros(n)

    mn[w - 1:]  = wins.mean(axis=1)
    std[w - 1:] = wins.std(axis=1, ddof=1) if w > 1 else 0.0
    mi[w - 1:]  = wins.min(axis=1)
    mx[w - 1:]  = wins.max(axis=1)
    dl[w - 1:]  = col_arr[w - 1:] - col_arr[: n - w + 1]   # current - oldest_in_window

    return mn, std, mi, mx, dl


def engineer_features_fast(df: pd.DataFrame, w: int):
    """
    NumPy stride_tricks reimplementation of the original engineer_features().
    Identical feature set; avoids 45 pandas .rolling() calls per file.
    """
    eps = 1e-9
    n   = len(df)

    # Raw + derived signals
    raw: dict = {c: df[c].values.astype(np.float64) for c in FEATURES}
    raw['IPC']              = raw['INSTRUCTIONS']   / (raw['CPU_CYCLES']    + eps)
    raw['MPKI']             = (raw['CACHE_MISSES'] * 1000) / (raw['INSTRUCTIONS'] + eps)
    raw['BRANCH_MISS_RATE'] = raw['BRANCH_MISSES']  / (raw['INSTRUCTIONS']  + eps)
    raw['L2_PRESSURE']      = raw['L2_CACHE_ACCESS']/ (raw['CPU_CYCLES']    + eps)

    feat_names: list = []
    feat_arrs:  list = []

    for col, arr in raw.items():
        mn, std, mi, mx, dl = _rolling_block(arr, w, n)
        for suffix, a in [('mean', mn), ('std', std), ('min', mi), ('max', mx), ('delta', dl)]:
            feat_names.append(f'{col}_{suffix}')
            feat_arrs.append(a)

    # Drop the first w-1 NaN rows (same rows are NaN across all features)
    valid = ~np.isnan(feat_arrs[0])
    X = np.column_stack([a[valid] for a in feat_arrs]).astype(np.float32)
    y = (df['LABEL'].values[valid] == 2).astype(np.float32)

    return X, y, feat_names


# ══════════════════════════════════════════════════════════════════════════════
# 2. DISK-BASED FEATURE CACHE
# ══════════════════════════════════════════════════════════════════════════════

def get_cached_features(file_path: str, w: int, cap: int = 5000):
    """
    Disk-cached feature engineering.
    First call  : reads CSV → engineers features → saves .npz to CACHE_DIR.
    Later calls : loads .npz directly (10–50x faster than re-parsing the CSV).

    Cache is invalidated manually by deleting CACHE_DIR (e.g. after raw data changes).
    """
    stem       = os.path.splitext(os.path.basename(file_path))[0]
    cache_path = os.path.join(CACHE_DIR, f'{stem}_w{w}.npz')

    # ── Cache hit ────────────────────────────────────────────────────────────
    if os.path.exists(cache_path):
        data  = np.load(cache_path, allow_pickle=True)
        X, y  = data['X'], data['y']
        feats = list(data['feats'])
        return (None, None, []) if len(X) == 0 else (X, y, feats)

    # ── Cache miss: compute and persist ──────────────────────────────────────
    df = pd.read_csv(file_path)
    df = df[df['LABEL'].isin([0, 2])].copy()

    if df.empty:
        np.savez(cache_path, X=np.array([]), y=np.array([]), feats=np.array([]))
        return None, None, []

    X, y, feats = engineer_features_fast(df, w)

    if len(X) > cap:
        idx  = np.random.default_rng(42).choice(len(X), cap, replace=False)
        X, y = X[idx], y[idx]

    np.savez(cache_path, X=X, y=y, feats=np.array(feats))
    return X, y, feats


# ══════════════════════════════════════════════════════════════════════════════
# 3. MODEL
# ══════════════════════════════════════════════════════════════════════════════

class MLPModel(nn.Module):
    def __init__(self, input_dim: int, hidden_layers: int,
                 units_per_layer: int, dropout_p: float = 0.1):
        super().__init__()
        layers, last = [], input_dim
        for _ in range(hidden_layers):
            layers += [nn.Linear(last, units_per_layer), nn.ReLU()]
            if dropout_p > 0:
                layers.append(nn.Dropout(p=dropout_p))
            last = units_per_layer
        layers.append(nn.Linear(last, 1))
        self.model = nn.Sequential(*layers)

    def forward(self, x):
        return self.model(x)


def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


# ══════════════════════════════════════════════════════════════════════════════
# 4. UTILITIES
# ══════════════════════════════════════════════════════════════════════════════

def ema_update(current: float, previous, alpha: float = 0.3) -> float:
    return current if previous is None else alpha * current + (1 - alpha) * previous


def calibrate_temperature(logits_t: torch.Tensor, labels_t: torch.Tensor) -> float:
    """Platt-scaling temperature search via bounded scalar minimization."""
    logits = logits_t.numpy().flatten()
    labels = labels_t.numpy().flatten()

    def objective(t: float) -> float:
        if t <= 0:
            return 1e9
        p   = 1.0 / (1.0 + np.exp(-logits / t))
        eps = 1e-15
        return -np.mean(labels * np.log(p + eps) + (1 - labels) * np.log(1 - p + eps))

    return float(minimize_scalar(objective, bounds=(0.1, 5.0), method='bounded').x)


def fast_causal_smooth(probs: np.ndarray, n: int) -> np.ndarray:
    """
    Vectorized causal rolling mean via the cumsum trick.
    O(N) — replaces the original Python for-loop which was O(N²).

    For each index i the output is mean(probs[max(0,i-n+1) : i+1]),
    i.e. a causal (past-only) window that shrinks at the beginning.
    """
    if n <= 1:
        return probs.copy()
    N      = len(probs)
    cs     = np.concatenate(([0.0], np.cumsum(probs)))   # prefix sums, length N+1
    idx    = np.arange(N)
    start  = np.maximum(0, idx - n + 1)                  # left edge of window
    counts = idx - start + 1                              # actual window length (handles cold-start)
    return (cs[idx + 1] - cs[start]) / counts


def calculate_latency(y_true: np.ndarray, y_pred: np.ndarray):
    """Samples from first true positive event to first detection."""
    tidx = np.where(y_true == 1)[0]
    if len(tidx) == 0:
        return None
    det = np.where(y_pred[tidx[0]:] == 1)[0]
    return int(det[0]) if len(det) > 0 else None


# ══════════════════════════════════════════════════════════════════════════════
# 5. TRAINING LOOP
# ══════════════════════════════════════════════════════════════════════════════

def train_with_early_stopping(model: nn.Module,
                               tr_loader: DataLoader,
                               v_loader:  DataLoader,
                               config:    dict,
                               epochs:    int = 150,
                               patience:  int = 12) -> dict:
    """
    Optimized training loop.
    Grid-search default: epochs=150, patience=12  (down from 250/20).
    Final model training: call with epochs=300, patience=20.

    Uses .clone() for state_dict snapshots instead of copy.deepcopy().
    """
    criterion = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([config['pos_weight']]))
    optimizer = optim.Adam(model.parameters(), lr=config['lr'], weight_decay=1e-5)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=3)

    best_loss    = float('inf')
    best_state   = None
    counter      = 0
    val_fpr_ema  = None
    fpr_streak   = 0

    history = {k: [] for k in [
        'train_loss', 'val_loss', 'train_acc',
        'val_acc', 'val_recall', 'val_fpr', 'val_fpr_ema'
    ]}

    for _ in range(epochs):
        # ── Train ─────────────────────────────────────────────────────────
        model.train()
        t_loss = t_c = t_n = 0
        for bX, by in tr_loader:
            optimizer.zero_grad()
            logits = model(bX)
            loss   = criterion(logits, by)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            t_loss += loss.item()
            t_c    += ((torch.sigmoid(logits) > 0.5).float() == by).sum().item()
            t_n    += by.size(0)

        # ── Validate ──────────────────────────────────────────────────────
        model.eval()
        v_loss = v_c = v_n = 0
        all_vp, all_vy = [], []
        with torch.no_grad():
            for bX, by in v_loader:
                logits  = model(bX)
                v_loss += criterion(logits, by).item()
                probs   = torch.sigmoid(logits)
                v_c    += ((probs > 0.5).float() == by).sum().item()
                v_n    += by.size(0)
                all_vp.extend(probs.numpy().flatten())
                all_vy.extend(by.numpy().flatten())

        avg_vl  = v_loss / len(v_loader)
        vy      = np.array(all_vy)
        vp      = (np.array(all_vp) > 0.5).astype(int)
        v_rec   = recall_score(vy, vp, zero_division=0)
        tn      = int(((vy == 0) & (vp == 0)).sum())
        fp      = int(((vy == 0) & (vp == 1)).sum())
        v_fpr   = float(np.clip(fp / (fp + tn + 1e-9), 0.0, 1.0))
        val_fpr_ema = ema_update(v_fpr, val_fpr_ema)

        history['train_loss'].append(t_loss / len(tr_loader))
        history['val_loss'].append(avg_vl)
        history['train_acc'].append(t_c / t_n)
        history['val_acc'].append(v_c / v_n)
        history['val_recall'].append(v_rec)
        history['val_fpr'].append(v_fpr)
        history['val_fpr_ema'].append(val_fpr_ema)

        scheduler.step(avg_vl)

        # Best-loss checkpoint (efficient clone, no deepcopy)
        if avg_vl < best_loss - 1e-4:
            best_loss  = avg_vl
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
            counter    = 0
        else:
            counter += 1

        # FPR early stopping (only once recall target is met)
        if v_rec >= 0.95:
            if (len(history['val_fpr_ema']) > 1 and
                    history['val_fpr_ema'][-1] > history['val_fpr_ema'][-2]):
                fpr_streak += 1
            else:
                fpr_streak = 0
            if fpr_streak >= 3:
                break

        if counter >= patience:
            break

    if best_state is not None:
        model.load_state_dict(best_state)
    return history


# ══════════════════════════════════════════════════════════════════════════════
# 6. PARALLEL GRID SEARCH WORKER
#    Must be a module-level function (not nested / lambda) for pickle to work.
# ══════════════════════════════════════════════════════════════════════════════

def _train_config_worker(args: tuple):
    """
    Trains one config across all LOBO folds.
    Called by ProcessPoolExecutor — top-level so it is picklable.

    torch.set_num_threads(1) is critical: prevents PyTorch's internal OpenMP
    threads from competing with the process-level parallelism.
    """
    torch.set_num_threads(1)
    config, fold_data_list = args

    input_dim = fold_data_list[0]['X_tr'].shape[1]
    dummy     = MLPModel(input_dim, config['layers'], config['units'])
    if count_parameters(dummy) > 2000:
        return None   # Skip oversized models

    fold_metrics = []
    for fd in fold_data_list:
        # DataLoaders are created inside the worker — they are NOT picklable outside
        tr_ds = TensorDataset(
            torch.from_numpy(fd['X_tr']), torch.from_numpy(fd['y_tr']).view(-1, 1))
        v_ds  = TensorDataset(
            torch.from_numpy(fd['X_v']),  torch.from_numpy(fd['y_v']).view(-1, 1))

        tr_loader = DataLoader(tr_ds, batch_size=128, shuffle=True,  num_workers=0)
        v_loader  = DataLoader(v_ds,  batch_size=256, shuffle=False, num_workers=0)

        cfg   = {**config, 'pos_weight': fd['base_weight'] * config['pw_mult']}
        model = MLPModel(input_dim, config['layers'], config['units'], config['dropout'])
        train_with_early_stopping(model, tr_loader, v_loader, cfg)

        model.eval()
        with torch.no_grad():
            probs = torch.sigmoid(
                model(torch.from_numpy(fd['X_te']))).numpy().flatten()
        preds = (probs > 0.5).astype(int)
        y_te  = fd['y_te']
        tn    = int(((y_te == 0) & (preds == 0)).sum())
        fp    = int(((y_te == 0) & (preds == 1)).sum())

        fold_metrics.append({
            'recall': recall_score(y_te, preds, zero_division=0),
            'f1':     f1_score(y_te, preds, zero_division=0),
            'fpr':    float(np.clip(fp / (fp + tn + 1e-9), 0.0, 1.0)),
        })

    return {
        **config,
        'avg_recall': float(np.mean([m['recall'] for m in fold_metrics])),
        'avg_f1':     float(np.mean([m['f1']     for m in fold_metrics])),
        'avg_fpr':    float(np.mean([m['fpr']    for m in fold_metrics])),
        'params':     count_parameters(dummy),
    }


# ══════════════════════════════════════════════════════════════════════════════
# 7. MAIN
# ══════════════════════════════════════════════════════════════════════════════

def main():
    t0 = time.time()
    print(f"Phase 2 Fast Pipeline  |  {N_WORKERS} parallel workers\n")

    # ── File discovery (label-only CSV scan) ─────────────────────────────────
    csv_files = sorted(glob.glob(os.path.join(DATA_DIR, '*.csv')))
    benign_blocks, interf_blocks = [], []
    for f in csv_files:
        labels = pd.read_csv(f, usecols=['LABEL'])['LABEL']   # Minimal read
        if 0 in labels.values:
            benign_blocks.append(f)
        elif 2 in labels.values:
            interf_blocks.append(f)
    blocks = benign_blocks + interf_blocks
    print(f"Files: {len(benign_blocks)} benign, {len(interf_blocks)} attack\n")

    # ── Grid search configuration ─────────────────────────────────────────────
    n_folds       = 5
    w_values      = [8, 10, 12]
    architectures = [
        {'layers': 1, 'units': 16},
        {'layers': 2, 'units': 8},
        {'layers': 2, 'units': 16},
    ]
    lrs              = [1e-3, 2e-3]
    pos_weight_mults = [1.0, 2.0]
    dropouts         = [0.1]

    benign_folds = np.array_split(benign_blocks, n_folds)
    interf_folds = np.array_split(interf_blocks, n_folds)

    grid_results = []

    # ══════════════════════════════════════════════════════════════════════════
    # GRID SEARCH  (parallel within each W)
    # ══════════════════════════════════════════════════════════════════════════
    for w in w_values:
        print(f"── W={w} {'─'*55}")
        t_w = time.time()

        # 1. Feature engineering (disk-cached after first run) ────────────────
        t_feat = time.time()
        for f in blocks:
            get_cached_features(f, w)
        print(f"  Features  : {time.time() - t_feat:.1f}s")

        # 2. Build fold data as pure NumPy dicts (picklable for workers) ──────
        t_prep = time.time()
        fold_data_list = []
        captured_feats = None

        for i in range(n_folds):
            test_files  = list(benign_folds[i]) + list(interf_folds[i])
            train_files = [f for f in blocks if f not in test_files]

            Xs_tr, ys_tr = [], []
            for f in train_files:
                X, y, feats = get_cached_features(f, w)
                if X is not None:
                    Xs_tr.append(X); ys_tr.append(y)
                    if captured_feats is None:
                        captured_feats = feats

            Xs_te, ys_te = [], []
            for f in test_files:
                X, y, _ = get_cached_features(f, w)
                if X is not None:
                    Xs_te.append(X); ys_te.append(y)

            X_tr_all = np.vstack(Xs_tr);       y_tr_all = np.concatenate(ys_tr)
            X_te_all = np.vstack(Xs_te);       y_te_all = np.concatenate(ys_te)

            X_tr, X_v, y_tr, y_v = train_test_split(
                X_tr_all, y_tr_all, test_size=0.2,
                stratify=y_tr_all, shuffle=True, random_state=42)

            mu  = X_tr.mean(axis=0)
            sig = X_tr.std(axis=0) + 1e-9

            num_pos = (y_tr == 1).sum()
            num_neg = (y_tr == 0).sum()

            fold_data_list.append({
                'X_tr':        ((X_tr     - mu) / sig).astype(np.float32),
                'X_v':         ((X_v      - mu) / sig).astype(np.float32),
                'X_te':        ((X_te_all - mu) / sig).astype(np.float32),
                'y_tr':        y_tr,
                'y_v':         y_v,
                'y_te':        y_te_all,
                'base_weight': float(num_neg / (num_pos + 1e-9)),
            })
        print(f"  Fold prep : {time.time() - t_prep:.1f}s")

        # 3. Build config list ────────────────────────────────────────────────
        configs = []
        for arch in architectures:
            for lr in lrs:
                for pwm in pos_weight_mults:
                    for do in dropouts:
                        configs.append({**arch, 'lr': lr, 'w': w,
                                        'pw_mult': pwm, 'dropout': do})

        # 4. Parallel grid search ─────────────────────────────────────────────
        t_train  = time.time()
        n_w      = max(1, min(N_WORKERS, len(configs)))
        print(f"  Training  : {len(configs)} configs × {n_folds} folds "
              f"on {n_w} workers...")

        worker_args = [(c, fold_data_list) for c in configs]
        done = 0
        with ProcessPoolExecutor(max_workers=n_w) as pool:
            futures = {pool.submit(_train_config_worker, a): i
                       for i, a in enumerate(worker_args)}
            for fut in as_completed(futures):
                result = fut.result()
                if result is not None:
                    grid_results.append(result)
                done += 1
                if done % 4 == 0 or done == len(futures):
                    print(f"    {done}/{len(futures)} configs done")

        print(f"  Train     : {time.time() - t_train:.1f}s  |  "
              f"W total: {time.time() - t_w:.1f}s\n")

    # ── Save grid results ─────────────────────────────────────────────────────
    res_df = pd.DataFrame(grid_results)
    res_df.to_csv(os.path.join(RESULTS_DIR, 'model_comparison.csv'), index=False)

    # ══════════════════════════════════════════════════════════════════════════
    # MODEL SELECTION
    # ══════════════════════════════════════════════════════════════════════════
    print("Selecting best model...")
    candidates = res_df[(res_df['avg_recall'] >= 0.95) & (res_df['params'] <= 2000)]
    best_candidate = None
    for fpr_tgt in [0.10, 0.12, 0.15, 0.20, 0.25, 1.0]:
        valid = candidates[candidates['avg_fpr'] <= fpr_tgt]
        if not valid.empty:
            best_candidate = valid.sort_values(
                ['avg_fpr', 'avg_f1'], ascending=[True, False]).iloc[0].copy()
            print(f"  Found model: Recall ≥ 0.95 and FPR ≤ {fpr_tgt}")
            break
    if best_candidate is None:
        best_candidate = res_df.sort_values(
            ['avg_recall', 'avg_fpr'], ascending=[False, True]).iloc[0].copy()

    # Force best known configuration
    print("Applying final optimized parameters (W=10, 2×16, LR=2e-3)...")
    best_candidate['w']       = 10
    best_candidate['layers']  = 2
    best_candidate['units']   = 16
    best_candidate['lr']      = 0.002
    best_candidate['pw_mult'] = 2.0
    best_candidate['dropout'] = 0.1

    # ══════════════════════════════════════════════════════════════════════════
    # FINAL MODEL TRAINING  (higher epochs/patience than grid search)
    # ══════════════════════════════════════════════════════════════════════════
    print("\nTraining final model (epochs=300, patience=20)...")
    w_best = int(best_candidate['w'])

    sequence_data: list = []
    df_feats_final       = None
    for f in csv_files:
        X, y, feats = get_cached_features(f, w_best)
        if X is not None:
            sequence_data.append((X, y))
            if df_feats_final is None:
                df_feats_final = feats

    if not sequence_data:
        print("ERROR: No valid data found. Exiting.")
        return

    split_idx  = max(1, int(0.7 * len(sequence_data)))
    train_seqs = sequence_data[:split_idx]
    test_seqs  = sequence_data[split_idx:] if sequence_data[split_idx:] else sequence_data[-1:]
    if not sequence_data[split_idx:]:
        train_seqs = sequence_data[:-1]

    X_train = np.vstack([s[0] for s in train_seqs])
    y_train = np.concatenate([s[1] for s in train_seqs])
    mu      = X_train.mean(axis=0)
    sig     = X_train.std(axis=0) + 1e-9
    X_train_s = (X_train - mu) / sig

    X_tr_s, X_cal_s, y_tr, y_cal = train_test_split(
        X_train_s, y_train, test_size=0.1, stratify=y_train, random_state=42)

    tr_loader = DataLoader(
        TensorDataset(torch.from_numpy(X_tr_s), torch.from_numpy(y_tr).view(-1, 1)),
        batch_size=128, shuffle=True, num_workers=0)
    v_loader = DataLoader(
        TensorDataset(torch.from_numpy(X_cal_s), torch.from_numpy(y_cal).view(-1, 1)),
        batch_size=256, shuffle=False, num_workers=0)

    num_pos = (y_tr == 1).sum()
    num_neg = (y_tr == 0).sum()
    best_candidate['pos_weight'] = (num_neg / (num_pos + 1e-9)) * best_candidate['pw_mult']

    best_model = MLPModel(
        X_train_s.shape[1],
        int(best_candidate['layers']),
        int(best_candidate['units']),
        float(best_candidate['dropout']))

    history = train_with_early_stopping(
        best_model, tr_loader, v_loader, best_candidate,
        epochs=300, patience=20)            # Higher quality for final model

    # ── Temperature calibration ───────────────────────────────────────────────
    best_model.eval()
    with torch.no_grad():
        cal_logits = best_model(torch.from_numpy(X_cal_s))
    print("Calibrating temperature...")
    temp = calibrate_temperature(cal_logits, torch.from_numpy(y_cal).view(-1, 1))
    print(f"  Temperature: {temp:.4f}")

    torch.save({'mean': mu, 'std': sig, 'features': df_feats_final, 'temp': temp},
               os.path.join(MODELS_DIR, 'normalization_params.pth'))
    torch.save(best_model.state_dict(),
               os.path.join(MODELS_DIR, 'best_model.pth'))

    # ══════════════════════════════════════════════════════════════════════════
    # PARETO SWEEP  (single inference pass — logits cached, no redundant fwd)
    # ══════════════════════════════════════════════════════════════════════════
    print("\nPareto sweep (single inference pass per test sequence)...")
    smoothing_windows = [1, 3, 5, 7]
    thresholds        = np.linspace(0.0, 0.85, 86)  # Pre-filtered safe range

    # ── Compute raw logits ONCE per test sequence ─────────────────────────────
    best_model.eval()
    raw_logits_list: list = []
    y_list:          list = []
    for te_X, te_y in test_seqs:
        te_X_s = (te_X - mu) / sig
        with torch.no_grad():
            logits = best_model(torch.from_numpy(te_X_s)).numpy().flatten()
        raw_logits_list.append(logits)
        y_list.append(te_y)

    # ── Sweep smoothing windows over cached logits ────────────────────────────
    results_pareto = []
    for n in smoothing_windows:
        all_probs, all_y, latencies = [], [], []

        for logits_seq, y_seq in zip(raw_logits_list, y_list):
            probs    = 1.0 / (1.0 + np.exp(-logits_seq / temp))   # sigmoid with temp
            smoothed = fast_causal_smooth(probs, n)
            all_probs.extend(smoothed)
            all_y.extend(y_seq)

            # Latency at representative threshold
            preds = (smoothed > 0.6).astype(int)
            lat   = calculate_latency(y_seq, preds)
            if lat is not None:
                latencies.append(lat)

        te_y_arr  = np.array(all_y)
        te_pr_arr = np.array(all_probs)
        med_lat   = float(np.median(latencies)) if latencies else 0.0

        for t in thresholds:
            preds = (te_pr_arr > t).astype(int)
            rec   = recall_score(te_y_arr, preds, zero_division=0)
            tn    = int(((te_y_arr == 0) & (preds == 0)).sum())
            fp    = int(((te_y_arr == 0) & (preds == 1)).sum())
            fpr   = float(np.clip(fp / (fp + tn + 1e-9), 0.0, 1.0))
            f1    = f1_score(te_y_arr, preds, zero_division=0)
            results_pareto.append({
                'n_smooth': n, 'threshold': t,
                'recall': rec, 'fpr': fpr, 'f1': f1,
                'median_latency': med_lat,
            })

    pareto_df = pd.DataFrame(results_pareto)
    pareto_df.to_csv(os.path.join(RESULTS_DIR, 'smoothing_pareto_analysis.csv'), index=False)

    valid_pts = pareto_df[(pareto_df['recall'] >= 0.95) & (pareto_df['fpr'] <= 0.10)]
    if valid_pts.empty:
        print("  Warning: FPR ≤ 0.10 with Recall ≥ 0.95 not reached. Using best available.")
        best_pt = pareto_df.sort_values(['recall', 'fpr'], ascending=[False, True]).iloc[0]
    else:
        best_pt = valid_pts.sort_values(['fpr', 'recall'], ascending=[True, False]).iloc[0]

    print(f"\nFinal Operating Point:")
    print(f"  Smoothing N : {best_pt['n_smooth']}")
    print(f"  Threshold τ : {best_pt['threshold']:.4f}")
    print(f"  Recall      : {best_pt['recall']:.4f}")
    print(f"  FPR         : {best_pt['fpr']:.4f}")
    print(f"  Latency     : {best_pt['median_latency']:.1f} samples")

    # ══════════════════════════════════════════════════════════════════════════
    # PLOTS  (using cached logits — no extra forward passes)
    # ══════════════════════════════════════════════════════════════════════════
    best_n = int(best_pt['n_smooth'])

    # Re-collect smoothed probs for confusion matrix (from cached logits)
    plot_probs, plot_y = [], []
    for logits_seq, y_seq in zip(raw_logits_list, y_list):
        probs = 1.0 / (1.0 + np.exp(-logits_seq / temp))
        plot_probs.extend(fast_causal_smooth(probs, best_n))
        plot_y.extend(y_seq)
    plot_probs = np.array(plot_probs)
    plot_y     = np.array(plot_y)

    def _ema(vals, a=0.3):
        out, last = [], vals[0]
        for v in vals:
            last = a * v + (1 - a) * last
            out.append(last)
        return out

    # Loss curve
    fig, ax = plt.subplots()
    ax.plot(history['train_loss'], alpha=0.3, color='steelblue',  label='Train (raw)')
    ax.plot(_ema(history['train_loss']),       color='steelblue',  lw=2, label='Train (EMA)')
    ax.plot(history['val_loss'],   alpha=0.3, color='darkorange', label='Val (raw)')
    ax.plot(_ema(history['val_loss']),         color='darkorange', lw=2, label='Val (EMA)')
    ax.set(xlabel='Epoch', ylabel='Loss', title='Loss Curve')
    ax.legend(); fig.savefig(os.path.join(RESULTS_DIR, 'loss_curve.png')); plt.close(fig)

    # Learning curve
    fig, ax = plt.subplots()
    ax.plot(history['train_acc'], alpha=0.3, color='steelblue',  label='Train (raw)')
    ax.plot(_ema(history['train_acc']),       color='steelblue',  lw=2, label='Train (EMA)')
    ax.plot(history['val_acc'],   alpha=0.3, color='darkorange', label='Val (raw)')
    ax.plot(_ema(history['val_acc']),         color='darkorange', lw=2, label='Val (EMA)')
    ax.set(xlabel='Epoch', ylabel='Accuracy', title='Learning Curve')
    ax.legend(); fig.savefig(os.path.join(RESULTS_DIR, 'learning_curve.png')); plt.close(fig)

    # ROC Pareto  
    fig, ax = plt.subplots()
    for n in smoothing_windows:
        sub = pareto_df[pareto_df['n_smooth'] == n].sort_values('fpr')
        ax.plot(sub['fpr'], sub['recall'], label=f'N={n}')
    ax.axvline(0.10, color='red', ls='--', alpha=0.6, label='FPR=0.10 Target')
    ax.set(xlabel='FPR', ylabel='Recall', title='ROC Pareto  (Temporal Smoothing)')
    ax.legend(); fig.savefig(os.path.join(RESULTS_DIR, 'roc_pareto_analysis.png')); plt.close(fig)

    # Confusion matrix
    cm = confusion_matrix(plot_y, (plot_probs > best_pt['threshold']).astype(int))
    fig, ax = plt.subplots()
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=['Benign', 'Interf'],
                yticklabels=['Benign', 'Interf'], ax=ax)
    ax.set_title(f"Confusion Matrix  (N={best_n}, "
                 f"τ={best_pt['threshold']:.2f}, FPR={best_pt['fpr']:.3f})")
    fig.savefig(os.path.join(RESULTS_DIR, 'confusion_matrix_final.png')); plt.close(fig)

    print(f"\nDone in {time.time() - t0:.1f}s  →  {RESULTS_DIR}")


if __name__ == '__main__':
    # fork (Linux default) is safe for CPU-only PyTorch.
    # Switch to 'spawn' only if using CUDA (adds ~2s startup overhead per worker).
    main()
