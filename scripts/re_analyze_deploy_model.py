import os
import ctypes
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, roc_curve, auc, precision_recall_curve, average_precision_score

# ── Configuration ─────────────────────────────────────────────────────────────
SCRIPT_DIR    = os.path.dirname(os.path.abspath(__file__))
WORKSPACE_DIR = os.path.dirname(SCRIPT_DIR)
DEPLOY_DIR    = os.path.join(WORKSPACE_DIR, 'deploy')
DATA_DIR      = os.path.join(WORKSPACE_DIR, 'data')
RESULTS_DIR   = os.path.join(WORKSPACE_DIR, 'results/deploy_model_analysis')
os.makedirs(RESULTS_DIR, exist_ok=True)

# ── 1. Compile C code as Shared Library ───────────────────────────────────────
print("[1/4] Compiling detector.c as shared library for binary-perfect evaluation...")
lib_path = os.path.join(DEPLOY_DIR, "libdetector.so")
# We do NOT define DETECTOR_PC_SIM to avoid the main() function conflict
compile_cmd = f"gcc -O3 -fPIC -shared -I{DEPLOY_DIR} -o {lib_path} {os.path.join(DEPLOY_DIR, 'detector.c')}"
if os.system(compile_cmd) != 0:
    print("Error: Compilation failed. Make sure gcc is installed.")
    exit(1)

# ── 2. Setup Ctypes Interface ─────────────────────────────────────────────────
class PMUSample(ctypes.Structure):
    _fields_ = [
        ("cpu_cycles", ctypes.c_uint64),
        ("instructions", ctypes.c_uint64),
        ("cache_misses", ctypes.c_uint64),
        ("branch_misses", ctypes.c_uint64),
        ("l2_cache_access", ctypes.c_uint64),
    ]

class DetOutput(ctypes.Structure):
    _fields_ = [
        ("status", ctypes.c_int),
        ("probability", ctypes.c_float),
    ]

try:
    lib = ctypes.CDLL(lib_path)
    lib.detector_init.argtypes = []
    lib.detector_process_sample.argtypes = [ctypes.c_int, ctypes.POINTER(PMUSample)]
    lib.detector_process_sample.restype = DetOutput
except Exception as e:
    print(f"Error loading library: {e}")
    exit(1)

# ── 3. Evaluation ─────────────────────────────────────────────────────────────
benign_files = ['data0_clean.csv', 'data1_clean.csv', 'data7_clean.csv', 'data10_clean.csv', 'data12_clean.csv', 'data15_clean.csv', 'data18_clean.csv', 'data19_clean.csv', 'data21_clean.csv', 'data23_clean.csv', 'data24_clean.csv']
attack_files = ['data3_clean.csv', 'data4_clean.csv', 'data5_clean.csv', 'data6_clean.csv', 'data13_clean.csv', 'data14_clean.csv', 'data16_clean.csv', 'data17_clean.csv', 'data20_clean.csv', 'data22_clean.csv', 'data25_clean.csv', 'data26_clean.csv', 'data27_clean.csv']

np.random.seed(42)
test_benign = list(np.random.choice(benign_files, size=int(len(benign_files)*0.3), replace=False))
test_attackArr = list(np.random.choice(attack_files, size=int(len(attack_files)*0.3), replace=False))
test_files = test_benign + test_attackArr

all_y_true, all_y_probs = [], []
case_studies = {}

print(f"[2/4] Evaluating {len(test_files)} files (C-runtime invocation)...")
for f_name in test_files:
    print(f"      Processing {f_name}...")
    lib.detector_init()
    df = pd.read_csv(os.path.join(DATA_DIR, f_name))
    # Filter to continuous core stream [0, 2] like the other evaluation files
    df = df[df['LABEL'].isin([0, 2])].copy()
    
    file_probs = []
    file_labels = []
    
    for _, row in df.iterrows():
        sample = PMUSample(
            int(row['CPU_CYCLES']), int(row['INSTRUCTIONS']),
            int(row['CACHE_MISSES']), int(row['BRANCH_MISSES']),
            int(row['L2_CACHE_ACCESS'])
        )
        res = lib.detector_process_sample(0, ctypes.byref(sample))
        
        if res.status != 0: # Skip DET_WARMUP (status=0)
            label = 1 if row['LABEL'] == 2 else 0
            file_probs.append(float(res.probability))
            file_labels.append(label)
            
    all_y_true.extend(file_labels)
    all_y_probs.extend(file_probs)
    
    # Store case study for one attack and one benign
    if f_name == test_attackArr[0] or f_name == test_benign[0]:
        case_studies[f_name] = {
            'probs': file_probs,
            'labels': file_labels,
            'type': 'Attack' if f_name in attack_files else 'Benign'
        }

y_true = np.array(all_y_true)
y_probs = np.array(all_y_probs)

# ── 4. Metrics & Plots ────────────────────────────────────────────────────────
print("[3/4] Calculating final metrics...")
tau = 0.5
y_pred = (y_probs >= tau).astype(int)

# Use labels=[0, 1] to avoid shape issues if one class is missing in predictions
cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
tn, fp, fn, tp = cm.ravel()

recall = tp / (tp + fn) if (tp + fn) > 0 else 0
fpr = fp / (fp + tn) if (fp + tn) > 0 else 0
f1 = 2 * tp / (2 * tp + fp + fn) if (2 * tp + fp + fn) > 0 else 0
fpr_vals, tpr_vals, _ = roc_curve(y_true, y_probs)
auc_score = auc(fpr_vals, tpr_vals)
auprc = average_precision_score(y_true, y_probs)

print(f"      ✓ Recall: {recall:.4f}")
print(f"      ✓ FPR:    {fpr:.4f}")
print(f"      ✓ F1:     {f1:.4f}")
print(f"      ✓ AUC:    {auc_score:.4f}")

# Save metrics
pd.DataFrame([{
    'Recall': recall, 'FPR': fpr, 'F1': f1, 'AUC': auc_score, 'AUPRC': auprc,
    'TP': tp, 'TN': tn, 'FP': fp, 'FN': fn
}]).to_csv(os.path.join(RESULTS_DIR, 'deploy_model_metrics.csv'), index=False)

print("[4/4] Generating publication-quality plots...")
sns.set_theme(style="whitegrid")
plt.rcParams.update({'font.family': 'serif', 'font.size': 12})

# 1. Confusion Matrix
plt.figure(figsize=(6, 5))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['Benign', 'Attack'], yticklabels=['Benign', 'Attack'])
plt.xlabel('Predicted Class')
plt.ylabel('Actual Class')
plt.title(f'Confusion Matrix (N=1, τ={tau})')
plt.tight_layout()
plt.savefig(os.path.join(RESULTS_DIR, 'confusion_matrix.png'), dpi=300)

# 2. ROC & PR Curves (Combined)
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

ax1.plot(fpr_vals, tpr_vals, color='steelblue', lw=2, label=f'ROC (AUC = {auc_score:.4f})')
ax1.plot([0, 1], [0, 1], color='gray', linestyle='--')
ax1.set_xlabel('False Positive Rate')
ax1.set_ylabel('True Positive Rate')
ax1.set_title('A. ROC Curve')
ax1.legend()

prec, rec, _ = precision_recall_curve(y_true, y_probs)
ax2.plot(rec, prec, color='steelblue', lw=2, label=f'PR (AUPRC = {auprc:.4f})')
ax2.set_xlabel('Recall')
ax2.set_ylabel('Precision')
ax2.set_title('B. Precision-Recall Curve')
ax2.legend()
plt.tight_layout()
plt.savefig(os.path.join(RESULTS_DIR, 'performance_curves.png'), dpi=300)

# 3. Case Studies
for f_name, data in case_studies.items():
    plt.figure(figsize=(12, 5))
    t = range(len(data['probs']))
    plt.plot(t, data['probs'], color='steelblue', lw=1.5, label='C-Model Probability')
    plt.axhline(y=tau, color='red', linestyle='--', label=f'Threshold τ={tau}')
    plt.fill_between(t, 0, data['labels'], color='gray', alpha=0.3, label='Ground Truth (Attack)')
    plt.ylim(-0.05, 1.05)
    plt.xlabel('Samples (Time)')
    plt.ylabel('Probability')
    plt.title(f'Temporal Detection Logic: {f_name} ({data["type"]})')
    plt.legend(loc='upper right')
    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_DIR, f'case_study_{data["type"].lower()}.png'), dpi=300)

print(f"\nSuccess! All reports and plots generated in {RESULTS_DIR}")
