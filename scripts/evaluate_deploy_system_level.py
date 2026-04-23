import os
import ctypes
import pandas as pd
import numpy as np
import glob
from sklearn.metrics import recall_score, confusion_matrix

# --- Configuration ---
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
BASE_DIR = os.path.dirname(SCRIPT_DIR)
DEPLOY_DIR = os.path.join(BASE_DIR, "deploy")
DATA_DIR = os.path.join(BASE_DIR, "data", "online validation data")
RESULTS_DIR = os.path.join(BASE_DIR, "results", "online_validation")
os.makedirs(RESULTS_DIR, exist_ok=True)

# --- C Interface ---
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

def setup_detector():
    so_path = os.path.join(DEPLOY_DIR, "libdetector.so")
    c_source = os.path.join(DEPLOY_DIR, "detector.c")
    cmd = f"gcc -O3 -fPIC -shared -o {so_path} {c_source}"
    os.system(cmd)
    
    lib = ctypes.CDLL(so_path)
    lib.detector_init.argtypes = []
    lib.detector_init.restype = None
    lib.detector_process_sample.argtypes = [ctypes.c_int, ctypes.POINTER(PMUSample)]
    lib.detector_process_sample.restype = DetOutput
    return lib

def run_system_level_validation():
    try:
        lib = setup_detector()
    except Exception as e:
        print(f"Error: {e}")
        return

    csv_files = sorted(glob.glob(os.path.join(DATA_DIR, 'data_new*_clean.csv')))
    all_summary = []

    for f_path in csv_files:
        f_name = os.path.basename(f_path)
        print(f"\nProcessing {f_name} (System-Level)...")
        df = pd.read_csv(f_path)
        
        # We need to simulate the detector state for each core independently
        # detector_process_sample(cpu_id, ...) handles this internally if implemented correctly
        lib.detector_init()
        
        # To simulate correctly, we need to know which core is which. 
        # Since the CSV doesn't have a CORE_ID column, but we know there are 3 instances per timestamp,
        # we can assign IDs 0, 1, 2 based on the order within the timestamp group.
        
        preds_status = []
        
        # Process in order to maintain temporal state in the C model
        for i, row in df.iterrows():
            cpu_id = i % 3 # Assuming rows are ordered: T1_C0, T1_C1, T1_C2, T2_C0, ...
            sample = PMUSample(
                int(row['CPU_CYCLES']),
                int(row['INSTRUCTIONS']),
                int(row['CACHE_MISSES']),
                int(row['BRANCH_MISSES']),
                int(row['L2_CACHE_ACCESS'])
            )
            out = lib.detector_process_sample(cpu_id, ctypes.byref(sample))
            preds_status.append(out.status)
            
        df['NEW_DET_STATUS'] = preds_status
        
        # --- System-Level Logic ---
        # Group by index floor (i // 3) to represent the same timestamp
        df['TS_ID'] = df.index // 3
        
        # System Ground Truth: 1 if ANY core is 1, 2, or 3
        sys_gt = df.groupby('TS_ID')['LABEL'].apply(lambda x: any(v in [1, 2, 3] for v in x)).astype(int)
        
        # System Prediction: 1 if ANY core is Status 2 (Attack)
        # Note: We ignore Status 0 (Warmup) - if any core says Attack, system says Attack.
        sys_pred = df.groupby('TS_ID')['NEW_DET_STATUS'].apply(lambda x: any(v == 2 for v in x)).astype(int)
        
        # Also check if warmup is active for the whole system
        sys_warmup = df.groupby('TS_ID')['NEW_DET_STATUS'].apply(lambda x: all(v == 0 for v in x))
        
        # Filter out warmup periods for metric calculation
        mask = ~sys_warmup
        y_true = sys_gt[mask]
        y_pred = sys_pred[mask]
        
        if len(y_true) > 0:
            rec = recall_score(y_true, y_pred, zero_division=0)
            cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
            tn, fp, fn, tp = cm.ravel()
            fpr = fp / (fp + tn) if (fp + tn) > 0 else 0
        else:
            rec, fpr = 0, 0

        stats = {
            "File": f_name,
            "Total Timestamps": len(sys_gt),
            "Attack States": sys_gt.sum(),
            "System Recall": f"{rec:.2%}",
            "System FPR": f"{fpr:.2%}",
        }
        all_summary.append(stats)
        print(stats)

    summary_df = pd.DataFrame(all_summary)
    print("\n--- System-Level Online Validation Summary ---")
    print(summary_df)
    summary_df.to_csv(os.path.join(RESULTS_DIR, "system_validation_summary.csv"), index=False)

if __name__ == "__main__":
    run_system_level_validation()
