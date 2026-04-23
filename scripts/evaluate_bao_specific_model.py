import os
import ctypes
import pandas as pd
import numpy as np
import glob
from sklearn.metrics import recall_score, confusion_matrix

# --- Configuration ---
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
BASE_DIR = os.path.dirname(SCRIPT_DIR)
# POINT TO THE NEW BAO-SPECIFIC FOLDER
DEPLOY_DIR = os.path.join(BASE_DIR, "deploy", "online_validation")
DATA_DIR = os.path.join(BASE_DIR, "data", "online validation data")
RESULTS_DIR = os.path.join(BASE_DIR, "results", "online_validation_recalibrated")
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
    # Using same compilation as before but in the new folder
    cmd = f"gcc -O3 -fPIC -shared -o {so_path} {c_source}"
    os.system(cmd)
    
    lib = ctypes.CDLL(so_path)
    lib.detector_init.argtypes = []
    lib.detector_init.restype = None
    lib.detector_process_sample.argtypes = [ctypes.c_int, ctypes.POINTER(PMUSample)]
    lib.detector_process_sample.restype = DetOutput
    return lib

def run_final_bao_validation():
    try:
        lib = setup_detector()
    except Exception as e:
        print(f"Error: {e}")
        return

    csv_files = sorted(glob.glob(os.path.join(DATA_DIR, 'data_new*_clean.csv')))
    all_summary = []

    for f_path in csv_files:
        f_name = os.path.basename(f_path)
        print(f"Evaluating {f_name} with Bao-Specific Model...")
        df = pd.read_csv(f_path)
        lib.detector_init()
        
        preds_status = []
        for i, row in df.iterrows():
            cpu_id = i % 3
            sample = PMUSample(
                int(row['CPU_CYCLES']),
                int(row['INSTRUCTIONS']),
                int(row['CACHE_MISSES']),
                int(row['BRANCH_MISSES']),
                int(row['L2_CACHE_ACCESS'])
            )
            out = lib.detector_process_sample(cpu_id, ctypes.byref(sample))
            preds_status.append(out.status)
            
        df['DET_STATUS_BAO'] = preds_status
        df['TS_ID'] = df.index // 3
        
        # System-Level metrics
        sys_gt = df.groupby('TS_ID')['LABEL'].apply(lambda x: any(v in [1, 2, 3] for v in x)).astype(int)
        sys_pred = df.groupby('TS_ID')['DET_STATUS_BAO'].apply(lambda x: any(v == 2 for v in x)).astype(int)
        sys_warmup = df.groupby('TS_ID')['DET_STATUS_BAO'].apply(lambda x: all(v == 0 for v in x))
        
        mask = ~sys_warmup
        y_true, y_pred = sys_gt[mask], sys_pred[mask]
        
        if len(y_true) > 0:
            rec = recall_score(y_true, y_pred, zero_division=0)
            tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
            fpr = fp / (fp + tn) if (fp + tn) > 0 else 0
        else:
            rec, fpr = 0, 0

        stats = {
            "File": f_name,
            "Recall": f"{rec:.2%}",
            "FPR": f"{fpr:.2%}",
            "Samples": len(df)
        }
        all_summary.append(stats)
        print(stats)

    summary_df = pd.DataFrame(all_summary)
    print("\n--- Final Bao-Specific Model Summary ---")
    print(summary_df)
    summary_df.to_csv(os.path.join(RESULTS_DIR, "final_validation_summary.csv"), index=False)

if __name__ == "__main__":
    run_final_bao_validation()
