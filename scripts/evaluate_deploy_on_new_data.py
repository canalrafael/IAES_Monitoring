import os
import ctypes
import pandas as pd
import numpy as np
import glob
from sklearn.metrics import confusion_matrix, recall_score

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
    
    # Recompile to ensure parity
    print(f"--- Compiling detector.c ---")
    c_source = os.path.join(DEPLOY_DIR, "detector.c")
    # Using same flags as deploy/README or usual embedded flags
    cmd = f"gcc -O3 -fPIC -shared -o {so_path} {c_source}"
    os.system(cmd)
    
    if not os.path.exists(so_path):
        raise FileNotFoundError(f"Could not find or build {so_path}")
        
    lib = ctypes.CDLL(so_path)
    lib.detector_init.argtypes = []
    lib.detector_init.restype = None
    lib.detector_process_sample.argtypes = [ctypes.c_int, ctypes.POINTER(PMUSample)]
    lib.detector_process_sample.restype = DetOutput
    return lib

def run_online_validation():
    try:
        lib = setup_detector()
    except Exception as e:
        print(f"Error: {e}")
        return

    csv_files = sorted(glob.glob(os.path.join(DATA_DIR, 'data_new*_clean.csv')))
    
    all_summary = []

    for f_path in csv_files:
        f_name = os.path.basename(f_path)
        print(f"\nProcessing {f_name}...")
        df = pd.read_csv(f_path)
        
        # Reset detector for each file to ensure independent evaluation
        lib.detector_init()
        
        preds_status = []
        preds_prob = []
        
        # Mapping: the C model outputs 0=WARMUP, 1=BENIGN, 2=ATTACK
        for _, row in df.iterrows():
            sample = PMUSample()
            sample.cpu_cycles = int(row['CPU_CYCLES'])
            sample.instructions = int(row['INSTRUCTIONS'])
            sample.cache_misses = int(row['CACHE_MISSES'])
            sample.branch_misses = int(row['BRANCH_MISSES'])
            sample.l2_cache_access = int(row['L2_CACHE_ACCESS'])
            
            # Using CPU ID 0 for simplicity (single core detector logic)
            out = lib.detector_process_sample(0, ctypes.byref(sample))
            preds_status.append(out.status)
            preds_prob.append(out.probability)
            
        df['NEW_DET_STATUS'] = preds_status
        df['NEW_DET_PROB'] = preds_prob
        
        # Analysis: Label 2 is Interference (Positive), Label 3 is Attack (Positive too, but on other core)
        # However, typically we want to see if we detect interference on the benchmark core (Label 2)
        # Let's filter to Label 2 and see the detection rate
        l2_df = df[df['LABEL'] == 2]
        l3_df = df[df['LABEL'] == 3]
        
        # Warmup samples (Status 0) should be excluded from performance metrics
        valid_l2 = l2_df[l2_df['NEW_DET_STATUS'] != 0]
        valid_l3 = l3_df[l3_df['NEW_DET_STATUS'] != 0]
        
        recall_l2 = (valid_l2['NEW_DET_STATUS'] == 2).mean() if not valid_l2.empty else 0
        recall_l3 = (valid_l3['NEW_DET_STATUS'] == 2).mean() if not valid_l3.empty else 0
        
        # Check against existing DET_STATUS
        parity = (df['NEW_DET_STATUS'] == df['DET_STATUS']).mean()
        
        stats = {
            "File": f_name,
            "Samples": len(df),
            "Label 2 Count": len(l2_df),
            "Label 3 Count": len(l3_df),
            "Recall (L2 Interference)": f"{recall_l2:.2%}",
            "Recall (L3 Attack Core)": f"{recall_l3:.2%}",
            "Parity with Previous Run": f"{parity:.2%}"
        }
        all_summary.append(stats)
        print(stats)
        
        # Save per-file detailed results
        out_csv = os.path.join(RESULTS_DIR, f"results_{f_name}")
        df.to_csv(out_csv, index=False)

    summary_df = pd.DataFrame(all_summary)
    print("\n--- Online Validation Summary ---")
    print(summary_df)
    summary_df.to_csv(os.path.join(RESULTS_DIR, "validation_summary.csv"), index=False)

if __name__ == "__main__":
    run_online_validation()
