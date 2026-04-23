import os
import ctypes
import pandas as pd
import numpy as np
from io import StringIO
from sklearn.metrics import confusion_matrix, accuracy_score

# --- Paths ---
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
BASE_DIR = os.path.dirname(SCRIPT_DIR)
DEPLOY_DIR = os.path.join(BASE_DIR, "deploy")
DATA_DIR = os.path.join(BASE_DIR, "data", "data_test", "data_detector")
NEW_FILES = ["data_detector3.txt", "data_detector4.txt"]

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
    
    # Recompile if necessary (to match architecture)
    print(f"--- Compiling detector.c ---")
    c_source = os.path.join(DEPLOY_DIR, "detector.c")
    cmd = f"gcc -O3 -fPIC -shared -o {so_path} {c_source}"
    ret = os.system(cmd)
    if ret != 0:
        print("Warning: Compilation failed. Attempting to load existing .so")
        
    if not os.path.exists(so_path):
        raise FileNotFoundError(f"Could not find or build {so_path}")
        
    lib = ctypes.CDLL(so_path)
    
    # void detector_init(void)
    lib.detector_init.argtypes = []
    lib.detector_init.restype = None
    
    # det_output_t detector_process_sample(cpuid_t cpu_id, const pmu_sample_t *sample)
    lib.detector_process_sample.argtypes = [ctypes.c_int, ctypes.POINTER(PMUSample)]
    lib.detector_process_sample.restype = DetOutput
    
    return lib

# --- Parsing ---
def parse_detector_file(filepath):
    print(f"Parsing {os.path.basename(filepath)}...")
    with open(filepath, 'r') as f:
        lines = f.readlines()
    
    pmu_lines = []
    current_section = None
    # Auto-detect start
    if lines and "CORE_ID" in lines[0]:
        current_section = "PMU"
        
    for line in lines:
        line = line.strip()
        if not line: continue
        if line == "PMU_START": current_section = "PMU"; continue
        elif line == "PMU_END": current_section = None; continue
        elif line == "DET_START": current_section = "DET"; continue
        elif line == "DET_END": current_section = None; continue
        
        if current_section == "PMU":
            pmu_lines.append(line)
            
    pmu_content = [pmu_lines[0]] + [l for l in pmu_lines[1:] if "CORE_ID" not in l]
    df = pd.read_csv(StringIO("\n".join(pmu_content)))
    return df

# --- Execution ---
def run_evaluation():
    try:
        lib = setup_detector()
    except Exception as e:
        print(f"Error setting up C detector: {e}")
        return

    results_all = []

    for f_name in NEW_FILES:
        f_path = os.path.join(DATA_DIR, f_name)
        if not os.path.exists(f_path):
            print(f"File not found: {f_path}")
            continue
            
        df = parse_detector_file(f_path)
        
        lib.detector_init()
        
        preds = []
        probs = []
        
        print(f"Running C inference on {len(df)} samples...")
        for _, row in df.iterrows():
            sample = PMUSample()
            sample.cpu_cycles = int(row['CPU_CYCLES'])
            sample.instructions = int(row['INSTRUCTIONS'])
            sample.cache_misses = int(row['CACHE_MISSES'])
            sample.branch_misses = int(row['BRANCH_MISSES'])
            sample.l2_cache_access = int(row['L2_CACHE_ACCESS'])
            
            out = lib.detector_process_sample(0, ctypes.byref(sample))
            
            # 0=WARMUP, 1=BENIGN, 2=ATTACK
            preds.append(out.status)
            probs.append(out.probability)
            
        df['C_PRED'] = preds
        df['C_PROB'] = probs
        
        # Metrics (Treat 0=WARMUP, 1=BENIGN as Negative (0), 2=ATTACK as Positive (1))
        y_true = df['LABEL'].map({0: 0, 2: 1})
        y_pred = df['C_PRED'].map({0: 0, 1: 0, 2: 1})
        
        cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
        tn, fp, fn, tp = cm.ravel()
        
        stats = {
            "File": f_name,
            "Total": len(df),
            "Recall": tp / (tp + fn) if (tp + fn) > 0 else 0,
            "FPR": fp / (fp + tn) if (fp + tn) > 0 else 0,
            "TP": tp, "FP": fp, "TN": tn, "FN": fn
        }
        results_all.append(stats)
        print(f"Results for {f_name}: {stats}")
        
    final_df = pd.DataFrame(results_all)
    print("\n--- Final C-Model Performance Summary ---")
    print(final_df)
    
    report_path = os.path.join(BASE_DIR, "results", "c_model_reclassification_summary.csv")
    final_df.to_csv(report_path, index=False)
    print(f"Report saved to {report_path}")

if __name__ == "__main__":
    run_evaluation()
