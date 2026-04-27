import os
import ctypes
import pandas as pd
import numpy as np
import glob
import re

# --- Configuration ---
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
BASE_DIR = os.path.dirname(SCRIPT_DIR)
DEPLOY_DIR = os.path.join(BASE_DIR, "deploy")
DATA_DIR = os.path.join(BASE_DIR, "data", "online validation data")
WEIGHTS_H = os.path.join(DEPLOY_DIR, "model_weights.h")

THRESHOLDS = [0.05, 0.50, 0.90]

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
    # Compile with the current model_weights.h
    cmd = f"gcc -O3 -fPIC -shared -o {so_path} {c_source}"
    os.system(cmd)
    
    lib = ctypes.CDLL(so_path)
    lib.detector_init.argtypes = []
    lib.detector_init.restype = None
    lib.detector_process_sample.argtypes = [ctypes.c_int, ctypes.POINTER(PMUSample)]
    lib.detector_process_sample.restype = DetOutput
    return lib

def set_threshold_in_header(threshold):
    with open(WEIGHTS_H, 'r') as f:
        lines = f.readlines()
    with open(WEIGHTS_H, 'w') as f:
        for line in lines:
            if "#define MDL_THRESHOLD" in line:
                f.write(f"#define MDL_THRESHOLD {threshold}f\n")
            else:
                f.write(line)

def run_evaluation(lib):
    csv_files = [f for f in sorted(glob.glob(os.path.join(DATA_DIR, '*.csv')))
                 if any(x in os.path.basename(f) for x in ["40", "41", "42", "43"])]
    
    # We'll use a representative subset if there are too many, 
    # but let's try all "clean" files first.
    
    total_tp = 0
    total_fn = 0
    total_fp = 0
    total_tn = 0
    
    for f_path in csv_files:
        df = pd.read_csv(f_path)
        if "LABEL" not in df.columns:
            continue
            
        lib.detector_init()
        
        for i, row in df.iterrows():
            cpu_id = int(row['CORE_ID']) % 4
            sample = PMUSample(
                int(row['CPU_CYCLES']),
                int(row['INSTRUCTIONS']),
                int(row['CACHE_MISSES']),
                int(row['BRANCH_MISSES']),
                int(row['L2_CACHE_ACCESS'])
            )
            out = lib.detector_process_sample(cpu_id, ctypes.byref(sample))
            
            if out.status == 0: # Warmup
                continue
            
            label = int(row['LABEL'])
            is_attack = label in [1, 2, 3] # Interference or Attack
            is_detected = (out.status == 2)
            
            if is_attack:
                if is_detected: total_tp += 1
                else: total_fn += 1
            else: # Benign (Label 0)
                if is_detected: total_fp += 1
                else: total_tn += 1
                
    recall = total_tp / (total_tp + total_fn) if (total_tp + total_fn) > 0 else 0
    fpr = total_fp / (total_fp + total_tn) if (total_fp + total_tn) > 0 else 0
    precision = total_tp / (total_tp + total_fp) if (total_tp + total_fp) > 0 else 0
    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
    
    return recall, fpr, f1

def main():
    # Save original threshold
    with open(WEIGHTS_H, 'r') as f:
        original_content = f.read()
    
    print(f"{'Configuration':<20} | {'tau':<6} | {'Recall':<10} | {'FPR':<10} | {'F1-Score':<10}")
    print("-" * 65)
    
    names = ["High-Sensitivity", "Operational", "Robust"]
    
    try:
        for name, tau in zip(names, THRESHOLDS):
            set_threshold_in_header(tau)
            lib = setup_detector()
            recall, fpr, f1 = run_evaluation(lib)
            print(f"{name:<20} | {tau:<6.2f} | {recall:<10.4f} | {fpr:<10.4f} | {f1:<10.4f}")
    finally:
        # Restore original weights
        with open(WEIGHTS_H, 'w') as f:
            f.write(original_content)

if __name__ == "__main__":
    main()
