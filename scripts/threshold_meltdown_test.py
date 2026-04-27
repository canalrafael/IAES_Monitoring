import os
import ctypes
import pandas as pd
import numpy as np

BASE_DIR = "/home/canal/github/IAES_Monitoring"
DEPLOY_DIR = os.path.join(BASE_DIR, "deploy")
VALID_DIR = os.path.join(BASE_DIR, "data", "online validation data")
GOLDEN_H = os.path.join(DEPLOY_DIR, "online_validation", "model_weights_golden.h")
TARGET_H = os.path.join(DEPLOY_DIR, "model_weights.h")

test_files = ["data_new28_clean.csv", "data_new29_clean.csv", "data_new30_clean.csv"]

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

def evaluate_threshold(thresh):
    with open(GOLDEN_H, 'r') as f:
        content = f.read()
    
    import re
    if "#define MDL_THRESHOLD" in content:
        content = re.sub(r"#define MDL_THRESHOLD\s+[\d\.]+f", f"#define MDL_THRESHOLD {thresh}f", content)
    else:
        content += f"\n#define MDL_THRESHOLD {thresh}f\n"
        
    with open(TARGET_H, 'w') as f:
        f.write(content)
        
    so_path = os.path.join(DEPLOY_DIR, "libdetector.so")
    c_source = os.path.join(DEPLOY_DIR, "detector.c")
    os.system(f"gcc -O3 -fPIC -shared -o {so_path} {c_source}")
    
    lib = ctypes.CDLL(so_path)
    lib.detector_init.argtypes = []
    lib.detector_init.restype = None
    lib.detector_process_sample.argtypes = [ctypes.c_int, ctypes.POINTER(PMUSample)]
    lib.detector_process_sample.restype = DetOutput
    
    # Store aggregated metrics
    total_metrics = []
    
    for f_name in test_files:
        df = pd.read_csv(os.path.join(VALID_DIR, f_name))
        preds = []
        
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
            preds.append(out.status)
            
        df['DET_STATUS'] = preds
        df_valid = df[df['DET_STATUS'] != 0].copy()
        
        def get_recall(bench, label):
            sub = df_valid[(df_valid['BENCH_ID'] == bench) & (df_valid['LABEL'] == label)]
            return (sub['DET_STATUS'] == 2).sum() / len(sub) if len(sub) > 0 else 0.0

        spec = get_recall(10, 3)
        arma = get_recall(11, 3)
        melt = get_recall(12, 3)
        
        fpr_df = df_valid[df_valid['LABEL'] == 0]
        fpr = (fpr_df['DET_STATUS'] == 2).sum() / len(fpr_df) if len(fpr_df) > 0 else 0.0
        
        total_metrics.append([spec, arma, melt, fpr])
        
    # Average across files
    avg_metrics = np.mean(total_metrics, axis=0)
    return avg_metrics[0], avg_metrics[1], avg_metrics[2], avg_metrics[3]

thresholds = [0.5, 0.9, 0.999]
print(f"{'Threshold':<10} | {'Spectre':<8} | {'Arma':<8} | {'Meltdown':<8} | {'Avg FPR':<10}")
print("-" * 60)

for t in thresholds:
    spec, arma, melt, fpr = evaluate_threshold(t)
    print(f"{t:<10.3f} | {spec:<8.2%} | {arma:<8.2%} | {melt:<8.2%} | {fpr:<10.2%}")
