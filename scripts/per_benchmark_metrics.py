import os
import ctypes
import pandas as pd
import numpy as np
import glob

# --- Configuration ---
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
BASE_DIR = os.path.dirname(SCRIPT_DIR)
DEPLOY_DIR = os.path.join(BASE_DIR, "deploy")
DATA_DIR = os.path.join(BASE_DIR, "data", "online validation data")
RESULTS_DIR = os.path.join(BASE_DIR, "results", "online_validation")
os.makedirs(RESULTS_DIR, exist_ok=True)

BENCH_NAMES = {
    0: "Idle", 1: "Bandwidth", 2: "Disparity", 3: "FFT", 4: "QSort", 
    5: "Dijkstra", 6: "SHA", 7: "Sorting", 10: "Spectre", 11: "Armageddon", 12: "Meltdown"
}

LABEL_MAP = {
    0: "Secure_Alone", 1: "Untrusted_Alone", 2: "Secure_Interfered", 3: "Attack_Core"
}

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

def run_per_benchmark_analysis():
    try:
        lib = setup_detector()
    except Exception as e:
        print(f"Error: {e}")
        return

    csv_files = sorted(glob.glob(os.path.join(DATA_DIR, 'data_new*_clean.csv')))
    all_results = []

    for f_path in csv_files:
        f_name = os.path.basename(f_path)
        print(f"Analyzing {f_name} per benchmark...")
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
            
        df['DET_STATUS_NEW'] = preds_status
        
        # Filter out Warmup samples (Status 0) for metric calculation
        df_valid = df[df['DET_STATUS_NEW'] != 0].copy()
        
        # Group by Bench_ID and Label to see metrics in each context
        groups = df_valid.groupby(['BENCH_ID', 'LABEL'])
        
        for (bench_id, label), group in groups:
            is_attack_label = label in [1, 2, 3]
            
            samples = len(group)
            detections = (group['DET_STATUS_NEW'] == 2).sum()
            
            metric_val = detections / samples if samples > 0 else 0
            metric_name = "Recall" if is_attack_label else "FPR"
            
            all_results.append({
                "File": f_name,
                "Bench_ID": bench_id,
                "Benchmark": BENCH_NAMES.get(bench_id, f"ID_{bench_id}"),
                "Label": label,
                "Context": LABEL_MAP.get(label, f"L_{label}"),
                "Samples": samples,
                "Metric_Type": metric_name,
                "Value": metric_val
            })

    results_df = pd.DataFrame(all_results)
    
    # Pivot for a better view
    print("\n--- Per-Benchmark Performance Metrics ---")
    
    # Format value as percentage
    results_df['Performance (%)'] = results_df['Value'].map(lambda x: f"{x:.2%}")
    
    # Final table display
    display_cols = ["File", "Benchmark", "Context", "Samples", "Metric_Type", "Performance (%)"]
    print(results_df[display_cols].to_string(index=False))
    
    results_df.to_csv(os.path.join(RESULTS_DIR, "per_benchmark_metrics.csv"), index=False)
    print(f"\nDetailed report saved to: {RESULTS_DIR}/per_benchmark_metrics.csv")

if __name__ == "__main__":
    run_per_benchmark_analysis()
