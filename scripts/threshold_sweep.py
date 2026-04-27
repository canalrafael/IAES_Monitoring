import os
import shutil
import subprocess
import pandas as pd
import numpy as np

# --- Configuration ---
VALID_DIR = "/home/canal/github/IAES_Monitoring/deploy/online_validation"
DEPLOY_H = "/home/canal/github/IAES_Monitoring/deploy/model_weights.h"
METRIC_SCRIPT = "/home/canal/github/IAES_Monitoring/scripts/per_benchmark_metrics.py"
RESULT_CSV = "/home/canal/github/IAES_Monitoring/results/online_validation/per_benchmark_metrics.csv"

# Auto-discover all weights in online_validation
variants = {}
for f in os.listdir(VALID_DIR):
    if f.endswith(".h"):
        # Basic check: does it look like a weight file?
        with open(os.path.join(VALID_DIR, f), 'r') as check_f:
            content = check_f.read()
            if "MDL_N_FEATURES" in content:
                name = f.replace(".h", "")
                variants[name] = f

target_variants = list(variants.keys())
print(f"Sweeping all variants: {target_variants}")
thresholds = [0.1, 0.3, 0.5, 0.7, 0.8, 0.9, 0.95, 0.99]

def set_threshold(file_path, threshold):
    with open(file_path, 'r') as f:
        lines = f.readlines()
    
    new_lines = []
    for line in lines:
        if "#define MDL_THRESHOLD" in line:
            new_lines.append(f"#define MDL_THRESHOLD {threshold}f\n")
        else:
            new_lines.append(line)
            
    with open(DEPLOY_H, 'w') as f:
        f.writelines(new_lines)

def run_sweep():
    sweep_results = []
    
    for var in target_variants:
        print(f"\n>>> Sweeping Variant: {var}")
        source_path = os.path.join(VALID_DIR, f"{var}.h")
        
        for thresh in thresholds:
            print(f"  Testing Threshold: {thresh}...")
            set_threshold(source_path, thresh)
            
            # Run metrics
            subprocess.run(["python3", METRIC_SCRIPT], capture_output=True)
            
            # Extract key metrics from CSV
            df = pd.read_csv(RESULT_CSV)
            
            def get_m(bench, context, m_type):
                return df[(df['Benchmark'] == bench) & (df['Context'] == context) & (df['Metric_Type'] == m_type)]['Value'].mean()

            res = {
                "Variant": var,
                "Threshold": thresh,
                "Spectre": get_m("Spectre", "Attack_Core", "Recall"),
                "Arma": get_m("Armageddon", "Attack_Core", "Recall"),
                "Melt": get_m("Meltdown", "Attack_Core", "Recall"),
                "Interfered_L2": df[df['Context'] == 'Secure_Interfered']['Value'].mean()
            }
            
            # Add all other benchmarks
            for b_name in ["Idle", "Bandwidth", "Disparity", "FFT", "QSort", "Dijkstra", "SHA", "Sorting"]:
                res[f"{b_name}_FPR"] = get_m(b_name, "Secure_Alone", "FPR")
                
            sweep_results.append(res)

    results_df = pd.DataFrame(sweep_results)
    
    # Print formatted results
    for var in target_variants:
        print(f"\n--- Detailed Threshold Sweep Results (Workloads Only) for {var} ---")
        subset = results_df[results_df['Variant'] == var].copy()
        # Metric columns: Recall for L3 (Spectre/Arma/Melt), Recall for L2 (Interfered), and Noise for L0
        cols = ["Spectre", "Arma", "Melt", "Interfered_L2"]
        noise_cols = ["Bandwidth_FPR", "Dijkstra_FPR", "SHA_FPR", "Sorting_FPR"]
        
        for col in cols + noise_cols:
            subset[col] = subset[col].map(lambda x: f"{x:.1%}" if not np.isnan(x) else "0.0%")
        print(subset[["Threshold"] + cols + noise_cols].to_string(index=False))

if __name__ == "__main__":
    run_sweep()
