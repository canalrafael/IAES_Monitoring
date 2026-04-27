import os
import shutil
import pandas as pd
import subprocess
import numpy as np

VALID_DIR = "/home/canal/github/IAES_Monitoring/deploy/online_validation"
DEPLOY_H = "/home/canal/github/IAES_Monitoring/deploy/model_weights.h"
METRIC_SCRIPT = "/home/canal/github/IAES_Monitoring/scripts/per_benchmark_metrics.py"
RESULT_CSV = "/home/canal/github/IAES_Monitoring/results/online_validation/per_benchmark_metrics.csv"

# Auto-discover all weights in online_validation
VALIDATION_DIR = "deploy/online_validation"
variants = {}
for f in os.listdir(VALIDATION_DIR):
    if f.endswith(".h"):
        # Basic check: does it look like a weight file?
        with open(os.path.join(VALIDATION_DIR, f), 'r') as check_f:
            content = check_f.read()
            if "MDL_N_FEATURES" in content:
                name = f.replace(".h", "")
                variants[name] = f

print(f"Discovered variants: {list(variants.keys())}")

def run_eval(name, file_path):
    print(f"\n--- Testing Variant: {name} ---")
    shutil.copy(os.path.join(VALID_DIR, file_path), DEPLOY_H)
    
    # Run the existing metrics script
    subprocess.run(["python3", METRIC_SCRIPT], check=True)
    
    # Load results and split by Batch
    df = pd.read_csv(RESULT_CSV)
    
    df['Variant'] = name
    return df

def main():
    all_results = []
    for name, f in variants.items():
        all_results.append(run_eval(name, f))
    
    results_df = pd.concat(all_results, ignore_index=True)
    
    # Grouping logic: Try to group files logically or just show individual file performance
    # For now, let's group by "All Data" to avoid hardcoding batch names
    summary_data = []
    
    # Get unique variants and files
    unique_variants = results_df['Variant'].unique()
    
    for var in unique_variants:
        var_subset = results_df[results_df['Variant'] == var]
        
        def get_metric(bench, context, m_type):
            val = var_subset[(var_subset['Benchmark'] == bench) & 
                             (var_subset['Context'] == context) & 
                             (var_subset['Metric_Type'] == m_type)]['Value'].mean()
            return f"{val:.2%}" if not np.isnan(val) else "N/A"

        summary_data.append({
            "Variant": var,
            "Spectre": get_metric("Spectre", "Attack_Core", "Recall"),
            "Arma": get_metric("Armageddon", "Attack_Core", "Recall"),
            "Melt": get_metric("Meltdown", "Attack_Core", "Recall"),
            "Idle_FPR": get_metric("Idle", "Secure_Alone", "FPR"),
            "Band_FPR": get_metric("Bandwidth", "Secure_Alone", "FPR")
        })

    summary_df = pd.DataFrame(summary_data)
    print("\n--- FINAL COMPARISON SUMMARY (Averaged across all CSVs) ---")
    print(summary_df.to_string(index=False))

if __name__ == "__main__":
    main()
