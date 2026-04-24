import pandas as pd
import numpy as np
import os

DATA_DIR = "/home/canal/github/IAES_Monitoring/data/online validation data/"
FILES = ["data_new19_clean.csv", "data_new21_clean.csv"]

def compare_datasets():
    eps = 1e-9
    results = []
    
    for f in FILES:
        df = pd.read_csv(os.path.join(DATA_DIR, f))
        # Filter for Core 3 (Attacker) and Meltdown (12)
        sub = df[(df.CORE_ID == 3) & (df.BENCH_ID == 12)].copy()
        
        if len(sub) == 0:
            print(f"No Spectre samples in {f}")
            continue
            
        sub['IPC'] = sub['INSTRUCTIONS'] / (sub['CPU_CYCLES'] + eps)
        sub['MPKI'] = (sub['CACHE_MISSES'] * 1000) / (sub['INSTRUCTIONS'] + eps)
        
        stats = sub[['IPC', 'MPKI', 'CPU_CYCLES']].describe().loc[['mean', 'std']]
        stats['file'] = f
        results.append(stats)
        
    print("\nComparison of Spectre (Bench 10) on Core 3:")
    for r in results:
        print(f"\n--- {r['file'].iloc[0]} ---")
        print(r.drop(columns='file'))

if __name__ == "__main__":
    compare_datasets()
