import pandas as pd
import numpy as np
import os
import glob

DATA_DIR = "/home/canal/github/IAES_Monitoring/data/online validation data"

def analyze_batch(batch_files):
    dfs = []
    for f in batch_files:
        dfs.append(pd.read_csv(f))
    df = pd.concat(dfs)
    
    # Calculate derived features used by the model
    # Note: These are rough estimates for comparison
    df['IPC'] = df['INSTRUCTIONS'] / df['CPU_CYCLES'].replace(0, 1)
    df['MPKI'] = (df['CACHE_MISSES'] * 1000) / df['INSTRUCTIONS'].replace(0, 1)
    df['L2_PRESSURE'] = df['L2_CACHE_ACCESS'] / df['INSTRUCTIONS'].replace(0, 1)
    df['BRANCH_MISS_RATE'] = df['BRANCH_MISSES'] / df['INSTRUCTIONS'].replace(0, 1)
    
    stats = df[['IPC', 'MPKI', 'L2_PRESSURE', 'BRANCH_MISS_RATE']].describe().loc[['mean', 'std', 'max']]
    return stats, df['LABEL'].value_counts(normalize=True)

batch_1 = sorted(glob.glob(os.path.join(DATA_DIR, "data_new2[01]_clean.csv")))
batch_2 = sorted(glob.glob(os.path.join(DATA_DIR, "data_new2[23]_clean.csv")))

print("--- Batch 1 (20, 21) ---")
stats1, labels1 = analyze_batch(batch_1)
print(stats1)
print("\nLabel Dist:\n", labels1)

print("\n--- Batch 2 (22, 23) ---")
stats2, labels2 = analyze_batch(batch_2)
print(stats2)
print("\nLabel Dist:\n", labels2)
