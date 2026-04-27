import pandas as pd
import numpy as np
import os

DATA_DIR = "/home/canal/github/IAES_Monitoring/data/online validation data"
files = ["data_new26_clean.csv", "data_new27_clean.csv"]

BENCH_NAMES = {
    0: "Idle", 1: "Bandwidth", 2: "Disparity", 3: "FFT", 4: "QSort", 
    5: "Dijkstra", 6: "SHA", 7: "Sorting", 10: "Spectre", 11: "Armageddon", 12: "Meltdown"
}

LABEL_MAP = {
    0: "Secure_Alone", 1: "Untrusted_Alone", 2: "Secure_Interfered", 3: "Attack_Core"
}

for f in files:
    path = os.path.join(DATA_DIR, f)
    if not os.path.exists(path):
        print(f"File not found: {path}")
        continue
    print(f"\n--- Analysis for {f} ---")
    df = pd.read_csv(path)
    
    print(f"Unique DET_STATUS values: {df['DET_STATUS'].unique()}")
    
    # Calculate metrics assuming DET_STATUS == 1 or DET_STATUS == 2 might be detections?
    # Let's print both to be sure.
    
    groups = df.groupby(['BENCH_ID', 'LABEL'])
    
    print(f"\nMetric: DET_STATUS == 1 (or whatever represents detection)")
    print(f"{'Benchmark':<12} {'Label':<5} {'Context':<18} {'Samples':<8} {'Metric':<6} {'DET==1':<10} {'DET==2':<10}")
    
    for (bench_id, label), group in groups:
        is_attack_label = label in [1, 2, 3]
        samples = len(group)
        det_1 = (group['DET_STATUS'] == 1).sum() / samples if samples > 0 else 0
        det_2 = (group['DET_STATUS'] == 2).sum() / samples if samples > 0 else 0
        
        metric_name = "Recall" if is_attack_label else "FPR"
        bench_name = BENCH_NAMES.get(bench_id, f"ID_{bench_id}")
        context = LABEL_MAP.get(label, f"L_{label}")
        
        print(f"{bench_name:<12} {label:<5} {context:<18} {samples:<8} {metric_name:<6} {det_1:.2%}     {det_2:.2%}")
