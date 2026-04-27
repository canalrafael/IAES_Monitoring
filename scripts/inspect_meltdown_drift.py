import pandas as pd
import os

BASE_DIR = "/home/canal/github/IAES_Monitoring"
VALID_DIR = os.path.join(BASE_DIR, "data", "online validation data")
files = ["data_new28_clean.csv", "data_new29_clean.csv", "data_new30_clean.csv"]

BENCH_NAMES = {
    0: "Idle", 1: "Bandwidth", 2: "Disparity", 3: "FFT", 4: "QSort", 
    5: "Dijkstra", 6: "SHA", 7: "Sorting", 10: "Spectre", 11: "Armageddon", 12: "Meltdown"
}

for f_name in files:
    df = pd.read_csv(os.path.join(VALID_DIR, f_name))
    print(f"\nBenchmark distribution in {f_name}:")
    counts = df['BENCH_ID'].value_counts()
    for b_id, count in counts.items():
        name = BENCH_NAMES.get(b_id, f"ID_{b_id}")
        print(f"  {name:<15}: {count:,}")
