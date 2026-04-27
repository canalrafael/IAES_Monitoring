import pandas as pd
import glob
import os

TRAIN_DIR = "/home/canal/github/IAES_Monitoring/data/train data"
files = sorted(glob.glob(os.path.join(TRAIN_DIR, "*.csv")))

print(f"{'File':<20} | {'Samples':<10} | {'Labels':<15} | {'Benchmarks':<20}")
print("-" * 75)

for f in files:
    try:
        df = pd.read_csv(f)
        samples = len(df)
        labels = sorted(df['LABEL'].unique().tolist()) if 'LABEL' in df.columns else ["N/A"]
        benchs = sorted(df['BENCH_ID'].unique().tolist()) if 'BENCH_ID' in df.columns else ["N/A"]
        print(f"{os.path.basename(f):<20} | {samples:<10} | {str(labels):<15} | {str(benchs):<20}")
    except Exception as e:
        print(f"{os.path.basename(f):<20} | Error: {e}")
