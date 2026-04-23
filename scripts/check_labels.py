import pandas as pd
import glob
import os

DATA_DIR = 'data/train data/'
csv_files = glob.glob(os.path.join(DATA_DIR, "*.csv"))

dist = []
for f in csv_files:
    try:
        df = pd.read_csv(f)
        counts = df['LABEL'].value_counts().to_dict()
        dist.append({'file': os.path.basename(f), **counts})
    except Exception as e:
        print(f"Error reading {f}: {e}")

df_dist = pd.DataFrame(dist).fillna(0)
print(df_dist)
