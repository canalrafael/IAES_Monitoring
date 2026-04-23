import pandas as pd
import glob
import os

DATA_DIR = 'data/online validation data/'
csv_files = sorted(glob.glob(os.path.join(DATA_DIR, '*.csv')))

for f in csv_files:
    df = pd.read_csv(f)
    print(f"File: {os.path.basename(f)}")
    print(df['LABEL'].value_counts())
    print("-" * 20)
