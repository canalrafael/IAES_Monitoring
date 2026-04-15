import pandas as pd
import glob, os

DATA_DIR = '/home/canal/github/IAES_Monitoring/data/'
for f in sorted(glob.glob(os.path.join(DATA_DIR, '*.csv'))):
    df = pd.read_csv(f)
    labels = sorted(df['LABEL'].unique().tolist())
    n_benign = (df['LABEL'] == 0).sum()
    n_attack = (df['LABEL'] == 2).sum()
    print(f"{os.path.basename(f):25s}  rows={len(df):7,}  labels={labels}  benign={n_benign:6,}  attack={n_attack:6,}")
