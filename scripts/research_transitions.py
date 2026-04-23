import pandas as pd
import glob
import os

DATA_DIR = 'data/train data/'
csv_files = glob.glob(os.path.join(DATA_DIR, "*.csv"))

transitions = []
for f in sorted(csv_files):
    try:
        df = pd.read_csv(f)
        unique_labels = df['LABEL'].unique()
        if 0 in unique_labels and 2 in unique_labels:
            # Find where transition happens
            first_2 = df[df['LABEL'] == 2].index[0]
            last_0 = df[df['LABEL'] == 0].index[-1]
            transitions.append({
                'file': os.path.basename(f),
                'first_2': int(first_2),
                'last_0': int(last_0),
                'total_samples': len(df)
            })
    except Exception as e:
        pass

if transitions:
    df_trans = pd.DataFrame(transitions)
    print(df_trans.to_string())
else:
    print("No transition files found.")
