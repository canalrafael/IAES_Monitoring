import pandas as pd
for f in ["data_new18_clean.csv", "data_new19_clean.csv", "data_new20_clean.csv", "data_new21_clean.csv"]:
    path = f"/home/canal/github/IAES_Monitoring/data/online validation data/{f}"
    df = pd.read_csv(path)
    print(f"\n--- {f} ---")
    print(f"Benchmarks: {sorted(df.BENCH_ID.unique())}")
    print(f"Total Rows: {len(df)}")
