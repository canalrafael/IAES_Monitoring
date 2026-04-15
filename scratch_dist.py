import pandas as pd
import glob
files = glob.glob('/home/canal/github/IAES_Monitoring/data/*clean.csv')
print('File \t IPC \t MPKI \t L2P')
for f in files:
    df = pd.read_csv(f)
    ipc = df.INSTRUCTIONS.sum() / df.CPU_CYCLES.sum()
    mpki = df.CACHE_MISSES.sum() * 1000 / df.INSTRUCTIONS.sum()
    l2p = df.L2_CACHE_ACCESS.sum() / df.CPU_CYCLES.sum()
    print(f"{f.split('/')[-1]:20} \t {ipc:.2f} \t {mpki:.2f} \t {l2p:.4f}")
