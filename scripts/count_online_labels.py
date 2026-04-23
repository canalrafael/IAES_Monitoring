import pandas as pd
import glob
import os

# Get the project root directory
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
BASE_DIR = os.path.dirname(SCRIPT_DIR)
DATA_DIR = os.path.join(BASE_DIR, 'data', 'online validation data')

csv_files = sorted(glob.glob(os.path.join(DATA_DIR, 'data_new*_clean.csv')))

results = []

print(f"Searching for CSV files in: {DATA_DIR}")
print(f"Found {len(csv_files)} files.")

for f_path in csv_files:
    f_name = os.path.basename(f_path)
    try:
        df = pd.read_csv(f_path)
        counts = df['LABEL'].value_counts().to_dict()
        
        row = {
            "File": f_name,
            "Total": len(df),
            "L0 (Benign Alone)": counts.get(0, 0),
            "L1 (Attack Alone)": counts.get(1, 0),
            "L2 (Interfered)": counts.get(2, 0),
            "L3 (Attacker)": counts.get(3, 0)
        }
        results.append(row)
    except Exception as e:
        print(f"Error reading {f_name}: {e}")

if results:
    summary_df = pd.DataFrame(results)
    print("\n--- Online Validation Label Counts ---")
    print(summary_df.to_string(index=False))
    
    out_dir = os.path.join(BASE_DIR, 'results', 'online_validation')
    os.makedirs(out_dir, exist_ok=True)
    summary_df.to_csv(os.path.join(out_dir, 'label_counts_summary.csv'), index=False)
else:
    print("No data processed.")
