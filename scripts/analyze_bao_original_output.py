import pandas as pd
import glob
import os

# --- Configuration ---
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
BASE_DIR = os.path.dirname(SCRIPT_DIR)
DATA_DIR = os.path.join(BASE_DIR, 'data', 'online validation data')
RESULTS_DIR = os.path.join(BASE_DIR, 'results', 'online_validation')
csv_files = sorted(glob.glob(os.path.join(DATA_DIR, 'data_new*_clean.csv')))

BENCH_MAP = {
    0: "Iddle",
    1: "Bandwidth",
    2: "Disparity",
    3: "FFT",
    4: "QSort",
    5: "Dijkstra",
    6: "SHA",
    7: "Sorting",
    10: "Spectre",
    11: "Armageddon",
    12: "Meltdown"
}

def analyze_original_output():
    results = []

    print(f"Found {len(csv_files)} files in: {DATA_DIR}")
    for f_path in csv_files:
        f_name = os.path.basename(f_path)
        print(f"Analyzing {f_name}...")
        
        try:
            df = pd.read_csv(f_path)
        except Exception as e:
            print(f"Error reading {f_path}: {e}")
            continue

        if 'DET_STATUS' not in df.columns:
            print(f"Warning: DET_STATUS not found in {f_name}")
            continue

        # Group by BENCH_ID and LABEL
        grouped = df.groupby(['BENCH_ID', 'LABEL'])
        
        for (bench_id, label), group in grouped:
            total_samples = len(group)
            detections = (group['DET_STATUS'] == 2).sum()
            avg_prob = group['DET_PROB'].mean()
            
            bench_name = BENCH_MAP.get(bench_id, f"Unknown({bench_id})")
            
            # Context description
            if label == 0:
                context = "Alone (Benign)"
            elif label == 1:
                context = "Alone (Attack)"
            elif label == 2:
                context = "Victim (Interfered)"
            elif label == 3:
                context = "Attacker"
            else:
                context = f"Label {label}"

            results.append({
                "File": f_name,
                "BENCH_ID": bench_id,
                "Benchmark": bench_name,
                "Context": context,
                "Samples": total_samples,
                "Detections": detections,
                "Detection_Rate": f"{(detections / total_samples):.2%}" if total_samples > 0 else "0.00%",
                "Avg_Prob": f"{avg_prob:.4f}"
            })

    summary_df = pd.DataFrame(results)
    print("\n--- Original Bao Model Output Analysis (DET_STATUS) ---")
    
    # Sort for better readability
    summary_df = summary_df.sort_values(by=['BENCH_ID', 'Context'])
    
    # Display per benchmark
    pd.set_option('display.max_rows', None)
    print(summary_df[['Benchmark', 'Context', 'Detection_Rate', 'Avg_Prob', 'Samples', 'File']])
    
    out_csv = os.path.join(RESULTS_DIR, 'original_bao_analysis.csv')
    summary_df.to_csv(out_csv, index=False)
    print(f"\nSaved to: {out_csv}")

if __name__ == "__main__":
    os.makedirs(RESULTS_DIR, exist_ok=True)
    analyze_original_output()
