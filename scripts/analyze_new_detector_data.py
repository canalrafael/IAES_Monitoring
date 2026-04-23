import os
import pandas as pd
import numpy as np
from sklearn.metrics import confusion_matrix, accuracy_score

# Paths
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
BASE_DIR = os.path.dirname(SCRIPT_DIR)
DATA_DIR = os.path.join(BASE_DIR, "data")
DETECTOR_DIR = os.path.join(DATA_DIR, "data_test", "data_detector")
NEW_FILES = ["data_detector3.txt", "data_detector4.txt"]

PREVIOUS_BENIGN_FILES = ['data0_clean.csv', 'data1_clean.csv', 'data7_clean.csv', 'data10_clean.csv', 
                         'data12_clean.csv', 'data15_clean.csv', 'data18_clean.csv', 'data21_clean.csv', 
                         'data23_clean.csv', 'data24_clean.csv']

PREVIOUS_ATTACK_FILES = ['data3_clean.csv', 'data4_clean.csv', 'data5_clean.csv', 'data6_clean.csv', 
                         'data13_clean.csv', 'data14_clean.csv', 'data16_clean.csv', 'data17_clean.csv', 
                         'data20_clean.csv', 'data22_clean.csv', 'data25_clean.csv', 'data26_clean.csv', 
                         'data27_clean.csv']

def parse_detector_file(filepath):
    with open(filepath, 'r') as f:
        lines = f.readlines()
    
    # Part 1: PMU Counters
    pmu_lines = []
    det_lines = []
    
    # Initial state: If file starts with a PMU header, start in PMU mode.
    # Otherwise, wait for markers.
    current_section = None
    if lines and "CORE_ID" in lines[0]:
        current_section = "PMU"
    
    for line in lines:
        line = line.strip()
        if not line: continue
        if line == "PMU_START":
            current_section = "PMU"
            continue
        elif line == "PMU_END":
            current_section = None
            continue
        elif line == "DET_START":
            current_section = "DET"
            continue
        elif line == "DET_END":
            current_section = None
            continue
        
        if current_section == "PMU":
            pmu_lines.append(line)
        elif current_section == "DET":
            det_lines.append(line)
            
    # Load into DataFrames
    from io import StringIO
    
    # Filter headers from lines if they repeat
    pmu_header = "CORE_ID,TIMESTAMP,CPU_CYCLES,INSTRUCTIONS,CACHE_MISSES,BRANCH_MISSES,L2_CACHE_ACCESS,LABEL"
    det_header = "SAMPLE_IDX,STATUS,PROBABILITY"
    
    pmu_content = [pmu_lines[0]] + [l for l in pmu_lines[1:] if "CORE_ID" not in l]
    det_content = [det_lines[0]] + [l for l in det_lines[1:] if "SAMPLE_IDX" not in l]

    pmu_df = pd.read_csv(StringIO("\n".join(pmu_content)))
    det_df = pd.read_csv(StringIO("\n".join(det_content)))
    
    # Merge on Sample Index if possible, but concat is safer if indices repeat per block
    # However, for a row-by-row report we want them aligned. 
    # Usually samples are sequential across blocks or reset.
    if len(pmu_df) != len(det_df):
        print(f"Warning: PMU samples ({len(pmu_df)}) != DET samples ({len(det_df)})")
        min_len = min(len(pmu_df), len(det_df))
        pmu_df = pmu_df.iloc[:min_len]
        det_df = det_df.iloc[:min_len]

    merged_df = pd.concat([pmu_df.reset_index(drop=True), det_df.reset_index(drop=True)], axis=1)
    return merged_df

def calculate_metrics(df):
    # Do NOT exclude WARMUP as per user request
    eval_df = df.copy()
    if len(eval_df) == 0: return None
    
    # Map Labels: 0 (Benign) -> 0, 2 (Attack) -> 1
    eval_df['y_true'] = eval_df['LABEL'].map({0: 0, 2: 1})
    # Map Status: BENIGN/WARMUP -> 0, ATTACK -> 1
    eval_df['y_pred'] = eval_df['STATUS'].map({'BENIGN': 0, 'WARMUP': 0, 'ATTACK': 1})
    
    # Drop rows with NaN if any mapping failed
    eval_df = eval_df.dropna(subset=['y_true', 'y_pred'])
    
    y_true = eval_df['y_true'].values
    y_pred = eval_df['y_pred'].values
    
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    tn, fp, fn, tp = cm.ravel()
    
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    fpr = fp / (fp + tn) if (fp + tn) > 0 else 0
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
    acc = accuracy_score(y_true, y_pred)
    
    return {
        'Recall': recall,
        'FPR': fpr,
        'Precision': precision,
        'F1': f1,
        'Accuracy': acc,
        'TP': tp, 'TN': tn, 'FP': fp, 'FN': fn,
        'Samples': len(eval_df)
    }

def aggregate_pmu_stats(df, label):
    features = ['CPU_CYCLES', 'INSTRUCTIONS', 'CACHE_MISSES', 'BRANCH_MISSES', 'L2_CACHE_ACCESS']
    sub_df = df[df['LABEL'] == label].copy()
    if len(sub_df) == 0: return None
    
    sub_df['IPC'] = sub_df['INSTRUCTIONS'] / sub_df['CPU_CYCLES']
    sub_df['Cache_Miss_Rate'] = sub_df['CACHE_MISSES'] / sub_df['L2_CACHE_ACCESS']
    
    stats = sub_df.describe().loc[['mean', 'std']]
    return stats

# Main Analysis
results = []
new_data_list = []

print("--- Evaluating New Data ---")
for f_name in NEW_FILES:
    f_path = os.path.join(DETECTOR_DIR, f_name)
    df = parse_detector_file(f_path)
    metrics = calculate_metrics(df)
    if metrics:
        metrics['File'] = f_name
        results.append(metrics)
        new_data_list.append(df)
        print(f"File: {f_name} | Recall: {metrics['Recall']:.4f} | FPR: {metrics['FPR']:.4f}")

# Overall metrics
combined_new = pd.concat(new_data_list)
overall_metrics = calculate_metrics(combined_new)
print(f"\nOverall Metric Summary:")
print(pd.Series(overall_metrics))

# --- Comparative Analysis ---
print("\n--- Comparing with Previous Data ---")

# Load Previous
prev_benign_list = []
for f in PREVIOUS_BENIGN_FILES:
    p = os.path.join(DATA_DIR, f)
    if os.path.exists(p):
        prev_benign_list.append(pd.read_csv(p))

prev_attack_list = []
for f in PREVIOUS_ATTACK_FILES:
    p = os.path.join(DATA_DIR, f)
    if os.path.exists(p):
        prev_attack_list.append(pd.read_csv(p))

prev_benign_df = pd.concat(prev_benign_list)
prev_attack_df = pd.concat(prev_attack_list)

def get_core_stats(df, label_val, name):
    sub = df[df['LABEL'] == label_val].copy()
    sub['IPC'] = sub['INSTRUCTIONS'] / sub['CPU_CYCLES']
    res = sub[['IPC', 'CPU_CYCLES', 'INSTRUCTIONS', 'CACHE_MISSES', 'BRANCH_MISSES', 'L2_CACHE_ACCESS']].mean()
    res.name = name
    return res

# Stats Compilation
comp_label0 = pd.DataFrame([
    get_core_stats(prev_benign_df, 0, "Previous_Benign"),
    get_core_stats(combined_new, 0, "New_Benign")
])

comp_label2 = pd.DataFrame([
    get_core_stats(prev_attack_df, 2, "Previous_Attack"),
    get_core_stats(combined_new, 2, "New_Attack")
])

print("\nLabel 0 (Benign) Comparison:")
print(comp_label0.T)

print("\nLabel 2 (Attack) Comparison:")
print(comp_label2.T)

# Save to artifacts
comp_label0.to_csv(os.path.join(BASE_DIR, "results", "comp_label0.csv"))
comp_label2.to_csv(os.path.join(BASE_DIR, "results", "comp_label2.csv"))
pd.DataFrame(results).to_csv(os.path.join(BASE_DIR, "results", "new_detector_metrics.csv"), index=False)

# --- New: Row-by-Row Reporting ---
REPORT_PATH = os.path.join(BASE_DIR, "results", "full_row_report.txt")
with open(REPORT_PATH, 'w') as f:
    f.write("Full Row-by-Row Detection Report\n")
    f.write("===============================\n\n")
    
    for df_idx, df in enumerate(new_data_list):
        f.write(f"File: {NEW_FILES[df_idx]}\n")
        f.write("-" * 30 + "\n")
        eval_df = df.copy()
        
        # Determine Result Status
        def get_result(row):
            # Treat WARMUP as Benign prediction as per user request
            gt = 'Attack' if row['LABEL'] == 2 else 'Benign'
            pred = 'Attack' if row['STATUS'] == 'ATTACK' else 'Benign'
            if gt == pred:
                return f'CORRECT({ "TP" if gt == "Attack" else "TN" })'
            else:
                # If WARMUP was during an Attack, it's a False Negative (FN). 
                # If WARMUP was during Benign, it's a True Negative (TN).
                return f'ERROR({ "FN" if gt == "Attack" else "FP" })'

        eval_df['RESULT'] = eval_df.apply(get_result, axis=1)
        
        report_cols = ['SAMPLE_IDX', 'LABEL', 'STATUS', 'RESULT', 'PROBABILITY']
        # Rename for clarity in txt
        display_df = eval_df[report_cols].rename(columns={
            'LABEL': 'GT_LABEL',
            'STATUS': 'PREDICTION'
        })
        # Map labels to names
        display_df['GT_LABEL'] = display_df['GT_LABEL'].map({0: 'Benign', 2: 'Attack'})
        
        f.write(display_df.to_string(index=False))
        f.write("\n\n" + "="*50 + "\n\n")

print(f"Full row-by-row report generated in {REPORT_PATH}")
