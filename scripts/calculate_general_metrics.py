import pandas as pd
import os

CSV_PATH = "/home/canal/github/IAES_Monitoring/results/online_validation/per_benchmark_metrics.csv"

if not os.path.exists(CSV_PATH):
    print(f"Error: CSV not found at {CSV_PATH}")
else:
    df = pd.read_csv(CSV_PATH)
    
    # 1. Direct Attack Recall (Label 3)
    direct_df = df[(df['Metric_Type'] == 'Recall') & (df['Label'] == 3)].copy()
    direct_df['Detections'] = direct_df['Value'] * direct_df['Samples']
    direct_samples = direct_df['Samples'].sum()
    direct_detections = direct_df['Detections'].sum()
    direct_recall = direct_detections / direct_samples if direct_samples > 0 else 0
    
    # 2. Interference Recall (Label 2)
    inter_df = df[(df['Metric_Type'] == 'Recall') & (df['Label'] == 2)].copy()
    inter_df['Detections'] = inter_df['Value'] * inter_df['Samples']
    inter_samples = inter_df['Samples'].sum()
    inter_detections = inter_df['Detections'].sum()
    inter_recall = inter_detections / inter_samples if inter_samples > 0 else 0
    
    # 3. Overall Benign FPR (Label 0)
    benign_df = df[df['Metric_Type'] == 'FPR'].copy()
    benign_df['False_Positives'] = benign_df['Value'] * benign_df['Samples']
    benign_samples = benign_df['Samples'].sum()
    benign_fps = benign_df['False_Positives'].sum()
    benign_fpr = benign_fps / benign_samples if benign_samples > 0 else 0
    
    print("=" * 45)
    print("           DETAILED GLOBAL METRICS         ")
    print("=" * 45)
    print(f"Malicious Execution Recall (Label 3) : {direct_recall:.2%}  ({int(direct_detections):,}/{direct_samples:,})")
    print(f"Cross-Core Interference Recall (Label 2): {inter_recall:.2%}  ({int(inter_detections):,}/{inter_samples:,})")
    print(f"System-Wide Benign FPR (Label 0)        : {benign_fpr:.2%}  ({int(benign_fps):,}/{benign_samples:,})")
    print("=" * 45)
