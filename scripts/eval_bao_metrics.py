"""
eval_bao_metrics.py
Evaluates the Bao hypervisor metrics (Recall / FPR) directly from the
DET_STATUS column already recorded inside the CSV files.
Usage:
    python3 scripts/eval_bao_metrics.py [file1.csv file2.csv ...]
    If no files are given, defaults to data_new40_clean.csv and data_new41_clean.csv.
"""
import os
import sys
import pandas as pd
import numpy as np

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, "data", "online validation data")

BENCH_NAMES = {
    0: "Idle",
    1: "Bandwidth",
    2: "Disparity",
    3: "FFT",
    4: "QSort",
    5: "Dijkstra",
    6: "SHA",
    7: "Sorting",
    10: "Spectre",
    11: "Armageddon",
    12: "Meltdown",
}

LABEL_MAP = {
    0: "Secure_Alone",
    1: "Untrusted_Alone",
    2: "Secure_Interfered",
    3: "Attack_Core",
}

# Bao DET_STATUS: 0 = no detection (warmup/idle), 1 = benign, 2 = anomaly
DETECTION_STATUS = 2   # Bao reports anomaly when DET_STATUS == 2


def evaluate_file(f_path):
    df = pd.read_csv(f_path)

    required = {"BENCH_ID", "LABEL", "DET_STATUS"}
    if not required.issubset(df.columns):
        print(f"  [SKIP] Missing columns in {os.path.basename(f_path)}: {required - set(df.columns)}")
        return []

    # Drop warmup rows (DET_STATUS == 0 means detector not yet initialised)
    df_valid = df[df["DET_STATUS"] != 0].copy()

    rows = []
    for (bench_id, label), group in df_valid.groupby(["BENCH_ID", "LABEL"]):
        samples = len(group)
        if samples == 0:
            continue

        detected = (group["DET_STATUS"] == DETECTION_STATUS).sum()
        metric_val = detected / samples

        # Attack labels → Recall; Benign labels → FPR
        is_attack = label in (1, 2, 3)
        metric_name = "Recall" if is_attack else "FPR"

        rows.append({
            "File":        os.path.basename(f_path),
            "Benchmark":   BENCH_NAMES.get(bench_id, f"ID_{bench_id}"),
            "BENCH_ID":    bench_id,
            "LABEL":       label,
            "Context":     LABEL_MAP.get(label, f"L{label}"),
            "Samples":     samples,
            "Detected":    int(detected),
            "Metric":      metric_name,
            "Value (%)":   f"{metric_val:.2%}",
        })
    return rows


def main():
    if len(sys.argv) > 1:
        files = [os.path.join(DATA_DIR, a) if not os.path.isabs(a) else a
                 for a in sys.argv[1:]]
    else:
        files = [
            os.path.join(DATA_DIR, "data_new40_clean.csv"),
            os.path.join(DATA_DIR, "data_new41_clean.csv"),
        ]

    all_rows = []
    for f in files:
        if not os.path.exists(f):
            print(f"[WARN] File not found: {f}")
            continue
        print(f"Evaluating {os.path.basename(f)} …")
        all_rows.extend(evaluate_file(f))

    if not all_rows:
        print("No results produced.")
        return

    df_res = pd.DataFrame(all_rows)

    # Pretty-print per file
    for fname, grp in df_res.groupby("File"):
        print(f"\n{'='*60}")
        print(f"  Bao Metrics — {fname}")
        print(f"{'='*60}")
        display = grp[["Benchmark", "Context", "Samples", "Detected", "Metric", "Value (%)"]].copy()
        display.sort_values(["Benchmark", "Context"], inplace=True)
        print(display.to_string(index=False))

    # Save CSV
    out_dir = os.path.join(BASE_DIR, "results", "online_validation")
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, "bao_metrics_40_41.csv")
    df_res.to_csv(out_path, index=False)
    print(f"\nResults saved to: {out_path}")


if __name__ == "__main__":
    main()
