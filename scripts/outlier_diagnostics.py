import pandas as pd
import numpy as np
import os
import glob
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score, recall_score, precision_score
from scipy import stats

DATA_DIR = 'data/'
RESULTS_DIR = 'results/phase1/'
os.makedirs(RESULTS_DIR, exist_ok=True)

FEATURES = ['CPU_CYCLES', 'INSTRUCTIONS', 'CACHE_MISSES', 'BRANCH_MISSES', 'L2_CACHE_ACCESS']

def load_data():
    csv_files = glob.glob(os.path.join(DATA_DIR, "*.csv"))
    all_dfs = []
    for f in csv_files:
        df = pd.read_csv(f)
        df['source'] = os.path.basename(f)
        all_dfs.append(df)
    return pd.concat(all_dfs, ignore_index=True)

def check_duplicates(df):
    dups = df.duplicated(subset=FEATURES + ['TIMESTAMP'], keep=False)
    dup_df = df[dups]
    
    benign_dups = len(dup_df[dup_df['LABEL'] == 0])
    interf_dups = len(dup_df[dup_df['LABEL'] == 2])
    
    print(f"Total Exact Duplicates: {len(dup_df)}")
    print(f"Benign Dups: {benign_dups}, Interference Dups: {interf_dups}")
    return dups

def check_wraps(df):
    wrap_counts = {}
    for feat in FEATURES:
        # Check within each source file to ensure temporal order
        total_wraps = 0
        for source, group in df.groupby('source'):
            group = group.sort_index() # Assume recorded order is temporal
            wraps = (group[feat].diff() < 0).sum()
            total_wraps += wraps
        wrap_counts[feat] = total_wraps
    print(f"Counter Wrap Signatures Found: {wrap_counts}")
    return wrap_counts

def check_spike_correlation(df):
    print("Analyzing Spike Correlation (Signal vs Noise)...")
    corrs = {}
    for feat in FEATURES:
        # Define spike as > P99
        threshold = df[feat].quantile(0.99)
        spikes = df[feat] > threshold
        # Correlation with Label 2
        corr = np.corrcoef(spikes.astype(int), (df['LABEL'] == 2).astype(int))[0, 1]
        corrs[feat] = corr
    print(f"Spike-to-Interference Correlation: {corrs}")
    return corrs

def evaluate_case(X_train, X_test, y_train, y_test, name):
    clf = RandomForestClassifier(n_estimators=50, random_state=42, n_jobs=-1)
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    
    f1 = f1_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    # FPR calculation
    tn = ((y_test == 0) & (y_pred == 0)).sum()
    fp = ((y_test == 0) & (y_pred == 1)).sum()
    fpr = fp / (fp + tn) if (fp + tn) > 0 else 0
    
    return {'case': name, 'f1': f1, 'recall': recall, 'fpr': fpr}

def main():
    print("Loading consolidated dataset...")
    df = load_data()
    # Filter only relevant labels
    df = df[df['LABEL'].isin([0, 2])].copy()
    df['target'] = (df['LABEL'] == 2).astype(int)

    # 1. Diagnostics
    check_duplicates(df)
    check_wraps(df)
    check_spike_correlation(df)

    results = []

    # CASE A: RAW
    print("\nEvaluating Case A: RAW...")
    X = df[FEATURES]
    y = df['target']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    results.append(evaluate_case(X_train, X_test, y_train, y_test, "RAW"))

    # CASE B: FILTERED (Blind Z-score removal)
    print("Evaluating Case B: FILTERED (Z > 3)...")
    z_scores = np.abs(stats.zscore(df[FEATURES]))
    filtered_df = df[(z_scores < 3).all(axis=1)]
    X = filtered_df[FEATURES]
    y = filtered_df['target']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    results.append(evaluate_case(X_train, X_test, y_train, y_test, "FILTERED"))

    # CASE C: ROBUST (Clipping P1-P99)
    print("Evaluating Case C: ROBUST (Clipped)...")
    clipped_df = df.copy()
    for feat in FEATURES:
        p1 = df[feat].quantile(0.01)
        p99 = df[feat].quantile(0.99)
        clipped_df[feat] = clipped_df[feat].clip(p1, p99)
    X = clipped_df[FEATURES]
    y = clipped_df['target']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    results.append(evaluate_case(X_train, X_test, y_train, y_test, "ROBUST"))

    # CASE D: PHYSICS-AWARE
    print("Evaluating Case D: PHYSICS-AWARE...")
    # Deduplicate logging artifacts (keep first)
    df_d = df.drop_duplicates(subset=FEATURES + ['TIMESTAMP'])
    # In Case D, we would apply wraps. For this diagnostic, we check if they exist.
    # If wrap check shows 0 wraps (which is common if handled during collection), we skip.
    X = df_d[FEATURES]
    y = df_d['target']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    results.append(evaluate_case(X_train, X_test, y_train, y_test, "PHYSICS-AWARE"))

    # Output Results
    results_df = pd.DataFrame(results)
    results_df.to_csv(os.path.join(RESULTS_DIR, 'outlier_impact_comparison.csv'), index=False)
    print("\nExperimental comparison complete.")
    print(results_df)

if __name__ == "__main__":
    main()
