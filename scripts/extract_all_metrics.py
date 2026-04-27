import pandas as pd
import numpy as np

files = [
    ('DS42', '/home/canal/github/IAES_Monitoring/data/online validation data/data_new42_clean.csv'),
    ('DS43', '/home/canal/github/IAES_Monitoring/data/online validation data/data_new43_clean.csv')
]

print(f"{'File':<6} | {'TP':<8} | {'FN':<8} | {'TN':<8} | {'FP':<8} | {'Recall':<10} | {'FPR':<10}")
print("-" * 75)

for name, path in files:
    df = pd.read_csv(path)
    
    # Ground Truth: Label 3 is Attack, everything else (0, 1, 2) is Benign
    is_attack = (df['LABEL'] == 3)
    is_benign = ~is_attack
    
    # Prediction: DET_STATUS 2 is Attack, everything else (0, 1) is Benign
    is_detected = (df['DET_STATUS'] == 2)
    
    tp = np.sum(is_attack & is_detected)
    fn = np.sum(is_attack & ~is_detected)
    fp = np.sum(is_benign & is_detected)
    tn = np.sum(is_benign & ~is_detected)
    
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    fpr = fp / (fp + tn) if (fp + tn) > 0 else 0
    
    print(f"{name:<6} | {tp:<8} | {fn:<8} | {tn:<8} | {fp:<8} | {recall:<10.2%} | {fpr:<10.2%}")
