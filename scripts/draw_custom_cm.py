import os
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

# ===============================================================
# USER INSTRUCTION:
# Open your favorite cm_N image (e.g., cm_N5.png) and copy the
# exact 4 numbers you see in the boxes into these variables:
# ===============================================================
TN = 85000  # True Negatives (Top-Left box)
FP = 120    # False Positives (Top-Right box)
FN = 50     # False Negatives (Bottom-Left box)
TP = 42000  # True Positives (Bottom-Right box)
# ===============================================================

RESULTS_DIR = 'results/phase2_simplified'

def draw_article_cm(tn, fp, fn, tp):
    # Calculate exact metrics
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    fpr = fp / (fp + tn) if (fp + tn) > 0 else 0
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0

    # Match exact Phase 2 layout (figsize=6x5)
    fig, ax = plt.subplots(figsize=(6, 5))
    
    data = np.array([[tn, fp], [fn, tp]])
    
    # Custom strings for Seaborn annotation
    labels = np.array([
        [f"TN\n{tn}", f"FP\n{fp}"],
        [f"FN\n{fn}", f"TP\n{tp}"]
    ])
    
    # Feed directly into seaborn to perfectly match the original visual style
    sns.heatmap(data, annot=labels, fmt='', cmap='Blues',
                xticklabels=['Benign', 'Attack'],
                yticklabels=['Benign', 'Attack'], ax=ax)
                
    # Place metrics exactly in the subtitle just like cm_N5.png
    ax.set_title(f'Confusion Matrix\nRecall={recall:.4f}  FPR={fpr:.4f}  F1={f1:.4f}')
    
    # Use exact same layout formatting
    plt.tight_layout()
    
    outpath = os.path.join(RESULTS_DIR, 'article_optimal_confusion_matrix.png')
    plt.savefig(outpath, dpi=150)
    plt.close()
    print(f"Success! Matrix saved matching original layout to: {outpath}")

if __name__ == "__main__":
    draw_article_cm(TN, FP, FN, TP)
