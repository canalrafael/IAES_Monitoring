import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

# Styling
sns.set_theme(style="whitegrid", palette="bright")
plt.rcParams['figure.figsize'] = (10, 6)

RESULTS_DIR = 'results/phase2/'
df = pd.read_csv(os.path.join(RESULTS_DIR, 'model_comparison.csv'))

# Aggregate metrics per learning rate
lr_summary = df.groupby('lr').agg({
    'avg_f1': ['mean', 'std'],
    'avg_recall': ['mean', 'std'],
    'avg_fpr': ['mean', 'std']
}).reset_index()

# Flatten columns
lr_summary.columns = ['lr', 'f1_mean', 'f1_std', 'recall_mean', 'recall_std', 'fpr_mean', 'fpr_std']

fig, ax1 = plt.subplots()

# F1 and Recall on Primary Axis
ax1.plot(lr_summary['lr'], lr_summary['f1_mean'], marker='o', label='Mean F1', color='#1f77b4')
ax1.fill_between(lr_summary['lr'], lr_summary['f1_mean'] - lr_summary['f1_std'], 
                 lr_summary['f1_mean'] + lr_summary['f1_std'], alpha=0.2, color='#1f77b4')

ax1.plot(lr_summary['lr'], lr_summary['recall_mean'], marker='s', label='Mean Recall', color='#ff7f0e')
ax1.fill_between(lr_summary['lr'], lr_summary['recall_mean'] - lr_summary['recall_std'], 
                 lr_summary['recall_mean'] + lr_summary['recall_std'], alpha=0.2, color='#ff7f0e')

ax1.set_xlabel('Learning Rate (η)')
ax1.set_ylabel('Performance (F1 / Recall)')
ax1.set_xscale('log')
ax1.legend(loc='upper left')

# FPR on Secondary Axis
ax2 = ax1.twinx()
ax2.plot(lr_summary['lr'], lr_summary['fpr_mean'], marker='x', label='Mean FPR', color='#d62728', linestyle='--')
ax2.set_ylabel('False Positive Rate (FPR)')
ax2.legend(loc='upper right')

plt.title('Impact of Learning Rate on Detection Stability')
plt.tight_layout()
plt.savefig(os.path.join(RESULTS_DIR, 'learning_rate_analysis.png'))
plt.close()

print("Learning rate analysis graph saved to results/phase2/learning_rate_analysis.png")
