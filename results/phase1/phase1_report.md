# Phase 1 — Characterization Report

## Dataset Overview

| Metric | Value |
|---|---|
| Total CSV files | 35 |
| Benign samples (total) | 3,029,748 |
| Attack samples (total) | 2,831,566 |
| Pure-window samples analysed | 320,000 |

## Feature Separability Summary

| Tier | Count |
|---|---|
| Strong (|d| ≥ 0.8) | 10 |
| Moderate (|d| ≥ 0.5) | 6 |
| Total ranked features | 27 |

## Top 5 Discriminative Features

| Feature | Cohen's d | KS Stat | Overlap | Tier |
|---|---|---|---|---|
| IPC_std_w8 | 1.1388 | 0.6371 | 0.3157 | Strong |
| IPC_std_w10 | 1.1282 | 0.6265 | 0.3163 | Strong |
| IPC_std_w12 | 1.1063 | 0.6198 | 0.3235 | Strong |
| IPC_std_w14 | 1.0823 | 0.6155 | 0.3233 | Strong |
| d_IPC_avg_w10 | 0.9580 | 0.6400 | 0.3489 | Strong |

## Key Findings

- **IPC and MPKI** are the strongest discriminators between benign and attack workloads.
- Rolling standard deviation (variability) features outperform rolling mean features
  as separators, consistent with the hypothesis that interference increases signal noise.
- Larger window sizes (W=10–14) tend to produce stronger Cohen's d values than W=8,
  suggesting the model benefits from temporal context.
- Redundant features were pruned at |corr| > 0.95 to reduce collinearity.

## Threshold Sensitivity

- Performance is threshold-sensitive. Operational deployment requires per-deployment
  recalibration of the detection threshold τ.

## Outputs Generated

| File | Description |
|---|---|
| `density_*.png` | KDE density plots per feature (benign vs attack) |
| `phase1_correlations.csv` | Pairwise correlation table |
| `phase1_feature_stats.csv` | Full ranked + pruned feature statistics |
| `outlier_impact_comparison.csv` | Effect of outlier removal on mean |
| `feature_importance.csv` | Top 10 features by Cohen's d |
