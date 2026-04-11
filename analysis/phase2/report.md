# Phase 2 Analysis: Model Design & Validation

## 1. Executive Summary

Phase 2 successfully identified a robust, real-time microarchitectural interference detector. Through an automated grid search over architectures, window sizes, and learning rates, we have discovered an optimal configuration that balances high recall with a low memory footprint suitable for the Bao hypervisor.

**Selected Configuration**:
- **Architecture**: 2-layer MLP (16 units/layer)
- **Window Size (W)**: 10 samples
- **Feature Set**: 45 features (5 base counters + 4 ratios, with 5 temporal stats each)
- **Hardware Footprint**: 4.0 KB (within the <5 KB constraint)
- **Inference Cost**: 992 MAC operations per sample

## 2. Experimental Discovery Results

### 2.1 Window Size (W) Impact
We observed that **W=10** provides the best trade-off between temporal aggregation and detection latency. While W=20 showed slightly more stability in some folds, it introduces an additional 10 samples of physical delay. W=10 sufficiently amplifies the variance signatures ($\sigma^2$) identified in Phase 1 without excessive lag.

### 2.2 Learning Rate & Stability
Convergence was most stable at $\eta = 0.005$. Lower learning rates ($1e-4, 5e-4$) often required more epochs than the early-stopping patience allowed, while higher rates sometimes led to jittery validation loss in the LOBO folds.

### 2.3 Model Architecture Comparison
| Layers | Units | W | Avg Recall ($\tau=0.5$) | Avg FPR ($\tau=0.5$) | Footprint (KB) |
| :--- | :--- | :--- | :--- | :--- | :--- |
| 1 | 8 | 10 | 0.911 | 0.291 | 1.47 KB |
| 1 | 16 | 10 | 0.931 | 0.256 | 2.94 KB |
| **2** | **16** | **10** | **0.939** | **0.264** | **4.00 KB** |

The 2-layer-16-unit model demonstrated superior non-linear mapping capabilities, capturing complex interactions between `BRANCH_MISSES` jitter and `L2_PRESSURE`.

## 3. Pareto Frontier and Threshold Selection

The following operating points were identified through a fine-grained threshold sweep ($\tau \in [0, 1]$):

| Operating Point | Threshold ($\tau$) | Recall | FPR | Precision | F1-Score | Justification |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **High-Safety** | **0.05** | **0.9992** | **0.049** | **0.791** | **0.883** | **Selected**: Prioritizes zero misses for critical safety. |
| **Balanced** | 0.32 | 0.9834 | 0.037 | 0.829 | 0.899 | Optimized for overall detection accuracy. |
| **Low-FPR** | 0.80 | 0.8170 | 0.026 | 0.852 | 0.834 | Minimizes false alarms in non-critical scenarios. |

> [!IMPORTANT]
> **Recommended Deployment Point: High-Safety ($\tau=0.05$)**. 
> For a microarchitectural security monitor, missing an attack is an order of magnitude more costly than a 4.9% false alarm rate in a high-frequency (1kHz) sampling environment.

## 4. Hardware Deployment Viability

The selected model is highly optimized for the Bao hypervisor:
- **Footprint**: At 4.0 KB, it resides fully in a single page or even a small fraction of the hypervisor's static memory area.
- **Latency**: 992 MACs per sample. On a 1.5GHz Cortex-A72, this inference time is $\ll 1\mu s$, well within the $100\mu s - 1ms$ sampling budget.
- **Complexity**: $O(1)$ feature extraction (using 45-slot ring buffers) and a purely feed-forward MLP (ReLU activations) ensure deterministic execution time.

## 5. Conclusion
The discovery process confirms that a small, temporal-aware MLP can effectively distinguish microarchitectural interference using only raw secure-core PMU data. The **High-Safety** operating point provides nearly perfect recall ($>0.999$) with a manageable false positive rate.

We are now ready to proceed to **Phase 3: Online Detection & Latency Analysis** to quantify the real-time performance and stability of this model under various contention scenarios.
