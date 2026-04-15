# deploy/ — Bao Bare-Metal Detector
## Three-step workflow

### Step 1 — Generate weights header (run once after every training run)
```bash
cd ~/github/IAES_Monitoring
python3 scripts/export_model_c.py
# → produces deploy/model_weights.h
```

### Step 2 — PC-side validation (simulate against a CSV file)
```bash
cd ~/github/IAES_Monitoring/deploy
gcc -DDETECTOR_PC_SIM -O2 -Wall -o detector_sim detector.c
./detector_sim ../data/data17_clean.csv   # attack file
./detector_sim ../data/data15_clean.csv   # benign file
```

### Step 3 — Embed in Bao (bare-metal, no OS)
Copy `detector.h`, `detector.c`, and `model_weights.h` into the Bao source tree.
In the hypervisor timer/PMU interrupt handler:

```c
#include "detector.h"

/* Boot-time (called once from EL2 setup) */
detector_init();

/* Inside PMU interrupt (every measurement interval) */
pmu_sample_t s = {
    .cpu_cycles      = read_pmevcntr(0),
    .instructions    = read_pmevcntr(1),
    .cache_misses    = read_pmevcntr(2),
    .branch_misses   = read_pmevcntr(3),
    .l2_cache_access = read_pmevcntr(4),
};
det_output_t r = detector_process_sample(cpu_id, &s);
if (r.status == DET_ATTACK) {
    /* isolate / migrate / alert the guest VM */
}
```

---

## File inventory

| File | Description |
|---|---|
| `detector.h` | Public API (include this in Bao source) |
| `detector.c` | Full bare-metal implementation |
| `model_weights.h` | **Auto-generated** — all weights, biases, norm params |

## Model summary (V1 — best balance)

| Property | Value |
|---|---|
| Architecture | 12 → 16 → 16 → 1 (ReLU hidden, sigmoid output) |
| Features | IPC, MPKI, L2_PRESSURE, BRANCH_MISS_RATE × {mean, std, delta} |
| Window size W | 10 PMU samples |
| Parameters | 497 |
| AUC | 0.9779 |
| Recall @ τ=0.50 | 0.9966 |
| FPR   @ τ=0.50 | 0.0865 |
| Threshold τ | 0.50 (stored in `MDL_THRESHOLD`) |
| Temperature T | from training (stored in `MDL_TEMPERATURE`) |

## Math parity with Python training

| Step | Python (`phase2_simplified.py`) | C (`detector.c`) |
|---|---|---|
| Ratio signals | `df['IPC'] = INSTRUCTIONS / CPU_CYCLES` | `sig_ipc = instr / (cycles + eps)` |
| Rolling mean | `df.rolling(W).mean()` | `sum / W` over ring buffer |
| Rolling std | `df.rolling(W).std()` (ddof=1) | `sqrt((sum_sq - W·mean²)/(W-1))` |
| Rolling delta | `df.diff(W)` | `new_val - ring[idx]` (pre-overwrite) |
| Normalise | `(x - mean) / std` | `(x - MDL_FEAT_MEAN[i]) / MDL_FEAT_STD[i]` |
| Layer act. | `nn.ReLU()` | `BM_RELU(x)` = `max(0, x)` |
| Output | `sigmoid(logit / T)` | `bm_sigmoidf(logit / MDL_TEMPERATURE)` |

## Bare-metal math functions

| Function | Method | Max error |
|---|---|---|
| `bm_expf(x)` | Repeated squaring `(1 + x/1024)^1024` | < 0.02% for \|x\| < 20 |
| `bm_sqrtf(x)` | Quake fast inverse sqrt + 2 Newton steps | < 0.01% |
| `bm_sigmoidf(x)` | `1 / (1 + bm_expf(-x))` | derived from expf |

All bare-metal functions are branchless-friendly and hardware-float capable on Cortex-A72.
