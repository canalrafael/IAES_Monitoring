/**
 * detector.h — Public API for the Bao bare-metal interference detector
 * ======================================================================
 * Include this header in the Bao guest VM monitor or hypervisor interrupt
 * handler.  Call detector_init() once at boot, then call
 * detector_process_sample() on every PMU snapshot.
 *
 * Model: Phase 2 Simplified V1 MLP (12 features, 2×16 hidden layers)
 * Features: IPC, MPKI, L2_PRESSURE, BRANCH_MISS_RATE × {mean, std, delta}
 * Window  : W = 10 consecutive PMU samples
 * Memory  : ~6 KB static (weights in ROM + per-CPU ring buffers in RAM)
 */

#ifndef DETECTOR_H
#define DETECTOR_H

#include <stdint.h>

/* ── Platform constants ─────────────────────────────────────────────────────── */
#ifndef PMU_MAX_CPUS
#define PMU_MAX_CPUS 4
#endif

typedef int cpuid_t;

/* ── Raw PMU counter snapshot ───────────────────────────────────────────────── */
typedef struct {
    uint64_t cpu_cycles;      /* PMCCNTR_EL0  — CPU clock cycles          */
    uint64_t instructions;    /* PMEVCNTR0_EL0 — retired instructions      */
    uint64_t cache_misses;    /* PMEVCNTR1_EL0 — LLC / L1D cache misses    */
    uint64_t branch_misses;   /* PMEVCNTR2_EL0 — branch mispredictions     */
    uint64_t l2_cache_access; /* PMEVCNTR3_EL0 — L2 cache accesses         */
} pmu_sample_t;

/* ── Detection result ───────────────────────────────────────────────────────── */
typedef enum {
    DET_WARMUP     = 0,  /* Still filling the W-sample window — no verdict  */
    DET_BENIGN     = 1,  /* p(attack) < threshold — system is safe          */
    DET_ATTACK     = 2,  /* p(attack) >= threshold — interference detected!  */
} det_result_t;

typedef struct {
    det_result_t status;
    float        probability;  /* Raw model output in [0, 1]                */
} det_output_t;

/* ── API ────────────────────────────────────────────────────────────────────── */

/**
 * detector_init() — Zero-initialise all per-CPU state.
 * Call once at hypervisor boot before enabling PMU interrupts.
 */
void detector_init(void);

/**
 * detector_process_sample() — Feed one PMU snapshot and get a verdict.
 *
 * @param cpu_id  CPU index in [0, PMU_MAX_CPUS)
 * @param sample  Pointer to a freshly-read PMU counter snapshot
 * @return        det_output_t containing the detection result and probability
 *
 * The first MDL_WINDOW_SIZE calls per CPU return DET_WARMUP.
 * From call W+1 onward a real verdict is returned on every call.
 * This function is designed to run inside an EL2 interrupt handler:
 *   — no heap allocation, no system calls, fully re-entrant per CPU.
 */
det_output_t detector_process_sample(cpuid_t cpu_id, const pmu_sample_t *sample);

#endif /* DETECTOR_H */
