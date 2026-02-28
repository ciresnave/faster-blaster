/**
 * @file exec_dag.h
 * @brief Execution DAG: optimal path selection across backends with transfer costs.
 *
 * Models an operation sequence as a layered directed acyclic graph where:
 *   - Each LAYER corresponds to one step in the sequence.
 *   - Each NODE in a layer is a (step, backend) pair with an execution cost
 *     estimated from probe scores (and judge profiles once available).
 *   - Each EDGE between adjacent layers carries a transfer cost when the
 *     source and destination backends reside on different devices.
 *
 * The planner runs forward dynamic programming (O(N * B^2)) to find the
 * minimum-cost path through the DAG and returns a concrete fb_exec_plan_t.
 *
 * During execution, measured latency is compared against the plan's estimate.
 * If a step runs >3x slower than predicted the planner is re-invoked from that
 * step forward to adjust for changed hardware conditions.
 *
 * Transfer cost model (PCIe 4.0 x16 bandwidth estimates):
 *   CPU -> GPU  (H2D): 12 GB/s + 5 µs fixed latency
 *   GPU -> CPU  (D2H): 12 GB/s + 5 µs fixed latency
 *   GPU -> GPU  (D2D): 6 GB/s  + 10 µs fixed latency (PCIe peer copy)
 *   Same device     :  0
 *
 * @copyright Copyright (c) 2025
 * @license MIT OR Apache-2.0
 */

#ifndef FASTER_BLASTER_EXEC_DAG_H
#define FASTER_BLASTER_EXEC_DAG_H

#include <stddef.h>
#include <stdint.h>
#include <stdbool.h>

#include "judge_select.h"       /* fb_select_objective_t */
#include "../dispatch_tables.h" /* fb_dispatch_constraints_t */

#ifdef __cplusplus
extern "C" {
#endif

/* =========================================================================
 * Device type — maps each backend to the memory address space it uses.
 * GPU backends with different drivers share FB_DEVICE_GPU_0 unless the
 * user explicitly manages multiple GPU contexts (future extension).
 * ========================================================================= */
typedef enum {
    FB_DEVICE_CPU   = 0,   /**< Shared CPU / host memory */
    FB_DEVICE_GPU_0 = 1,   /**< First GPU device */
    FB_DEVICE_GPU_1 = 2,   /**< Second GPU device (multi-GPU) */
    FB_DEVICE_GPU_2 = 3,
    FB_DEVICE_GPU_3 = 4,
    FB_DEVICE_COUNT = 5,
} fb_device_type_t;

/* =========================================================================
 * Data transfer type required on a DAG edge between adjacent plan steps.
 * ========================================================================= */
typedef enum {
    FB_XFER_NONE = 0,  /**< No transfer — both steps on the same device */
    FB_XFER_H2D  = 1,  /**< Host-to-device: CPU memory → GPU memory    */
    FB_XFER_D2H  = 2,  /**< Device-to-host: GPU memory → CPU memory    */
    FB_XFER_D2D  = 3,  /**< Device-to-device: GPU0 → GPU1 (peer copy)  */
} fb_xfer_type_t;

/* =========================================================================
 * Minimal per-step input needed by the planner.
 * op_chain.c translates fb_seq_step_t[] into this before calling the DAG.
 * ========================================================================= */
typedef struct {
    uint32_t                  op_id;           /**< FB_OP_* constant          */
    fb_dispatch_constraints_t constraints;     /**< Dispatch constraints       */
    size_t                    size_bytes_hint; /**< Data volume for xfer cost  */
} fb_dag_step_input_t;

/* =========================================================================
 * One resolved step in the execution plan.
 * Produced by fb_exec_dag_plan() and stored inside fb_exec_plan_t.
 * ========================================================================= */
typedef struct {
    uint32_t         backend_id;    /**< Index into plugin registry           */
    fb_device_type_t device_type;   /**< Device this backend runs on          */
    fb_xfer_type_t   xfer_before;   /**< Transfer required before this step   */
    float            est_exec_ns;   /**< Predicted execution time (ns)        */
    float            est_xfer_ns;   /**< Predicted transfer cost (ns)         */
} fb_dag_plan_step_t;

/* =========================================================================
 * Complete execution plan for an operation sequence.
 * ========================================================================= */
typedef struct {
    fb_dag_plan_step_t   *steps;          /**< Array of steps (length count)  */
    uint32_t              count;          /**< Number of steps                */
    float                 total_cost_ns;  /**< Sum of exec + xfer costs       */
    fb_select_objective_t objective;      /**< Optimization objective used    */
} fb_exec_plan_t;

/* =========================================================================
 * Public API
 * ========================================================================= */

/**
 * Build the DAG and find the optimal execution plan.
 *
 * Enumerates registered plugins by probing each one, constructs the layered
 * DAG, and runs forward DP to minimise total cost according to @p objective.
 *
 * If no plugins are registered, returns a trivial single-backend plan with
 * backend_id=0 everywhere and zero transfer costs.
 *
 * @param steps       Array of step descriptors (length nsteps).
 * @param nsteps      Number of steps.  Must be >= 1.
 * @param objective   Optimisation goal (MAXIMIZE_SPEED is the common choice).
 * @return Heap-allocated plan (caller must call fb_exec_plan_free), or NULL
 *         on allocation failure.
 */
fb_exec_plan_t *fb_exec_dag_plan(const fb_dag_step_input_t *steps,
                                  uint32_t                   nsteps,
                                  fb_select_objective_t      objective);

/**
 * Replan from step @p from_step forward, preserving the already-executed
 * portion of @p current.
 *
 * Useful for dynamic replanning during execution: after detecting that a step
 * ran significantly slower than predicted (e.g. GPU throttling), call this to
 * obtain an updated plan for the remaining steps.
 *
 * @param current    Existing plan (steps < from_step are preserved in result).
 * @param from_step  First step index to re-plan (must be < nsteps).
 * @param steps      Full step descriptor array (all nsteps entries).
 * @param nsteps     Total number of steps.
 * @param objective  Optimisation goal — may differ from original plan.
 * @return New merged plan (caller owns it; fb_exec_plan_free the old one),
 *         or NULL on failure.
 */
fb_exec_plan_t *fb_exec_dag_replan(const fb_exec_plan_t     *current,
                                    uint32_t                  from_step,
                                    const fb_dag_step_input_t *steps,
                                    uint32_t                  nsteps,
                                    fb_select_objective_t     objective);

/**
 * Free a plan returned by fb_exec_dag_plan() or fb_exec_dag_replan().
 * Safe to call on NULL.
 */
void fb_exec_plan_free(fb_exec_plan_t *plan);

/** Return a short string for @p xfer (e.g. "H2D", "D2H", "D2D", "NONE"). */
const char *fb_xfer_type_name(fb_xfer_type_t xfer);

/** Return a short string for @p device (e.g. "CPU", "GPU_0"). */
const char *fb_device_type_name(fb_device_type_t device);

#ifdef __cplusplus
}
#endif

#endif /* FASTER_BLASTER_EXEC_DAG_H */
