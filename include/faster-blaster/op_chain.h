/**
 * @file op_chain.h
 * @brief Operation sequence executor — wired to FB_OP_* IDs and ranked dispatch.
 *
 * Provides a lightweight sequential execution framework that:
 *   1. Accepts a list of {op_id, constraints, exec_fn} steps.
 *   2. At compile() time, selects the best backend per step via
 *      fb_select_backend_with_constraints() and annotates adjacent steps
 *      eligible for fusion.
 *   3. At execute() time, calls each step's caller-supplied exec_fn with
 *      the currently active vtable pointer so the caller can invoke the
 *      real backend function.
 *
 * Usage:
 * @code
 *   // Declare per-step argument structs as needed:
 *   struct my_gemm_args { float alpha, beta; float *A, *B, *C; int m,n,k; };
 *
 *   // Provide an exec_fn:
 *   static int run_gemm(const fb_backend_vtable_t *vt, void *ud) {
 *       struct my_gemm_args *a = (struct my_gemm_args *)ud;
 *       if (vt && vt->sgemm)
 *           vt->sgemm(FB_ROW_MAJOR, FB_NO_TRANS, FB_NO_TRANS,
 *                     a->m, a->n, a->k, a->alpha, a->A, a->k,
 *                     a->B, a->n, a->beta, a->C, a->n);
 *       return 0;
 *   }
 *
 *   fb_op_sequence_t *seq = fb_op_sequence_create(16);
 *   fb_seq_step_t step = {
 *       .op_id   = FB_OP_SGEMM,
 *       .exec_fn = run_gemm,
 *       .user_data = &my_args,
 *   };
 *   strncpy(step.name, "gemm0", sizeof(step.name) - 1);
 *   fb_op_sequence_add(seq, &step);
 *   fb_op_sequence_compile(seq);
 *   fb_op_sequence_execute(seq);
 *   fb_op_sequence_free(seq);
 * @endcode
 *
 * @copyright Copyright (c) 2025
 * @license MIT OR Apache-2.0
 */

#ifndef FASTER_BLASTER_OP_CHAIN_H
#define FASTER_BLASTER_OP_CHAIN_H

#include <stdint.h>
#include <stdbool.h>

#include "vtable_autofill.h"        /* fb_backend_vtable_t (opaque), fb_get_active_vtable() */
#include "../dispatch_tables.h"     /* fb_dispatch_constraints_t, fb_select_backend_with_constraints() */
#include "exec_dag.h"               /* fb_exec_plan_t, fb_device_type_t, fb_xfer_type_t */

#ifdef __cplusplus
extern "C" {
#endif

/* =========================================================================
 * Step descriptor
 * ========================================================================= */

/**
 * A single step in an operation sequence.
 *
 * The caller fills in op_id, constraints, exec_fn, user_data and name before
 * passing the struct to fb_op_sequence_add().  The remaining fields are
 * populated by fb_op_sequence_compile().
 */
typedef struct {
    /* --- Caller-provided (before add()) --- */

    /** Operation identifier (FB_OP_SAXPY, FB_OP_SGEMM, …). */
    uint32_t  op_id;

    /** Runtime constraints.  Zero-initialise for unconstrained dispatch. */
    fb_dispatch_constraints_t  constraints;

    /**
     * Execution callback.  Called once per step during execute().
     * Receives the currently active vtable and the user_data pointer.
     * Return 0 for success; any other value aborts sequence execution.
     * May be NULL — the step is then skipped silently.
     */
    int (*exec_fn)(const fb_backend_vtable_t *vtable, void *user_data);

    /** Caller-owned, passed verbatim to exec_fn. */
    void     *user_data;

    /** Human-readable label (for debug output). */
    char      name[32];

    /* --- Set by fb_op_sequence_compile() --- */

    /**
     * Estimated data volume for this step in bytes.  Used by the DAG
     * planner to price memory-transfer edges between devices.
     * Zero means "unknown" — the planner will treat transfer cost as
     * fixed latency only (no bandwidth term).
     */
    size_t    size_bytes_hint;

    /* --- Set by fb_op_sequence_compile() via the DAG planner --- */

    /** Backend index (plugin registry position) elected by the DAG. */
    uint32_t  selected_backend_id;

    /** Device type the DAG placed this step on (FB_DEVICE_CPU / GPU_*). */
    fb_device_type_t planned_device_type;

    /**
     * Transfer the DAG requires before this step executes.
     * FB_XFER_NONE means no transfer needed (same device as previous step).
     * The caller's exec_fn is responsible for performing the described
     * transfer before accessing output data from the previous step.
     */
    fb_xfer_type_t   planned_xfer_before;

    /**
     * Advisory: this step and the next step may be fused into a single
     * kernel call (same device, compatible operation pattern).
     * True only when both steps share a device and the op pair matches
     * a known fusible pattern (e.g. GEMM->SAXPY, GEMM->SCAL).
     */
    bool      may_fuse_with_next;
} fb_seq_step_t;

/* =========================================================================
 * Sequence
 * ========================================================================= */

/** Opaque sequence object. */
typedef struct fb_op_sequence fb_op_sequence_t;

/**
 * Create an empty operation sequence with room for up to @p max_steps steps.
 *
 * @param max_steps  Upper bound on the number of steps.  Must be ≥ 1.
 * @return New sequence, or NULL on allocation failure.
 */
fb_op_sequence_t *fb_op_sequence_create(uint32_t max_steps);

/**
 * Free a sequence and all associated memory.
 * Safe to call on NULL.
 */
void fb_op_sequence_free(fb_op_sequence_t *seq);

/**
 * Append a step to the sequence.
 *
 * @param seq   Target sequence (must not be NULL; must not yet be compiled).
 * @param step  Step descriptor.  The struct is copied by value.
 * @return 0 on success; -1 if seq is NULL, already compiled, or capacity exceeded.
 */
int fb_op_sequence_add(fb_op_sequence_t *seq, const fb_seq_step_t *step);

/**
 * Compile the sequence.
 *
 * For each step:
 *   - Calls fb_select_backend_with_constraints() to populate
 *     step->selected_backend_id.
 *   - Runs fusion heuristics and sets step->may_fuse_with_next where
 *     applicable.
 *
 * After a successful compile, steps may be added again only after calling
 * fb_op_sequence_reset().
 *
 * @param seq  Sequence to compile (must not be NULL).
 * @return 0 on success; -1 if seq is NULL or empty.
 */
int fb_op_sequence_compile(fb_op_sequence_t *seq);

/**
 * Execute a compiled sequence.
 *
 * For each step (in order) that has a non-NULL exec_fn:
 *   1. Fetches the currently active vtable via fb_get_active_vtable().
 *   2. Calls step->exec_fn(vtable, step->user_data).
 *   3. If the return value is non-zero, stops and returns that value.
 *
 * A NULL active vtable is legal (exec_fn receives NULL and can respond
 * accordingly — typically skip or use a fallback).
 *
 * @param seq  A compiled sequence (must not be NULL).
 * @return 0 if all steps succeed; -1 if seq is NULL or not compiled;
 *         the first non-zero exec_fn return value otherwise.
 */
int fb_op_sequence_execute(fb_op_sequence_t *seq);

/**
 * Reset the sequence to the empty, uncompiled state (keeps capacity).
 * Useful for rebuilding the sequence without freeing and reallocating.
 *
 * @param seq  Sequence to reset (NULL is a no-op).
 */
void fb_op_sequence_reset(fb_op_sequence_t *seq);

/* =========================================================================
 * Inspection helpers
 * ========================================================================= */

/** Returns the number of steps currently in the sequence. */
uint32_t fb_op_sequence_step_count(const fb_op_sequence_t *seq);

/**
 * Return a pointer to the step at index @p idx, or NULL if out of range.
 * The pointer is valid until the next call to fb_op_sequence_add() or
 * fb_op_sequence_reset().
 */
const fb_seq_step_t *fb_op_sequence_get_step(const fb_op_sequence_t *seq,
                                              uint32_t                idx);

/** Returns true if the sequence has been compiled and not yet reset. */
bool fb_op_sequence_is_compiled(const fb_op_sequence_t *seq);

/**
 * Print a human-readable summary of the sequence to stdout.
 * Shows each step's name, op_id, planned backend, device, transfer type,
 * and fusion hints.  Suitable for debugging; output format is not stable.
 */
void fb_op_sequence_print(const fb_op_sequence_t *seq);

/* =========================================================================
 * DAG planning controls
 * ========================================================================= */

/**
 * Set the optimisation objective used by the DAG planner at compile() time.
 * Must be called before fb_op_sequence_compile().
 * Default is FB_SELECT_MAXIMIZE_SPEED.
 *
 * @param seq        Target sequence.  NULL is a no-op.
 * @param objective  Objective from fb_select_objective_t.
 */
void fb_op_sequence_set_objective(fb_op_sequence_t     *seq,
                                   fb_select_objective_t objective);

/**
 * Return the execution plan most recently produced by compile().
 * The returned pointer is owned by the sequence and invalidated by
 * fb_op_sequence_reset() or fb_op_sequence_free().
 * Returns NULL if the sequence has not been compiled.
 */
const fb_exec_plan_t *fb_op_sequence_get_plan(const fb_op_sequence_t *seq);

#ifdef __cplusplus
}
#endif

#endif /* FASTER_BLASTER_OP_CHAIN_H */
