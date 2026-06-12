/**
 * @file op_chain.c
 * @brief Operation sequence executor -- wired to DAG planner and ranked dispatch.
 *
 * compile() now builds a layered DAG via fb_exec_dag_plan(), which finds the
 * globally optimal backend path through the operation sequence accounting for:
 *   - Per-operation execution cost (from probe scores, future: judge profiles)
 *   - Data transfer cost at cross-device step boundaries (H2D/D2H/D2D)
 *
 * execute() follows the plan, times each step, and dynamically replans from
 * the next step when actual latency exceeds 3x the predicted cost.
 *
 * See op_chain.h for the full API contract.
 */

#include <stdlib.h>
#include <string.h>
#include <stdio.h>
#include <stdint.h>
#include <stdbool.h>

#ifdef _WIN32
#  include <windows.h>
#else
#  include <time.h>
#endif

#include "../include/faster-blaster/op_chain.h"
#include "../include/faster-blaster/backend_plugin.h"   /* fb_get_vtable_by_id */

/* Include real op-id constants so fusion heuristics use the canonical values. */
#include "judge/judge_op_ids.h"

/* =========================================================================
 * Timing helper
 * ========================================================================= */

/** Returns a monotonic timestamp in nanoseconds. */
static float now_ns(void)
{
#ifdef _WIN32
    LARGE_INTEGER freq, ctr;
    QueryPerformanceFrequency(&freq);
    QueryPerformanceCounter(&ctr);
    return (float)((double)ctr.QuadPart * 1.0e9 / (double)freq.QuadPart);
#else
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (float)((double)ts.tv_sec * 1.0e9 + (double)ts.tv_nsec);
#endif
}

/* =========================================================================
 * Replan deviation threshold
 *
 * If a step takes more than FB_REPLAN_THRESHOLD times its predicted cost
 * the planner is re-invoked for all remaining steps.
 * ========================================================================= */
#define FB_REPLAN_THRESHOLD 3.0f

/* Minimum predicted cost (ns) before the threshold is applied.
 * Avoids spurious replanning when est_exec_ns is near zero. */
#define FB_REPLAN_MIN_PREDICTED_NS 200.0f

/* =========================================================================
 * Internal structure
 * ========================================================================= */

struct fb_op_sequence {
    fb_seq_step_t        *steps;        /* heap-allocated flat array          */
    uint32_t              step_count;   /* how many steps have been added     */
    uint32_t              max_steps;    /* allocated capacity                 */
    bool                  is_compiled;  /* compile() has been called          */

    /* DAG plan -- allocated at compile(), freed at reset()/free() */
    fb_exec_plan_t       *plan;
    fb_select_objective_t objective;    /* set via set_objective(), default SPEED */
};

/* =========================================================================
 * Fusion helpers (used after DAG planning to set may_fuse_with_next)
 * ========================================================================= */

static inline bool is_gemm_family(uint32_t op_id)
{
    return (op_id == FB_OP_SGEMM || op_id == FB_OP_DGEMM);
}

static inline bool is_axpy_or_scal(uint32_t op_id)
{
    return (op_id == FB_OP_SAXPY  || op_id == FB_OP_DAXPY ||
            op_id == FB_OP_SSCAL  || op_id == FB_OP_DSCAL);
}

/**
 * Returns true when two adjacent steps that share the same device are
 * candidates for kernel-level fusion.
 */
static bool check_fusion(const fb_seq_step_t *cur, const fb_seq_step_t *next,
                          bool same_device)
{
    if (!same_device) {
        return false;   /* cross-device steps cannot be fused */
    }
    if (is_gemm_family(cur->op_id) && is_axpy_or_scal(next->op_id)) {
        return true;    /* GEMM -> AXPY/SCAL: fold into beta/alpha */
    }
    if (is_gemm_family(cur->op_id) && is_gemm_family(next->op_id)) {
        return true;    /* consecutive GEMMs: batched/tiled candidate */
    }
    return false;
}

/* =========================================================================
 * DAG integration helpers
 * ========================================================================= */

/**
 * Build a fb_dag_step_input_t[] from the current steps array and call
 * fb_exec_dag_plan().  Stores the resulting plan in seq->plan.
 * Also populates the per-step DAG-derived fields in each fb_seq_step_t.
 *
 * Returns 0 on success, -1 on failure (plan remains NULL).
 */
static int compile_via_dag(fb_op_sequence_t *seq)
{
    /* Build the input array for the DAG planner */
    fb_dag_step_input_t *inputs =
        (fb_dag_step_input_t *)calloc(seq->step_count, sizeof(*inputs));
    if (!inputs) {
        return -1;
    }

    for (uint32_t i = 0; i < seq->step_count; i++) {
        inputs[i].op_id           = seq->steps[i].op_id;
        inputs[i].constraints     = seq->steps[i].constraints;
        inputs[i].size_bytes_hint = seq->steps[i].size_bytes_hint;
    }

    /* Free any previous plan */
    fb_exec_plan_free(seq->plan);
    seq->plan = NULL;

    seq->plan = fb_exec_dag_plan(inputs, seq->step_count, seq->objective);
    free(inputs);

    if (!seq->plan) {
        return -1;
    }

    /* Populate per-step fields from the plan */
    for (uint32_t i = 0; i < seq->step_count; i++) {
        fb_seq_step_t      *s = &seq->steps[i];
        fb_dag_plan_step_t *p = &seq->plan->steps[i];

        s->selected_backend_id   = p->backend_id;
        s->planned_device_type   = p->device_type;
        s->planned_xfer_before   = p->xfer_before;

        /* Fusion: only makes sense when adjacent steps share a device
         * and the op pattern is fusible. */
        s->may_fuse_with_next = false;
        if (i + 1 < seq->step_count) {
            bool same_dev = (p->device_type == seq->plan->steps[i + 1].device_type);
            s->may_fuse_with_next =
                check_fusion(s, &seq->steps[i + 1], same_dev);
        }
    }

    return 0;
}

/**
 * Re-run the DAG planner from step @p from_step forward and update seq.
 * Called automatically from execute() when actual latency diverges
 * significantly from the plan's prediction.
 */
static void replan_from(fb_op_sequence_t *seq, uint32_t from_step)
{
    if (!seq || !seq->plan || from_step >= seq->step_count) {
        return;
    }

    /* Build input sub-array for the remaining steps */
    fb_dag_step_input_t *inputs =
        (fb_dag_step_input_t *)calloc(seq->step_count, sizeof(*inputs));
    if (!inputs) {
        return;
    }

    for (uint32_t i = 0; i < seq->step_count; i++) {
        inputs[i].op_id           = seq->steps[i].op_id;
        inputs[i].constraints     = seq->steps[i].constraints;
        inputs[i].size_bytes_hint = seq->steps[i].size_bytes_hint;
    }

    fb_exec_plan_t *new_plan = fb_exec_dag_replan(
        seq->plan, from_step, inputs, seq->step_count, seq->objective);

    free(inputs);

    if (!new_plan) {
        return;
    }

    /* Replace the plan and refresh per-step fields for remaining steps */
    fb_exec_plan_free(seq->plan);
    seq->plan = new_plan;

    for (uint32_t i = from_step; i < seq->step_count; i++) {
        fb_seq_step_t      *s = &seq->steps[i];
        fb_dag_plan_step_t *p = &new_plan->steps[i];

        s->selected_backend_id = p->backend_id;
        s->planned_device_type = p->device_type;
        s->planned_xfer_before = p->xfer_before;

        s->may_fuse_with_next = false;
        if (i + 1 < seq->step_count) {
            bool same_dev =
                (p->device_type == seq->plan->steps[i + 1].device_type);
            s->may_fuse_with_next =
                check_fusion(s, &seq->steps[i + 1], same_dev);
        }
    }
}

/* =========================================================================
 * Public API
 * ========================================================================= */

fb_op_sequence_t *fb_op_sequence_create(uint32_t max_steps)
{
    if (max_steps == 0) {
        return NULL;
    }

    fb_op_sequence_t *seq = (fb_op_sequence_t *)calloc(1, sizeof(*seq));
    if (!seq) {
        return NULL;
    }

    seq->steps = (fb_seq_step_t *)calloc(max_steps, sizeof(fb_seq_step_t));
    if (!seq->steps) {
        free(seq);
        return NULL;
    }

    seq->max_steps   = max_steps;
    seq->step_count  = 0;
    seq->is_compiled = false;
    seq->plan        = NULL;
    seq->objective   = FB_SELECT_MAXIMIZE_SPEED;   /* default */

    return seq;
}

void fb_op_sequence_free(fb_op_sequence_t *seq)
{
    if (!seq) {
        return;
    }
    fb_exec_plan_free(seq->plan);
    free(seq->steps);
    free(seq);
}

int fb_op_sequence_add(fb_op_sequence_t *seq, const fb_seq_step_t *step)
{
    if (!seq || !step) {
        return -1;
    }
    if (seq->is_compiled) {
        return -1;
    }
    if (seq->step_count >= seq->max_steps) {
        return -1;
    }

    fb_seq_step_t *dst = &seq->steps[seq->step_count];
    memcpy(dst, step, sizeof(*step));

    /* Clear all compile-time fields to prevent stale reads */
    dst->selected_backend_id = 0;
    dst->planned_device_type = FB_DEVICE_CPU;
    dst->planned_xfer_before = FB_XFER_NONE;
    dst->may_fuse_with_next  = false;

    /* Guarantee NUL-termination */
    dst->name[sizeof(dst->name) - 1] = '\0';

    seq->step_count++;
    return 0;
}

int fb_op_sequence_compile(fb_op_sequence_t *seq)
{
    if (!seq) {
        return -1;
    }
    if (seq->step_count == 0) {
        seq->is_compiled = true;
        return 0;
    }

    if (compile_via_dag(seq) != 0) {
        /* DAG planning failed (OOM, etc.) -- fall back to simple dispatch */
        for (uint32_t i = 0; i < seq->step_count; i++) {
            fb_seq_step_t *s = &seq->steps[i];
            int bid = fb_select_backend_with_constraints(
                          (int)s->op_id, &s->constraints);
            s->selected_backend_id   = (bid >= 0) ? (uint32_t)bid : 0;
            s->planned_device_type   = FB_DEVICE_CPU;
            s->planned_xfer_before   = FB_XFER_NONE;
            s->may_fuse_with_next    = false;
        }
        /* Fusion pass (fallback path -- no device info, treat all as same) */
        for (uint32_t i = 0; i + 1 < seq->step_count; i++) {
            seq->steps[i].may_fuse_with_next =
                check_fusion(&seq->steps[i], &seq->steps[i + 1], true);
        }
    }

    seq->is_compiled = true;
    return 0;
}

int fb_op_sequence_execute(fb_op_sequence_t *seq)
{
    if (!seq) {
        return -1;
    }
    if (!seq->is_compiled) {
        return -1;
    }

    for (uint32_t i = 0; i < seq->step_count; i++) {
        const fb_seq_step_t *s = &seq->steps[i];

        /* Resolve the vtable for THIS step's selected backend.
         * The DAG planner stores stable FB_BACKEND_ID_* values (not
         * snapshot-position indices), so fb_get_vtable_by_backend_id() is
         * the correct resolver here.  Falls back to the global active vtable
         * when the backend hasn't been lazily initialised yet. */
        const fb_backend_vtable_t *vtable =
            fb_get_vtable_by_backend_id(s->selected_backend_id);
        if (!vtable) {
            vtable = fb_get_active_vtable(); /* last-resort fallback */
        }

        /* Log transfer requirement.
         * A full implementation would execute the transfer here using the
         * cudaMemcpy / hipMemcpy / clEnqueueCopyBuffer APIs already present
         * in the GPU backends.  For now: surface it so the caller's exec_fn
         * can act on s->planned_xfer_before. */
        if (seq->plan && s->planned_xfer_before != FB_XFER_NONE) {
#ifdef FB_DEBUG
            printf("[exec_dag] step %u: %s transfer required before exec_fn\n",
                   i, fb_xfer_type_name(s->planned_xfer_before));
#endif
        }

        if (!s->exec_fn) {
            continue;   /* skip silently */
        }

        /* Time the step for dynamic replanning */
        float t0  = now_ns();
        int   rc  = s->exec_fn(vtable, s->user_data);
        float t1  = now_ns();

        if (rc != 0) {
            return rc;
        }

        /* Dynamic replanning: if this step ran much slower than predicted
         * and there are remaining steps, replan from the next step. */
        if (seq->plan && i + 1 < seq->step_count) {
            float predicted = seq->plan->steps[i].est_exec_ns;
            float actual_ns = t1 - t0;

            if (predicted >= FB_REPLAN_MIN_PREDICTED_NS &&
                actual_ns > FB_REPLAN_THRESHOLD * predicted) {
#ifdef FB_DEBUG
                printf("[exec_dag] step %u actual=%.0f ns  predicted=%.0f ns"
                       "  => replanning from step %u\n",
                       i, actual_ns, predicted, i + 1);
#endif
                replan_from(seq, i + 1);
            }
        }
    }

    return 0;
}

void fb_op_sequence_reset(fb_op_sequence_t *seq)
{
    if (!seq) {
        return;
    }
    fb_exec_plan_free(seq->plan);
    seq->plan        = NULL;
    seq->step_count  = 0;
    seq->is_compiled = false;
    memset(seq->steps, 0, seq->max_steps * sizeof(fb_seq_step_t));
}

/* ---- Inspection --------------------------------------------------------- */

uint32_t fb_op_sequence_step_count(const fb_op_sequence_t *seq)
{
    return seq ? seq->step_count : 0u;
}

const fb_seq_step_t *fb_op_sequence_get_step(const fb_op_sequence_t *seq,
                                              uint32_t                idx)
{
    if (!seq || idx >= seq->step_count) {
        return NULL;
    }
    return &seq->steps[idx];
}

bool fb_op_sequence_is_compiled(const fb_op_sequence_t *seq)
{
    return seq ? seq->is_compiled : false;
}

void fb_op_sequence_print(const fb_op_sequence_t *seq)
{
    if (!seq) {
        printf("[fb_op_sequence] (null)\n");
        return;
    }

    printf("[fb_op_sequence] %u/%u steps  compiled=%s  total_cost=%.0f ns\n",
           seq->step_count, seq->max_steps,
           seq->is_compiled ? "yes" : "no",
           seq->plan ? seq->plan->total_cost_ns : 0.0f);

    for (uint32_t i = 0; i < seq->step_count; i++) {
        const fb_seq_step_t *s = &seq->steps[i];
        printf("  [%02u] op_id=%-4u  backend=%-3u  device=%-5s"
               "  xfer=%-4s  exec=%.0f ns  fusion=%s  name=\"%s\"\n",
               i,
               s->op_id,
               s->selected_backend_id,
               fb_device_type_name(s->planned_device_type),
               fb_xfer_type_name(s->planned_xfer_before),
               seq->plan ? seq->plan->steps[i].est_exec_ns : 0.0f,
               s->may_fuse_with_next ? "yes" : " no",
               s->name[0] ? s->name : "(unnamed)");
    }
}

/* ---- DAG planning controls --------------------------------------------- */

void fb_op_sequence_set_objective(fb_op_sequence_t     *seq,
                                   fb_select_objective_t objective)
{
    if (seq) {
        seq->objective = objective;
    }
}

const fb_exec_plan_t *fb_op_sequence_get_plan(const fb_op_sequence_t *seq)
{
    return seq ? seq->plan : NULL;
}
