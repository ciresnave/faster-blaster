/**
 * @file op_chain.c
 * @brief Operation sequence executor implementation.
 *
 * Wires together:
 *   - FB_OP_* constant space (judge_op_ids.h)
 *   - ranked dispatch / constraint selection (dispatch_tables.h)
 *   - active vtable (vtable_autofill.h)
 *
 * See op_chain.h for the full API contract.
 */

#include <stdlib.h>
#include <string.h>
#include <stdio.h>
#include <stdint.h>
#include <stdbool.h>

#include "../include/faster-blaster/op_chain.h"

/* Include real op-id constants so fusion heuristics use the canonical values. */
#include "judge/judge_op_ids.h"

/* =========================================================================
 * Internal structure
 * ========================================================================= */

struct fb_op_sequence {
    fb_seq_step_t *steps;       /* heap-allocated flat array          */
    uint32_t       step_count;  /* how many steps have been added     */
    uint32_t       max_steps;   /* allocated capacity                 */
    bool           is_compiled; /* compile() has been called          */
};

/* =========================================================================
 * Fusion helpers
 * ========================================================================= */

/** Returns true if op_id belongs to the SGEMM / DGEMM family. */
static inline bool is_gemm_family(uint32_t op_id)
{
    return (op_id == FB_OP_SGEMM || op_id == FB_OP_DGEMM);
}

/**
 * Returns true if op_id is a scaling/add-into type that can be folded
 * into a preceding GEMM's alpha/beta parameters.
 */
static inline bool is_axpy_or_scal(uint32_t op_id)
{
    return (op_id == FB_OP_SAXPY  || op_id == FB_OP_DAXPY ||
            op_id == FB_OP_SSCAL  || op_id == FB_OP_DSCAL);
}

/**
 * Heuristic: decide whether step at index @p i may be fused with the
 * following step.
 *
 *  Rule 1 - GEMM >> AXPY/SCAL:
 *    A GEMM followed by an AXPY or SCAL on the output matrix can be
 *    expressed as a modified beta or alpha in a single fused GEMM call.
 *
 *  Rule 2 - GEMM >> GEMM:
 *    Two consecutive GEMMs on compatible shapes can sometimes be fused
 *    into a batched or tiled kernel.  We flag as potentially fusible;
 *    the caller must verify dimension compatibility.
 */
static bool check_fusion(const fb_seq_step_t *cur, const fb_seq_step_t *next)
{
    if (is_gemm_family(cur->op_id) && is_axpy_or_scal(next->op_id)) {
        return true;   /* Rule 1 */
    }
    if (is_gemm_family(cur->op_id) && is_gemm_family(next->op_id)) {
        return true;   /* Rule 2 */
    }
    return false;
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

    return seq;
}

void fb_op_sequence_free(fb_op_sequence_t *seq)
{
    if (!seq) {
        return;
    }
    free(seq->steps);
    free(seq);
}

int fb_op_sequence_add(fb_op_sequence_t *seq, const fb_seq_step_t *step)
{
    if (!seq || !step) {
        return -1;
    }
    if (seq->is_compiled) {
        /* Refuse mutation of a compiled sequence; caller must reset() first. */
        return -1;
    }
    if (seq->step_count >= seq->max_steps) {
        return -1;
    }

    fb_seq_step_t *dst = &seq->steps[seq->step_count];
    memcpy(dst, step, sizeof(*step));

    /* Clear compile-time fields so stale data is never visible. */
    dst->selected_backend_id = 0;
    dst->may_fuse_with_next  = false;

    /* Guarantee NUL-termination regardless of what the caller put in name[]. */
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
        /* Nothing to compile; mark compiled so execute() is still valid. */
        seq->is_compiled = true;
        return 0;
    }

    /* ---- Pass 1: select backend per step -------------------------------- */
    for (uint32_t i = 0; i < seq->step_count; i++) {
        fb_seq_step_t *s = &seq->steps[i];

        int backend_id = fb_select_backend_with_constraints(
                             (int)s->op_id, &s->constraints);
        s->selected_backend_id = (backend_id >= 0) ? (uint32_t)backend_id : 0;
        s->may_fuse_with_next  = false;   /* reset before fusion pass */
    }

    /* ---- Pass 2: detect adjacent fusion opportunities ------------------- */
    for (uint32_t i = 0; i + 1 < seq->step_count; i++) {
        seq->steps[i].may_fuse_with_next =
            check_fusion(&seq->steps[i], &seq->steps[i + 1]);
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

    const fb_backend_vtable_t *vtable = fb_get_active_vtable();

    for (uint32_t i = 0; i < seq->step_count; i++) {
        const fb_seq_step_t *s = &seq->steps[i];
        if (!s->exec_fn) {
            /* Step has no execution body - skip silently. */
            continue;
        }

        int rc = s->exec_fn(vtable, s->user_data);
        if (rc != 0) {
            return rc;
        }
    }

    return 0;
}

void fb_op_sequence_reset(fb_op_sequence_t *seq)
{
    if (!seq) {
        return;
    }
    seq->step_count  = 0;
    seq->is_compiled = false;
    /* Zero out the steps array to avoid any stale data reads. */
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

    printf("[fb_op_sequence] %u/%u steps, compiled=%s\n",
           seq->step_count, seq->max_steps,
           seq->is_compiled ? "yes" : "no");

    for (uint32_t i = 0; i < seq->step_count; i++) {
        const fb_seq_step_t *s = &seq->steps[i];
        printf("  [%02u] op_id=%-4u  backend_id=%-4u  fusion=%s  name=\"%s\"\n",
               i,
               s->op_id,
               s->selected_backend_id,
               s->may_fuse_with_next ? "yes" : " no",
               s->name[0] ? s->name : "(unnamed)");
    }
}
