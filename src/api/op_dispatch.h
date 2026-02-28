/**
 * @file op_dispatch.h
 * @brief Per-operation vtable dispatch for the public BLAS API hot path.
 *
 * Provides fb_get_vtable_for_op(uint32_t op_id), an inline function used by
 * level1.c, level2.c and level3.c to select the best available backend for
 * each individual operation.
 *
 * Selection strategy
 * ------------------
 *   1. If no ranked dispatch table has been built yet (the common case until
 *      the judge/benchmark pipeline runs), fall back immediately to the
 *      globally active vtable.  This is identical to the original behaviour
 *      of fb_dispatch_get_backend(fb_dispatch_global()) and guarantees that
 *      the single-backend path has zero extra overhead.
 *
 *   2. When a ranked table is available, consult fb_select_backend_with_constraints()
 *      with NULL constraints (balanced default).  If the result is NONE or the
 *      reference backend (possible when no benchmark data exists for this op),
 *      fall back to the active vtable — never silently downgrade to the slow
 *      reference implementation.
 *
 *   3. Map the returned FB_BACKEND_ID_* to a concrete vtable via
 *      fb_get_vtable_by_backend_id(), which lazy-inits the target plugin on
 *      first use.  If the plugin cannot be initialised, fall back gracefully.
 *
 * The FB_OP_* constants used at each call site come from judge_op_ids.h
 * (included transitively via dispatch_tables.h).
 *
 * @copyright Copyright (c) 2025
 * @license MIT OR Apache-2.0
 */

#ifndef FASTER_BLASTER_OP_DISPATCH_H
#define FASTER_BLASTER_OP_DISPATCH_H

/* dispatch_tables.h brings in fb_select_backend_with_constraints(),
 * fb_dispatch_constraints_t, and (transitively) judge/judge_op_ids.h
 * for all FB_OP_* constants.                                          */
#include "../../include/dispatch_tables.h"

/* ranked_dispatch.h provides fb_op_dispatch_get_table() and
 * backend_ids.h (FB_BACKEND_ID_*, FB_BACKEND_ID_NONE, …).            */
#include "../../include/faster-blaster/ranked_dispatch.h"

/* fb_get_vtable_by_backend_id() and related registry API. */
#include "../../include/faster-blaster/backend_plugin.h"

/* fb_get_active_vtable() — used as the fallback on every path. */
#include "../../include/faster-blaster/vtable_autofill.h"

/* fb_data_recommend_device() / fb_data_record_access() — data locality. */
#include "../../include/faster-blaster/data_tracker.h"

/* NOTE: fb_backend_vtable_t (full struct) must be visible before this header
 * is included.  Including files should have already included dispatch.h or
 * backends/backend_interface.h which provides the typedef.            */

#ifdef __cplusplus
extern "C" {
#endif

/**
 * Return the best vtable for @p op_id, consulting the ranked dispatch table
 * when one is available and falling back to the globally active vtable
 * otherwise.
 *
 * This is designed to be inlined by the compiler and compiled to a single
 * function-pointer load in the common case (no ranked table built yet).
 *
 * @param op_id  FB_OP_* constant from judge/judge_op_ids.h.
 * @return       Non-NULL vtable pointer (falls back to active vtable).
 */
static inline const fb_backend_vtable_t *
fb_get_vtable_for_op(uint32_t op_id)
{
    /* Fast path: no ranked table built — use the active (best) backend. */
    if (!fb_op_dispatch_get_table())
        return fb_get_active_vtable();

    /* Ranked table available: find the best backend for this operation. */
    uint32_t bid = fb_select_backend_with_constraints(op_id, NULL);

    /* NONE means the table exists but has no data for this op.
     * REFERENCE is returned when the table falls back generically —
     * prefer the loaded optimised backend over the reference in that case. */
    if (bid == FB_BACKEND_ID_NONE || bid == FB_BACKEND_ID_REFERENCE)
        return fb_get_active_vtable();

    const fb_backend_vtable_t *vt = fb_get_vtable_by_backend_id(bid);
    return vt ? vt : fb_get_active_vtable();
}

/**
 * Variant of fb_get_vtable_for_op() that additionally records data locality
 * information and consults fb_data_recommend_device() for future device-aware
 * dispatch.
 *
 * @p inputs / @p outputs may be NULL (or @p n_in / @p n_out may be 0) when
 * the caller cannot supply pointer information; the function then behaves
 * identically to fb_get_vtable_for_op().
 *
 * The recommended device is currently advisory only — it is recorded through
 * fb_data_record_access() to keep the locality tracker up to date, but the
 * actual vtable returned is still CPU-centric until fb_dispatch_constraints_t
 * gains a preferred_device field.
 *
 * @param op_id    FB_OP_* constant from judge/judge_op_ids.h.
 * @param inputs   Array of @p n_in input data pointers (may be NULL).
 * @param n_in     Number of entries in @p inputs.
 * @param outputs  Array of @p n_out output data pointers (may be NULL).
 * @param n_out    Number of entries in @p outputs.
 * @return         Non-NULL vtable pointer (same guarantees as fb_get_vtable_for_op).
 */
static inline const fb_backend_vtable_t *
fb_get_vtable_for_op_with_locality(uint32_t op_id,
    const void **inputs,  int n_in,
    const void **outputs, int n_out)
{
    if (inputs || outputs) {
        int dev = fb_data_recommend_device(inputs,  n_in  ? n_in  : 0,
                                           outputs, n_out ? n_out : 0);
        /* Record access for each pointer so the locality map stays
         * up-to-date even before device-aware dispatch is wired in. */
        for (int i = 0; i < n_in;  i++)
            if (inputs  && inputs[i])  fb_data_record_access(inputs[i],  dev);
        for (int i = 0; i < n_out; i++)
            if (outputs && outputs[i]) fb_data_record_access(outputs[i], dev);

        /* Device locality is a *cost component*, not a preference constraint.
         * The recommended device should not be passed as a "preferred device"
         * to the selector — the right backend might still be on a different
         * device if its compute savings exceed the transfer penalty.
         *
         * For op_chain / DAG users: populate fb_dag_step_input_t.input_ptrs
         * and .output_ptrs; exec_dag.c charges fb_data_estimate_transfer_cost()
         * at DP layer 0 so each candidate backend competes on total cost
         * (ingress + compute) rather than compute alone.
         *
         * For direct level1/2/3 dispatch (this path): fb_get_vtable_for_op()
         * consults ranked dispatch tables built offline by the judge pipeline.
         * Those tables reflect measured hardware performance; per-call pointer
         * locality is recorded here so the data_tracker map stays fresh for
         * future planning by the DAG planner or a future ranked-table rebuild.
         */
        (void)dev;
    }
    return fb_get_vtable_for_op(op_id);
}

#ifdef __cplusplus
}
#endif

#endif /* FASTER_BLASTER_OP_DISPATCH_H */
