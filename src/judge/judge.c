/**
 * @file judge.c
 * @brief faster-blaster judge module — main entry points and orchestration.
 *
 * The judge module:
 *  1. Maintains a singleton (profile_dir + backend registry) initialized via
 *     fb_judge_init().
 *  2. On fb_judge_run(), generates a deterministic corpus, runs oracle +
 *     candidate through the archetype evaluator, accumulates precision and
 *     timing metrics, assembles an fb_precision_profile_t, and saves it.
 *  3. Exposes query helpers and profile-currency checks declared in judge.h.
 *
 * @copyright Copyright (c) 2025
 * @license MIT OR Apache-2.0
 */

#include "../../include/faster-blaster/judge.h"
#include "judge_types.h"
#include "judge_corpus.h"
#include "judge_metadata.h"
#include "judge_direct.h"
#include "judge_index.h"
#include "judge_factorization.h"
#include "judge_profile.h"
#include "judge_store.h"
#include "../backends/backend_interface.h"

#include <stdint.h>
#include <stdio.h>
#include <string.h>
#include <stdbool.h>
#include <stdlib.h>

/* =========================================================================
 * Module singleton
 * ========================================================================= */

/** Maximum number of simultaneously registered candidate backends. */
#define FB_JUDGE_MAX_BACKENDS  32

typedef struct {
    uint32_t                    id;
    const fb_backend_vtable_t  *vtable;
} fb_judge_backend_entry_t;

static struct {
    bool                     initialized;
    char                     profile_dir[512];
    const fb_backend_vtable_t *oracle_vtable;   /**< Reference backend (NULL = unset) */
    fb_judge_backend_entry_t  backends[FB_JUDGE_MAX_BACKENDS];
    uint32_t                  backend_count;
} fb_judge_g;

/* =========================================================================
 * Internal registration helpers (not declared in public judge.h)
 * ========================================================================= */

/**
 * Register a candidate backend vtable.  Called by plugin_init.c after a
 * backend is successfully loaded.
 */
void fb_judge_register_backend(uint32_t backend_id, const fb_backend_vtable_t *vtable)
{
    if (!fb_judge_g.initialized || !vtable)
        return;
    if (fb_judge_g.backend_count >= FB_JUDGE_MAX_BACKENDS)
        return;
    /* Remove existing entry for this id, if any. */
    for (uint32_t i = 0; i < fb_judge_g.backend_count; i++) {
        if (fb_judge_g.backends[i].id == backend_id) {
            fb_judge_g.backends[i].vtable = vtable;
            return;
        }
    }
    fb_judge_g.backends[fb_judge_g.backend_count++] =
        (fb_judge_backend_entry_t){ .id = backend_id, .vtable = vtable };
}

/**
 * Set the oracle vtable (reference backend).  The oracle must be registered
 * before fb_judge_run() is callable.
 */
void fb_judge_register_oracle(const fb_backend_vtable_t *vtable)
{
    fb_judge_g.oracle_vtable = vtable;
}

/** Look up a candidate vtable by id. Returns NULL if not found. */
static const fb_backend_vtable_t *find_candidate(uint32_t backend_id)
{
    for (uint32_t i = 0; i < fb_judge_g.backend_count; i++) {
        if (fb_judge_g.backends[i].id == backend_id)
            return fb_judge_g.backends[i].vtable;
    }
    return NULL;
}

/* =========================================================================
 * fb_judge_init / fb_judge_shutdown
 * ========================================================================= */

fb_judge_status_t fb_judge_init(const char *profile_dir)
{
    if (!profile_dir)
        return FB_JUDGE_ERR_INVALID_OP;

    memset(&fb_judge_g, 0, sizeof(fb_judge_g));
    snprintf(fb_judge_g.profile_dir, sizeof(fb_judge_g.profile_dir),
             "%s", profile_dir);
    fb_judge_g.initialized = true;
    return FB_JUDGE_OK;
}

void fb_judge_shutdown(void)
{
    memset(&fb_judge_g, 0, sizeof(fb_judge_g));
}

/* =========================================================================
 * fb_judge_run — core orchestration
 * ========================================================================= */

fb_judge_status_t fb_judge_run(
    uint32_t                op_id,
    uint32_t                backend_id,
    uint32_t                device_id,
    fb_size_class_t         size_class,
    uint8_t                 dtype,
    bool                    deep_audit,
    fb_precision_profile_t *profile_out)
{
    if (!fb_judge_g.initialized)
        return FB_JUDGE_ERR_NOT_INITIALIZED;
    if (!profile_out)
        return FB_JUDGE_ERR_INVALID_OP;

    /* Lookup vtables. */
    const fb_backend_vtable_t *oracle    = fb_judge_g.oracle_vtable;
    const fb_backend_vtable_t *candidate = find_candidate(backend_id);

    if (!oracle)
        return FB_JUDGE_ERR_NOT_INITIALIZED;   /* oracle not registered */
    if (!candidate)
        return FB_JUDGE_ERR_INVALID_OP;        /* backend not registered */

    /* Metadata. */
    const fb_op_judge_meta_t *meta = fb_judge_meta_get(op_id);
    if (!meta)
        return FB_JUDGE_ERR_INVALID_OP;

    /* Generate corpus. */
    fb_corpus_case_t cases[FB_CORPUS_TOTAL_CASES];
    fb_judge_status_t cs = fb_corpus_generate(op_id, size_class,
                                               (fb_dtype_t)dtype, cases);
    if (cs != FB_JUDGE_OK)
        return cs;

    /* Zero the output now that we have a valid input setup. */
    memset(profile_out, 0, sizeof(*profile_out));
    profile_out->op_id      = op_id;
    profile_out->backend_id = backend_id;
    profile_out->device_id  = device_id;
    profile_out->size_class = (uint8_t)size_class;
    profile_out->dtype      = dtype;
    profile_out->archetype  = (uint8_t)meta->archetype;
    profile_out->is_deep_audit = deep_audit;

    /* ---- DIRECT archetype -------------------------------------------- */
    if (meta->archetype == FB_JUDGE_DIRECT) {

        fb_metric_accum_t  direct_accum;
        fb_timing_accum_t  timing_accum;
        fb_metric_accum_init(&direct_accum);
        fb_timing_accum_init(&timing_accum);

        for (int ci = 0; ci < FB_CORPUS_TOTAL_CASES; ci++) {
            fb_judge_case_result_t result;
            uint64_t ns = 0;

            fb_judge_status_t rs = fb_judge_run_direct_case(
                oracle, candidate, &cases[ci], &result, &ns);

            if (rs == FB_JUDGE_ERR_NOT_IMPL) {
                continue;
            }
            if (rs != FB_JUDGE_OK) {
                for (int fi = ci; fi < FB_CORPUS_TOTAL_CASES; fi++)
                    fb_corpus_case_free(&cases[fi]);
                return rs;
            }

            if (result.is_oracle_fatal) {
                for (int fi = ci; fi < FB_CORPUS_TOTAL_CASES; fi++)
                    fb_corpus_case_free(&cases[fi]);
                for (int fi = 0; fi < ci; fi++)
                    fb_corpus_case_free(&cases[fi]);
                return FB_JUDGE_ERR_ORACLE_FAILURE;
            }

            fb_metric_accum_add(&direct_accum, &result);

            if (!cases[ci].meta.is_edge_case && ns > 0)
                fb_timing_accum_add(&timing_accum, ns);

            fb_corpus_case_free(&cases[ci]);
        }

        fb_profile_finish_direct(&direct_accum, &timing_accum, meta, profile_out);
        profile_out->oracle_max_certifiable =
            fb_judge_meta_oracle_ceiling(op_id, (fb_dtype_t)dtype);

    /* ---- INDEX archetype --------------------------------------------- */
    } else if (meta->archetype == FB_JUDGE_INDEX) {

        /* Two accumulators: index match → profile->direct,
         *                   value accuracy → profile->values.           */
        fb_metric_accum_t  index_accum;
        fb_metric_accum_t  values_accum;
        fb_timing_accum_t  timing_accum;
        fb_metric_accum_init(&index_accum);
        fb_metric_accum_init(&values_accum);
        fb_timing_accum_init(&timing_accum);

        for (int ci = 0; ci < FB_CORPUS_TOTAL_CASES; ci++) {
            fb_judge_index_result_t idx_res;
            uint64_t ns = 0;

            fb_judge_status_t rs = fb_judge_run_index_case(
                oracle, candidate, &cases[ci], &idx_res, &ns);

            if (rs == FB_JUDGE_ERR_NOT_IMPL) {
                continue;
            }
            if (rs != FB_JUDGE_OK) {
                for (int fi = ci; fi < FB_CORPUS_TOTAL_CASES; fi++)
                    fb_corpus_case_free(&cases[fi]);
                return rs;
            }

            /* Oracle fatal on either sub-result → halt. */
            if (idx_res.index.is_oracle_fatal || idx_res.value.is_oracle_fatal) {
                for (int fi = ci; fi < FB_CORPUS_TOTAL_CASES; fi++)
                    fb_corpus_case_free(&cases[fi]);
                for (int fi = 0; fi < ci; fi++)
                    fb_corpus_case_free(&cases[fi]);
                return FB_JUDGE_ERR_ORACLE_FAILURE;
            }

            fb_metric_accum_add(&index_accum,  &idx_res.index);
            fb_metric_accum_add(&values_accum, &idx_res.value);

            if (!cases[ci].meta.is_edge_case && ns > 0)
                fb_timing_accum_add(&timing_accum, ns);

            fb_corpus_case_free(&cases[ci]);
        }

        /* Finish primary metrics into their respective profile slots. */
        fb_metric_accum_finish(&index_accum,  &profile_out->direct);
        fb_metric_accum_finish(&values_accum, &profile_out->values);
        fb_timing_accum_finish(&timing_accum, &profile_out->timing);

        profile_out->archetype        = (uint8_t)FB_JUDGE_INDEX;
        profile_out->limiting_metric  = FB_JUDGE_LIMIT_DIRECT;
        profile_out->oracle_max_certifiable =
            fb_judge_meta_oracle_ceiling(op_id, (fb_dtype_t)dtype);

    } else if (meta->archetype == FB_JUDGE_FACTORIZATION) {

        /* Two accumulators: reconstruction → profile->reconstruction,
         *                   orthogonality  → profile->orthogonality (QR only). */
        fb_metric_accum_t  reconstruction_accum;
        fb_metric_accum_t  orthogonality_accum;
        fb_timing_accum_t  timing_accum;
        fb_metric_accum_init(&reconstruction_accum);
        fb_metric_accum_init(&orthogonality_accum);
        fb_timing_accum_init(&timing_accum);

        for (int ci = 0; ci < FB_CORPUS_TOTAL_CASES; ci++) {
            fb_judge_factorization_result_t factor_res;
            uint64_t ns = 0;

            fb_judge_status_t rs = fb_judge_run_factorization_case(
                oracle, candidate, &cases[ci], &factor_res, &ns);

            if (rs == FB_JUDGE_ERR_NOT_IMPL) {
                continue;
            }
            if (rs != FB_JUDGE_OK) {
                for (int fi = ci; fi < FB_CORPUS_TOTAL_CASES; fi++)
                    fb_corpus_case_free(&cases[fi]);
                return rs;
            }

            /* Oracle fatal on either sub-result → halt. */
            if (factor_res.reconstruction.is_oracle_fatal ||
                factor_res.orthogonality.is_oracle_fatal) {
                for (int fi = ci; fi < FB_CORPUS_TOTAL_CASES; fi++)
                    fb_corpus_case_free(&cases[fi]);
                for (int fi = 0; fi < ci; fi++)
                    fb_corpus_case_free(&cases[fi]);
                return FB_JUDGE_ERR_ORACLE_FAILURE;
            }

            fb_metric_accum_add(&reconstruction_accum,  &factor_res.reconstruction);
            fb_metric_accum_add(&orthogonality_accum,  &factor_res.orthogonality);

            if (!cases[ci].meta.is_edge_case && ns > 0)
                fb_timing_accum_add(&timing_accum, ns);

            fb_corpus_case_free(&cases[ci]);
        }

        /* Finish primary metrics into their respective profile slots. */
        fb_metric_accum_finish(&reconstruction_accum, &profile_out->reconstruction);
        fb_metric_accum_finish(&orthogonality_accum,  &profile_out->orthogonality);
        fb_timing_accum_finish(&timing_accum, &profile_out->timing);

        profile_out->archetype        = (uint8_t)FB_JUDGE_FACTORIZATION;
        profile_out->limiting_metric  = FB_JUDGE_LIMIT_RECONSTRUCTION;
        profile_out->oracle_max_certifiable =
            fb_judge_meta_oracle_ceiling(op_id, (fb_dtype_t)dtype);

    } else {
        /* Other archetypes (FACTORIZATION, SOLVE, SPECTRAL) stub for Phase 2. */
        for (int ci = 0; ci < FB_CORPUS_TOTAL_CASES; ci++)
            fb_corpus_case_free(&cases[ci]);
        return FB_JUDGE_ERR_NOT_IMPL;
    }

    /* Save profile. */
    return fb_judge_store_save(fb_judge_g.profile_dir, profile_out);
}

/* =========================================================================
 * fb_judge_load_profile
 * ========================================================================= */

fb_judge_status_t fb_judge_load_profile(
    uint32_t                op_id,
    uint32_t                backend_id,
    uint32_t                device_id,
    fb_size_class_t         size_class,
    uint8_t                 dtype,
    fb_precision_profile_t *profile_out)
{
    if (!fb_judge_g.initialized)
        return FB_JUDGE_ERR_NOT_INITIALIZED;
    if (!profile_out)
        return FB_JUDGE_ERR_INVALID_OP;

    return fb_judge_store_load(fb_judge_g.profile_dir,
                               op_id, backend_id, device_id,
                               (uint8_t)size_class, dtype,
                               profile_out);
}

/* =========================================================================
 * fb_judge_profiles_are_current / fb_judge_invalidate
 * (implements the public stubs deferred from judge_store.c)
 * ========================================================================= */

bool fb_judge_profiles_are_current(uint32_t backend_id, uint32_t device_id)
{
    if (!fb_judge_g.initialized)
        return false;
    return fb_judge_store_profiles_are_current(
        fb_judge_g.profile_dir, backend_id, device_id);
}

void fb_judge_invalidate(uint32_t backend_id, uint32_t device_id)
{
    if (!fb_judge_g.initialized)
        return;
    fb_judge_store_invalidate(fb_judge_g.profile_dir, backend_id, device_id);
}
