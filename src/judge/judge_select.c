/**
 * @file judge_select.c
 * @brief Backend selection and dispatch table construction.
 *
 * Implements fb_judge_select_best_backend() and fb_judge_build_dispatch_table()
 * declared in include/faster-blaster/judge_select.h.
 *
 * Scoring model
 * -------------
 * For each candidate backend we extract two scalar values from its stored
 * profile:
 *
 *   effective_digits — the weakest digit score across all required_metrics
 *                      at the caller-specified min_pass_rate.  Corresponds to
 *                      the guaranteed floor behaviour for the operation.
 *
 *   latency_ns       — timing.p50_ns for the chosen latency_size_class.
 *                      UINT32_MAX when no timing data is available.
 *
 * These are then combined according to the objective:
 *
 *   MAXIMIZE_PRECISION          sort desc by digits, asc by latency (tie)
 *   MAXIMIZE_SPEED              sort asc by latency, desc by digits (tie)
 *   PRECISION_FLOOR_THEN_SPEED  qualify: digits >= min_digits; sort asc latency
 *   SPEED_FLOOR_THEN_PRECISION  qualify: latency <= max_latency_ns; sort desc digits
 *   WEIGHTED                    score = P * (d / ceiling) + S * (min_lat / lat)
 *                               where P=precision_weight, S=speed_weight,
 *                               ceiling = oracle max-certifiable for the dtype,
 *                               min_lat = minimum latency observed across all backends
 *
 * @copyright Copyright (c) 2025
 * @license MIT OR Apache-2.0
 */

#include "../../include/faster-blaster/judge_select.h"
#include "judge_store.h"
#include "judge_types.h"
#include "judge_metadata.h"

#include <stddef.h>
#include <stdint.h>
#include <string.h>
#include <stdbool.h>

/* =========================================================================
 * Preset constant definitions
 * ========================================================================= */

const fb_select_criteria_t FB_SELECT_MAX_PRECISION = {
    .objective          = FB_SELECT_MAXIMIZE_PRECISION,
    .min_digits         = 0,
    .min_pass_rate      = 0.0f,
    .required_metrics   = (uint32_t)(FB_JUDGE_LIMIT_DIRECT |
                                     FB_JUDGE_LIMIT_VALUES  |
                                     FB_JUDGE_LIMIT_RECONSTRUCTION),
    .max_latency_ns     = 0,
    .latency_size_class = 0,    /* unused */
    .precision_weight   = 0.0f,
    .speed_weight       = 0.0f,
    .allow_degraded     = true,
};

const fb_select_criteria_t FB_SELECT_MAX_SPEED = {
    .objective          = FB_SELECT_MAXIMIZE_SPEED,
    .min_digits         = 0,
    .min_pass_rate      = 0.50f,    /* used only if speed ties */
    .required_metrics   = (uint32_t)(FB_JUDGE_LIMIT_DIRECT |
                                     FB_JUDGE_LIMIT_VALUES  |
                                     FB_JUDGE_LIMIT_RECONSTRUCTION),
    .max_latency_ns     = 0,
    .latency_size_class = FB_SIZE_MEDIUM,  /* timing driven at medium matrices */
    .precision_weight   = 0.0f,
    .speed_weight       = 0.0f,
    .allow_degraded     = true,
};

const fb_select_criteria_t FB_SELECT_BALANCED = {
    .objective          = FB_SELECT_WEIGHTED,
    .min_digits         = 0,
    .min_pass_rate      = 0.50f,
    .required_metrics   = (uint32_t)(FB_JUDGE_LIMIT_DIRECT |
                                     FB_JUDGE_LIMIT_VALUES  |
                                     FB_JUDGE_LIMIT_RECONSTRUCTION),
    .max_latency_ns     = 0,
    .latency_size_class = FB_SIZE_MEDIUM,
    .precision_weight   = 0.5f,
    .speed_weight       = 0.5f,
    .allow_degraded     = true,
};

/* =========================================================================
 * Internal helpers
 * ========================================================================= */

/* Maximum number of backends we'll score in one call.
   Exceeding this is a programming error (not a user error). */
#define FB_SELECT_MAX_BACKENDS  32u

/*
 * Metric-bitmask → profile field mapping table.
 * Used when computing effective_digits across required_metrics.
 */
typedef struct {
    uint32_t                    bit;
    size_t                      offset;  /* offsetof fb_precision_profile_t */
} metric_field_t;

#define MFIELD(bit_, member_) \
    { (uint32_t)(bit_), offsetof(fb_precision_profile_t, member_) }

static const metric_field_t k_metric_fields[] = {
    MFIELD(FB_JUDGE_LIMIT_DIRECT,         direct),
    MFIELD(FB_JUDGE_LIMIT_VALUES,         values),
    MFIELD(FB_JUDGE_LIMIT_RECONSTRUCTION, reconstruction),
    MFIELD(FB_JUDGE_LIMIT_ORTHOGONALITY,  orthogonality),
    MFIELD(FB_JUDGE_LIMIT_RESIDUAL,       residual),
    MFIELD(FB_JUDGE_LIMIT_PAIRS,          pairs),
    MFIELD(FB_JUDGE_LIMIT_SUBSPACE,       subspace),
};
#define K_METRIC_FIELD_COUNT  (sizeof(k_metric_fields) / sizeof(k_metric_fields[0]))

/*
 * Compute effective_digits: the minimum digit score across every metric
 * in required_metrics, evaluated at min_pass_rate.
 *
 * Returns 0 when no required metric has been populated.
 */
static uint8_t compute_effective_digits(const fb_precision_profile_t *p,
                                        float    min_pass_rate,
                                        uint32_t required_metrics)
{
    uint8_t min_d    = 255u;
    bool    any_seen = false;

    for (size_t i = 0; i < K_METRIC_FIELD_COUNT; i++) {
        if (!(required_metrics & k_metric_fields[i].bit))
            continue;

        const fb_metric_profile_t *m =
            (const fb_metric_profile_t *)
            ((const char *)p + k_metric_fields[i].offset);

        if (m->test_case_count == 0)
            continue;   /* metric not populated for this archetype — skip */

        any_seen = true;

        /* Hard failure in any case → 0 digits for this metric */
        if (m->has_fatal_failure) {
            /* The curve implicitly captures fatal failures via low pass rates.
             * The fb_judge_metric_digits_at_rate() call handles this correctly.
             * No early-out needed here. */
        }

        uint8_t d = fb_judge_metric_digits_at_rate(m, min_pass_rate);
        if (d < min_d) min_d = d;
    }

    return any_seen ? min_d : 0u;
}

/*
 * Per-backend scoring scratch record.
 */
typedef struct {
    uint32_t backend_id;
    bool     has_profile;
    uint8_t  effective_digits;
    uint32_t latency_ns;         /* p50 at latency_size_class; UINT32_MAX = unknown */
} backend_entry_t;

/*
 * Load profile and populate a backend_entry_t.
 * The float pass_rate and metric mask are used early to compute
 * effective_digits; everything else is filled from the profile.
 */
static void load_entry(const char           *profile_dir,
                       const char           *op_name,
                       uint32_t              backend_id,
                       uint32_t              device_id,
                       fb_size_class_t       size_class,
                       uint8_t               dtype,
                       float                 min_pass_rate,
                       uint32_t              required_metrics,
                       backend_entry_t      *out)
{
    out->backend_id       = backend_id;
    out->has_profile      = false;
    out->effective_digits = 0;
    out->latency_ns       = UINT32_MAX;

    fb_precision_profile_t profile;
    memset(&profile, 0, sizeof(profile));

    fb_judge_status_t st = fb_judge_store_load(
        profile_dir, op_name, backend_id, device_id,
        (uint8_t)size_class, dtype, &profile);

    if (st != FB_JUDGE_OK)
        return;

    out->has_profile      = true;
    out->effective_digits = compute_effective_digits(&profile,
                                                     min_pass_rate,
                                                     required_metrics);
    /* Pick latency at the comparison size class.
     * The stored size_class in the profile tells us which size_class
     * this profile was run at — must match the requested one. */
    if (profile.timing.sample_count > 0)
        out->latency_ns = profile.timing.p50_ns;
}

/* =========================================================================
 * fb_judge_select_best_backend — public API
 * ========================================================================= */

fb_select_status_t fb_judge_select_best_backend(
    const char                  *profile_dir,
    uint32_t                     op_id,
    uint32_t                     device_id,
    fb_size_class_t              size_class,
    uint8_t                      dtype,
    const fb_select_criteria_t  *criteria,
    const uint32_t              *backend_ids,
    uint32_t                     n_backends,
    uint32_t                    *best_backend_out,
    uint8_t                     *effective_digits_out)
{
    if (!profile_dir || !criteria || !backend_ids || !best_backend_out)
        return FB_SELECT_ERR_INVALID;
    if (n_backends == 0)
        return FB_SELECT_ERR_NO_BACKENDS;
    if (n_backends > FB_SELECT_MAX_BACKENDS)
        n_backends = FB_SELECT_MAX_BACKENDS;   /* silently clamp */

    /* Resolve op name from metadata. */
    const fb_op_judge_meta_t *meta = fb_judge_meta_get(op_id);
    const char *op_name = (meta && meta->name) ? meta->name : "";

    /* Effective pass_rate used when computing digit scores.
     * For MAXIMIZE_PRECISION / MAXIMIZE_SPEED we still need a pass_rate to
     * meaningfully compare digits; fall back to 0.50 if not specified. */
    float pass_rate = (criteria->min_pass_rate > 0.0f)
                      ? criteria->min_pass_rate
                      : 0.50f;

    uint32_t req_metrics = criteria->required_metrics;
    if (req_metrics == 0)
        req_metrics = (uint32_t)FB_JUDGE_LIMIT_DIRECT; /* safe default */

    /* Load all profiles. */
    backend_entry_t entries[FB_SELECT_MAX_BACKENDS];
    uint32_t n_found = 0;

    for (uint32_t i = 0; i < n_backends; i++) {
        load_entry(profile_dir, op_name, backend_ids[i],
                   device_id, size_class, dtype,
                   pass_rate, req_metrics, &entries[i]);
        if (entries[i].has_profile)
            n_found++;
    }

    if (n_found == 0)
        return FB_SELECT_ERR_NO_PROFILES;

    /* ---------------------------------------------------------------- */
    /* Rank according to objective.                                      */
    /* ---------------------------------------------------------------- */

    uint32_t best_idx       = UINT32_MAX;
    uint8_t  best_digits    = 0;
    uint32_t best_latency   = UINT32_MAX;
    bool     any_qualifying = false;

    switch (criteria->objective) {

    /* ---- MAXIMIZE_PRECISION: most digits first, latency as tie-breaker ---- */
    case FB_SELECT_MAXIMIZE_PRECISION:
        for (uint32_t i = 0; i < n_backends; i++) {
            if (!entries[i].has_profile) continue;
            uint8_t  d = entries[i].effective_digits;
            uint32_t l = entries[i].latency_ns;
            if (best_idx == UINT32_MAX ||
                d > best_digits ||
                (d == best_digits && l < best_latency))
            {
                best_idx     = i;
                best_digits  = d;
                best_latency = l;
            }
        }
        break;

    /* ---- MAXIMIZE_SPEED: lowest latency first, digits as tie-breaker ---- */
    case FB_SELECT_MAXIMIZE_SPEED:
        for (uint32_t i = 0; i < n_backends; i++) {
            if (!entries[i].has_profile) continue;
            uint8_t  d = entries[i].effective_digits;
            uint32_t l = entries[i].latency_ns;
            if (best_idx == UINT32_MAX ||
                l < best_latency ||
                (l == best_latency && d > best_digits))
            {
                best_idx     = i;
                best_digits  = d;
                best_latency = l;
            }
        }
        break;

    /* ---- PRECISION_FLOOR_THEN_SPEED ---- */
    case FB_SELECT_PRECISION_FLOOR_THEN_SPEED:
        /* Pass 1: find fastest among qualifying backends. */
        for (uint32_t i = 0; i < n_backends; i++) {
            if (!entries[i].has_profile) continue;
            if (entries[i].effective_digits < criteria->min_digits) continue;
            any_qualifying = true;
            uint32_t l = entries[i].latency_ns;
            uint8_t  d = entries[i].effective_digits;
            if (best_idx == UINT32_MAX ||
                l < best_latency ||
                (l == best_latency && d > best_digits))
            {
                best_idx     = i;
                best_digits  = d;
                best_latency = l;
            }
        }
        /* Pass 2: degraded — largest digits regardless of floor. */
        if (!any_qualifying) {
            if (!criteria->allow_degraded)
                return FB_SELECT_ERR_CONSTRAINTS_NOT_MET;
            /* Re-run as MAXIMIZE_PRECISION. */
            for (uint32_t i = 0; i < n_backends; i++) {
                if (!entries[i].has_profile) continue;
                uint8_t  d = entries[i].effective_digits;
                uint32_t l = entries[i].latency_ns;
                if (best_idx == UINT32_MAX ||
                    d > best_digits ||
                    (d == best_digits && l < best_latency))
                {
                    best_idx     = i;
                    best_digits  = d;
                    best_latency = l;
                }
            }
        }
        break;

    /* ---- SPEED_FLOOR_THEN_PRECISION ---- */
    case FB_SELECT_SPEED_FLOOR_THEN_PRECISION:
        /* Pass 1: find most precise among backends meeting the latency budget. */
        for (uint32_t i = 0; i < n_backends; i++) {
            if (!entries[i].has_profile) continue;
            if (entries[i].latency_ns > criteria->max_latency_ns &&
                entries[i].latency_ns != UINT32_MAX) continue;
            /* Backends with no timing data (UINT32_MAX) are excluded — we
             * cannot verify they meet the latency constraint. */
            if (entries[i].latency_ns == UINT32_MAX &&
                criteria->max_latency_ns != 0) continue;
            any_qualifying = true;
            uint8_t  d = entries[i].effective_digits;
            uint32_t l = entries[i].latency_ns;
            if (best_idx == UINT32_MAX ||
                d > best_digits ||
                (d == best_digits && l < best_latency))
            {
                best_idx     = i;
                best_digits  = d;
                best_latency = l;
            }
        }
        /* Pass 2: degraded — fastest regardless of constraint. */
        if (!any_qualifying) {
            if (!criteria->allow_degraded)
                return FB_SELECT_ERR_CONSTRAINTS_NOT_MET;
            for (uint32_t i = 0; i < n_backends; i++) {
                if (!entries[i].has_profile) continue;
                uint32_t l = entries[i].latency_ns;
                uint8_t  d = entries[i].effective_digits;
                if (best_idx == UINT32_MAX ||
                    l < best_latency ||
                    (l == best_latency && d > best_digits))
                {
                    best_idx     = i;
                    best_digits  = d;
                    best_latency = l;
                }
            }
        }
        break;

    /* ---- WEIGHTED ---- */
    case FB_SELECT_WEIGHTED: {
        /* Normalise:
         *   precision: digits / oracle_ceiling   (oracle ceiling from metadata)
         *   speed:     min_latency / latency      (min across loaded backends)
         *
         * A backend with no timing data contributes 0 for speed. */
        uint8_t ceiling = fb_judge_meta_oracle_ceiling(op_id, (fb_dtype_t)dtype);
        if (ceiling == 0) ceiling = 16;  /* fallback: FP64 practical ceiling */

        /* Find minimum latency across backends that have timing data. */
        uint32_t min_lat = UINT32_MAX;
        for (uint32_t i = 0; i < n_backends; i++) {
            if (!entries[i].has_profile) continue;
            if (entries[i].latency_ns < min_lat)
                min_lat = entries[i].latency_ns;
        }

        float pw = criteria->precision_weight;
        float sw = criteria->speed_weight;
        float best_score = -1.0f;

        for (uint32_t i = 0; i < n_backends; i++) {
            if (!entries[i].has_profile) continue;

            float prec_score  = (float)entries[i].effective_digits / (float)ceiling;
            float speed_score = 0.0f;
            if (entries[i].latency_ns < UINT32_MAX && entries[i].latency_ns > 0 &&
                min_lat < UINT32_MAX)
            {
                speed_score = (float)min_lat / (float)entries[i].latency_ns;
            }

            float score = pw * prec_score + sw * speed_score;
            if (best_idx == UINT32_MAX || score > best_score) {
                best_idx     = i;
                best_score   = score;
                best_digits  = entries[i].effective_digits;
                best_latency = entries[i].latency_ns;
            }
        }
        break;
    }

    default:
        return FB_SELECT_ERR_INVALID;
    }

    if (best_idx == UINT32_MAX)
        return FB_SELECT_ERR_NO_PROFILES;   /* no profile had data */

    *best_backend_out = entries[best_idx].backend_id;
    if (effective_digits_out)
        *effective_digits_out = entries[best_idx].effective_digits;

    /* Report degraded if applicable (PRECISION_FLOOR or SPEED_FLOOR paths). */
    if ((criteria->objective == FB_SELECT_PRECISION_FLOOR_THEN_SPEED ||
         criteria->objective == FB_SELECT_SPEED_FLOOR_THEN_PRECISION) &&
        !any_qualifying)
    {
        return FB_SELECT_WARN_DEGRADED;
    }

    return FB_SELECT_OK;
}

/* =========================================================================
 * fb_judge_build_dispatch_table — public API
 * ========================================================================= */

fb_select_status_t fb_judge_build_dispatch_table(
    const char                   *profile_dir,
    uint32_t                      device_id,
    uint8_t                       primary_dtype,
    const fb_select_criteria_t   *default_criteria,
    const fb_select_override_t   *overrides,
    uint32_t                      n_overrides,
    const uint32_t               *backend_ids,
    uint32_t                      n_backends,
    uint32_t                      fallback_backend_id,
    uint32_t                     *dispatch_table_out)
{
    if (!profile_dir || !default_criteria || !backend_ids || !dispatch_table_out)
        return FB_SELECT_ERR_INVALID;
    if (n_backends == 0)
        return FB_SELECT_ERR_NO_BACKENDS;

    /* Build a sorted array of override op_ids for binary-search or linear scan.
     * n_overrides is expected to be small (< 100) so linear scan is fine. */

    for (uint32_t op = 0; op < (uint32_t)FB_JUDGE_MAX_OPERATIONS; op++) {

        /* Skip ops with no metadata name — nothing to profile. */
        const fb_op_judge_meta_t *meta = fb_judge_meta_get(op);
        if (!meta || !meta->name || meta->name[0] == '\0') {
            dispatch_table_out[op] = fallback_backend_id;
            continue;
        }

        /* Choose criteria: override if one exists for this op, else default. */
        const fb_select_criteria_t *crit = default_criteria;
        if (overrides) {
            for (uint32_t oi = 0; oi < n_overrides; oi++) {
                if (overrides[oi].op_id == op) {
                    crit = &overrides[oi].criteria;
                    break;
                }
            }
        }

        uint32_t best   = fallback_backend_id;
        uint8_t  digits = 0;

        /* Use the latency_size_class from criteria for the selection pass. */
        fb_size_class_t sc = crit->latency_size_class;

        fb_select_status_t st = fb_judge_select_best_backend(
            profile_dir, op, device_id, sc, primary_dtype,
            crit, backend_ids, n_backends, &best, &digits);

        /* Any result (including WARN_DEGRADED) gives a valid backend.
         * Only hard errors fall through to the fallback. */
        if (st < 0)
            best = fallback_backend_id;

        dispatch_table_out[op] = best;
    }

    return FB_SELECT_OK;
}
