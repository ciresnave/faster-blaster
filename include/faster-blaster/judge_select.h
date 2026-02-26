/**
 * @file judge_select.h
 * @brief Backend selection API for the faster-blaster judge module.
 *
 * Provides the developer-facing types and functions for choosing the best
 * backend for an operation (or building a whole-table dispatch map) based
 * on an explicit objective — maximize precision, maximize speed, or any
 * combination with hard constraints on the other axis.
 *
 * Typical workflow:
 *  1. Run fb_judge_run() for each {op, backend, device, size_class, dtype}
 *     tuple to produce on-disk profiles (.fbjp files).
 *  2. Call fb_judge_build_dispatch_table() once at startup (or after a new
 *     benchmarking pass) to elect a winner per operation.
 *  3. Use the resulting table to route operation calls at runtime.
 *
 * @copyright Copyright (c) 2025
 * @license MIT OR Apache-2.0
 */

#ifndef FASTER_BLASTER_JUDGE_SELECT_H
#define FASTER_BLASTER_JUDGE_SELECT_H

#include <stdint.h>
#include <stdbool.h>
#include "judge.h"       /* fb_precision_profile_t, fb_size_class_t, fb_dtype_t */

#ifdef __cplusplus
extern "C" {
#endif

/* =========================================================================
 * Selection objective
 * ========================================================================= */

/**
 * Determines how candidates are ranked among those that pass the hard
 * constraint (if any) specified in fb_select_criteria_t.
 *
 *  MAXIMIZE_PRECISION          — most digits wins; speed is a tie-breaker only.
 *  MAXIMIZE_SPEED              — lowest latency wins; digits are a tie-breaker.
 *  PRECISION_FLOOR_THEN_SPEED  — must meet min_digits@min_pass_rate; among
 *                                qualifying backends, lowest latency wins.
 *  SPEED_FLOOR_THEN_PRECISION  — must fit within max_latency_ns at
 *                                latency_size_class; among qualifying
 *                                backends, most digits wins.
 *  WEIGHTED                    — caller supplies precision_weight and
 *                                speed_weight; score is a linear combination
 *                                of normalised precision and speed.  The two
 *                                weights need not sum to 1 but both should be
 *                                non-negative.
 */
typedef enum {
    FB_SELECT_MAXIMIZE_PRECISION           = 0,
    FB_SELECT_MAXIMIZE_SPEED               = 1,
    FB_SELECT_PRECISION_FLOOR_THEN_SPEED   = 2,
    FB_SELECT_SPEED_FLOOR_THEN_PRECISION   = 3,
    FB_SELECT_WEIGHTED                     = 4,
} fb_select_objective_t;

/* =========================================================================
 * Selection criteria
 * ========================================================================= */

/**
 * Complete description of what "best" means for one operation slot.
 *
 * Not every field is used by every objective:
 *
 *  Objective                  | Required fields
 *  ---------------------------+------------------------------------------
 *  MAXIMIZE_PRECISION         | (none beyond objective)
 *  MAXIMIZE_SPEED             | (none beyond objective)
 *  PRECISION_FLOOR_THEN_SPEED | min_digits, min_pass_rate, required_metrics,
 *                             | latency_size_class
 *  SPEED_FLOOR_THEN_PRECISION | max_latency_ns, latency_size_class,
 *                             | (min_pass_rate for digit ranking)
 *  WEIGHTED                   | precision_weight, speed_weight,
 *                             | latency_size_class
 *
 * For all objectives, allow_degraded controls behaviour when no backend
 * passes the hard constraint:
 *   false → return FB_SELECT_ERR_CONSTRAINTS_NOT_MET
 *   true  → return the best available and set status FB_SELECT_WARN_DEGRADED
 */
typedef struct {
    /** How to rank candidates. */
    fb_select_objective_t  objective;

    /* ---- Precision constraint ---- */
    /** Minimum acceptable digit count (PRECISION_FLOOR_THEN_SPEED, partly WEIGHTED). */
    uint8_t   min_digits;
    /** Fraction of corpus cases that must achieve min_digits [0.0, 1.0]. */
    float     min_pass_rate;
    /** Which metric(s) to require — fb_judge_limit_t bitmask.
     *  Use FB_JUDGE_POLICY_DIRECT_ONLY for BLAS, FB_JUDGE_POLICY_ALL_PRIMARY
     *  for factorisation/spectral ops. */
    uint32_t  required_metrics;

    /* ---- Speed constraint ---- */
    /** Maximum acceptable wallclock time per call in nanoseconds.
     *  Used as hard constraint for SPEED_FLOOR_THEN_PRECISION.
     *  0 means "no constraint". */
    uint32_t  max_latency_ns;
    /** Size class whose timing profile drives the speed comparison.
     *  Typically FB_SIZE_MEDIUM for throughput-sensitive workloads;
     *  FB_SIZE_TINY for latency-sensitive loops. */
    fb_size_class_t  latency_size_class;

    /* ---- Weights for WEIGHTED objective ---- */
    float  precision_weight;   /**< Contribution of normalised digit score. */
    float  speed_weight;       /**< Contribution of normalised reciprocal latency. */

    /* ---- Fallback policy ---- */
    /** When true and no backend meets the hard constraint, elect the best
     *  available and return FB_SELECT_WARN_DEGRADED instead of an error. */
    bool  allow_degraded;
} fb_select_criteria_t;

/* =========================================================================
 * Preset criteria (non-modifiable constants)
 * ========================================================================= */

/**
 * Maximise precision; speed breaks ties.
 * allow_degraded = true (always returns something).
 */
extern const fb_select_criteria_t FB_SELECT_MAX_PRECISION;

/**
 * Maximise speed; digits break ties.
 * allow_degraded = true.
 */
extern const fb_select_criteria_t FB_SELECT_MAX_SPEED;

/**
 * Equal-weight precision+speed combination over FB_SIZE_MEDIUM.
 * allow_degraded = true.
 */
extern const fb_select_criteria_t FB_SELECT_BALANCED;

/**
 * Helper: construct a PRECISION_FLOOR_THEN_SPEED criteria.
 *
 * @param min_digits    Minimum significant decimal digits required.
 * @param min_pass_rate Fraction of corpus that must achieve them.
 * @param metrics       fb_judge_limit_t bitmask (e.g. FB_JUDGE_POLICY_DIRECT_ONLY).
 * @param size_class    Size class to compare latency on.
 */
static inline fb_select_criteria_t
fb_select_precision_floor(uint8_t min_digits, float min_pass_rate,
                           uint32_t metrics, fb_size_class_t size_class)
{
    fb_select_criteria_t c = {
        .objective          = FB_SELECT_PRECISION_FLOOR_THEN_SPEED,
        .min_digits         = min_digits,
        .min_pass_rate      = min_pass_rate,
        .required_metrics   = metrics,
        .max_latency_ns     = 0,
        .latency_size_class = size_class,
        .precision_weight   = 0.0f,
        .speed_weight       = 0.0f,
        .allow_degraded     = false,
    };
    return c;
}

/**
 * Helper: construct a SPEED_FLOOR_THEN_PRECISION criteria.
 *
 * @param max_latency_ns   Hard latency ceiling in nanoseconds.
 * @param size_class       Size class that applies for the latency check.
 * @param min_pass_rate    Pass rate used when ranking survivors by digits.
 * @param metrics          Metrics to rank digits against.
 */
static inline fb_select_criteria_t
fb_select_speed_floor(uint32_t max_latency_ns, fb_size_class_t size_class,
                      float min_pass_rate, uint32_t metrics)
{
    fb_select_criteria_t c = {
        .objective          = FB_SELECT_SPEED_FLOOR_THEN_PRECISION,
        .min_digits         = 0,
        .min_pass_rate      = min_pass_rate,
        .required_metrics   = metrics,
        .max_latency_ns     = max_latency_ns,
        .latency_size_class = size_class,
        .precision_weight   = 0.0f,
        .speed_weight       = 0.0f,
        .allow_degraded     = false,
    };
    return c;
}

/* =========================================================================
 * Per-operation override for dispatch table builds
 * ========================================================================= */

/**
 * Override the default criteria for a specific operation when calling
 * fb_judge_build_dispatch_table().
 *
 * Example — insist on 12-digit DGEMM but allow any-speed any-precision
 * for everything else:
 *   fb_select_override_t overrides[] = {
 *       { FB_OP_DGEMM, fb_select_precision_floor(12, 1.0f,
 *                          FB_JUDGE_POLICY_DIRECT_ONLY, FB_SIZE_MEDIUM) }
 *   };
 */
typedef struct {
    uint32_t              op_id;
    fb_select_criteria_t  criteria;
} fb_select_override_t;

/* =========================================================================
 * Status codes
 * ========================================================================= */

typedef enum {
    FB_SELECT_OK                     =  0,
    /** No backend fully met the hard constraint; best-effort was returned.
     *  Only possible when allow_degraded = true.  Positive value so callers
     *  can distinguish warning (> 0) from error (< 0) with a single check. */
    FB_SELECT_WARN_DEGRADED          =  1,
    /** No .fbjp profiles were found for any candidate backend. */
    FB_SELECT_ERR_NO_PROFILES        = -1,
    /** n_backends was 0 or backend_ids was NULL. */
    FB_SELECT_ERR_NO_BACKENDS        = -2,
    /** A required parameter was NULL or out of range. */
    FB_SELECT_ERR_INVALID            = -3,
    /** No backend met the hard constraint and allow_degraded was false. */
    FB_SELECT_ERR_CONSTRAINTS_NOT_MET = -4,
} fb_select_status_t;

/* =========================================================================
 * Public API
 * ========================================================================= */

/**
 * Select the best backend for one (op, device, size_class, dtype) tuple.
 *
 * Loads profiles from @p profile_dir for each backend in @p backend_ids,
 * ranks them according to @p criteria, and returns the winner.
 *
 * @param profile_dir          Directory passed to fb_judge_init().
 * @param op_id                Operation ID (FB_OP_*).
 * @param device_id            Device ordinal.
 * @param size_class           Size class to compare on.
 * @param dtype                Data type (fb_dtype_t cast to uint8_t).
 * @param criteria             Selection objective and constraints.
 * @param backend_ids          Array of candidate backend IDs.
 * @param n_backends           Length of backend_ids[].
 * @param best_backend_out     Receives the elected backend ID.
 * @param effective_digits_out (nullable) Receives the effective digit score
 *                             of the winning backend at criteria.min_pass_rate.
 * @return FB_SELECT_OK, FB_SELECT_WARN_DEGRADED, or a negative error code.
 */
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
    uint8_t                     *effective_digits_out   /* nullable */
);

/**
 * Build a complete per-operation dispatch table in a single pass.
 *
 * For every op_id in [0, FB_JUDGE_MAX_OPERATIONS), the elected backend ID
 * is written to dispatch_table_out[op_id].  If no profile exists for any
 * backend on that op, the entry is set to @p fallback_backend_id.
 *
 * @param profile_dir          Directory passed to fb_judge_init().
 * @param device_id            Device ordinal.
 * @param primary_dtype        Data type to drive the selection comparison.
 * @param default_criteria     Criteria applied to every op without an override.
 * @param overrides            (nullable) Per-op criteria overrides.
 * @param n_overrides          Length of overrides[].
 * @param backend_ids          Candidate backend IDs to consider.
 * @param n_backends           Length of backend_ids[].
 * @param fallback_backend_id  Backend to use when no profile is found.
 * @param dispatch_table_out   Caller-allocated uint32_t[FB_JUDGE_MAX_OPERATIONS].
 * @return FB_SELECT_OK, or a negative error code on fatal failure.
 *         Per-op selection warnings are silently absorbed (fallback is used).
 */
fb_select_status_t fb_judge_build_dispatch_table(
    const char                   *profile_dir,
    uint32_t                      device_id,
    uint8_t                       primary_dtype,
    const fb_select_criteria_t   *default_criteria,
    const fb_select_override_t   *overrides,           /* nullable */
    uint32_t                      n_overrides,
    const uint32_t               *backend_ids,
    uint32_t                      n_backends,
    uint32_t                      fallback_backend_id,
    uint32_t                     *dispatch_table_out
);

#ifdef __cplusplus
}
#endif

#endif /* FASTER_BLASTER_JUDGE_SELECT_H */
