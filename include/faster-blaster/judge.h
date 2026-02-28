/**
 * @file judge.h
 * @brief faster-blaster Judge Module — public API
 *
 * The judge runs once at startup (or on backend/device change) and produces
 * precision + timing profiles for every (op, backend, device, size_class, dtype)
 * combination. These profiles feed the operation ranking tables used by the router.
 * The judge does NOT run during normal dispatch.
 *
 * Developer-facing usage: set precision requirements on an op-sequence descriptor
 * via fb_judge_query_t. The router reads pre-computed profiles to satisfy the query.
 *
 * @copyright Copyright (c) 2025
 * @license MIT OR Apache-2.0
 */

#ifndef FASTER_BLASTER_JUDGE_H
#define FASTER_BLASTER_JUDGE_H

#include <stdint.h>
#include <stdbool.h>
#include <stddef.h>   /* size_t */
#include "../benchmark_types.h"  /* fb_size_class_t */

/* For backend selection API (fb_judge_select_best_backend, FB_SELECT_* presets,
 * fb_judge_build_dispatch_table), include "judge_select.h" which builds on
 * this header.  They are kept separate to avoid circular inclusion. */

#ifdef __cplusplus
extern "C" {
#endif

/* =========================================================================
 * Version — increment when profile format changes to force re-profiling
 * ========================================================================= */

#define FB_JUDGE_MODULE_VERSION  1

/**
 * Maximum number of distinct operation IDs tracked by the judge.
 * Must match FB_JUDGE_MAX_OPERATIONS in src/judge/judge_op_ids.h.
 * The dispatch table allocated by fb_judge_build_dispatch_table() has this
 * many uint32_t slots.
 */
#define FB_JUDGE_MAX_OPERATIONS  2400u

/* =========================================================================
 * Accuracy curve
 * ========================================================================= */

/** Maximum number of (digits, pass_rate) points stored per metric profile. */
#define FB_JUDGE_MAX_CURVE_POINTS  16

/**
 * One point on the accuracy curve.
 * Semantics: "pass_rate fraction of test cases achieved at least 'digits'
 * significant decimal digits."
 */
typedef struct {
    uint8_t  digits;     /**< Number of correct decimal digits */
    float    pass_rate;  /**< Fraction of test cases [0.0, 1.0] */
} fb_digits_point_t;

/**
 * Precision profile for a single metric (e.g., reconstruction, orthogonality).
 * The curve[] array is sorted descending by digits, so curve[0] is the best
 * observed performance and guaranteed_digits is the floor.
 */
typedef struct {
    fb_digits_point_t  curve[FB_JUDGE_MAX_CURVE_POINTS];
    uint8_t            curve_len;            /**< Number of valid points in curve[] */
    uint8_t            guaranteed_digits;    /**< Digits where pass_rate == 1.0 */
    uint8_t            typical_digits;       /**< Digits at pass_rate >= 0.50 */
    uint8_t            max_observed_digits;  /**< Best digits seen on any test case */
    bool               has_fatal_failure;    /**< NaN/Inf/crash seen in any test case */
    uint32_t           test_case_count;      /**< Number of test cases run */
} fb_metric_profile_t;

/* =========================================================================
 * Timing profile (produced in the same judge pass as precision)
 * ========================================================================= */

/** Number of calls discarded as warmup per (backend, op, size_class). */
#define FB_JUDGE_WARMUP_RUNS  2

/**
 * Timing statistics for one (op, backend, device, size_class, dtype).
 * Populated from standard (non-edge-case) corpus entries only.
 * For deep-audit mode, only the candidate op call is timed; extra residual
 * computation added by the judge is excluded.
 */
typedef struct {
    uint32_t  mean_ns;        /**< Mean wall-clock time per call (nanoseconds) */
    uint32_t  p50_ns;         /**< Median */
    uint32_t  p95_ns;         /**< 95th percentile */
    uint32_t  min_ns;         /**< Best observed (steady-state throughput indicator) */
    uint32_t  stddev_ns;      /**< Standard deviation */
    uint32_t  sample_count;   /**< Calls timed (after warmup, non-edge-case only) */
} fb_timing_profile_t;

/* =========================================================================
 * Limiting metric bitmask
 * ========================================================================= */

/**
 * Indicates which metric(s) are binding the overall precision score.
 * Stored in the profile and surfaced to the router so it can distinguish
 * "values-limited" from "reconstruction-limited" implementations.
 */
typedef enum {
    FB_JUDGE_LIMIT_NONE           = 0,
    FB_JUDGE_LIMIT_VALUES         = 1 << 0, /**< Eigenvalues / singular values */
    FB_JUDGE_LIMIT_DIRECT         = 1 << 1, /**< Element-wise output (GEMM, AXPY) */
    FB_JUDGE_LIMIT_RECONSTRUCTION = 1 << 2, /**< ||A - factors|| / ||A|| */
    FB_JUDGE_LIMIT_ORTHOGONALITY  = 1 << 3, /**< ||Q^H Q - I|| / ||I|| */
    FB_JUDGE_LIMIT_RESIDUAL       = 1 << 4, /**< Solve backward error */
    FB_JUDGE_LIMIT_PAIRS          = 1 << 5, /**< Eigenpair / triplet residuals */
    FB_JUDGE_LIMIT_SUBSPACE       = 1 << 6, /**< Degenerate subspace metric */
    FB_JUDGE_LIMIT_NONCONVERGENCE = 1 << 7, /**< Algorithm did not converge */
    FB_JUDGE_LIMIT_NAN_INF        = 1 << 8, /**< NaN or Inf in output */
} fb_judge_limit_t;

/* =========================================================================
 * Full precision + timing profile
 * ========================================================================= */

/**
 * Complete profile for one (op_name, backend_id, device_id, size_class, dtype).
 * Produced by the judge in a single pass; stored on disk between runs.
 */
typedef struct {
    /* Precision metrics — only those relevant to the archetype are populated. */
    fb_metric_profile_t  values;         /**< Eigenvalues / singular values */
    fb_metric_profile_t  direct;         /**< Element-wise output (DIRECT archetype) */
    fb_metric_profile_t  reconstruction; /**< Factorization / spectral reconstruction */
    fb_metric_profile_t  orthogonality;  /**< Q^H Q ≈ I check */
    fb_metric_profile_t  residual;       /**< Solve backward error */
    fb_metric_profile_t  pairs;          /**< Eigenpair residuals (deep audit only) */
    fb_metric_profile_t  subspace;       /**< Degenerate cluster subspace metric */

    /* Timing — standard corpus only, warmup excluded. */
    fb_timing_profile_t  timing;

    /* Summary */
    uint8_t              archetype;              /**< fb_judge_archetype_t */
    uint32_t             limiting_metric;        /**< fb_judge_limit_t bitmask */
    bool                 is_deep_audit;          /**< Deep audit mode was run */
    uint8_t              oracle_max_certifiable; /**< Ceiling from PRECISION_GUARANTEES */

    /* Provenance */
    char                 op_name[32];     /**< Operation name — serialization key (e.g. "sgemm") */
    uint32_t             backend_id;
    uint32_t             device_id;
    uint8_t              size_class;   /**< fb_size_class_t */
    uint8_t              dtype;        /**< fb_precision_t */
} fb_precision_profile_t;

/* =========================================================================
 * Judge archetype
 * ========================================================================= */

typedef enum {
    FB_JUDGE_DIRECT        = 0, /**< GEMM, AXPY, COPY — unique output */
    FB_JUDGE_INDEX         = 1, /**< ISAMAX — discrete, tie-aware */
    FB_JUDGE_FACTORIZATION = 2, /**< LU, QR, Chol — reconstruction residual */
    FB_JUDGE_SOLVE         = 3, /**< GESV, GELS — backward error + conditioning */
    FB_JUDGE_SPECTRAL      = 4, /**< SYEV, GESVD — subspace + reconstruction */
} fb_judge_archetype_t;

/* =========================================================================
 * Judge query — developer-facing precision request
 * ========================================================================= */

/**
 * Metric policy presets — most developers use one of these; advanced users
 * can compose their own bitmask from fb_judge_limit_t values.
 */
#define FB_JUDGE_POLICY_ALL_PRIMARY \
    ((uint32_t)(FB_JUDGE_LIMIT_VALUES | FB_JUDGE_LIMIT_RECONSTRUCTION | \
                FB_JUDGE_LIMIT_ORTHOGONALITY))

#define FB_JUDGE_POLICY_RECONSTRUCTION_ONLY \
    ((uint32_t)(FB_JUDGE_LIMIT_RECONSTRUCTION))

#define FB_JUDGE_POLICY_VALUES_ONLY \
    ((uint32_t)(FB_JUDGE_LIMIT_VALUES))

#define FB_JUDGE_POLICY_RESIDUAL_ONLY \
    ((uint32_t)(FB_JUDGE_LIMIT_RESIDUAL))

#define FB_JUDGE_POLICY_DIRECT_ONLY \
    ((uint32_t)(FB_JUDGE_LIMIT_DIRECT))

/**
 * Precision requirement for one operation slot in a developer's op-sequence.
 *
 * Example — "I need DGEMM to be accurate to 12 digits, always":
 *   fb_judge_query_t q = {
 *       .requested_digits   = 12,
 *       .required_pass_rate = 1.0f,
 *       .required_metrics   = FB_JUDGE_POLICY_DIRECT_ONLY,
 *       .deep_audit         = false
 *   };
 *
 * Example — "I need SVD eigenvalues to 8 digits, 90% of the time":
 *   fb_judge_query_t q = {
 *       .requested_digits   = 8,
 *       .required_pass_rate = 0.90f,
 *       .required_metrics   = FB_JUDGE_POLICY_VALUES_ONLY,
 *       .deep_audit         = false
 *   };
 */
typedef struct {
    uint8_t   requested_digits;     /**< Minimum significant decimal digits required */
    float     required_pass_rate;   /**< Fraction of calls that must achieve it [0,1] */
    uint32_t  required_metrics;     /**< fb_judge_limit_t bitmask of required checks */
    bool      deep_audit;           /**< Request eigenpair/triplet residuals (slow) */
} fb_judge_query_t;

/* =========================================================================
 * Status codes
 * ========================================================================= */

typedef enum {
    FB_JUDGE_OK                  =  0,
    FB_JUDGE_ERR_NOT_INITIALIZED = -1,
    FB_JUDGE_ERR_INVALID_OP      = -2,
    FB_JUDGE_ERR_NO_PROFILE      = -3,  /**< Profile not yet computed */
    FB_JUDGE_ERR_ORACLE_FAILURE  = -4,  /**< Reference oracle returned NaN/Inf on
                                             a valid input — bug in reference */
    FB_JUDGE_ERR_ALLOC           = -5,
    FB_JUDGE_ERR_IO              = -6,  /**< Profile store read/write error */
    FB_JUDGE_ERR_NOT_IMPL        = -7,  /**< Op exists but runner not yet implemented */
} fb_judge_status_t;

/* =========================================================================
 * Public API
 * ========================================================================= */

/**
 * Initialize the judge module.
 * Must be called before any other judge function.
 * profile_dir: directory where judge profiles are persisted (created if absent).
 * Returns FB_JUDGE_OK on success.
 */
fb_judge_status_t fb_judge_init(const char* profile_dir);

/**
 * Shut down the judge module and free resources.
 */
void fb_judge_shutdown(void);

/**
 * Run the judge for a specific (op_id, backend_id, device_id, size_class, dtype).
 *
 * Calls the reference oracle and the candidate backend, computes precision and
 * timing metrics, and stores the result in the profile store.
 *
 * deep_audit: if true, also compute eigenpair/triplet residuals (slow; use once
 *             per new backend installation, not on every startup).
 *
 * Returns FB_JUDGE_OK on success.
 * Returns FB_JUDGE_ERR_ORACLE_FAILURE if the reference oracle fails on a valid
 * input — the caller must fix the reference before retrying.
 */
fb_judge_status_t fb_judge_run(
    uint32_t          op_id,
    uint32_t          backend_id,
    uint32_t          device_id,
    fb_size_class_t   size_class,
    uint8_t           dtype,
    bool              deep_audit,
    fb_precision_profile_t* profile_out  /**< Caller-allocated, filled on success */
);

/**
 * Load a previously-computed profile from the profile store.
 * Returns FB_JUDGE_ERR_NO_PROFILE if no profile exists for the given key.
 */
fb_judge_status_t fb_judge_load_profile(
    uint32_t          op_id,
    uint32_t          backend_id,
    uint32_t          device_id,
    fb_size_class_t   size_class,
    uint8_t           dtype,
    fb_precision_profile_t* profile_out
);

/**
 * Check whether a stored profile satisfies a developer's query.
 *
 * effective_digits_out (nullable): set to the weakest digit score across all
 * required metrics at the requested pass_rate.
 *
 * Returns true if the profile meets the query; false otherwise.
 */
bool fb_judge_profile_meets_query(
    const fb_precision_profile_t* profile,
    const fb_judge_query_t*       query,
    uint8_t*                      effective_digits_out
);

/**
 * Walk an accuracy curve to find the digits achievable at a given pass_rate.
 * Returns 0 if no test cases were run or the metric was not populated.
 */
uint8_t fb_judge_metric_digits_at_rate(
    const fb_metric_profile_t* metric,
    float                       required_pass_rate
);

/**
 * Return true if stored profiles for all (op_id, backend_id, device_id) tuples
 * are current (no backend changes, no corpus version bump, no module version bump).
 * If false, callers should re-run fb_judge_run for affected entries.
 */
bool fb_judge_profiles_are_current(uint32_t backend_id, uint32_t device_id);

/**
 * Invalidate all stored profiles for a backend (call when a backend is updated
 * or when the judge module version increments).
 */
void fb_judge_invalidate(uint32_t backend_id, uint32_t device_id);

/* =========================================================================
 * Internal-use helpers (also useful for auto_benchmark and testing)
 * ========================================================================= */

/**
 * Fill @p ids_out with the IDs of all backends registered via
 * fb_judge_register_backend().  At most @p max entries are written;
 * *count_out receives the actual count.
 */
void fb_judge_get_registered_ids(uint32_t *ids_out, uint32_t max,
                                  uint32_t *count_out);

/**
 * Copy the profile directory path (set by fb_judge_init()) into @p buf.
 * At most @p capacity bytes including the NUL terminator are written.
 */
void fb_judge_get_profile_dir(char *buf, size_t capacity);

/** Return true if fb_judge_init() has been called and not yet shut down. */
bool fb_judge_is_initialized(void);

/**
 * Persist @p profile to the configured profile directory.
 * Convenience wrapper so callers outside the judge module do not need
 * to include private judge_store.h.
 * Returns FB_JUDGE_OK, or FB_JUDGE_ERR_NOT_INITIALIZED / FB_JUDGE_ERR_IO.
 */
fb_judge_status_t fb_judge_save_profile(const fb_precision_profile_t *profile);

#ifdef __cplusplus
}
#endif

#endif /* FASTER_BLASTER_JUDGE_H */
