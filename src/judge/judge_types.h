/**
 * @file judge_types.h
 * @brief Internal types shared across all judge module source files.
 *
 * Not part of the public API. Include judge.h for the public interface.
 *
 * @copyright Copyright (c) 2025
 * @license MIT OR Apache-2.0
 */

#ifndef FB_JUDGE_TYPES_H
#define FB_JUDGE_TYPES_H

#include <stdint.h>
#include <stdbool.h>
#include <stddef.h>
#include "../../include/faster-blaster/judge.h"

#ifdef __cplusplus
extern "C" {
#endif

/* =========================================================================
 * Precision / dtype enum
 * ========================================================================= */

/**
 * Data type for an operation's floating-point precision.
 * Values map directly to the BLAS S/D/C/Z convention and extensions.
 */
typedef enum {
    FB_DTYPE_F32  = 0,  /**< Single precision real (BLAS 's') */
    FB_DTYPE_F64  = 1,  /**< Double precision real (BLAS 'd') */
    FB_DTYPE_CF32 = 2,  /**< Single precision complex (BLAS 'c') */
    FB_DTYPE_CF64 = 3,  /**< Double precision complex (BLAS 'z') */
    FB_DTYPE_F16  = 4,  /**< Half precision real (extensions) */
    FB_DTYPE_BF16 = 5,  /**< Brain float 16 (extensions) */
    FB_DTYPE_I8   = 6,  /**< 8-bit integer (LPGEMM) */
    FB_DTYPE_I32  = 7,  /**< 32-bit integer (LPGEMM accumulators) */
    FB_DTYPE_COUNT = 8
} fb_dtype_t;

/** Machine epsilon for each dtype (0.0 for integer types). */
extern const double fb_dtype_eps[FB_DTYPE_COUNT];

/** Human-readable dtype name. */
extern const char* const fb_dtype_name[FB_DTYPE_COUNT];

/**
 * Size in bytes of one element for each dtype.
 */
extern const size_t fb_dtype_element_size[FB_DTYPE_COUNT];

/* =========================================================================
 * Output uniqueness
 * ========================================================================= */

typedef enum {
    FB_OUTPUT_UNIQUE         = 0, /**< Mathematically unique result */
    FB_OUTPUT_SIGN_FLIP      = 1, /**< Vectors may have sign/phase ambiguity */
    FB_OUTPUT_SUBSPACE       = 2, /**< Only subspace is well-defined (degenerate) */
    FB_OUTPUT_PIVOT          = 3, /**< Permutation ambiguity (LU pivoting) */
    FB_OUTPUT_CONJUGATE_PAIR = 4, /**< Real geev: conjugate pair ordering */
    FB_OUTPUT_DISCRETE       = 5, /**< Integer index output (argmax/argmin) */
} fb_output_uniqueness_t;

/* =========================================================================
 * Score policy
 * ========================================================================= */

typedef enum {
    FB_SCORE_MIN_PRIMARY          = 0, /**< min(required metrics) is overall score */
    FB_SCORE_RECONSTRUCTION_ONLY  = 1, /**< Only reconstruction matters */
    FB_SCORE_VALUES_ONLY          = 2, /**< Only value accuracy matters (rare) */
} fb_score_policy_t;

/* =========================================================================
 * Operation metadata registry entry
 * ========================================================================= */

/**
 * Metadata for one operation ID, defining how the judge evaluates it.
 * The static table fb_op_judge_table[] is indexed by op_id.
 */
typedef struct {
    fb_judge_archetype_t    archetype;
    fb_output_uniqueness_t  uniqueness;
    fb_score_policy_t       default_score_policy;
    uint32_t                primary_metrics;     /**< fb_judge_limit_t bitmask */
    uint8_t                 max_certifiable_f32; /**< Oracle ceiling, FP32 */
    uint8_t                 max_certifiable_f64; /**< Oracle ceiling, FP64 */
    float                   ortho_slack_f32;     /**< Allowed gap vs value digits */
    float                   ortho_slack_f64;
    double                  cluster_threshold_multiplier; /**< n×eps for spectral */
    bool                    is_in_place;         /**< Input must be snapshotted */
    const char*             name;                /**< Human-readable op name */
} fb_op_judge_meta_t;

/* Maximum number of tracked operations (matches benchmark_cache.c MAX_OPERATIONS). */
#define FB_JUDGE_MAX_OPERATIONS  1248

extern const fb_op_judge_meta_t fb_op_judge_table[FB_JUDGE_MAX_OPERATIONS];

/* =========================================================================
 * Corpus entry
 * ========================================================================= */

/**
 * Describes one test case in the judge corpus.
 * Corpus is generated deterministically from (op_id, size_class, dtype, seed).
 */
typedef struct {
    uint32_t  op_id;
    uint8_t   dtype;         /**< fb_dtype_t */
    uint8_t   size_class;    /**< fb_size_class_t */
    uint64_t  seed;          /**< RNG seed for deterministic reproduction */
    double    kappa_estimate; /**< Condition number estimate; -1 if not applicable */
    bool      is_degenerate_spectrum; /**< Triggers subspace judge path */
    bool      is_edge_case;           /**< Down-weighted in guaranteed_digits stats */
} fb_corpus_entry_t;

/* =========================================================================
 * Raw single-case judge result
 * ========================================================================= */

/**
 * Judge result for a single test case and a single metric.
 * The profile accumulator collects these into fb_metric_profile_t.
 */
typedef struct {
    double    digits;          /**< -log10(relative_error), or -1.0 on fatal failure */
    double    relative_error;  /**< Raw relative error value */
    bool      is_fatal;        /**< NaN, Inf, or crash */
    bool      is_oracle_fatal; /**< Oracle (reference) failed — must halt and fix */
} fb_judge_case_result_t;

/* =========================================================================
 * Platform timing helper (inline)
 * ========================================================================= */

#ifdef _WIN32
#  include <windows.h>
static inline uint64_t fb_judge_time_ns(void) {
    LARGE_INTEGER freq, count;
    QueryPerformanceFrequency(&freq);
    QueryPerformanceCounter(&count);
    return (uint64_t)((count.QuadPart * UINT64_C(1000000000)) / freq.QuadPart);
}
#else
#  include <time.h>
static inline uint64_t fb_judge_time_ns(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (uint64_t)ts.tv_sec * UINT64_C(1000000000) + (uint64_t)ts.tv_nsec;
}
#endif

/* =========================================================================
 * Convenience: safe log10 for digit computation
 * ========================================================================= */

#include <math.h>
#include <float.h>

/**
 * Convert a relative error value to a "digits correct" score.
 * Returns 0.0 on NaN/Inf input. Cap at 16 to avoid log(0) singularity.
 */
static inline double fb_judge_digits(double relative_error) {
    if (!isfinite(relative_error) || relative_error <= 0.0) {
        return (relative_error == 0.0) ? 16.0 : 0.0;
    }
    double d = -log10(relative_error);
    return (d > 16.0) ? 16.0 : (d < 0.0 ? 0.0 : d);
}

#ifdef __cplusplus
}
#endif

#endif /* FB_JUDGE_TYPES_H */
