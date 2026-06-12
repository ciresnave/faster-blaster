/**
 * @file judge_profile.h
 * @brief Profile accumulator internals for the judge module.
 *
 * Provides in-memory accumulators that collect per-case results and timing
 * samples, then produce the public fb_metric_profile_t / fb_timing_profile_t /
 * fb_precision_profile_t structures declared in judge.h.
 *
 * Not part of the public API.
 *
 * @copyright Copyright (c) 2025
 * @license MIT OR Apache-2.0
 */

#ifndef FB_JUDGE_PROFILE_H
#define FB_JUDGE_PROFILE_H

#include <stdint.h>
#include <stdbool.h>
#include "judge_types.h"
#include "judge_corpus.h"           /* FB_CORPUS_TOTAL_CASES */
#include "../../include/faster-blaster/judge.h"

#ifdef __cplusplus
extern "C" {
#endif

/* Maximum test cases a single accumulator can hold per profiling pass. */
#define FB_PROFILE_ACCUM_MAX_CASES  32

/* =========================================================================
 * Metric accumulator (one per metric, e.g. direct, residual, ortho)
 * ========================================================================= */

/**
 * Collects raw digit scores and failure flags from individual test cases.
 * Call fb_metric_accum_finish() to build the final fb_metric_profile_t.
 */
typedef struct {
    double    digits[FB_PROFILE_ACCUM_MAX_CASES]; /**< Score per case (-1.0 = fatal) */
    bool      is_fatal[FB_PROFILE_ACCUM_MAX_CASES];
    bool      is_oracle_fatal[FB_PROFILE_ACCUM_MAX_CASES];
    uint32_t  count;               /**< Number of add() calls */
    uint32_t  fatal_count;         /**< Calls that produced NaN/Inf/crash */
    bool      has_oracle_fatal;    /**< Any oracle failure seen */
} fb_metric_accum_t;

/** Zero-initialise accumulator for a fresh profiling pass. */
void fb_metric_accum_init(fb_metric_accum_t *a);

/** Add one test-case result to the accumulator. */
void fb_metric_accum_add(
    fb_metric_accum_t         *a,
    const fb_judge_case_result_t *r
);

/**
 * Finalise: sort collected samples and build the accuracy curve.
 * 'out' is fully overwritten.
 */
void fb_metric_accum_finish(
    const fb_metric_accum_t *a,
    fb_metric_profile_t     *out
);

/* =========================================================================
 * Timing accumulator
 * ========================================================================= */

/**
 * Collects best-of-N timing samples (one per non-edge-case corpus entry).
 */
typedef struct {
    uint64_t  ns[FB_PROFILE_ACCUM_MAX_CASES]; /**< Per-case best timing in ns */
    uint32_t  count;
} fb_timing_accum_t;

/** Zero-initialise. */
void fb_timing_accum_init(fb_timing_accum_t *a);

/** Add one timing sample (already the minimum of several timing runs). */
void fb_timing_accum_add(fb_timing_accum_t *a, uint64_t ns);

/**
 * Finalise: compute mean, p50, p95, min, stddev.
 * 'out' is fully overwritten.
 */
void fb_timing_accum_finish(
    const fb_timing_accum_t *a,
    fb_timing_profile_t     *out
);

/* =========================================================================
 * Combined precision profile builder
 * ========================================================================= */

/**
 * Combine one metric accumulator (direct) and one timing accumulator into
 * a full fb_precision_profile_t.  Provenance fields must be filled by caller.
 *
 * Used by judge.c for the DIRECT archetype.
 */
void fb_profile_finish_direct(
    const fb_metric_accum_t  *direct_accum,
    const fb_timing_accum_t  *timing_accum,
    const fb_op_judge_meta_t *meta,
    fb_precision_profile_t   *out
);

#ifdef __cplusplus
}
#endif

#endif /* FB_JUDGE_PROFILE_H */
