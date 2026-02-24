/**
 * @file judge_index.h
 * @brief FB_JUDGE_INDEX archetype evaluator — discrete argmax / argmin ops.
 *
 * Operations covered: ISAMAX, IDAMAX, ICAMAX, IZAMAX.
 *
 * The INDEX archetype differs from DIRECT in two ways:
 *
 *   1.  The output is a discrete integer index, not a continuous vector.
 *       "Correctness" is not a distance — it is pass/fail with tie awareness.
 *
 *   2.  Two sub-results are reported per test case:
 *       - index : did the candidate find the right element position?
 *         (16 digits = exact, tie-credit = value digits, 0 = clearly wrong)
 *       - value : how closely does |x[cand_idx]| ≈ |x[oracle_idx]|?
 *         (continuous digits from relative error of magnitudes)
 *
 * In the profile:
 *   - profile->direct  receives the index sub-results.
 *   - profile->values  receives the value sub-results.
 *
 * @copyright Copyright (c) 2025
 * @license MIT OR Apache-2.0
 */

#ifndef FB_JUDGE_INDEX_H
#define FB_JUDGE_INDEX_H

#include "judge_types.h"
#include "judge_corpus.h"
#include "../backends/backend_interface.h"

/* =========================================================================
 * Result type
 * ========================================================================= */

/**
 * Combined result for one INDEX-archetype test case.
 *
 * @member index   Pass/fail for the element position.
 *                 digits == 16 → exact match.
 *                 digits ∈ (0,16) → tie-correct (value credit).
 *                 digits == 0 → wrong (candidate missed the max).
 * @member value   Digit count measuring |x[cand_idx]| vs |x[oracle_idx]|.
 *                 Always available; valid even when index is wrong, to indicate
 *                 whether the candidate found an element of comparable magnitude.
 */
typedef struct {
    fb_judge_case_result_t  index;  /**< Discrete position result            */
    fb_judge_case_result_t  value;  /**< Magnitude agreement (continuous)    */
} fb_judge_index_result_t;

/* =========================================================================
 * Public API
 * ========================================================================= */

/**
 * Run one INDEX-archetype test case against an oracle and a candidate.
 *
 * Corpus layout expected (L1 vector operations):
 *   tc->A[0 .. n-1]  : input vector (float / double / complex per dtype)
 *   tc->n            : vector length
 *
 * Tie detection uses a threshold of 4 × machine-epsilon (dtype-appropriate).
 * Elements within this threshold are considered indistinguishable.
 *
 * For complex variants (ICAMAX / IZAMAX) the magnitude used for tie detection
 * and value comparison is the BLAS sum-of-absolutes norm:
 *     |z|₁ = |Re(z)| + |Im(z)|
 * This matches the definition used by most reference BLAS implementations.
 *
 * @param oracle          Oracle vtable (faster-blaster-reference).
 * @param cand            Candidate backend vtable.
 * @param tc              Corpus test case.
 * @param result_out      Filled with both index and value sub-results.
 * @param elapsed_ns_out  Best-of-TIMING_RUNS candidate wall time, nullable.
 * @return FB_JUDGE_OK on success.
 *         FB_JUDGE_ERR_NOT_IMPL if op not yet in the dispatch table.
 *         FB_JUDGE_ERR_ALLOC on allocation failure (rare — only timing path).
 */
fb_judge_status_t fb_judge_run_index_case(
    const fb_backend_vtable_t    *oracle,
    const fb_backend_vtable_t    *cand,
    const fb_corpus_case_t       *tc,
    fb_judge_index_result_t      *result_out,
    uint64_t                     *elapsed_ns_out);

#endif /* FB_JUDGE_INDEX_H */
