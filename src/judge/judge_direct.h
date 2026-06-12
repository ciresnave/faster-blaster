/**
 * @file judge_direct.h
 * @brief FB_JUDGE_DIRECT archetype evaluator.
 *
 * Runs oracle + candidate side-by-side for operations whose correctness
 * can be measured by a single forward relative error (BLAS L1/L2/L3, direct
 * solvers before specialized solve judge).
 *
 * Not part of the public API.
 *
 * @copyright Copyright (c) 2025
 * @license MIT OR Apache-2.0
 */

#ifndef FB_JUDGE_DIRECT_H
#define FB_JUDGE_DIRECT_H

#include <stdint.h>
#include "judge_types.h"
#include "judge_corpus.h"
#include "../../include/faster-blaster/judge.h"
#include "../backends/backend_interface.h"

#ifdef __cplusplus
extern "C" {
#endif

/**
 * Run one direct-comparison judge case.
 *
 * Calls oracle (reference backend) and candidate backend with the same
 * inputs from @p tc, measures relative error, and optionally records the
 * best-of-N timing for the candidate.
 *
 * @param oracle_vtable     Reference backend vtable (must not be NULL).
 * @param candidate_vtable  Candidate backend vtable (must not be NULL).
 * @param tc                Test case; ownership not transferred.
 * @param result_out        Filled with digit score and fatal flags.
 * @param elapsed_ns_out    If non-NULL, receives best observed candidate
 *                          wall-clock time in nanoseconds.
 *
 * @return FB_JUDGE_OK on success (even if the candidate is inaccurate).
 *         FB_JUDGE_ERR_NOT_IMPL if op_id has no registered runner.
 *         FB_JUDGE_ERR_ALLOC on allocation failure.
 */
fb_judge_status_t fb_judge_run_direct_case(
    const fb_backend_vtable_t *oracle_vtable,
    const fb_backend_vtable_t *candidate_vtable,
    const fb_corpus_case_t    *tc,
    fb_judge_case_result_t    *result_out,
    uint64_t                  *elapsed_ns_out   /* nullable */
);

#ifdef __cplusplus
}
#endif

#endif /* FB_JUDGE_DIRECT_H */
