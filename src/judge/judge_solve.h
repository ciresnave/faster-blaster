/**
 * FB_JUDGE_SOLVE — Solver operation evaluation.
 *
 * Covers: GESV, POSV, GELS, GETRS, POTRS and driver variants.
 *
 * Metrics:
 *   - residual: backward error η = ||b - Ax|| / (||A|| ||x|| + ||b||)
 *   - condition: kappa_estimate from factorization or from condition number routine
 *
 * Phase 4: Residual accuracy evaluation for linear system solutions.
 */

#ifndef FB_JUDGE_SOLVE_H
#define FB_JUDGE_SOLVE_H

#include "judge_types.h"
#include "judge_corpus.h"
#include "backend_interface.h"

#ifdef __cplusplus
extern "C" {
#endif

/**
 * Result structure for solve operation evaluation.
 *
 * For a solve op like GESV or GELS, we evaluate:
 *   1. Residual: backward error ||b - A*x|| / (||A|| ||x|| + ||b||)
 *      - This is the primary metric; stored in result.residual
 *   2. Condition number: estimate of κ(A) given to router for filtering
 *      - Stored separately; not a profile metric (informational only)
 *
 * The orthogonality field (for consistency with factorization results) is set to
 * 16 digits (N/A for solve operations) and carries no useful information.
 */
typedef struct {
    fb_judge_case_result_t residual;      /**< Primary: backward error ||b - A*x|| */
    fb_judge_case_result_t orthogonality; /**< Always 16 digits (N/A for solve); consistency with factorization. */
    double                 kappa_estimate; /**< Condition number estimate; used to filter "ill-conditioned" cases. */
} fb_judge_solve_result_t;

/**
 * fb_judge_run_solve_case — Evaluate one solve operation against reference.
 *
 * Generates or uses provided data (A, b, x from corpus), runs both oracle and
 * candidate backends, computes residual, and fills result_out.
 *
 * @param oracle      Reference backend vtable (faster-blaster-reference)
 * @param cand        Candidate backend vtable
 * @param tc          Test case (corpus entry providing seed, dtype, sizes)
 * @param result_out  Output: residual metrics and condition estimate
 * @param ns_out      Output: wall-clock nanoseconds for candidate call (warmup excluded)
 *
 * @return FB_JUDGE_OK on success
 *         FB_JUDGE_ERR_NOT_IMPL if operation not yet implemented
 *         FB_JUDGE_ERR_ORACLE_FAILURE if oracle() returns non-zero info or NaN/Inf
 *         FB_JUDGE_ERR_INVALID_OP if invalid inputs
 */
fb_judge_status_t fb_judge_run_solve_case(
    const fb_backend_vtable_t *oracle,
    const fb_backend_vtable_t *cand,
    const fb_corpus_case_t    *tc,
    fb_judge_solve_result_t   *result_out,
    uint64_t                  *ns_out
);

#ifdef __cplusplus
}
#endif

#endif /* FB_JUDGE_SOLVE_H */
