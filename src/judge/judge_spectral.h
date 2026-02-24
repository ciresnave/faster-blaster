/**
 * FB_JUDGE_SPECTRAL — Eigenvalue and singular value problem evaluation.
 *
 * Covers: SYEV, HEEV, GESVD, GESDD, GEEV and expert driver variants.
 * Most complex archetype with 5-metric stack.
 *
 * Phase 5: Comprehensive evaluation of spectral decomposition operations.
 */

#ifndef FB_JUDGE_SPECTRAL_H
#define FB_JUDGE_SPECTRAL_H

#include "backend_interface.h"
#include "judge_corpus.h"
#include "judge_types.h"


#ifdef __cplusplus
extern "C" {
#endif

/**
 * Result structure for spectral operation evaluation.
 *
 * Spectral operations (eigenvalue, SVD, generalized eigenvalue) have multiple
 * accuracy dimensions that must be tracked independently:
 *
 *   1. Values (primary) — eigenvalue/singular value accuracy
 *      digits = -log10( ||λ_candidate - λ_oracle|| / ||λ_oracle|| )
 *
 *   2. Reconstruction — factorization quality
 *      For SYEV: ||A - V*Λ*V^H|| / ||A||
 *      For GESVD: ||A - U*Σ*V^H|| / ||A||
 *      For GEEV: ||A - V*Λ*V^{-1}|| / ||A||
 *
 *   3. Orthogonality — factor unitarity (SYEV/GESVD only)
 *      ||V^H*V - I|| / ||I|| (symmetric eigenvectors)
 *      ||U^H*U - I|| / ||I|| (SVD left singular vectors)
 *      For GEEV: N/A (right eigenvectors not necessarily orthogonal)
 *
 *   4. Subspace (conditional) — for degenerate clusters only
 *      Comparison of subspace spanned by cluster vs. oracle subspace
 *
 *   5. Pairs (deep audit) — eigenpair/singular-triplet residuals
 *      ||A*v - λ*v|| / (||A|| ||v||) for each eigenpair
 *      Computed only when deep_audit flag is set
 *
 * Safe canonicalization (for display, doesn't affect metrics):
 *   - Values: sort by magnitude (for real); real/conjugate pairs ordered by
 * imag part
 *   - Vectors: sign/phase alignment only for well-separated clusters
 *   - Always uses relative gap threshold to detect "separate" vs "clustered"
 */
typedef struct {
  fb_judge_case_result_t
      values; /**< Eigenvalue/singular value accuracy (primary) */
  fb_judge_case_result_t reconstruction; /**< Factorization residual */
  fb_judge_case_result_t
      orthogonality; /**< Factor unitarity (0 for general eigenvalue) */
  fb_judge_case_result_t subspace; /**< Subspace comparison (degenerate only) */
  fb_judge_case_result_t pairs;    /**< Eigenpair residuals (deep audit only) */
} fb_judge_spectral_result_t;

/**
 * fb_judge_run_spectral_case — Evaluate one spectral operation against
 * reference.
 *
 * Covers symmetric eigenvalue (SYEV/HEEV), SVD (GESVD/GESDD), and general
 * eigenvalue (GEEV) operations. Computes full 5-metric accuracy profile with
 * cluster detection and optional deep audit for eigenpair residuals.
 *
 * @param oracle      Reference backend vtable (faster-blaster-reference)
 * @param cand        Candidate backend vtable
 * @param tc          Test case (corpus entry providing dtype, matrix, seed)
 * @param result_out  Output: 5-metric result structure
 * @param ns_out      Output: wall-clock nanoseconds for candidate call
 *
 * @return FB_JUDGE_OK on success
 *         FB_JUDGE_ERR_NOT_IMPL if operation not yet implemented
 *         FB_JUDGE_ERR_ORACLE_FAILURE if oracle returns non-zero info or
 * NaN/Inf FB_JUDGE_ERR_INVALID_OP if invalid inputs
 */
fb_judge_status_t fb_judge_run_spectral_case(
    const fb_backend_vtable_t *oracle, const fb_backend_vtable_t *cand,
    const fb_corpus_case_t *tc, fb_judge_spectral_result_t *result_out,
    uint64_t *ns_out);

#ifdef __cplusplus
}
#endif

#endif /* FB_JUDGE_SPECTRAL_H */
