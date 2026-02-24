/**
 * @file judge_factorization.h
 * @brief FB_JUDGE_FACTORIZATION archetype evaluator — LU, Cholesky, QR.
 *
 * Operations covered (Phase 3):
 *   LU (GETRF):      SGETRF, DGETRF, CGETRF, ZGETRF
 *   Cholesky (POTRF): SPOTRF, DPOTRF, CPOTRF, ZPOTRF
 *   QR (GEQRF):      SGEQRF, DGEQRF, CGEQRF, ZGEQRF
 *
 * For each operation, two metrics are computed:
 *
 *   1. Reconstruction residual (always)
 *      For LU:       ||A - LU|| / ||A||
 *      For Cholesky: ||A - LL^H|| / ||A||
 *      For QR:       ||A - QR|| / ||A||
 *
 *   2. Orthogonality (QR only)
 *      ||Q^H Q - I|| / ||I||
 *
 * In the profile:
 *   - profile->reconstruction receives reconstruction error digits
 *   - profile->orthogonality receives orthogonality error (QR only)
 *
 * Safe canonicalization (for display only, does not affect reconstruction):
 *   - LU:       Sort diagonal of U to match oracle (for visual debugging)
 *   - Cholesky: No canonicalization (upper/lower already canonical)
 *   - QR:       No Q canonicalization (reconstruction is canonical)
 *
 * @copyright Copyright (c) 2025
 * @license MIT OR Apache-2.0
 */

#ifndef FB_JUDGE_FACTORIZATION_H
#define FB_JUDGE_FACTORIZATION_H

#include "../backends/backend_interface.h"
#include "judge_corpus.h"
#include "judge_types.h"


/* =========================================================================
 * Result type
 * ========================================================================= */

/**
 * Combined result for one FACTORIZATION-archetype test case.
 *
 * @member reconstruction  Reconstruction error digits (all factorizations).
 * @member orthogonality   Orthogonality error digits (QR only; 0 for others).
 */
typedef struct {
  fb_judge_case_result_t reconstruction;
  fb_judge_case_result_t orthogonality;
} fb_judge_factorization_result_t;

/* =========================================================================
 * Public API
 * ========================================================================= */

/**
 * Run one FACTORIZATION-archetype test case against an oracle and a candidate.
 *
 * Corpus layout expected:
 *   tc->A[0 .. m*lda-1]  : original matrix (m rows, n cols)
 *   tc->m, tc->n, tc->lda: dimensions and leading dimension
 *   tc->meta.op_id       : operation identifier (SGETRF, SPOTRF, SGEQRF, etc.)
 *
 * Both oracle and candidate modify A in-place to store the factorization.
 * The judge clones A before each call to preserve the original for comparison.
 *
 * For QR:
 *   - tau array must be allocated and passed in tc->B (size ≥ min(m, n))
 *   - Q is reconstructed via ORGQR/UNGQR (extracted from compact form)
 *
 * @param oracle          Oracle vtable (faster-blaster-reference).
 * @param cand            Candidate backend vtable.
 * @param tc              Corpus test case.
 * @param result_out      Filled with reconstruction and orthogonality results.
 * @param elapsed_ns_out  Best-of-TIMING_RUNS candidate wall time, nullable.
 * @return FB_JUDGE_OK on success.
 *         FB_JUDGE_ERR_NOT_IMPL if op not yet in the dispatch table.
 *         FB_JUDGE_ERR_ALLOC on memory allocation failure.
 */
fb_judge_status_t fb_judge_run_factorization_case(
    const fb_backend_vtable_t *oracle, const fb_backend_vtable_t *cand,
    const fb_corpus_case_t *tc, fb_judge_factorization_result_t *result_out,
    uint64_t *elapsed_ns_out);

#endif /* FB_JUDGE_FACTORIZATION_H */
