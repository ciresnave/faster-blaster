/**
 * @file judge_corpus.h
 * @brief Test corpus generator for the judge module.
 *
 * Generates deterministic test inputs per (op_id, dtype, size_class) combination.
 * The corpus covers normal, structured, and edge-case inputs.
 *
 * @copyright Copyright (c) 2025
 * @license MIT OR Apache-2.0
 */

#ifndef FB_JUDGE_CORPUS_H
#define FB_JUDGE_CORPUS_H

#include "judge_types.h"
#include <benchmark_types.h>

#ifdef __cplusplus
extern "C" {
#endif

/* Number of standard test cases per (op, size_class, dtype). */
#define FB_CORPUS_STANDARD_CASES   8
/* Number of edge-case test cases per (op, size_class, dtype).
 * 5 edge cases:
 *   1. Extreme-scale underflow
 *   2. Extreme-scale overflow
 *   3. Near-zero alpha/beta
 *   4. Large alpha
 *   5. Degenerate spectrum (F64/CF64 only; array slot allocated for all dtypes)
 */
#define FB_CORPUS_EDGE_CASES       5
/* Total cases per configuration. */
#define FB_CORPUS_TOTAL_CASES      (FB_CORPUS_STANDARD_CASES + FB_CORPUS_EDGE_CASES)

/**
 * Categories of test inputs produced by the corpus generator.
 */
typedef enum {
    FB_CORPUS_CAT_NORMAL          = 0, /**< Random, no special structure */
    FB_CORPUS_CAT_WELL_COND       = 1, /**< κ ≤ 10 */
    FB_CORPUS_CAT_MOD_COND        = 2, /**< κ ≈ 10³–10⁶ */
    FB_CORPUS_CAT_NEAR_SINGULAR   = 3, /**< κ → 1/eps */
    FB_CORPUS_CAT_STRUCTURED      = 4, /**< SPD, triangular, banded, etc. */
    FB_CORPUS_CAT_EXTREME_SCALE   = 5, /**< Elements spanning 10 orders of magnitude */
    FB_CORPUS_CAT_RANK_DEFICIENT  = 6, /**< Near-rank-deficient */
    FB_CORPUS_CAT_DEGENERATE_SPEC = 7, /**< Repeated eigenvalues/singular values */
    FB_CORPUS_CAT_COUNT
} fb_corpus_category_t;

/**
 * Allocated buffer holding the raw data for one test case.
 * Freed with fb_corpus_case_free().
 */
typedef struct {
    fb_corpus_entry_t   meta;
    fb_corpus_category_t category;

    /* Primary matrix / vector inputs (A is always present). */
    void*   A;          /**< Main input array (matrix or vector) */
    void*   B;          /**< Second input (e.g., B in GEMM, or b in Ax=b) */
    void*   C_init;     /**< Initial value of output (alpha/beta semantics) */

    /* Scalars (stored as double, cast to dtype before use). */
    double  alpha;
    double  beta;

    /* Dimensions */
    int     m, n, k;
    int     lda, ldb, ldc;

    /* Copy of A before call — used for in-place operators and solve judge. */
    void*   A_snapshot;

    /* Expected leading-dim memory sizes in elements. */
    size_t  A_elems;
    size_t  B_elems;
    size_t  C_elems;
} fb_corpus_case_t;

/**
 * Generate the full corpus for one (op_id, size_class, dtype) triple.
 *
 * 'cases_out' must point to an array of FB_CORPUS_TOTAL_CASES fb_corpus_case_t.
 * The function fills them in order: standard cases first, edge cases last.
 *
 * Returns FB_JUDGE_OK on success.
 */
fb_judge_status_t fb_corpus_generate(
    uint32_t          op_id,
    fb_size_class_t   size_class,
    fb_dtype_t        dtype,
    fb_corpus_case_t  cases_out[FB_CORPUS_TOTAL_CASES]
);

/**
 * Free all allocations inside a fb_corpus_case_t produced by fb_corpus_generate().
 * Does NOT free 'c' itself (caller controls the array lifetime).
 */
void fb_corpus_case_free(fb_corpus_case_t* c);

/**
 * Return the representative (m, n, k) dimensions for a BLAS-like operation
 * at a given size class. k is set to 0 for vector operations.
 */
void fb_corpus_dims_for_size_class(
    fb_size_class_t size_class,
    int*  m_out,
    int*  n_out,
    int*  k_out
);

/**
 * Estimate the condition number of a freshly-generated random double matrix.
 * Returns -1.0 if not applicable (e.g., vector input).
 * Uses a cheap Gershgorin-circle bound, not a true SVD.
 */
double fb_corpus_estimate_kappa(const double* A, int n, int lda);

#ifdef __cplusplus
}
#endif

#endif /* FB_JUDGE_CORPUS_H */
