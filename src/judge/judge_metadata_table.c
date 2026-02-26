/**
 * @file judge_metadata_table.c
 * @brief Static metadata registry for all 1248 operations.
 *
 * Phase 1 fills in BLAS Level 1 and Level 3 entries with precise
 * oracle‐ceiling values derived from PRECISION_GUARANTEES.md.
 * All other entries use conservative Phase 1 defaults and will be
 * refined as later judge archetypes are implemented.
 *
 * @copyright Copyright (c) 2025
 * @license MIT OR Apache-2.0
 */

#include "judge_types.h"
#include "judge_op_ids.h"

/* =========================================================================
 * Convenience macros for table initialisation
 *
 * Usage:
 *   DIRECT_ENTRY(op_id, max_f32, max_f64)
 *   INDEX_ENTRY(op_id)
 *   FACTOR_ENTRY(op_id, max_f32, max_f64)
 *   SOLVE_ENTRY(op_id, max_f32, max_f64)
 *   SPECTRAL_ENTRY(op_id, max_f32, max_f64)
 *   DEFAULT_ENTRY(op_id)          — conservative fallback
 * ========================================================================= */

#define _COMMON_FIELDS(id_)  \
    .name  = #id_,                        \
    .is_in_place = false,                 \
    .ortho_slack_f32 = 1e-5f,             \
    .ortho_slack_f64 = 1e-10,             \
    .cluster_threshold_multiplier = 4.0

#define DIRECT_ENTRY(id_, f32_ceil_, f64_ceil_) [id_] = { \
    _COMMON_FIELDS(id_),                                    \
    .archetype             = FB_JUDGE_DIRECT,               \
    .uniqueness            = FB_OUTPUT_UNIQUE,              \
    .default_score_policy  = FB_SCORE_MIN_PRIMARY,          \
    .primary_metrics       = FB_JUDGE_LIMIT_DIRECT,         \
    .max_certifiable_f32   = (f32_ceil_),                   \
    .max_certifiable_f64   = (f64_ceil_),                   \
}

#define INDEX_ENTRY(id_) [id_] = {                          \
    _COMMON_FIELDS(id_),                                    \
    .archetype             = FB_JUDGE_INDEX,                \
    .uniqueness            = FB_OUTPUT_DISCRETE,            \
    .default_score_policy  = FB_SCORE_MIN_PRIMARY,          \
    .primary_metrics       = FB_JUDGE_LIMIT_DIRECT,         \
    .max_certifiable_f32   = 0,  /* exact integer result */ \
    .max_certifiable_f64   = 0,                             \
}

#define FACTOR_ENTRY(id_, f32_ceil_, f64_ceil_) [id_] = {  \
    _COMMON_FIELDS(id_),                                    \
    .archetype             = FB_JUDGE_FACTORIZATION,        \
    .uniqueness            = FB_OUTPUT_PIVOT,               \
    .default_score_policy  = FB_SCORE_RECONSTRUCTION_ONLY,  \
    .primary_metrics       = FB_JUDGE_LIMIT_RECONSTRUCTION, \
    .max_certifiable_f32   = (f32_ceil_),                   \
    .max_certifiable_f64   = (f64_ceil_),                   \
}

#define SOLVE_ENTRY(id_, f32_ceil_, f64_ceil_) [id_] = {   \
    _COMMON_FIELDS(id_),                                    \
    .archetype             = FB_JUDGE_SOLVE,                \
    .uniqueness            = FB_OUTPUT_UNIQUE,              \
    .default_score_policy  = FB_SCORE_MIN_PRIMARY,          \
    .primary_metrics       = FB_JUDGE_LIMIT_RESIDUAL,       \
    .max_certifiable_f32   = (f32_ceil_),                   \
    .max_certifiable_f64   = (f64_ceil_),                   \
}

#define SPECTRAL_ENTRY(id_, f32_ceil_, f64_ceil_) [id_] = { \
    _COMMON_FIELDS(id_),                                     \
    .archetype             = FB_JUDGE_SPECTRAL,              \
    .uniqueness            = FB_OUTPUT_SIGN_FLIP,            \
    .default_score_policy  = FB_SCORE_MIN_PRIMARY,           \
    .primary_metrics       = FB_JUDGE_LIMIT_VALUES           \
                           | FB_JUDGE_LIMIT_ORTHOGONALITY,   \
    .max_certifiable_f32   = (f32_ceil_),                    \
    .max_certifiable_f64   = (f64_ceil_),                    \
}

/* Conservative Phase 1 defaults (PRECISION_GUARANTEES.md §4) */
#define DEFAULT_ENTRY(id_) DIRECT_ENTRY(id_, 4, 10)

/* =========================================================================
 * Table — FB_JUDGE_MAX_OPERATIONS entries
 *
 * Designated initialisers; C allows partial initialisation of arrays.
 * Unspecified slots are zero‐initialised (archetype==0 == FB_ARCHETYPE_DIRECT).
 * That is safe because fb_judge_meta_get() patches zeros at lookup time to
 * the conservative defaults defined in judge_metadata.c.
 * ========================================================================= */

const fb_op_judge_meta_t fb_op_judge_table[FB_JUDGE_MAX_OPERATIONS] = {

    /* -----------------------------------------------------------------
     * BLAS Level 1 — element‐wise vector operations
     *
     * max_certifiable_f64 = 13 (Kahan in dot; simple elementwise
     *   operations roundtrip within 1 ULP, so ~7 for f32, ~15 for f64,
     *   but conservative estimate cited in PRECISION_GUARANTEES §2 is 6/13)
     * ----------------------------------------------------------------- */

    /* AXPY */
    DIRECT_ENTRY(FB_OP_SAXPY, 6, 13),
    DIRECT_ENTRY(FB_OP_DAXPY, 6, 13),
    DIRECT_ENTRY(FB_OP_CAXPY, 6, 13),
    DIRECT_ENTRY(FB_OP_ZAXPY, 6, 13),

    /* SCAL */
    DIRECT_ENTRY(FB_OP_SSCAL, 6, 13),
    DIRECT_ENTRY(FB_OP_DSCAL, 6, 13),
    DIRECT_ENTRY(FB_OP_CSCAL, 6, 13),
    DIRECT_ENTRY(FB_OP_ZSCAL, 6, 13),
    DIRECT_ENTRY(FB_OP_CSSCAL, 6, 13),
    DIRECT_ENTRY(FB_OP_ZDSCAL, 6, 13),

    /* COPY */
    DIRECT_ENTRY(FB_OP_SCOPY, 7,
                 15), /* exact copy → near‐infinite but cap at type max */
    DIRECT_ENTRY(FB_OP_DCOPY, 7, 15),
    DIRECT_ENTRY(FB_OP_CCOPY, 7, 15),
    DIRECT_ENTRY(FB_OP_ZCOPY, 7, 15),

    /* SWAP */
    DIRECT_ENTRY(FB_OP_SSWAP, 7, 15),
    DIRECT_ENTRY(FB_OP_DSWAP, 7, 15),
    DIRECT_ENTRY(FB_OP_CSWAP, 7, 15),
    DIRECT_ENTRY(FB_OP_ZSWAP, 7, 15),

    /* DOT — Kahan accumulation → f32=5, f64=12 (PRECISION_GUARANTEES §2.1) */
    DIRECT_ENTRY(FB_OP_SDOT, 5, 12),
    DIRECT_ENTRY(FB_OP_DDOT, 5, 12),
    DIRECT_ENTRY(FB_OP_CDOTC, 5, 12),
    DIRECT_ENTRY(FB_OP_CDOTU, 5, 12),
    DIRECT_ENTRY(FB_OP_ZDOTC, 5, 12),
    DIRECT_ENTRY(FB_OP_ZDOTU, 5, 12),
    DIRECT_ENTRY(FB_OP_SDSDOT, 5, 12),
    DIRECT_ENTRY(FB_OP_DSDOT, 5, 12),

    /* ASUM — scaled accumulation → f32=4, f64=10 (PRECISION_GUARANTEES §2.3) */
    DIRECT_ENTRY(FB_OP_SASUM, 4, 10),
    DIRECT_ENTRY(FB_OP_DASUM, 4, 10),
    DIRECT_ENTRY(FB_OP_SCASUM, 4, 10),
    DIRECT_ENTRY(FB_OP_DZASUM, 4, 10),

    /* NRM2 — scaled accumulation → f32=4, f64=10 (PRECISION_GUARANTEES §2.2) */
    DIRECT_ENTRY(FB_OP_SNRM2, 4, 10),
    DIRECT_ENTRY(FB_OP_DNRM2, 4, 10),
    DIRECT_ENTRY(FB_OP_SCNRM2, 4, 10),
    DIRECT_ENTRY(FB_OP_DZNRM2, 4, 10),

    /* AMAX index — discrete (exact) result */
    INDEX_ENTRY(FB_OP_ISAMAX),
    INDEX_ENTRY(FB_OP_IDAMAX),
    INDEX_ENTRY(FB_OP_ICAMAX),
    INDEX_ENTRY(FB_OP_IZAMAX),

    /* ROT / ROTG — simple Givens arithmetic, f32=6 / f64=13 */
    DIRECT_ENTRY(FB_OP_SROT, 6, 13),
    DIRECT_ENTRY(FB_OP_DROT, 6, 13),
    DIRECT_ENTRY(FB_OP_CROT, 6, 13),
    DIRECT_ENTRY(FB_OP_ZROT, 6, 13),
    DIRECT_ENTRY(FB_OP_ZDROT, 6, 13),
    DIRECT_ENTRY(FB_OP_SROTG, 5, 12),
    DIRECT_ENTRY(FB_OP_DROTG, 5, 12),
    DIRECT_ENTRY(FB_OP_SROTM, 6, 13),
    DIRECT_ENTRY(FB_OP_DROTM, 6, 13),
    DIRECT_ENTRY(FB_OP_SROTMG, 5, 12),

    /* -----------------------------------------------------------------
     * BLAS Level 2 — matrix‐vector operations
     *
     * Inner products over n terms → treat as DOT‐level quality: f32=5, f64=12.
     * Rank‑1 update (GER, HER) inherits same bounds.
     * ----------------------------------------------------------------- */

    /* GEMV */
    DIRECT_ENTRY(FB_OP_SGEMV, 5, 12),
    DIRECT_ENTRY(FB_OP_DGEMV, 5, 12),
    DIRECT_ENTRY(FB_OP_CGEMV, 5, 12),
    DIRECT_ENTRY(FB_OP_ZGEMV, 5, 12),

    /* SYMV / HEMV */
    DIRECT_ENTRY(FB_OP_SSYMV, 5, 12),
    DIRECT_ENTRY(FB_OP_DSYMV, 5, 12),
    DIRECT_ENTRY(FB_OP_CHEMV, 5, 12),
    DIRECT_ENTRY(FB_OP_ZHEMV, 5, 12),

    /* TRMV — triangular matvec, no accumulation error beyond GEMV */
    DIRECT_ENTRY(FB_OP_STRMV, 5, 12),
    DIRECT_ENTRY(FB_OP_DTRMV, 5, 12),
    DIRECT_ENTRY(FB_OP_CTRMV, 5, 12),
    DIRECT_ENTRY(FB_OP_ZTRMV, 5, 12),

    /* TRSV — triangular solve: residual-based; treat as SOLVE archetype */
    SOLVE_ENTRY(FB_OP_STRSV, 4, 10),
    SOLVE_ENTRY(FB_OP_DTRSV, 4, 10),
    SOLVE_ENTRY(FB_OP_CTRSV, 4, 10),
    SOLVE_ENTRY(FB_OP_ZTRSV, 4, 10),

    /* GER / GERU / GERC */
    DIRECT_ENTRY(FB_OP_SGER, 6, 13),
    DIRECT_ENTRY(FB_OP_DGER, 6, 13),
    DIRECT_ENTRY(FB_OP_CGERU, 6, 13),
    DIRECT_ENTRY(FB_OP_CGERC, 6, 13),
    DIRECT_ENTRY(FB_OP_ZGERU, 6, 13),
    DIRECT_ENTRY(FB_OP_ZGERC, 6, 13),

    /* SYR / HER rank‑1 updates */
    DIRECT_ENTRY(FB_OP_SSYR, 6, 13),
    DIRECT_ENTRY(FB_OP_DSYR, 6, 13),
    DIRECT_ENTRY(FB_OP_CHER, 6, 13),
    DIRECT_ENTRY(FB_OP_ZHER, 6, 13),
    DIRECT_ENTRY(FB_OP_SSYR2, 5, 12),
    DIRECT_ENTRY(FB_OP_DSYR2, 5, 12),
    DIRECT_ENTRY(FB_OP_CHER2, 5, 12),
    DIRECT_ENTRY(FB_OP_ZHER2, 5, 12),

    /* Packed / banded variants — same bounds as their full counterparts */
    DIRECT_ENTRY(FB_OP_SSPMV, 5, 12),
    DIRECT_ENTRY(FB_OP_DSPMV, 5, 12),
    DIRECT_ENTRY(FB_OP_CHPMV, 5, 12),
    DIRECT_ENTRY(FB_OP_ZHPMV, 5, 12),
    DIRECT_ENTRY(FB_OP_SSBMV, 5, 12),
    DIRECT_ENTRY(FB_OP_DSBMV, 5, 12),
    DIRECT_ENTRY(FB_OP_CHBMV, 5, 12),
    DIRECT_ENTRY(FB_OP_ZHBMV, 5, 12),
    DIRECT_ENTRY(FB_OP_STBMV, 5, 12),
    DIRECT_ENTRY(FB_OP_DTBMV, 5, 12),
    DIRECT_ENTRY(FB_OP_CTBMV, 5, 12),
    DIRECT_ENTRY(FB_OP_ZTBMV, 5, 12),
    SOLVE_ENTRY(FB_OP_STBSV, 4, 10),
    SOLVE_ENTRY(FB_OP_DTBSV, 4, 10),
    SOLVE_ENTRY(FB_OP_CTBSV, 4, 10),
    SOLVE_ENTRY(FB_OP_ZTBSV, 4, 10),
    DIRECT_ENTRY(FB_OP_STPMV, 5, 12),
    DIRECT_ENTRY(FB_OP_DTPMV, 5, 12),
    DIRECT_ENTRY(FB_OP_CTPMV, 5, 12),
    DIRECT_ENTRY(FB_OP_ZTPMV, 5, 12),
    SOLVE_ENTRY(FB_OP_STPSV, 4, 10),
    SOLVE_ENTRY(FB_OP_DTPSV, 4, 10),
    SOLVE_ENTRY(FB_OP_CTPSV, 4, 10),
    SOLVE_ENTRY(FB_OP_ZTPSV, 4, 10),
    DIRECT_ENTRY(FB_OP_SSPR, 6, 13),
    DIRECT_ENTRY(FB_OP_DSPR, 6, 13),
    DIRECT_ENTRY(FB_OP_CHPR, 6, 13),
    DIRECT_ENTRY(FB_OP_ZHPR, 6, 13),
    DIRECT_ENTRY(FB_OP_SSPR2, 5, 12),
    DIRECT_ENTRY(FB_OP_DSPR2, 5, 12),
    DIRECT_ENTRY(FB_OP_CHPR2, 5, 12),
    DIRECT_ENTRY(FB_OP_ZHPR2, 5, 12),

    /* -----------------------------------------------------------------
     * BLAS Level 3 — matrix‐matrix operations
     *
     * GEMM with Kahan: f32=5, f64=13 (PRECISION_GUARANTEES §3.1)
     * TRSM: SOLVE archetype, f32=4, f64=10
     * TRMM: DIRECT archetype (no solve), f32=5, f64=12
     * SYRK/HERK: DIRECT, as GEMM-quality accumulation, f32=5, f64=12
     * ----------------------------------------------------------------- */

    /* GEMM */
    DIRECT_ENTRY(FB_OP_SGEMM, 5, 13),
    DIRECT_ENTRY(FB_OP_DGEMM, 5, 13),
    DIRECT_ENTRY(FB_OP_CGEMM, 5, 13),
    DIRECT_ENTRY(FB_OP_ZGEMM, 5, 13),

    /* SYMM / HEMM */
    DIRECT_ENTRY(FB_OP_SSYMM, 5, 13),
    DIRECT_ENTRY(FB_OP_DSYMM, 5, 13),
    DIRECT_ENTRY(FB_OP_CSYMM, 5, 13),
    DIRECT_ENTRY(FB_OP_ZSYMM, 5, 13),
    DIRECT_ENTRY(FB_OP_CHEMM, 5, 13),
    DIRECT_ENTRY(FB_OP_ZHEMM, 5, 13),

    /* SYRK / HERK */
    DIRECT_ENTRY(FB_OP_SSYRK, 5, 12),
    DIRECT_ENTRY(FB_OP_DSYRK, 5, 12),
    DIRECT_ENTRY(FB_OP_CSYRK, 5, 12),
    DIRECT_ENTRY(FB_OP_ZSYRK, 5, 12),
    DIRECT_ENTRY(FB_OP_CHERK, 5, 12),
    DIRECT_ENTRY(FB_OP_ZHERK, 5, 12),

    /* SYR2K / HER2K */
    DIRECT_ENTRY(FB_OP_SSYR2K, 5, 12),
    DIRECT_ENTRY(FB_OP_DSYR2K, 5, 12),
    DIRECT_ENTRY(FB_OP_CSYR2K, 5, 12),
    DIRECT_ENTRY(FB_OP_ZSYR2K, 5, 12),
    DIRECT_ENTRY(FB_OP_CHER2K, 5, 12),
    DIRECT_ENTRY(FB_OP_ZHER2K, 5, 12),

    /* TRMM — triangular matrix‐matrix multiply */
    DIRECT_ENTRY(FB_OP_STRMM, 5, 12),
    DIRECT_ENTRY(FB_OP_DTRMM, 5, 12),
    DIRECT_ENTRY(FB_OP_CTRMM, 5, 12),
    DIRECT_ENTRY(FB_OP_ZTRMM, 5, 12),

    /* TRSM — triangular solve */
    SOLVE_ENTRY(FB_OP_STRSM, 4, 10),
    SOLVE_ENTRY(FB_OP_DTRSM, 4, 10),
    SOLVE_ENTRY(FB_OP_CTRSM, 4, 10),
    SOLVE_ENTRY(FB_OP_ZTRSM, 4, 10),

    /* Batched GEMM */
    DIRECT_ENTRY(FB_OP_SGEMM_BATCH, 5, 13),
    DIRECT_ENTRY(FB_OP_DGEMM_BATCH, 5, 13),
    DIRECT_ENTRY(FB_OP_CGEMM_BATCH, 5, 13),
    DIRECT_ENTRY(FB_OP_ZGEMM_BATCH, 5, 13),
    DIRECT_ENTRY(FB_OP_SGEMM_STRIDED, 5, 13),
    DIRECT_ENTRY(FB_OP_DGEMM_STRIDED, 5, 13),
    DIRECT_ENTRY(FB_OP_CGEMM_STRIDED, 5, 13),
    DIRECT_ENTRY(FB_OP_ZGEMM_STRIDED, 5, 13),

    /* -----------------------------------------------------------------
     * LAPACK — Phase 3/4/5
     *
     * Complex variants added alongside real: same archetype + ceilings.
     * FACTOR: archetype measured by reconstruction residual (||PA-LU||/||A||).
     * SOLVE:  archetype measured by solve residual (||Ax-b||/(||A||*||x||)).
     * SPECTRAL: archetype measured by value accuracy (eigenvalues/singular
     * values).
     * ----------------------------------------------------------------- */

    /* LU factorisation (GETRF) */
    FACTOR_ENTRY(FB_OP_SGETRF, 4, 10),
    FACTOR_ENTRY(FB_OP_DGETRF, 4, 10),
    FACTOR_ENTRY(FB_OP_CGETRF, 4, 10),
    FACTOR_ENTRY(FB_OP_ZGETRF, 4, 10),

    /* LU solve (GETRS) */
    SOLVE_ENTRY(FB_OP_SGETRS, 4, 10),
    SOLVE_ENTRY(FB_OP_DGETRS, 4, 10),
    SOLVE_ENTRY(FB_OP_CGETRS, 4, 10),
    SOLVE_ENTRY(FB_OP_ZGETRS, 4, 10),

    /* Cholesky factorisation (POTRF) */
    FACTOR_ENTRY(FB_OP_SPOTRF, 4, 10),
    FACTOR_ENTRY(FB_OP_DPOTRF, 4, 10),
    FACTOR_ENTRY(FB_OP_CPOTRF, 4, 10),
    FACTOR_ENTRY(FB_OP_ZPOTRF, 4, 10),

    /* Cholesky solve (POTRS) */
    SOLVE_ENTRY(FB_OP_SPOTRS, 4, 10),
    SOLVE_ENTRY(FB_OP_DPOTRS, 4, 10),
    SOLVE_ENTRY(FB_OP_CPOTRS, 4, 10),
    SOLVE_ENTRY(FB_OP_ZPOTRS, 4, 10),

    /* QR factorisation (GEQRF) */
    FACTOR_ENTRY(FB_OP_SGEQRF, 4, 10),
    FACTOR_ENTRY(FB_OP_DGEQRF, 4, 10),
    FACTOR_ENTRY(FB_OP_CGEQRF, 4, 10),
    FACTOR_ENTRY(FB_OP_ZGEQRF, 4, 10),

    /* Symmetric/Hermitian indefinite factorisation (SYTRF) */
    FACTOR_ENTRY(FB_OP_SSYTRF, 4, 10),
    FACTOR_ENTRY(FB_OP_DSYTRF, 4, 10),
    FACTOR_ENTRY(FB_OP_CSYTRF, 4, 10),
    FACTOR_ENTRY(FB_OP_ZSYTRF, 4, 10),

    /* Symmetric/Hermitian indefinite solve (SYTRS) */
    SOLVE_ENTRY(FB_OP_SSYTRS, 4, 10),
    SOLVE_ENTRY(FB_OP_DSYTRS, 4, 10),
    SOLVE_ENTRY(FB_OP_CSYTRS, 4, 10),
    SOLVE_ENTRY(FB_OP_ZSYTRS, 4, 10),

    /* Expert driver: general system (GESV) */
    SOLVE_ENTRY(FB_OP_SGESV, 4, 10),
    SOLVE_ENTRY(FB_OP_DGESV, 4, 10),
    SOLVE_ENTRY(FB_OP_CGESV, 4, 10),
    SOLVE_ENTRY(FB_OP_ZGESV, 4, 10),

    /* Expert driver: positive-definite system (POSV) */
    SOLVE_ENTRY(FB_OP_SPOSV, 4, 10),
    SOLVE_ENTRY(FB_OP_DPOSV, 4, 10),
    SOLVE_ENTRY(FB_OP_CPOSV, 4, 10),
    SOLVE_ENTRY(FB_OP_ZPOSV, 4, 10),

    /* Least-squares: minimum-norm (GELS) */
    SOLVE_ENTRY(FB_OP_SGELS, 4, 10),
    SOLVE_ENTRY(FB_OP_DGELS, 4, 10),
    SOLVE_ENTRY(FB_OP_CGELS, 4, 10),
    SOLVE_ENTRY(FB_OP_ZGELS, 4, 10),

    /* Least-squares: divide-and-conquer SVD (GELSD) */
    SOLVE_ENTRY(FB_OP_SGELSD, 4, 10),
    SOLVE_ENTRY(FB_OP_DGELSD, 4, 10),
    SOLVE_ENTRY(FB_OP_CGELSD, 4, 10),
    SOLVE_ENTRY(FB_OP_ZGELSD, 4, 10),

    /* Least-squares: iterative refinement via SVD (GELSS) */
    SOLVE_ENTRY(FB_OP_SGELSS, 4, 10),
    SOLVE_ENTRY(FB_OP_DGELSS, 4, 10),
    SOLVE_ENTRY(FB_OP_CGELSS, 4, 10),
    SOLVE_ENTRY(FB_OP_ZGELSS, 4, 10),

    /* Least-squares: complete orthogonal decomposition (GELSY) */
    SOLVE_ENTRY(FB_OP_SGELSY, 4, 10),
    SOLVE_ENTRY(FB_OP_DGELSY, 4, 10),
    SOLVE_ENTRY(FB_OP_CGELSY, 4, 10),
    SOLVE_ENTRY(FB_OP_ZGELSY, 4, 10),

    /* Symmetric/Hermitian eigenvalue (SYEV/HEEV) */
    SPECTRAL_ENTRY(FB_OP_SSYEV, 4, 10),
    SPECTRAL_ENTRY(FB_OP_DSYEV, 4, 10),
    SPECTRAL_ENTRY(FB_OP_CHEEV, 4, 10),
    SPECTRAL_ENTRY(FB_OP_ZHEEV, 4, 10),

    /* Singular value decomposition (GESVD) */
    SPECTRAL_ENTRY(FB_OP_SGESVD, 4, 10),
    SPECTRAL_ENTRY(FB_OP_DGESVD, 4, 10),
    SPECTRAL_ENTRY(FB_OP_CGESVD, 4, 10),
    SPECTRAL_ENTRY(FB_OP_ZGESVD, 4, 10),

    /* Non-symmetric eigenvalue (GEEV — all 4 precisions) */
    SPECTRAL_ENTRY(FB_OP_SGEEV, 4, 10),
    SPECTRAL_ENTRY(FB_OP_DGEEV, 4, 10),
    SPECTRAL_ENTRY(FB_OP_CGEEV, 4, 10),
    SPECTRAL_ENTRY(FB_OP_ZGEEV, 4, 10),

    /*
     * Entries 312–1247 are zero‐initialised.
     * fb_judge_meta_get() treats any entry with max_certifiable_f64==0
     * and archetype==0 as "not yet registered" and replaces it with
     * the conservative default at lookup time.
     */
};
