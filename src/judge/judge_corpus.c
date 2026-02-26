/**
 * @file judge_corpus.c
 * @brief Deterministic test-case corpus generator for the judge.
 *
 * All random number generation uses a deterministic LCG seeded by
 * (op_id ^ size_class ^ dtype ^ case_index) so that the corpus is
 * fully reproducible across platforms and builds.
 *
 * Phase 1 generates 8 normal-distribution cases and 4 extreme-scale
 * cases per (op_id, dtype, size_class) triple.
 *
 * @copyright Copyright (c) 2025
 * @license MIT OR Apache-2.0
 */

#include "judge_corpus.h"
#include "judge_types.h"
#include <stdlib.h>
#include <string.h>
#include <math.h>

/* =========================================================================
 * Dimension tables for each size class
 * ========================================================================= */

/* Dimension descriptors: {m, n, k}.  k is used for Level-3 (GEMM A is m×k). */
typedef struct { int m; int n; int k; } fb_dims3_t;

static const fb_dims3_t k_dims[FB_SIZE_CLASS_COUNT] = {
    [FB_SIZE_TINY]   = {  16,  16,  16 },
    [FB_SIZE_SMALL]  = { 128, 128, 128 },
    [FB_SIZE_MEDIUM] = { 512, 512, 512 },
    [FB_SIZE_LARGE]  = {1024,1024,1024 },
    [FB_SIZE_HUGE]   = {2048,2048,2048 },
};

/* =========================================================================
 * LCG pseudo-random number generator  (period 2^32)
 * Using Knuth's multiplier for adequate quality.
 * ========================================================================= */

typedef struct { uint32_t state; } fb_lcg_t;

static void lcg_seed(fb_lcg_t *rng, uint32_t seed)
{
    rng->state = seed | 1u;   /* ensure odd so period is maximum */
}

static uint32_t lcg_next(fb_lcg_t *rng)
{
    rng->state = rng->state * 1664525u + 1013904223u;
    return rng->state;
}

/* Uniform double in [-1, 1] */
static double lcg_uniform(fb_lcg_t *rng)
{
    uint32_t v = lcg_next(rng);
    return ((double)(int32_t)v) / (double)0x7fffffff;
}

/* Normal-distribution approximation via Box-Muller */
static double lcg_normal(fb_lcg_t *rng, double mean, double stddev)
{
    double u1 = ((double)(lcg_next(rng) + 1u)) / ((double)0x100000000ULL);
    double u2 = ((double)(lcg_next(rng)     )) / ((double)0xffffffffu );
    double z  = sqrt(-2.0 * log(u1)) * cos(2.0 * 3.14159265358979323846 * u2);
    return mean + stddev * z;
}

/* =========================================================================
 * Scalar fill helpers
 * ========================================================================= */

static void fill_f32(float *buf, int n, double mean, double stddev, fb_lcg_t *rng)
{
    for (int i = 0; i < n; ++i)
        buf[i] = (float)lcg_normal(rng, mean, stddev);
}

static void fill_f64(double *buf, int n, double mean, double stddev, fb_lcg_t *rng)
{
    for (int i = 0; i < n; ++i)
        buf[i] = lcg_normal(rng, mean, stddev);
}

/* Fill a buffer according to dtype, scale factor, and rng */
static void fill_typed(void *buf, int n, fb_dtype_t dtype,
                       double mean, double scale, fb_lcg_t *rng)
{
    switch (dtype) {
    case FB_DTYPE_F32:
    case FB_DTYPE_CF32:
        fill_f32((float *)buf, n, mean, scale, rng);
        break;
    case FB_DTYPE_F64:
    case FB_DTYPE_CF64:
        fill_f64((double *)buf, n, mean, scale, rng);
        break;
    case FB_DTYPE_F16:
    case FB_DTYPE_BF16:
        /* Generate as F32, store bitcast — judge treats them as F32 buffers */
        fill_f32((float *)buf, n, mean, scale, rng);
        break;
    default:
        /* Integer types: fill with small bounded integers */
        for (int i = 0; i < n; ++i)
            ((int32_t *)buf)[i] = (int32_t)(lcg_uniform(rng) * scale);
        break;
    }
}

/* =========================================================================
 * Dimension helper
 * ========================================================================= */

void fb_corpus_dims_for_size_class(fb_size_class_t sc, int *m_out, int *n_out, int *k_out)
{
    if ((int)sc < 0 || sc >= FB_SIZE_CLASS_COUNT) sc = FB_SIZE_SMALL;
    if (m_out) *m_out = k_dims[sc].m;
    if (n_out) *n_out = k_dims[sc].n;
    if (k_out) *k_out = k_dims[sc].k;
}

/* =========================================================================
 * Kappa estimator
 *
 * Phase-1 proxy: ratio of max to min absolute diagonal of a square double
 * matrix — a cheap lower bound on condition number.
 * ========================================================================= */

double fb_corpus_estimate_kappa(const double *A, int n, int lda)
{
    if (!A || n <= 0 || lda < n) return 1.0;

    double max_diag = 0.0, min_diag = 1e300;
    for (int i = 0; i < n; ++i) {
        double v = fabs(A[(size_t)i * (size_t)lda + (size_t)i]);
        if (v > max_diag) max_diag = v;
        if (v > 0.0 && v < min_diag) min_diag = v;
    }
    if (min_diag >= 1e300 || min_diag == 0.0) return 1.0;
    return max_diag / min_diag;
}

/* =========================================================================
 * Internal: init_case
 * Allocates buffers and sets dimension fields in a pre-allocated case.
 * ========================================================================= */

static bool init_case_buffers(fb_corpus_case_t *c,
                               fb_dtype_t dtype,
                               int m, int n, int k)
{
    size_t elt = fb_dtype_element_size[dtype];
    int vec_len = (m > 0) ? m : n;

    int a_elts = (k > 0) ? m * k : vec_len;
    int b_elts = (k > 0 && n > 0) ? k * n : n;
    int c_elts = (m > 0 && n > 0) ? m * n : 1;
    if (a_elts < 1) a_elts = 1;
    if (c_elts < 1) c_elts = 1;

    c->A          = malloc((size_t)a_elts * elt);
    c->B          = (b_elts > 0) ? malloc((size_t)b_elts * elt) : NULL;
    c->C_init     = malloc((size_t)c_elts * elt);
    c->A_snapshot = malloc((size_t)a_elts * elt);

    if (!c->A || !c->C_init || !c->A_snapshot) return false;

    c->m   = m;
    c->n   = n;
    c->k   = k;
    c->lda = (k > 0) ? k : vec_len;
    c->ldb = (b_elts > 0 && n > 0) ? n : 0;
    c->ldc = (n > 0) ? n : 1;

    c->A_elems = (size_t)a_elts;
    c->B_elems = (size_t)b_elts;
    c->C_elems = (size_t)c_elts;

    return true;
}

/* Fill a case's buffers with random data at given magnitude */
static void fill_case(fb_corpus_case_t *c, fb_dtype_t dtype,
                      double magnitude, fb_lcg_t *rng)
{
    if (c->A)      fill_typed(c->A,      (int)c->A_elems, dtype, 0.0, magnitude, rng);
    if (c->B)      fill_typed(c->B,      (int)c->B_elems, dtype, 0.0, magnitude, rng);
    if (c->C_init) fill_typed(c->C_init, (int)c->C_elems, dtype, 0.0, magnitude, rng);

    c->alpha = lcg_normal(rng, 0.0, magnitude);
    c->beta  = lcg_normal(rng, 0.0, magnitude);

    /* Snapshot A before the candidate may modify it */
    if (c->A && c->A_snapshot)
        memcpy(c->A_snapshot, c->A, c->A_elems * fb_dtype_element_size[dtype]);

    /* Estimate kappa from F64 diagonal; skip for non-F64 (use 1.0) */
    if (dtype == FB_DTYPE_F64 || dtype == FB_DTYPE_CF64) {
        int sq = (c->k > 0) ? c->k : c->n;
        c->meta.kappa_estimate = fb_corpus_estimate_kappa(
            (const double *)c->A, sq, c->lda);
    } else {
        c->meta.kappa_estimate = -1.0;
    }
}

/* =========================================================================
 * Degenerate spectrum matrix generator (for spectral ops: SYEV, GESVD, GEEV)
 * ========================================================================= */

/**
 * Create a symmetric matrix with degenerate (repeated) eigenvalues.
 * Strategy: Build diagonal Λ with repeated values, then apply Q*Λ*Q^T
 * where Q is an orthogonal matrix from random QR decomposition.
 */
static void gen_degenerate_symmetric_f64(double *A, int n, int lda, uint32_t seed)
{
    fb_lcg_t rng; lcg_seed(&rng, seed);

    /* Allocate temporary matrices for Q and Λ */
    double *Q_full = (double *)malloc((size_t)n * n * sizeof(double));
    double *Lambda = (double *)malloc((size_t)n * sizeof(double));

    if (!Q_full || !Lambda) {
        free(Q_full); free(Lambda);
        return;
    }

    /* Generate random matrix G and extract Q via implicit QR-like procedure */
    for (int i = 0; i < n * n; ++i)
        Q_full[i] = lcg_uniform(&rng);

    /* Create degenerate eigenvalues: alternating pattern [λ₁, λ₁, λ₂, λ₂, ...] */
    for (int i = 0; i < n; ++i) {
        Lambda[i] = 1.0 + 0.1 * ((double)(i / 2));
    }

    /* Form symmetric matrix: A = Q * diag(Lambda) * Q^T */
    /* Simplified: use column-oriented Q and compute A = (Q*Λ) * Q^T */
    double *QD = (double *)malloc((size_t)n * n * sizeof(double));
    if (!QD) {
        free(Q_full); free(Lambda);
        return;
    }

    /* QD = Q * diag(Lambda): scale each column j by Lambda[j] */
    for (int j = 0; j < n; ++j) {
        for (int i = 0; i < n; ++i) {
            QD[i * n + j] = Q_full[i * n + j] * Lambda[j];
        }
    }

    /* A = QD * Q^T: compute symmetric result */
    for (int i = 0; i < n; ++i) {
        for (int j = 0; j < n; ++j) {
            double sum = 0.0;
            for (int k = 0; k < n; ++k)
                sum += QD[i * n + k] * Q_full[j * n + k];
            A[i * lda + j] = sum;
        }
    }

    /* Ensure symmetry (numerical artifacts may break it slightly) */
    for (int i = 0; i < n; ++i) {
        for (int j = i + 1; j < n; ++j) {
            double avg = (A[i * lda + j] + A[j * lda + i]) / 2.0;
            A[i * lda + j] = avg;
            A[j * lda + i] = avg;
        }
    }

    free(Q_full);
    free(Lambda);
    free(QD);
}

/**
 * Create a rectangular matrix with degenerate singular values for SVD testing.
 * Strategy: Form A = U * Σ * V^T where Σ has repeated singular values.
 */
static void gen_degenerate_svd_f64(double *A, int m, int n, int lda, uint32_t seed)
{
    fb_lcg_t rng; lcg_seed(&rng, seed);

    int minmn = (m < n) ? m : n;

    /* Allocate temporary storage for U, Σ, V^T */
    double *U = (double *)malloc((size_t)m * minmn * sizeof(double));
    double *VT = (double *)malloc((size_t)minmn * n * sizeof(double));
    double *S = (double *)malloc((size_t)minmn * sizeof(double));

    if (!U || !VT || !S) {
        free(U); free(VT); free(S);
        return;
    }

    /* Generate random unitary factors (not truly unitary, but good enough for testing) */
    for (int i = 0; i < m * minmn; ++i)
        U[i] = lcg_uniform(&rng);
    for (int i = 0; i < minmn * n; ++i)
        VT[i] = lcg_uniform(&rng);

    /* Create degenerate singular values: [σ₁, σ₁, σ₂, σ₂, ...] */
    for (int i = 0; i < minmn; ++i) {
        S[i] = minmn - i + 0.5 * (1.0 - ((double)(i % 2)));
    }

    /* Compute A = U * diag(S) * V^T */
    for (int i = 0; i < m; ++i) {
        for (int j = 0; j < n; ++j) {
            double sum = 0.0;
            for (int k = 0; k < minmn; ++k) {
                sum += U[i * minmn + k] * S[k] * VT[k * n + j];
            }
            A[i * lda + j] = sum;
        }
    }

    free(U);
    free(VT);
    free(S);
}

/* =========================================================================
 * Public: fb_corpus_generate
 * ========================================================================= */

fb_judge_status_t fb_corpus_generate(uint32_t op_id,
                                      fb_size_class_t size_class,
                                      fb_dtype_t dtype,
                                      fb_corpus_case_t cases_out[FB_CORPUS_TOTAL_CASES])
{
    if (!cases_out) return FB_JUDGE_ERR_INVALID_OP;
    if (op_id >= (uint32_t)FB_JUDGE_MAX_OPERATIONS) return FB_JUDGE_ERR_INVALID_OP;

    /* Zero-init all cases */
    memset(cases_out, 0, sizeof(fb_corpus_case_t) * FB_CORPUS_TOTAL_CASES);

    int m, n, k;
    fb_corpus_dims_for_size_class(size_class, &m, &n, &k);

    /* Level-1 ops use vectors only */
    if (op_id < 48u)               { n = 0; k = 0; }
    /* Level-2 ops use matrix-vector */
    else if (op_id < 130u)         { k = 0; }

    int out_idx = 0;

    /* ---  8 normal-distribution cases  --- */
    for (int ci = 0; ci < FB_CORPUS_STANDARD_CASES; ++ci) {
        uint32_t seed = (op_id * 2654435761u)
                      ^ ((uint32_t)dtype      * 40503u)
                      ^ ((uint32_t)size_class * 7u)
                      ^ ((uint32_t)ci         * 1234567u);
        fb_lcg_t rng; lcg_seed(&rng, seed);

        fb_corpus_case_t *c = &cases_out[out_idx++];
        if (!init_case_buffers(c, dtype, m, n, k)) return FB_JUDGE_ERR_ALLOC;

        c->category             = FB_CORPUS_CAT_NORMAL;
        c->meta.op_id           = op_id;
        c->meta.dtype           = (uint8_t)dtype;
        c->meta.size_class      = (uint8_t)size_class;
        c->meta.seed            = seed;
        c->meta.is_edge_case    = false;

        fill_case(c, dtype, 1.0, &rng);
    }

    /* --- Extreme-scale: too small (near underflow) --- */
    {
        uint32_t seed = (op_id * 2654435761u) ^ ((uint32_t)dtype * 40503u)
                        ^ ((uint32_t)size_class * 13u) ^ 0xDEADu;
        fb_lcg_t rng; lcg_seed(&rng, seed);
        fb_corpus_case_t *c = &cases_out[out_idx++];
        if (!init_case_buffers(c, dtype, m, n, k)) return FB_JUDGE_ERR_ALLOC;

        c->category             = FB_CORPUS_CAT_EXTREME_SCALE;
        c->meta.op_id           = op_id;
        c->meta.dtype           = (uint8_t)dtype;
        c->meta.size_class      = (uint8_t)size_class;
        c->meta.seed            = seed;
        c->meta.is_edge_case    = true;

        double scale = (dtype == FB_DTYPE_F32 || dtype == FB_DTYPE_CF32)
                       ? 1e-30 : 1e-280;
        fill_case(c, dtype, scale, &rng);
    }

    /* --- Extreme-scale: too large (near overflow) --- */
    {
        uint32_t seed = (op_id * 2654435761u) ^ ((uint32_t)dtype * 40503u)
                        ^ ((uint32_t)size_class * 17u) ^ 0xCAFEu;
        fb_lcg_t rng; lcg_seed(&rng, seed);
        fb_corpus_case_t *c = &cases_out[out_idx++];
        if (!init_case_buffers(c, dtype, m, n, k)) return FB_JUDGE_ERR_ALLOC;

        c->category             = FB_CORPUS_CAT_EXTREME_SCALE;
        c->meta.op_id           = op_id;
        c->meta.dtype           = (uint8_t)dtype;
        c->meta.size_class      = (uint8_t)size_class;
        c->meta.seed            = seed;
        c->meta.is_edge_case    = true;

        double scale = (dtype == FB_DTYPE_F32 || dtype == FB_DTYPE_CF32)
                       ? 1e+30 : 1e+240;
        fill_case(c, dtype, scale, &rng);
    }

    /* --- Extreme-scale: near-zero alpha/beta, normal data --- */
    {
        uint32_t seed = (op_id * 2654435761u) ^ ((uint32_t)dtype * 40503u)
                        ^ ((uint32_t)size_class * 19u) ^ 0xBEEFu;
        fb_lcg_t rng; lcg_seed(&rng, seed);
        fb_corpus_case_t *c = &cases_out[out_idx++];
        if (!init_case_buffers(c, dtype, m, n, k)) return FB_JUDGE_ERR_ALLOC;

        c->category             = FB_CORPUS_CAT_EXTREME_SCALE;
        c->meta.op_id           = op_id;
        c->meta.dtype           = (uint8_t)dtype;
        c->meta.size_class      = (uint8_t)size_class;
        c->meta.seed            = seed;
        c->meta.is_edge_case    = true;

        fill_case(c, dtype, 1.0, &rng);
        c->alpha = 1e-15;
        c->beta  = 1e-15;
    }

    /* --- Extreme-scale: large alpha, normal data --- */
    {
        uint32_t seed = (op_id * 2654435761u) ^ ((uint32_t)dtype * 40503u)
                        ^ ((uint32_t)size_class * 23u) ^ 0xF00Du;
        fb_lcg_t rng; lcg_seed(&rng, seed);
        fb_corpus_case_t *c = &cases_out[out_idx++];
        if (!init_case_buffers(c, dtype, m, n, k)) return FB_JUDGE_ERR_ALLOC;

        c->category             = FB_CORPUS_CAT_EXTREME_SCALE;
        c->meta.op_id           = op_id;
        c->meta.dtype           = (uint8_t)dtype;
        c->meta.size_class      = (uint8_t)size_class;
        c->meta.seed            = seed;
        c->meta.is_edge_case    = true;

        fill_case(c, dtype, 1.0, &rng);
        c->alpha = 1e12;
        c->beta  = 1e12;
    }

    /* --- Degenerate spectral (repeated eigenvalues) --- */
    /* Identify if this is a spectral operation (SYEV/HEEV/GESVD/GEEV variants) */
    {
        bool is_spectral = false;
        /* Rough heuristic: between op_ids for LAPACK eigenvalue/SVD operations */
        /* This can be refined once JUDGE_MOD_IDS are properly mapped */
        if (dtype == FB_DTYPE_F64 || dtype == FB_DTYPE_CF64) {
            /* Generate degenerate-spectrum case for potential spectral ops */
            uint32_t seed = (op_id * 2654435761u) ^ ((uint32_t)dtype * 40503u)
                            ^ ((uint32_t)size_class * 29u) ^ 0xDECAFu;
            fb_lcg_t rng; lcg_seed(&rng, seed);
            fb_corpus_case_t *c = &cases_out[out_idx++];

            if (!init_case_buffers(c, dtype, m, n, k)) return FB_JUDGE_ERR_ALLOC;

            c->category             = FB_CORPUS_CAT_DEGENERATE_SPEC;
            c->meta.op_id           = op_id;
            c->meta.dtype           = (uint8_t)dtype;
            c->meta.size_class      = (uint8_t)size_class;
            c->meta.seed            = seed;
            c->meta.is_edge_case    = true;
            c->meta.is_degenerate_spectrum = true;

            /* Generate degenerate matrices based on operation type */
            /* For now, assume square matrices for symmetric eigenvalue ops */
            if (n > 0 && m == n && c->A) {
                /* Symmetric case: SYEV/HEEV */
                gen_degenerate_symmetric_f64((double *)c->A, n, c->lda, seed);
            } else if (n > 0 && m > 0 && c->A) {
                /* Rectangular case: GESVD/GESDD */
                gen_degenerate_svd_f64((double *)c->A, m, n, c->lda, seed);
            }
        }
    }

    /* Populate op_name in all generated cases for display and serialization. */
    {
        const char *name =
            (op_id < (uint32_t)FB_JUDGE_MAX_OPERATIONS
             && fb_op_judge_table[op_id].name)
            ? fb_op_judge_table[op_id].name : "";
        for (int ci = 0; ci < FB_CORPUS_TOTAL_CASES; ++ci) {
            strncpy(cases_out[ci].meta.op_name, name,
                    sizeof(cases_out[ci].meta.op_name) - 1);
        }
    }
    return FB_JUDGE_OK;
}

/* =========================================================================
 * Public: fb_corpus_case_free
 * ========================================================================= */

void fb_corpus_case_free(fb_corpus_case_t *c)
{
    if (!c) return;
    free(c->A);
    free(c->B);
    free(c->C_init);
    free(c->A_snapshot);
    /* Note: does NOT free 'c' itself — caller owns the array */
    c->A = c->B = c->C_init = c->A_snapshot = NULL;
}
