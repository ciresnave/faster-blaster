/**
 * @file level3.c
 * @brief BLAS Level 3 API implementation
 * 
 * @copyright Copyright (c) 2025
 * @license MIT OR Apache-2.0
 */

#include "faster_blaster.h"
#include "dispatch.h"
#include <stdio.h>

/* ==== GEMM: C := alpha*op(A)*op(B) + beta*C ==== */

void fb_sgemm(
    const fb_layout_t layout,
    const fb_transpose_t transa,
    const fb_transpose_t transb,
    const int64_t m,
    const int64_t n,
    const int64_t k,
    const float alpha,
    const float *A,
    const int64_t lda,
    const float *B,
    const int64_t ldb,
    const float beta,
    float *C,
    const int64_t ldc)
{
    const fb_backend_vtable_t *backend = fb_dispatch_get_backend(fb_dispatch_global());
    if (backend && backend->sgemm) {
        backend->sgemm(layout, transa, transb, m, n, k, alpha, A, lda, B, ldb, beta, C, ldc);
    } else {
        fprintf(stderr, "fb_sgemm: No backend available\n");
    }
}

void fb_dgemm(
    const fb_layout_t layout,
    const fb_transpose_t transa,
    const fb_transpose_t transb,
    const int64_t m,
    const int64_t n,
    const int64_t k,
    const double alpha,
    const double *A,
    const int64_t lda,
    const double *B,
    const int64_t ldb,
    const double beta,
    double *C,
    const int64_t ldc)
{
    const fb_backend_vtable_t *backend = fb_dispatch_get_backend(fb_dispatch_global());
    if (backend && backend->dgemm) {
        backend->dgemm(layout, transa, transb, m, n, k, alpha, A, lda, B, ldb, beta, C, ldc);
    } else {
        fprintf(stderr, "fb_dgemm: No backend available\n");
    }
}

/* ==== SYMM: C := alpha*A*B + beta*C or C := alpha*B*A + beta*C (A symmetric) ==== */

void fb_ssymm(
    const fb_layout_t layout,
    const fb_side_t side,
    const fb_uplo_t uplo,
    const int64_t m,
    const int64_t n,
    const float alpha,
    const float *A,
    const int64_t lda,
    const float *B,
    const int64_t ldb,
    const float beta,
    float *C,
    const int64_t ldc)
{
    const fb_backend_vtable_t *backend = fb_dispatch_get_backend(fb_dispatch_global());
    if (backend && backend->ssymm) {
        backend->ssymm(layout, side, uplo, m, n, alpha, A, lda, B, ldb, beta, C, ldc);
    } else {
        fprintf(stderr, "fb_ssymm: No backend available\n");
    }
}

void fb_dsymm(
    const fb_layout_t layout,
    const fb_side_t side,
    const fb_uplo_t uplo,
    const int64_t m,
    const int64_t n,
    const double alpha,
    const double *A,
    const int64_t lda,
    const double *B,
    const int64_t ldb,
    const double beta,
    double *C,
    const int64_t ldc)
{
    const fb_backend_vtable_t *backend = fb_dispatch_get_backend(fb_dispatch_global());
    if (backend && backend->dsymm) {
        backend->dsymm(layout, side, uplo, m, n, alpha, A, lda, B, ldb, beta, C, ldc);
    } else {
        fprintf(stderr, "fb_dsymm: No backend available\n");
    }
}

/* ==== SYRK: C := alpha*A*A^T + beta*C or C := alpha*A^T*A + beta*C ==== */

void fb_ssyrk(
    const fb_layout_t layout,
    const fb_uplo_t uplo,
    const fb_transpose_t trans,
    const int64_t n,
    const int64_t k,
    const float alpha,
    const float *A,
    const int64_t lda,
    const float beta,
    float *C,
    const int64_t ldc)
{
    const fb_backend_vtable_t *backend = fb_dispatch_get_backend(fb_dispatch_global());
    if (backend && backend->ssyrk) {
        backend->ssyrk(layout, uplo, trans, n, k, alpha, A, lda, beta, C, ldc);
    } else {
        fprintf(stderr, "fb_ssyrk: No backend available\n");
    }
}

void fb_dsyrk(
    const fb_layout_t layout,
    const fb_uplo_t uplo,
    const fb_transpose_t trans,
    const int64_t n,
    const int64_t k,
    const double alpha,
    const double *A,
    const int64_t lda,
    const double beta,
    double *C,
    const int64_t ldc)
{
    const fb_backend_vtable_t *backend = fb_dispatch_get_backend(fb_dispatch_global());
    if (backend && backend->dsyrk) {
        backend->dsyrk(layout, uplo, trans, n, k, alpha, A, lda, beta, C, ldc);
    } else {
        fprintf(stderr, "fb_dsyrk: No backend available\n");
    }
}

/* ==== SYR2K: C := alpha*A*B^T + alpha*B*A^T + beta*C or C := alpha*A^T*B + alpha*B^T*A + beta*C ==== */

void fb_ssyr2k(
    const fb_layout_t layout,
    const fb_uplo_t uplo,
    const fb_transpose_t trans,
    const int64_t n,
    const int64_t k,
    const float alpha,
    const float *A,
    const int64_t lda,
    const float *B,
    const int64_t ldb,
    const float beta,
    float *C,
    const int64_t ldc)
{
    const fb_backend_vtable_t *backend = fb_dispatch_get_backend(fb_dispatch_global());
    if (backend && backend->ssyr2k) {
        backend->ssyr2k(layout, uplo, trans, n, k, alpha, A, lda, B, ldb, beta, C, ldc);
    } else {
        fprintf(stderr, "fb_ssyr2k: No backend available\n");
    }
}

void fb_dsyr2k(
    const fb_layout_t layout,
    const fb_uplo_t uplo,
    const fb_transpose_t trans,
    const int64_t n,
    const int64_t k,
    const double alpha,
    const double *A,
    const int64_t lda,
    const double *B,
    const int64_t ldb,
    const double beta,
    double *C,
    const int64_t ldc)
{
    const fb_backend_vtable_t *backend = fb_dispatch_get_backend(fb_dispatch_global());
    if (backend && backend->dsyr2k) {
        backend->dsyr2k(layout, uplo, trans, n, k, alpha, A, lda, B, ldb, beta, C, ldc);
    } else {
        fprintf(stderr, "fb_dsyr2k: No backend available\n");
    }
}

/* ==== TRMM: B := alpha*op(A)*B or B := alpha*B*op(A) (A triangular) ==== */

void fb_strmm(
    const fb_layout_t layout,
    const fb_side_t side,
    const fb_uplo_t uplo,
    const fb_transpose_t trans,
    const fb_diag_t diag,
    const int64_t m,
    const int64_t n,
    const float alpha,
    const float *A,
    const int64_t lda,
    float *B,
    const int64_t ldb)
{
    const fb_backend_vtable_t *backend = fb_dispatch_get_backend(fb_dispatch_global());
    if (backend && backend->strmm) {
        backend->strmm(layout, side, uplo, trans, diag, m, n, alpha, A, lda, B, ldb);
    } else {
        fprintf(stderr, "fb_strmm: No backend available\n");
    }
}

void fb_dtrmm(
    const fb_layout_t layout,
    const fb_side_t side,
    const fb_uplo_t uplo,
    const fb_transpose_t trans,
    const fb_diag_t diag,
    const int64_t m,
    const int64_t n,
    const double alpha,
    const double *A,
    const int64_t lda,
    double *B,
    const int64_t ldb)
{
    const fb_backend_vtable_t *backend = fb_dispatch_get_backend(fb_dispatch_global());
    if (backend && backend->dtrmm) {
        backend->dtrmm(layout, side, uplo, trans, diag, m, n, alpha, A, lda, B, ldb);
    } else {
        fprintf(stderr, "fb_dtrmm: No backend available\n");
    }
}

/* ==== TRSM: Solve op(A)*X = alpha*B or X*op(A) = alpha*B (A triangular) ==== */

void fb_strsm(
    const fb_layout_t layout,
    const fb_side_t side,
    const fb_uplo_t uplo,
    const fb_transpose_t trans,
    const fb_diag_t diag,
    const int64_t m,
    const int64_t n,
    const float alpha,
    const float *A,
    const int64_t lda,
    float *B,
    const int64_t ldb)
{
    const fb_backend_vtable_t *backend = fb_dispatch_get_backend(fb_dispatch_global());
    if (backend && backend->strsm) {
        backend->strsm(layout, side, uplo, trans, diag, m, n, alpha, A, lda, B, ldb);
    } else {
        fprintf(stderr, "fb_strsm: No backend available\n");
    }
}

void fb_dtrsm(
    const fb_layout_t layout,
    const fb_side_t side,
    const fb_uplo_t uplo,
    const fb_transpose_t trans,
    const fb_diag_t diag,
    const int64_t m,
    const int64_t n,
    const double alpha,
    const double *A,
    const int64_t lda,
    double *B,
    const int64_t ldb)
{
    const fb_backend_vtable_t *backend = fb_dispatch_get_backend(fb_dispatch_global());
    if (backend && backend->dtrsm) {
        backend->dtrsm(layout, side, uplo, trans, diag, m, n, alpha, A, lda, B, ldb);
    } else {
        fprintf(stderr, "fb_dtrsm: No backend available\n");
    }
}
