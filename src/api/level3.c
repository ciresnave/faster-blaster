/**
 * @file level3.c
 * @brief BLAS Level 3 API implementation
 * 
 * @copyright Copyright (c) 2025
 * @license MIT OR Apache-2.0
 */

#include "faster_blaster.h"
#include "dispatch.h"
#include "op_dispatch.h"
#include <stdio.h>

/* ==== GEMM: C := alpha*op(A)*op(B) + beta*C ==== */

void fb_sgemm(
    const FB_LAYOUT layout,
    const FB_TRANSPOSE transa,
    const FB_TRANSPOSE transb,
    const int m,
    const int n,
    const int k,
    const float alpha,
    const float *A,
    const int lda,
    const float *B,
    const int ldb,
    const float beta,
    float *C,
    const int ldc)
{
    const fb_backend_vtable_t *backend = fb_get_vtable_for_op(FB_OP_SGEMM);
    if (backend && backend->sgemm) {
        backend->sgemm(layout, transa, transb, m, n, k, alpha, A, lda, B, ldb, beta, C, ldc);
    } else {
        fprintf(stderr, "fb_sgemm: No backend available\n");
    }
}

void fb_dgemm(
    const FB_LAYOUT layout,
    const FB_TRANSPOSE transa,
    const FB_TRANSPOSE transb,
    const int m,
    const int n,
    const int k,
    const double alpha,
    const double *A,
    const int lda,
    const double *B,
    const int ldb,
    const double beta,
    double *C,
    const int ldc)
{
    const fb_backend_vtable_t *backend = fb_get_vtable_for_op(FB_OP_DGEMM);
    if (backend && backend->dgemm) {
        backend->dgemm(layout, transa, transb, m, n, k, alpha, A, lda, B, ldb, beta, C, ldc);
    } else {
        fprintf(stderr, "fb_dgemm: No backend available\n");
    }
}

/* ==== SYMM: C := alpha*A*B + beta*C or C := alpha*B*A + beta*C (A symmetric) ==== */

void fb_ssymm(
    const FB_LAYOUT layout,
    const FB_SIDE side,
    const FB_UPLO uplo,
    const int m,
    const int n,
    const float alpha,
    const float *A,
    const int lda,
    const float *B,
    const int ldb,
    const float beta,
    float *C,
    const int ldc)
{
    const fb_backend_vtable_t *backend = fb_get_vtable_for_op(FB_OP_SSYMM);
    if (backend && backend->ssymm) {
        backend->ssymm(layout, side, uplo, m, n, alpha, A, lda, B, ldb, beta, C, ldc);
    } else {
        fprintf(stderr, "fb_ssymm: No backend available\n");
    }
}

void fb_dsymm(
    const FB_LAYOUT layout,
    const FB_SIDE side,
    const FB_UPLO uplo,
    const int m,
    const int n,
    const double alpha,
    const double *A,
    const int lda,
    const double *B,
    const int ldb,
    const double beta,
    double *C,
    const int ldc)
{
    const fb_backend_vtable_t *backend = fb_get_vtable_for_op(FB_OP_DSYMM);
    if (backend && backend->dsymm) {
        backend->dsymm(layout, side, uplo, m, n, alpha, A, lda, B, ldb, beta, C, ldc);
    } else {
        fprintf(stderr, "fb_dsymm: No backend available\n");
    }
}

/* ==== SYRK: C := alpha*A*A^T + beta*C or C := alpha*A^T*A + beta*C ==== */

void fb_ssyrk(
    const FB_LAYOUT layout,
    const FB_UPLO uplo,
    const FB_TRANSPOSE trans,
    const int n,
    const int k,
    const float alpha,
    const float *A,
    const int lda,
    const float beta,
    float *C,
    const int ldc)
{
    const fb_backend_vtable_t *backend = fb_get_vtable_for_op(FB_OP_SSYRK);
    if (backend && backend->ssyrk) {
        backend->ssyrk(layout, uplo, trans, n, k, alpha, A, lda, beta, C, ldc);
    } else {
        fprintf(stderr, "fb_ssyrk: No backend available\n");
    }
}

void fb_dsyrk(
    const FB_LAYOUT layout,
    const FB_UPLO uplo,
    const FB_TRANSPOSE trans,
    const int n,
    const int k,
    const double alpha,
    const double *A,
    const int lda,
    const double beta,
    double *C,
    const int ldc)
{
    const fb_backend_vtable_t *backend = fb_get_vtable_for_op(FB_OP_DSYRK);
    if (backend && backend->dsyrk) {
        backend->dsyrk(layout, uplo, trans, n, k, alpha, A, lda, beta, C, ldc);
    } else {
        fprintf(stderr, "fb_dsyrk: No backend available\n");
    }
}

/* ==== SYR2K: C := alpha*A*B^T + alpha*B*A^T + beta*C or C := alpha*A^T*B + alpha*B^T*A + beta*C ==== */

void fb_ssyr2k(
    const FB_LAYOUT layout,
    const FB_UPLO uplo,
    const FB_TRANSPOSE trans,
    const int n,
    const int k,
    const float alpha,
    const float *A,
    const int lda,
    const float *B,
    const int ldb,
    const float beta,
    float *C,
    const int ldc)
{
    const fb_backend_vtable_t *backend = fb_get_vtable_for_op(FB_OP_SSYR2K);
    if (backend && backend->ssyr2k) {
        backend->ssyr2k(layout, uplo, trans, n, k, alpha, A, lda, B, ldb, beta, C, ldc);
    } else {
        fprintf(stderr, "fb_ssyr2k: No backend available\n");
    }
}

void fb_dsyr2k(
    const FB_LAYOUT layout,
    const FB_UPLO uplo,
    const FB_TRANSPOSE trans,
    const int n,
    const int k,
    const double alpha,
    const double *A,
    const int lda,
    const double *B,
    const int ldb,
    const double beta,
    double *C,
    const int ldc)
{
    const fb_backend_vtable_t *backend = fb_get_vtable_for_op(FB_OP_DSYR2K);
    if (backend && backend->dsyr2k) {
        backend->dsyr2k(layout, uplo, trans, n, k, alpha, A, lda, B, ldb, beta, C, ldc);
    } else {
        fprintf(stderr, "fb_dsyr2k: No backend available\n");
    }
}

/* ==== TRMM: B := alpha*op(A)*B or B := alpha*B*op(A) (A triangular) ==== */

void fb_strmm(
    const FB_LAYOUT layout,
    const FB_SIDE side,
    const FB_UPLO uplo,
    const FB_TRANSPOSE trans,
    const FB_DIAG diag,
    const int m,
    const int n,
    const float alpha,
    const float *A,
    const int lda,
    float *B,
    const int ldb)
{
    const fb_backend_vtable_t *backend = fb_get_vtable_for_op(FB_OP_STRMM);
    if (backend && backend->strmm) {
        backend->strmm(layout, side, uplo, trans, diag, m, n, alpha, A, lda, B, ldb);
    } else {
        fprintf(stderr, "fb_strmm: No backend available\n");
    }
}

void fb_dtrmm(
    const FB_LAYOUT layout,
    const FB_SIDE side,
    const FB_UPLO uplo,
    const FB_TRANSPOSE trans,
    const FB_DIAG diag,
    const int m,
    const int n,
    const double alpha,
    const double *A,
    const int lda,
    double *B,
    const int ldb)
{
    const fb_backend_vtable_t *backend = fb_get_vtable_for_op(FB_OP_DTRMM);
    if (backend && backend->dtrmm) {
        backend->dtrmm(layout, side, uplo, trans, diag, m, n, alpha, A, lda, B, ldb);
    } else {
        fprintf(stderr, "fb_dtrmm: No backend available\n");
    }
}

/* ==== TRSM: Solve op(A)*X = alpha*B or X*op(A) = alpha*B (A triangular) ==== */

void fb_strsm(
    const FB_LAYOUT layout,
    const FB_SIDE side,
    const FB_UPLO uplo,
    const FB_TRANSPOSE trans,
    const FB_DIAG diag,
    const int m,
    const int n,
    const float alpha,
    const float *A,
    const int lda,
    float *B,
    const int ldb)
{
    const fb_backend_vtable_t *backend = fb_get_vtable_for_op(FB_OP_STRSM);
    if (backend && backend->strsm) {
        backend->strsm(layout, side, uplo, trans, diag, m, n, alpha, A, lda, B, ldb);
    } else {
        fprintf(stderr, "fb_strsm: No backend available\n");
    }
}

void fb_dtrsm(
    const FB_LAYOUT layout,
    const FB_SIDE side,
    const FB_UPLO uplo,
    const FB_TRANSPOSE trans,
    const FB_DIAG diag,
    const int m,
    const int n,
    const double alpha,
    const double *A,
    const int lda,
    double *B,
    const int ldb)
{
    const fb_backend_vtable_t *backend = fb_get_vtable_for_op(FB_OP_DTRSM);
    if (backend && backend->dtrsm) {
        backend->dtrsm(layout, side, uplo, trans, diag, m, n, alpha, A, lda, B, ldb);
    } else {
        fprintf(stderr, "fb_dtrsm: No backend available\n");
    }
}
