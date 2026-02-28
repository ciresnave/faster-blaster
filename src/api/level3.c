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

/* ==== Complex single-precision (C) Level 3 ==== */

void fb_cgemm(
    const FB_LAYOUT layout,
    const FB_TRANSPOSE transa, const FB_TRANSPOSE transb,
    const int m, const int n, const int k,
    const void *alpha,
    const void *A, const int lda,
    const void *B, const int ldb,
    const void *beta,
    void *C, const int ldc)
{
    const fb_backend_vtable_t *backend = fb_get_vtable_for_op(FB_OP_CGEMM);
    if (backend && backend->cgemm) {
        backend->cgemm(layout, transa, transb, m, n, k,
                       *(const fb_complex_float_t *)alpha,
                       (const fb_complex_float_t *)A, lda,
                       (const fb_complex_float_t *)B, ldb,
                       *(const fb_complex_float_t *)beta,
                       (fb_complex_float_t *)C, ldc);
    } else {
        fprintf(stderr, "fb_cgemm: No backend available\n");
    }
}

void fb_csymm(
    const FB_LAYOUT layout, const FB_SIDE side, const FB_UPLO uplo,
    const int m, const int n,
    const void *alpha,
    const void *A, const int lda,
    const void *B, const int ldb,
    const void *beta,
    void *C, const int ldc)
{
    const fb_backend_vtable_t *backend = fb_get_vtable_for_op(FB_OP_CSYMM);
    if (backend && backend->csymm) {
        backend->csymm(layout, side, uplo, m, n,
                       *(const fb_complex_float_t *)alpha,
                       (const fb_complex_float_t *)A, lda,
                       (const fb_complex_float_t *)B, ldb,
                       *(const fb_complex_float_t *)beta,
                       (fb_complex_float_t *)C, ldc);
    } else {
        fprintf(stderr, "fb_csymm: No backend available\n");
    }
}

void fb_chemm(
    const FB_LAYOUT layout, const FB_SIDE side, const FB_UPLO uplo,
    const int m, const int n,
    const void *alpha,
    const void *A, const int lda,
    const void *B, const int ldb,
    const void *beta,
    void *C, const int ldc)
{
    const fb_backend_vtable_t *backend = fb_get_vtable_for_op(FB_OP_CHEMM);
    if (backend && backend->chemm) {
        backend->chemm(layout, side, uplo, m, n,
                       *(const fb_complex_float_t *)alpha,
                       (const fb_complex_float_t *)A, lda,
                       (const fb_complex_float_t *)B, ldb,
                       *(const fb_complex_float_t *)beta,
                       (fb_complex_float_t *)C, ldc);
    } else {
        fprintf(stderr, "fb_chemm: No backend available\n");
    }
}

void fb_csyrk(
    const FB_LAYOUT layout, const FB_UPLO uplo, const FB_TRANSPOSE trans,
    const int n, const int k,
    const void *alpha,
    const void *A, const int lda,
    const void *beta,
    void *C, const int ldc)
{
    const fb_backend_vtable_t *backend = fb_get_vtable_for_op(FB_OP_CSYRK);
    if (backend && backend->csyrk) {
        backend->csyrk(layout, uplo, trans, n, k,
                       *(const fb_complex_float_t *)alpha,
                       (const fb_complex_float_t *)A, lda,
                       *(const fb_complex_float_t *)beta,
                       (fb_complex_float_t *)C, ldc);
    } else {
        fprintf(stderr, "fb_csyrk: No backend available\n");
    }
}

void fb_cherk(
    const FB_LAYOUT layout, const FB_UPLO uplo, const FB_TRANSPOSE trans,
    const int n, const int k,
    const float alpha,
    const void *A, const int lda,
    const float beta,
    void *C, const int ldc)
{
    const fb_backend_vtable_t *backend = fb_get_vtable_for_op(FB_OP_CHERK);
    if (backend && backend->cherk) {
        backend->cherk(layout, uplo, trans, n, k, alpha,
                       (const fb_complex_float_t *)A, lda, beta,
                       (fb_complex_float_t *)C, ldc);
    } else {
        fprintf(stderr, "fb_cherk: No backend available\n");
    }
}

void fb_csyr2k(
    const FB_LAYOUT layout, const FB_UPLO uplo, const FB_TRANSPOSE trans,
    const int n, const int k,
    const void *alpha,
    const void *A, const int lda,
    const void *B, const int ldb,
    const void *beta,
    void *C, const int ldc)
{
    const fb_backend_vtable_t *backend = fb_get_vtable_for_op(FB_OP_CSYR2K);
    if (backend && backend->csyr2k) {
        backend->csyr2k(layout, uplo, trans, n, k,
                        *(const fb_complex_float_t *)alpha,
                        (const fb_complex_float_t *)A, lda,
                        (const fb_complex_float_t *)B, ldb,
                        *(const fb_complex_float_t *)beta,
                        (fb_complex_float_t *)C, ldc);
    } else {
        fprintf(stderr, "fb_csyr2k: No backend available\n");
    }
}

void fb_cher2k(
    const FB_LAYOUT layout, const FB_UPLO uplo, const FB_TRANSPOSE trans,
    const int n, const int k,
    const void *alpha,
    const void *A, const int lda,
    const void *B, const int ldb,
    const float beta,
    void *C, const int ldc)
{
    const fb_backend_vtable_t *backend = fb_get_vtable_for_op(FB_OP_CHER2K);
    if (backend && backend->cher2k) {
        backend->cher2k(layout, uplo, trans, n, k,
                        *(const fb_complex_float_t *)alpha,
                        (const fb_complex_float_t *)A, lda,
                        (const fb_complex_float_t *)B, ldb,
                        beta, (fb_complex_float_t *)C, ldc);
    } else {
        fprintf(stderr, "fb_cher2k: No backend available\n");
    }
}

void fb_ctrmm(
    const FB_LAYOUT layout, const FB_SIDE side,
    const FB_UPLO uplo, const FB_TRANSPOSE trans, const FB_DIAG diag,
    const int m, const int n,
    const void *alpha,
    const void *A, const int lda,
    void *B, const int ldb)
{
    const fb_backend_vtable_t *backend = fb_get_vtable_for_op(FB_OP_CTRMM);
    if (backend && backend->ctrmm) {
        backend->ctrmm(layout, side, uplo, trans, diag, m, n,
                       *(const fb_complex_float_t *)alpha,
                       (const fb_complex_float_t *)A, lda,
                       (fb_complex_float_t *)B, ldb);
    } else {
        fprintf(stderr, "fb_ctrmm: No backend available\n");
    }
}

void fb_ctrsm(
    const FB_LAYOUT layout, const FB_SIDE side,
    const FB_UPLO uplo, const FB_TRANSPOSE trans, const FB_DIAG diag,
    const int m, const int n,
    const void *alpha,
    const void *A, const int lda,
    void *B, const int ldb)
{
    const fb_backend_vtable_t *backend = fb_get_vtable_for_op(FB_OP_CTRSM);
    if (backend && backend->ctrsm) {
        backend->ctrsm(layout, side, uplo, trans, diag, m, n,
                       *(const fb_complex_float_t *)alpha,
                       (const fb_complex_float_t *)A, lda,
                       (fb_complex_float_t *)B, ldb);
    } else {
        fprintf(stderr, "fb_ctrsm: No backend available\n");
    }
}

/* ==== Complex double-precision (Z) Level 3 ==== */

void fb_zgemm(
    const FB_LAYOUT layout,
    const FB_TRANSPOSE transa, const FB_TRANSPOSE transb,
    const int m, const int n, const int k,
    const void *alpha,
    const void *A, const int lda,
    const void *B, const int ldb,
    const void *beta,
    void *C, const int ldc)
{
    const fb_backend_vtable_t *backend = fb_get_vtable_for_op(FB_OP_ZGEMM);
    if (backend && backend->zgemm) {
        backend->zgemm(layout, transa, transb, m, n, k,
                       *(const fb_complex_double_t *)alpha,
                       (const fb_complex_double_t *)A, lda,
                       (const fb_complex_double_t *)B, ldb,
                       *(const fb_complex_double_t *)beta,
                       (fb_complex_double_t *)C, ldc);
    } else {
        fprintf(stderr, "fb_zgemm: No backend available\n");
    }
}

void fb_zsymm(
    const FB_LAYOUT layout, const FB_SIDE side, const FB_UPLO uplo,
    const int m, const int n,
    const void *alpha,
    const void *A, const int lda,
    const void *B, const int ldb,
    const void *beta,
    void *C, const int ldc)
{
    const fb_backend_vtable_t *backend = fb_get_vtable_for_op(FB_OP_ZSYMM);
    if (backend && backend->zsymm) {
        backend->zsymm(layout, side, uplo, m, n,
                       *(const fb_complex_double_t *)alpha,
                       (const fb_complex_double_t *)A, lda,
                       (const fb_complex_double_t *)B, ldb,
                       *(const fb_complex_double_t *)beta,
                       (fb_complex_double_t *)C, ldc);
    } else {
        fprintf(stderr, "fb_zsymm: No backend available\n");
    }
}

void fb_zhemm(
    const FB_LAYOUT layout, const FB_SIDE side, const FB_UPLO uplo,
    const int m, const int n,
    const void *alpha,
    const void *A, const int lda,
    const void *B, const int ldb,
    const void *beta,
    void *C, const int ldc)
{
    const fb_backend_vtable_t *backend = fb_get_vtable_for_op(FB_OP_ZHEMM);
    if (backend && backend->zhemm) {
        backend->zhemm(layout, side, uplo, m, n,
                       *(const fb_complex_double_t *)alpha,
                       (const fb_complex_double_t *)A, lda,
                       (const fb_complex_double_t *)B, ldb,
                       *(const fb_complex_double_t *)beta,
                       (fb_complex_double_t *)C, ldc);
    } else {
        fprintf(stderr, "fb_zhemm: No backend available\n");
    }
}

void fb_zsyrk(
    const FB_LAYOUT layout, const FB_UPLO uplo, const FB_TRANSPOSE trans,
    const int n, const int k,
    const void *alpha,
    const void *A, const int lda,
    const void *beta,
    void *C, const int ldc)
{
    const fb_backend_vtable_t *backend = fb_get_vtable_for_op(FB_OP_ZSYRK);
    if (backend && backend->zsyrk) {
        backend->zsyrk(layout, uplo, trans, n, k,
                       *(const fb_complex_double_t *)alpha,
                       (const fb_complex_double_t *)A, lda,
                       *(const fb_complex_double_t *)beta,
                       (fb_complex_double_t *)C, ldc);
    } else {
        fprintf(stderr, "fb_zsyrk: No backend available\n");
    }
}

void fb_zherk(
    const FB_LAYOUT layout, const FB_UPLO uplo, const FB_TRANSPOSE trans,
    const int n, const int k,
    const double alpha,
    const void *A, const int lda,
    const double beta,
    void *C, const int ldc)
{
    const fb_backend_vtable_t *backend = fb_get_vtable_for_op(FB_OP_ZHERK);
    if (backend && backend->zherk) {
        backend->zherk(layout, uplo, trans, n, k, alpha,
                       (const fb_complex_double_t *)A, lda, beta,
                       (fb_complex_double_t *)C, ldc);
    } else {
        fprintf(stderr, "fb_zherk: No backend available\n");
    }
}

void fb_zsyr2k(
    const FB_LAYOUT layout, const FB_UPLO uplo, const FB_TRANSPOSE trans,
    const int n, const int k,
    const void *alpha,
    const void *A, const int lda,
    const void *B, const int ldb,
    const void *beta,
    void *C, const int ldc)
{
    const fb_backend_vtable_t *backend = fb_get_vtable_for_op(FB_OP_ZSYR2K);
    if (backend && backend->zsyr2k) {
        backend->zsyr2k(layout, uplo, trans, n, k,
                        *(const fb_complex_double_t *)alpha,
                        (const fb_complex_double_t *)A, lda,
                        (const fb_complex_double_t *)B, ldb,
                        *(const fb_complex_double_t *)beta,
                        (fb_complex_double_t *)C, ldc);
    } else {
        fprintf(stderr, "fb_zsyr2k: No backend available\n");
    }
}

void fb_zher2k(
    const FB_LAYOUT layout, const FB_UPLO uplo, const FB_TRANSPOSE trans,
    const int n, const int k,
    const void *alpha,
    const void *A, const int lda,
    const void *B, const int ldb,
    const double beta,
    void *C, const int ldc)
{
    const fb_backend_vtable_t *backend = fb_get_vtable_for_op(FB_OP_ZHER2K);
    if (backend && backend->zher2k) {
        backend->zher2k(layout, uplo, trans, n, k,
                        *(const fb_complex_double_t *)alpha,
                        (const fb_complex_double_t *)A, lda,
                        (const fb_complex_double_t *)B, ldb,
                        beta, (fb_complex_double_t *)C, ldc);
    } else {
        fprintf(stderr, "fb_zher2k: No backend available\n");
    }
}

void fb_ztrmm(
    const FB_LAYOUT layout, const FB_SIDE side,
    const FB_UPLO uplo, const FB_TRANSPOSE trans, const FB_DIAG diag,
    const int m, const int n,
    const void *alpha,
    const void *A, const int lda,
    void *B, const int ldb)
{
    const fb_backend_vtable_t *backend = fb_get_vtable_for_op(FB_OP_ZTRMM);
    if (backend && backend->ztrmm) {
        backend->ztrmm(layout, side, uplo, trans, diag, m, n,
                       *(const fb_complex_double_t *)alpha,
                       (const fb_complex_double_t *)A, lda,
                       (fb_complex_double_t *)B, ldb);
    } else {
        fprintf(stderr, "fb_ztrmm: No backend available\n");
    }
}

void fb_ztrsm(
    const FB_LAYOUT layout, const FB_SIDE side,
    const FB_UPLO uplo, const FB_TRANSPOSE trans, const FB_DIAG diag,
    const int m, const int n,
    const void *alpha,
    const void *A, const int lda,
    void *B, const int ldb)
{
    const fb_backend_vtable_t *backend = fb_get_vtable_for_op(FB_OP_ZTRSM);
    if (backend && backend->ztrsm) {
        backend->ztrsm(layout, side, uplo, trans, diag, m, n,
                       *(const fb_complex_double_t *)alpha,
                       (const fb_complex_double_t *)A, lda,
                       (fb_complex_double_t *)B, ldb);
    } else {
        fprintf(stderr, "fb_ztrsm: No backend available\n");
    }
}
