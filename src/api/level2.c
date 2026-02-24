/**
 * @file level2.c
 * @brief BLAS Level 2 API implementation
 * 
 * @copyright Copyright (c) 2025
 * @license MIT OR Apache-2.0
 */

#include "faster_blaster.h"
#include "dispatch.h"
#include <stdio.h>

/* ==== GEMV: y := alpha*A*x + beta*y or y := alpha*A^T*x + beta*y ==== */

void fb_sgemv(
    const fb_layout_t layout,
    const fb_transpose_t trans,
    const int64_t m,
    const int64_t n,
    const float alpha,
    const float *A,
    const int64_t lda,
    const float *x,
    const int64_t incx,
    const float beta,
    float *y,
    const int64_t incy)
{
    const fb_backend_vtable_t *backend = fb_dispatch_get_backend(fb_dispatch_global());
    if (backend && backend->sgemv) {
        backend->sgemv(layout, trans, m, n, alpha, A, lda, x, incx, beta, y, incy);
    } else {
        fprintf(stderr, "fb_sgemv: No backend available\n");
    }
}

void fb_dgemv(
    const fb_layout_t layout,
    const fb_transpose_t trans,
    const int64_t m,
    const int64_t n,
    const double alpha,
    const double *A,
    const int64_t lda,
    const double *x,
    const int64_t incx,
    const double beta,
    double *y,
    const int64_t incy)
{
    const fb_backend_vtable_t *backend = fb_dispatch_get_backend(fb_dispatch_global());
    if (backend && backend->dgemv) {
        backend->dgemv(layout, trans, m, n, alpha, A, lda, x, incx, beta, y, incy);
    } else {
        fprintf(stderr, "fb_dgemv: No backend available\n");
    }
}

/* ==== GER: A := alpha*x*y^T + A ==== */

void fb_sger(
    const fb_layout_t layout,
    const int64_t m,
    const int64_t n,
    const float alpha,
    const float *x,
    const int64_t incx,
    const float *y,
    const int64_t incy,
    float *A,
    const int64_t lda)
{
    const fb_backend_vtable_t *backend = fb_dispatch_get_backend(fb_dispatch_global());
    if (backend && backend->sger) {
        backend->sger(layout, m, n, alpha, x, incx, y, incy, A, lda);
    } else {
        fprintf(stderr, "fb_sger: No backend available\n");
    }
}

void fb_dger(
    const fb_layout_t layout,
    const int64_t m,
    const int64_t n,
    const double alpha,
    const double *x,
    const int64_t incx,
    const double *y,
    const int64_t incy,
    double *A,
    const int64_t lda)
{
    const fb_backend_vtable_t *backend = fb_dispatch_get_backend(fb_dispatch_global());
    if (backend && backend->dger) {
        backend->dger(layout, m, n, alpha, x, incx, y, incy, A, lda);
    } else {
        fprintf(stderr, "fb_dger: No backend available\n");
    }
}

/* ==== TRMV: x := A*x or x := A^T*x (A triangular) ==== */

void fb_strmv(
    const fb_layout_t layout,
    const fb_uplo_t uplo,
    const fb_transpose_t trans,
    const fb_diag_t diag,
    const int64_t n,
    const float *A,
    const int64_t lda,
    float *x,
    const int64_t incx)
{
    const fb_backend_vtable_t *backend = fb_dispatch_get_backend(fb_dispatch_global());
    if (backend && backend->strmv) {
        backend->strmv(layout, uplo, trans, diag, n, A, lda, x, incx);
    } else {
        fprintf(stderr, "fb_strmv: No backend available\n");
    }
}

void fb_dtrmv(
    const fb_layout_t layout,
    const fb_uplo_t uplo,
    const fb_transpose_t trans,
    const fb_diag_t diag,
    const int64_t n,
    const double *A,
    const int64_t lda,
    double *x,
    const int64_t incx)
{
    const fb_backend_vtable_t *backend = fb_dispatch_get_backend(fb_dispatch_global());
    if (backend && backend->dtrmv) {
        backend->dtrmv(layout, uplo, trans, diag, n, A, lda, x, incx);
    } else {
        fprintf(stderr, "fb_dtrmv: No backend available\n");
    }
}

/* ==== SYMV: y := alpha*A*x + beta*y (A symmetric) ==== */

void fb_ssymv(
    const FB_LAYOUT Layout,
    const FB_UPLO Uplo,
    const int N,
    const float alpha,
    const float *A,
    const int lda,
    const float *X,
    const int incX,
    const float beta,
    float *Y,
    const int incY)
{
    const fb_backend_vtable_t *backend = fb_dispatch_get_backend(fb_dispatch_global());
    if (backend && backend->ssymv) {
        backend->ssymv(Layout, Uplo, N, alpha, A, lda, X, incX, beta, Y, incY);
    } else {
        fprintf(stderr, "fb_ssymv: No backend available\n");
    }
}

void fb_dsymv(
    const FB_LAYOUT Layout,
    const FB_UPLO Uplo,
    const int N,
    const double alpha,
    const double *A,
    const int lda,
    const double *X,
    const int incX,
    const double beta,
    double *Y,
    const int incY)
{
    const fb_backend_vtable_t *backend = fb_dispatch_get_backend(fb_dispatch_global());
    if (backend && backend->dsymv) {
        backend->dsymv(Layout, Uplo, N, alpha, A, lda, X, incX, beta, Y, incY);
    } else {
        fprintf(stderr, "fb_dsymv: No backend available\n");
    }
}

/* ==== GBMV: y := alpha*op(A)*x + beta*y (A general banded) ==== */

void fb_sgbmv(
    const FB_LAYOUT Layout,
    const FB_TRANSPOSE TransA,
    const int M,
    const int N,
    const int kl,
    const int ku,
    const float alpha,
    const float *A,
    const int lda,
    const float *X,
    const int incX,
    const float beta,
    float *Y,
    const int incY)
{
    const fb_backend_vtable_t *backend = fb_dispatch_get_backend(fb_dispatch_global());
    if (backend && backend->sgbmv) {
        backend->sgbmv(Layout, TransA, M, N, kl, ku, alpha, A, lda, X, incX, beta, Y, incY);
    } else {
        fprintf(stderr, "fb_sgbmv: No backend available\n");
    }
}

void fb_dgbmv(
    const FB_LAYOUT Layout,
    const FB_TRANSPOSE TransA,
    const int M,
    const int N,
    const int kl,
    const int ku,
    const double alpha,
    const double *A,
    const int lda,
    const double *X,
    const int incX,
    const double beta,
    double *Y,
    const int incY)
{
    const fb_backend_vtable_t *backend = fb_dispatch_get_backend(fb_dispatch_global());
    if (backend && backend->dgbmv) {
        backend->dgbmv(Layout, TransA, M, N, kl, ku, alpha, A, lda, X, incX, beta, Y, incY);
    } else {
        fprintf(stderr, "fb_dgbmv: No backend available\n");
    }
}

/* ==== SBMV: y := alpha*A*x + beta*y (A symmetric banded) ==== */

void fb_ssbmv(
    const FB_LAYOUT Layout,
    const FB_UPLO Uplo,
    const int N,
    const int k,
    const float alpha,
    const float *A,
    const int lda,
    const float *X,
    const int incX,
    const float beta,
    float *Y,
    const int incY)
{
    const fb_backend_vtable_t *backend = fb_dispatch_get_backend(fb_dispatch_global());
    if (backend && backend->ssbmv) {
        backend->ssbmv(Layout, Uplo, N, k, alpha, A, lda, X, incX, beta, Y, incY);
    } else {
        fprintf(stderr, "fb_ssbmv: No backend available\n");
    }
}

void fb_dsbmv(
    const FB_LAYOUT Layout,
    const FB_UPLO Uplo,
    const int N,
    const int k,
    const double alpha,
    const double *A,
    const int lda,
    const double *X,
    const int incX,
    const double beta,
    double *Y,
    const int incY)
{
    const fb_backend_vtable_t *backend = fb_dispatch_get_backend(fb_dispatch_global());
    if (backend && backend->dsbmv) {
        backend->dsbmv(Layout, Uplo, N, k, alpha, A, lda, X, incX, beta, Y, incY);
    } else {
        fprintf(stderr, "fb_dsbmv: No backend available\n");
    }
}

/* ==== SPMV: y := alpha*A*x + beta*y (A symmetric packed) ==== */

void fb_sspmv(
    const FB_LAYOUT Layout,
    const FB_UPLO Uplo,
    const int N,
    const float alpha,
    const float *Ap,
    const float *X,
    const int incX,
    const float beta,
    float *Y,
    const int incY)
{
    const fb_backend_vtable_t *backend = fb_dispatch_get_backend(fb_dispatch_global());
    if (backend && backend->sspmv) {
        backend->sspmv(Layout, Uplo, N, alpha, Ap, X, incX, beta, Y, incY);
    } else {
        fprintf(stderr, "fb_sspmv: No backend available\n");
    }
}

void fb_dspmv(
    const FB_LAYOUT Layout,
    const FB_UPLO Uplo,
    const int N,
    const double alpha,
    const double *Ap,
    const double *X,
    const int incX,
    const double beta,
    double *Y,
    const int incY)
{
    const fb_backend_vtable_t *backend = fb_dispatch_get_backend(fb_dispatch_global());
    if (backend && backend->dspmv) {
        backend->dspmv(Layout, Uplo, N, alpha, Ap, X, incX, beta, Y, incY);
    } else {
        fprintf(stderr, "fb_dspmv: No backend available\n");
    }
}

/* ==== TBMV: x := A*x or x := A^T*x (A triangular banded) ==== */

void fb_stbmv(
    const FB_LAYOUT Layout,
    const FB_UPLO Uplo,
    const FB_TRANSPOSE TransA,
    const FB_DIAG Diag,
    const int N,
    const int k,
    const float *A,
    const int lda,
    float *X,
    const int incX)
{
    const fb_backend_vtable_t *backend = fb_dispatch_get_backend(fb_dispatch_global());
    if (backend && backend->stbmv) {
        backend->stbmv(Layout, Uplo, TransA, Diag, N, k, A, lda, X, incX);
    } else {
        fprintf(stderr, "fb_stbmv: No backend available\n");
    }
}

void fb_dtbmv(
    const FB_LAYOUT Layout,
    const FB_UPLO Uplo,
    const FB_TRANSPOSE TransA,
    const FB_DIAG Diag,
    const int N,
    const int k,
    const double *A,
    const int lda,
    double *X,
    const int incX)
{
    const fb_backend_vtable_t *backend = fb_dispatch_get_backend(fb_dispatch_global());
    if (backend && backend->dtbmv) {
        backend->dtbmv(Layout, Uplo, TransA, Diag, N, k, A, lda, X, incX);
    } else {
        fprintf(stderr, "fb_dtbmv: No backend available\n");
    }
}

/* ==== TPMV: x := A*x or x := A^T*x (A triangular packed) ==== */

void fb_stpmv(
    const FB_LAYOUT Layout,
    const FB_UPLO Uplo,
    const FB_TRANSPOSE TransA,
    const FB_DIAG Diag,
    const int N,
    const float *Ap,
    float *X,
    const int incX)
{
    const fb_backend_vtable_t *backend = fb_dispatch_get_backend(fb_dispatch_global());
    if (backend && backend->stpmv) {
        backend->stpmv(Layout, Uplo, TransA, Diag, N, Ap, X, incX);
    } else {
        fprintf(stderr, "fb_stpmv: No backend available\n");
    }
}

void fb_dtpmv(
    const FB_LAYOUT Layout,
    const FB_UPLO Uplo,
    const FB_TRANSPOSE TransA,
    const FB_DIAG Diag,
    const int N,
    const double *Ap,
    double *X,
    const int incX)
{
    const fb_backend_vtable_t *backend = fb_dispatch_get_backend(fb_dispatch_global());
    if (backend && backend->dtpmv) {
        backend->dtpmv(Layout, Uplo, TransA, Diag, N, Ap, X, incX);
    } else {
        fprintf(stderr, "fb_dtpmv: No backend available\n");
    }
}

/* ==== TRSV: op(A)*x = b (A triangular) ==== */

void fb_strsv(
    const FB_LAYOUT Layout,
    const FB_UPLO Uplo,
    const FB_TRANSPOSE TransA,
    const FB_DIAG Diag,
    const int N,
    const float *A,
    const int lda,
    float *X,
    const int incX)
{
    const fb_backend_vtable_t *backend = fb_dispatch_get_backend(fb_dispatch_global());
    if (backend && backend->strsv) {
        backend->strsv(Layout, Uplo, TransA, Diag, N, A, lda, X, incX);
    } else {
        fprintf(stderr, "fb_strsv: No backend available\n");
    }
}

void fb_dtrsv(
    const FB_LAYOUT Layout,
    const FB_UPLO Uplo,
    const FB_TRANSPOSE TransA,
    const FB_DIAG Diag,
    const int N,
    const double *A,
    const int lda,
    double *X,
    const int incX)
{
    const fb_backend_vtable_t *backend = fb_dispatch_get_backend(fb_dispatch_global());
    if (backend && backend->dtrsv) {
        backend->dtrsv(Layout, Uplo, TransA, Diag, N, A, lda, X, incX);
    } else {
        fprintf(stderr, "fb_dtrsv: No backend available\n");
    }
}

/* ==== TBSV: op(A)*x = b (A triangular banded) ==== */

void fb_stbsv(
    const FB_LAYOUT Layout,
    const FB_UPLO Uplo,
    const FB_TRANSPOSE TransA,
    const FB_DIAG Diag,
    const int N,
    const int k,
    const float *A,
    const int lda,
    float *X,
    const int incX)
{
    const fb_backend_vtable_t *backend = fb_dispatch_get_backend(fb_dispatch_global());
    if (backend && backend->stbsv) {
        backend->stbsv(Layout, Uplo, TransA, Diag, N, k, A, lda, X, incX);
    } else {
        fprintf(stderr, "fb_stbsv: No backend available\n");
    }
}

void fb_dtbsv(
    const FB_LAYOUT Layout,
    const FB_UPLO Uplo,
    const FB_TRANSPOSE TransA,
    const FB_DIAG Diag,
    const int N,
    const int k,
    const double *A,
    const int lda,
    double *X,
    const int incX)
{
    const fb_backend_vtable_t *backend = fb_dispatch_get_backend(fb_dispatch_global());
    if (backend && backend->dtbsv) {
        backend->dtbsv(Layout, Uplo, TransA, Diag, N, k, A, lda, X, incX);
    } else {
        fprintf(stderr, "fb_dtbsv: No backend available\n");
    }
}

/* ==== TPSV: op(A)*x = b (A triangular packed) ==== */

void fb_stpsv(
    const FB_LAYOUT Layout,
    const FB_UPLO Uplo,
    const FB_TRANSPOSE TransA,
    const FB_DIAG Diag,
    const int N,
    const float *Ap,
    float *X,
    const int incX)
{
    const fb_backend_vtable_t *backend = fb_dispatch_get_backend(fb_dispatch_global());
    if (backend && backend->stpsv) {
        backend->stpsv(Layout, Uplo, TransA, Diag, N, Ap, X, incX);
    } else {
        fprintf(stderr, "fb_stpsv: No backend available\n");
    }
}

void fb_dtpsv(
    const FB_LAYOUT Layout,
    const FB_UPLO Uplo,
    const FB_TRANSPOSE TransA,
    const FB_DIAG Diag,
    const int N,
    const double *Ap,
    double *X,
    const int incX)
{
    const fb_backend_vtable_t *backend = fb_dispatch_get_backend(fb_dispatch_global());
    if (backend && backend->dtpsv) {
        backend->dtpsv(Layout, Uplo, TransA, Diag, N, Ap, X, incX);
    } else {
        fprintf(stderr, "fb_dtpsv: No backend available\n");
    }
}

/* ==== SYR: A := alpha*x*x^T + A (A symmetric) ==== */

void fb_ssyr(
    const FB_LAYOUT Layout,
    const FB_UPLO Uplo,
    const int N,
    const float alpha,
    const float *X,
    const int incX,
    float *A,
    const int lda)
{
    const fb_backend_vtable_t *backend = fb_dispatch_get_backend(fb_dispatch_global());
    if (backend && backend->ssyr) {
        backend->ssyr(Layout, Uplo, N, alpha, X, incX, A, lda);
    } else {
        fprintf(stderr, "fb_ssyr: No backend available\n");
    }
}

void fb_dsyr(
    const FB_LAYOUT Layout,
    const FB_UPLO Uplo,
    const int N,
    const double alpha,
    const double *X,
    const int incX,
    double *A,
    const int lda)
{
    const fb_backend_vtable_t *backend = fb_dispatch_get_backend(fb_dispatch_global());
    if (backend && backend->dsyr) {
        backend->dsyr(Layout, Uplo, N, alpha, X, incX, A, lda);
    } else {
        fprintf(stderr, "fb_dsyr: No backend available\n");
    }
}

/* ==== SPR: A := alpha*x*x^T + A (A symmetric packed) ==== */

void fb_sspr(
    const FB_LAYOUT Layout,
    const FB_UPLO Uplo,
    const int N,
    const float alpha,
    const float *X,
    const int incX,
    float *Ap)
{
    const fb_backend_vtable_t *backend = fb_dispatch_get_backend(fb_dispatch_global());
    if (backend && backend->sspr) {
        backend->sspr(Layout, Uplo, N, alpha, X, incX, Ap);
    } else {
        fprintf(stderr, "fb_sspr: No backend available\n");
    }
}

void fb_dspr(
    const FB_LAYOUT Layout,
    const FB_UPLO Uplo,
    const int N,
    const double alpha,
    const double *X,
    const int incX,
    double *Ap)
{
    const fb_backend_vtable_t *backend = fb_dispatch_get_backend(fb_dispatch_global());
    if (backend && backend->dspr) {
        backend->dspr(Layout, Uplo, N, alpha, X, incX, Ap);
    } else {
        fprintf(stderr, "fb_dspr: No backend available\n");
    }
}

/* ==== SYR2: A := alpha*x*y^T + alpha*y*x^T + A (A symmetric) ==== */

void fb_ssyr2(
    const FB_LAYOUT Layout,
    const FB_UPLO Uplo,
    const int N,
    const float alpha,
    const float *X,
    const int incX,
    const float *Y,
    const int incY,
    float *A,
    const int lda)
{
    const fb_backend_vtable_t *backend = fb_dispatch_get_backend(fb_dispatch_global());
    if (backend && backend->ssyr2) {
        backend->ssyr2(Layout, Uplo, N, alpha, X, incX, Y, incY, A, lda);
    } else {
        fprintf(stderr, "fb_ssyr2: No backend available\n");
    }
}

void fb_dsyr2(
    const FB_LAYOUT Layout,
    const FB_UPLO Uplo,
    const int N,
    const double alpha,
    const double *X,
    const int incX,
    const double *Y,
    const int incY,
    double *A,
    const int lda)
{
    const fb_backend_vtable_t *backend = fb_dispatch_get_backend(fb_dispatch_global());
    if (backend && backend->dsyr2) {
        backend->dsyr2(Layout, Uplo, N, alpha, X, incX, Y, incY, A, lda);
    } else {
        fprintf(stderr, "fb_dsyr2: No backend available\n");
    }
}

/* ==== SPR2: A := alpha*x*y^T + alpha*y*x^T + A (A symmetric packed) ==== */

void fb_sspr2(
    const FB_LAYOUT Layout,
    const FB_UPLO Uplo,
    const int N,
    const float alpha,
    const float *X,
    const int incX,
    const float *Y,
    const int incY,
    float *Ap)
{
    const fb_backend_vtable_t *backend = fb_dispatch_get_backend(fb_dispatch_global());
    if (backend && backend->sspr2) {
        backend->sspr2(Layout, Uplo, N, alpha, X, incX, Y, incY, Ap);
    } else {
        fprintf(stderr, "fb_sspr2: No backend available\n");
    }
}

void fb_dspr2(
    const FB_LAYOUT Layout,
    const FB_UPLO Uplo,
    const int N,
    const double alpha,
    const double *X,
    const int incX,
    const double *Y,
    const int incY,
    double *Ap)
{
    const fb_backend_vtable_t *backend = fb_dispatch_get_backend(fb_dispatch_global());
    if (backend && backend->dspr2) {
        backend->dspr2(Layout, Uplo, N, alpha, X, incX, Y, incY, Ap);
    } else {
        fprintf(stderr, "fb_dspr2: No backend available\n");
    }
}
