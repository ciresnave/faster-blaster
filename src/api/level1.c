/**
 * @file level1.c
 * @brief BLAS Level 1 API implementation
 * 
 * @copyright Copyright (c) 2025
 * @license MIT OR Apache-2.0
 */

#include "faster_blaster.h"
#include "dispatch.h"
#include "op_dispatch.h"
#include <stdio.h>

/* ==== AXPY: y := alpha*x + y ==== */

void fb_saxpy(
    const int n,
    const float alpha,
    const float *x,
    const int incx,
    float *y,
    const int incy)
{
    const fb_backend_vtable_t *backend = fb_get_vtable_for_op(FB_OP_SAXPY);
    if (backend && backend->saxpy) {
        backend->saxpy(n, alpha, x, incx, y, incy);
    } else {
        fprintf(stderr, "fb_saxpy: No backend available\n");
    }
}

void fb_daxpy(
    const int n,
    const double alpha,
    const double *x,
    const int incx,
    double *y,
    const int incy)
{
    const fb_backend_vtable_t *backend = fb_get_vtable_for_op(FB_OP_DAXPY);
    if (backend && backend->daxpy) {
        backend->daxpy(n, alpha, x, incx, y, incy);
    } else {
        fprintf(stderr, "fb_daxpy: No backend available\n");
    }
}

/* ==== COPY: y := x ==== */

void fb_scopy(
    const int n,
    const float *x,
    const int incx,
    float *y,
    const int incy)
{
    const fb_backend_vtable_t *backend = fb_get_vtable_for_op(FB_OP_SCOPY);
    if (backend && backend->scopy) {
        backend->scopy(n, x, incx, y, incy);
    } else {
        fprintf(stderr, "fb_scopy: No backend available\n");
    }
}

void fb_dcopy(
    const int n,
    const double *x,
    const int incx,
    double *y,
    const int incy)
{
    const fb_backend_vtable_t *backend = fb_get_vtable_for_op(FB_OP_DCOPY);
    if (backend && backend->dcopy) {
        backend->dcopy(n, x, incx, y, incy);
    } else {
        fprintf(stderr, "fb_dcopy: No backend available\n");
    }
}

/* ==== DOT: result := x^T * y ==== */

float fb_sdot(
    const int n,
    const float *x,
    const int incx,
    const float *y,
    const int incy)
{
    const fb_backend_vtable_t *backend = fb_get_vtable_for_op(FB_OP_SDOT);
    if (backend && backend->sdot) {
        return backend->sdot(n, x, incx, y, incy);
    } else {
        fprintf(stderr, "fb_sdot: No backend available\n");
        return 0.0f;
    }
}

double fb_ddot(
    const int n,
    const double *x,
    const int incx,
    const double *y,
    const int incy)
{
    const fb_backend_vtable_t *backend = fb_get_vtable_for_op(FB_OP_DDOT);
    if (backend && backend->ddot) {
        return backend->ddot(n, x, incx, y, incy);
    } else {
        fprintf(stderr, "fb_ddot: No backend available\n");
        return 0.0;
    }
}

/* ==== NRM2: result := ||x||_2 ==== */

float fb_snrm2(
    const int n,
    const float *x,
    const int incx)
{
    const fb_backend_vtable_t *backend = fb_get_vtable_for_op(FB_OP_SNRM2);
    if (backend && backend->snrm2) {
        return backend->snrm2(n, x, incx);
    } else {
        fprintf(stderr, "fb_snrm2: No backend available\n");
        return 0.0f;
    }
}

double fb_dnrm2(
    const int n,
    const double *x,
    const int incx)
{
    const fb_backend_vtable_t *backend = fb_get_vtable_for_op(FB_OP_DNRM2);
    if (backend && backend->dnrm2) {
        return backend->dnrm2(n, x, incx);
    } else {
        fprintf(stderr, "fb_dnrm2: No backend available\n");
        return 0.0;
    }
}

/* ==== SCAL: x := alpha*x ==== */

void fb_sscal(
    const int n,
    const float alpha,
    float *x,
    const int incx)
{
    const fb_backend_vtable_t *backend = fb_get_vtable_for_op(FB_OP_SSCAL);
    if (backend && backend->sscal) {
        backend->sscal(n, alpha, x, incx);
    } else {
        fprintf(stderr, "fb_sscal: No backend available\n");
    }
}

void fb_dscal(
    const int n,
    const double alpha,
    double *x,
    const int incx)
{
    const fb_backend_vtable_t *backend = fb_get_vtable_for_op(FB_OP_DSCAL);
    if (backend && backend->dscal) {
        backend->dscal(n, alpha, x, incx);
    } else {
        fprintf(stderr, "fb_dscal: No backend available\n");
    }
}

/* ==== ROT: Apply Givens rotation ==== */

void fb_srot(
    const int n,
    float *x,
    const int incx,
    float *y,
    const int incy,
    const float c,
    const float s)
{
    const fb_backend_vtable_t *backend = fb_get_vtable_for_op(FB_OP_SROT);
    if (backend && backend->srot) {
        backend->srot(n, x, incx, y, incy, c, s);
    } else {
        fprintf(stderr, "fb_srot: No backend available\n");
    }
}

void fb_drot(
    const int n,
    double *x,
    const int incx,
    double *y,
    const int incy,
    const double c,
    const double s)
{
    const fb_backend_vtable_t *backend = fb_get_vtable_for_op(FB_OP_DROT);
    if (backend && backend->drot) {
        backend->drot(n, x, incx, y, incy, c, s);
    } else {
        fprintf(stderr, "fb_drot: No backend available\n");
    }
}

/* ==== ROTG: Construct Givens rotation ==== */

void fb_srotg(float *a, float *b, float *c, float *s)
{
    const fb_backend_vtable_t *backend = fb_get_vtable_for_op(FB_OP_SROTG);
    if (backend && backend->srotg) {
        backend->srotg(a, b, c, s);
    } else {
        fprintf(stderr, "fb_srotg: No backend available\n");
    }
}

void fb_drotg(double *a, double *b, double *c, double *s)
{
    const fb_backend_vtable_t *backend = fb_get_vtable_for_op(FB_OP_DROTG);
    if (backend && backend->drotg) {
        backend->drotg(a, b, c, s);
    } else {
        fprintf(stderr, "fb_drotg: No backend available\n");
    }
}

/* ==== ROTM: Apply modified Givens rotation ==== */

void fb_srotm(
    const int n,
    float *x,
    const int incx,
    float *y,
    const int incy,
    const float *param)
{
    const fb_backend_vtable_t *backend = fb_get_vtable_for_op(FB_OP_SROTM);
    if (backend && backend->srotm) {
        backend->srotm(n, x, incx, y, incy, param);
    } else {
        fprintf(stderr, "fb_srotm: No backend available\n");
    }
}

void fb_drotm(
    const int n,
    double *x,
    const int incx,
    double *y,
    const int incy,
    const double *param)
{
    const fb_backend_vtable_t *backend = fb_get_vtable_for_op(FB_OP_DROTM);
    if (backend && backend->drotm) {
        backend->drotm(n, x, incx, y, incy, param);
    } else {
        fprintf(stderr, "fb_drotm: No backend available\n");
    }
}

/* ==== ROTMG: Construct modified Givens rotation ==== */

void fb_srotmg(
    float *d1,
    float *d2,
    float *x1,
    const float y1,
    float *param)
{
    const fb_backend_vtable_t *backend = fb_get_vtable_for_op(FB_OP_SROTMG);
    if (backend && backend->srotmg) {
        backend->srotmg(d1, d2, x1, y1, param);
    } else {
        fprintf(stderr, "fb_srotmg: No backend available\n");
    }
}

void fb_drotmg(
    double *d1,
    double *d2,
    double *x1,
    const double y1,
    double *param)
{
    const fb_backend_vtable_t *backend = fb_get_vtable_for_op(FB_OP_DROTMG);
    if (backend && backend->drotmg) {
        backend->drotmg(d1, d2, x1, y1, param);
    } else {
        fprintf(stderr, "fb_drotmg: No backend available\n");
    }
}

/* ==== SWAP: exchange two vectors ==== */

void fb_sswap(const int N, float *X, const int incX, float *Y, const int incY)
{
    const fb_backend_vtable_t *backend = fb_get_vtable_for_op(FB_OP_SSWAP);
    if (backend && backend->sswap) {
        backend->sswap(N, X, incX, Y, incY);
    } else {
        fprintf(stderr, "fb_sswap: No backend available\n");
    }
}

void fb_dswap(const int N, double *X, const int incX, double *Y, const int incY)
{
    const fb_backend_vtable_t *backend = fb_get_vtable_for_op(FB_OP_DSWAP);
    if (backend && backend->dswap) {
        backend->dswap(N, X, incX, Y, incY);
    } else {
        fprintf(stderr, "fb_dswap: No backend available\n");
    }
}

/* ==== ASUM: sum of absolute values ==== */

float fb_sasum(const int N, const float *X, const int incX)
{
    const fb_backend_vtable_t *backend = fb_get_vtable_for_op(FB_OP_SASUM);
    if (backend && backend->sasum) {
        return backend->sasum(N, X, incX);
    }
    fprintf(stderr, "fb_sasum: No backend available\n");
    return 0.0f;
}

double fb_dasum(const int N, const double *X, const int incX)
{
    const fb_backend_vtable_t *backend = fb_get_vtable_for_op(FB_OP_DASUM);
    if (backend && backend->dasum) {
        return backend->dasum(N, X, incX);
    }
    fprintf(stderr, "fb_dasum: No backend available\n");
    return 0.0;
}

/* ==== IAMAX: index of maximum absolute value ==== */

size_t fb_isamax(const int N, const float *X, const int incX)
{
    const fb_backend_vtable_t *backend = fb_get_vtable_for_op(FB_OP_ISAMAX);
    if (backend && backend->isamax) {
        return (size_t)backend->isamax(N, X, incX);
    }
    fprintf(stderr, "fb_isamax: No backend available\n");
    return 0;
}

size_t fb_idamax(const int N, const double *X, const int incX)
{
    const fb_backend_vtable_t *backend = fb_get_vtable_for_op(FB_OP_IDAMAX);
    if (backend && backend->idamax) {
        return (size_t)backend->idamax(N, X, incX);
    }
    fprintf(stderr, "fb_idamax: No backend available\n");
    return 0;
}

/* ==== Complex single-precision (C) Level 1 ==== */

void fb_cswap(const int N, void *X, const int incX, void *Y, const int incY)
{
    const fb_backend_vtable_t *backend = fb_get_vtable_for_op(FB_OP_CSWAP);
    if (backend && backend->cswap) {
        backend->cswap(N, (fb_complex_float_t *)X, incX,
                       (fb_complex_float_t *)Y, incY);
    } else {
        fprintf(stderr, "fb_cswap: No backend available\n");
    }
}

void fb_cscal(const int N, const void *alpha, void *X, const int incX)
{
    const fb_backend_vtable_t *backend = fb_get_vtable_for_op(FB_OP_CSCAL);
    if (backend && backend->cscal) {
      backend->cscal(N, (const fb_complex_float_t *)alpha,
                     (fb_complex_float_t *)X, incX);
    } else {
        fprintf(stderr, "fb_cscal: No backend available\n");
    }
}

void fb_ccopy(const int N, const void *X, const int incX, void *Y, const int incY)
{
    const fb_backend_vtable_t *backend = fb_get_vtable_for_op(FB_OP_CCOPY);
    if (backend && backend->ccopy) {
        backend->ccopy(N, (const fb_complex_float_t *)X, incX,
                       (fb_complex_float_t *)Y, incY);
    } else {
        fprintf(stderr, "fb_ccopy: No backend available\n");
    }
}

void fb_caxpy(const int N, const void *alpha, const void *X, const int incX,
              void *Y, const int incY)
{
    const fb_backend_vtable_t *backend = fb_get_vtable_for_op(FB_OP_CAXPY);
    if (backend && backend->caxpy) {
      backend->caxpy(N, (const fb_complex_float_t *)alpha,
                     (const fb_complex_float_t *)X, incX,
                     (fb_complex_float_t *)Y, incY);
    } else {
        fprintf(stderr, "fb_caxpy: No backend available\n");
    }
}

/* cdotu/cdotc: public API places result last; vtable places it first */
void fb_cdotu(const int N, const void *X, const int incX,
              const void *Y, const int incY, void *result)
{
    const fb_backend_vtable_t *backend = fb_get_vtable_for_op(FB_OP_CDOTU);
    if (backend && backend->cdotu) {
        backend->cdotu((fb_complex_float_t *)result, N,
                       (const fb_complex_float_t *)X, incX,
                       (const fb_complex_float_t *)Y, incY);
    } else {
        fprintf(stderr, "fb_cdotu: No backend available\n");
    }
}

void fb_cdotc(const int N, const void *X, const int incX,
              const void *Y, const int incY, void *result)
{
    const fb_backend_vtable_t *backend = fb_get_vtable_for_op(FB_OP_CDOTC);
    if (backend && backend->cdotc) {
        backend->cdotc((fb_complex_float_t *)result, N,
                       (const fb_complex_float_t *)X, incX,
                       (const fb_complex_float_t *)Y, incY);
    } else {
        fprintf(stderr, "fb_cdotc: No backend available\n");
    }
}

float fb_scnrm2(const int N, const void *X, const int incX)
{
    const fb_backend_vtable_t *backend = fb_get_vtable_for_op(FB_OP_SCNRM2);
    if (backend && backend->scnrm2) {
        return backend->scnrm2(N, (const fb_complex_float_t *)X, incX);
    }
    fprintf(stderr, "fb_scnrm2: No backend available\n");
    return 0.0f;
}

float fb_scasum(const int N, const void *X, const int incX)
{
    const fb_backend_vtable_t *backend = fb_get_vtable_for_op(FB_OP_SCASUM);
    if (backend && backend->scasum) {
        return backend->scasum(N, (const fb_complex_float_t *)X, incX);
    }
    fprintf(stderr, "fb_scasum: No backend available\n");
    return 0.0f;
}

size_t fb_icamax(const int N, const void *X, const int incX)
{
    const fb_backend_vtable_t *backend = fb_get_vtable_for_op(FB_OP_ICAMAX);
    if (backend && backend->icamax) {
        return (size_t)backend->icamax(N, (const fb_complex_float_t *)X, incX);
    }
    fprintf(stderr, "fb_icamax: No backend available\n");
    return 0;
}

/* ==== Complex double-precision (Z) Level 1 ==== */

void fb_zswap(const int N, void *X, const int incX, void *Y, const int incY)
{
    const fb_backend_vtable_t *backend = fb_get_vtable_for_op(FB_OP_ZSWAP);
    if (backend && backend->zswap) {
        backend->zswap(N, (fb_complex_double_t *)X, incX,
                       (fb_complex_double_t *)Y, incY);
    } else {
        fprintf(stderr, "fb_zswap: No backend available\n");
    }
}

void fb_zscal(const int N, const void *alpha, void *X, const int incX)
{
    const fb_backend_vtable_t *backend = fb_get_vtable_for_op(FB_OP_ZSCAL);
    if (backend && backend->zscal) {
      backend->zscal(N, (const fb_complex_double_t *)alpha,
                     (fb_complex_double_t *)X, incX);
    } else {
        fprintf(stderr, "fb_zscal: No backend available\n");
    }
}

void fb_zcopy(const int N, const void *X, const int incX, void *Y, const int incY)
{
    const fb_backend_vtable_t *backend = fb_get_vtable_for_op(FB_OP_ZCOPY);
    if (backend && backend->zcopy) {
        backend->zcopy(N, (const fb_complex_double_t *)X, incX,
                       (fb_complex_double_t *)Y, incY);
    } else {
        fprintf(stderr, "fb_zcopy: No backend available\n");
    }
}

void fb_zaxpy(const int N, const void *alpha, const void *X, const int incX,
              void *Y, const int incY)
{
    const fb_backend_vtable_t *backend = fb_get_vtable_for_op(FB_OP_ZAXPY);
    if (backend && backend->zaxpy) {
      backend->zaxpy(N, (const fb_complex_double_t *)alpha,
                     (const fb_complex_double_t *)X, incX,
                     (fb_complex_double_t *)Y, incY);
    } else {
        fprintf(stderr, "fb_zaxpy: No backend available\n");
    }
}

void fb_zdotu(const int N, const void *X, const int incX,
              const void *Y, const int incY, void *result)
{
    const fb_backend_vtable_t *backend = fb_get_vtable_for_op(FB_OP_ZDOTU);
    if (backend && backend->zdotu) {
        backend->zdotu((fb_complex_double_t *)result, N,
                       (const fb_complex_double_t *)X, incX,
                       (const fb_complex_double_t *)Y, incY);
    } else {
        fprintf(stderr, "fb_zdotu: No backend available\n");
    }
}

void fb_zdotc(const int N, const void *X, const int incX,
              const void *Y, const int incY, void *result)
{
    const fb_backend_vtable_t *backend = fb_get_vtable_for_op(FB_OP_ZDOTC);
    if (backend && backend->zdotc) {
        backend->zdotc((fb_complex_double_t *)result, N,
                       (const fb_complex_double_t *)X, incX,
                       (const fb_complex_double_t *)Y, incY);
    } else {
        fprintf(stderr, "fb_zdotc: No backend available\n");
    }
}

double fb_dznrm2(const int N, const void *X, const int incX)
{
    const fb_backend_vtable_t *backend = fb_get_vtable_for_op(FB_OP_DZNRM2);
    if (backend && backend->dznrm2) {
        return backend->dznrm2(N, (const fb_complex_double_t *)X, incX);
    }
    fprintf(stderr, "fb_dznrm2: No backend available\n");
    return 0.0;
}

double fb_dzasum(const int N, const void *X, const int incX)
{
    const fb_backend_vtable_t *backend = fb_get_vtable_for_op(FB_OP_DZASUM);
    if (backend && backend->dzasum) {
        return backend->dzasum(N, (const fb_complex_double_t *)X, incX);
    }
    fprintf(stderr, "fb_dzasum: No backend available\n");
    return 0.0;
}

size_t fb_izamax(const int N, const void *X, const int incX)
{
    const fb_backend_vtable_t *backend = fb_get_vtable_for_op(FB_OP_IZAMAX);
    if (backend && backend->izamax) {
        return (size_t)backend->izamax(N, (const fb_complex_double_t *)X, incX);
    }
    fprintf(stderr, "fb_izamax: No backend available\n");
    return 0;
}
