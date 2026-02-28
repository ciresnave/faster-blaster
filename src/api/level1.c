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
