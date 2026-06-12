#include <stdio.h>
#include <string.h>

#include "../src/core/conv_thunks.h"

typedef void (*test_sgbmvx_cblas_fn)(fb_layout_t layout, char trans, int m,
                                     int n, int kl, int ku, float alpha,
                                     float *ab, int ldab, float *x, int incx,
                                     float beta, float *y, int incy);
typedef void (*test_dgbmvx_cblas_fn)(fb_layout_t layout, char trans, int m,
                                     int n, int kl, int ku, double alpha,
                                     double *ab, int ldab, double *x,
                                     int incx, double beta, double *y,
                                     int incy);
typedef void (*test_cgbmvx_cblas_fn)(fb_layout_t layout, char trans, int m,
                                     int n, int kl, int ku,
                                     fb_complex_float_t alpha,
                                     fb_complex_float_t *ab, int ldab,
                                     fb_complex_float_t *x, int incx,
                                     fb_complex_float_t beta,
                                     fb_complex_float_t *y, int incy);
typedef void (*test_zgbmvx_cblas_fn)(fb_layout_t layout, char trans, int m,
                                     int n, int kl, int ku,
                                     fb_complex_double_t alpha,
                                     fb_complex_double_t *ab, int ldab,
                                     fb_complex_double_t *x, int incx,
                                     fb_complex_double_t beta,
                                     fb_complex_double_t *y, int incy);

typedef void (*fb_sgbmvx_fortran_slot_fn)(char *trans, int *m, int *n,
                                          int *kl, int *ku, float *alpha,
                                          float *ab, int *ldab, float *x,
                                          int *incx, float *beta, float *y,
                                          int *incy);
typedef void (*fb_dgbmvx_fortran_slot_fn)(char *trans, int *m, int *n,
                                          int *kl, int *ku, double *alpha,
                                          double *ab, int *ldab, double *x,
                                          int *incx, double *beta,
                                          double *y, int *incy);
typedef void (*fb_cgbmvx_fortran_slot_fn)(char *trans, int *m, int *n,
                                          int *kl, int *ku,
                                          fb_complex_float_t *alpha,
                                          fb_complex_float_t *ab, int *ldab,
                                          fb_complex_float_t *x, int *incx,
                                          fb_complex_float_t *beta,
                                          fb_complex_float_t *y, int *incy);
typedef void (*fb_zgbmvx_fortran_slot_fn)(char *trans, int *m, int *n,
                                          int *kl, int *ku,
                                          fb_complex_double_t *alpha,
                                          fb_complex_double_t *ab, int *ldab,
                                          fb_complex_double_t *x, int *incx,
                                          fb_complex_double_t *beta,
                                          fb_complex_double_t *y, int *incy);

static struct {
    int called;
    char trans;
    int m;
    int n;
    int kl;
    int ku;
    int ldab;
    int incx;
    int incy;
    float alpha;
    float beta;
    float ab[9];
} g_sgbmvx_fortran_call;

static struct {
    int called;
    fb_layout_t layout;
    char trans;
    int m;
    int n;
    int kl;
    int ku;
    int ldab;
    int incx;
    int incy;
    float alpha;
    float beta;
    float *ab;
    float *x;
    float *y;
} g_sgbmvx_cblas_call;

static struct {
    int called;
    char trans;
    int m;
    int n;
    int kl;
    int ku;
    int ldab;
    int incx;
    int incy;
    double alpha;
    double beta;
    double ab[9];
} g_dgbmvx_fortran_call;

static struct {
    int called;
    fb_layout_t layout;
    char trans;
    int m;
    int n;
    int kl;
    int ku;
    int ldab;
    int incx;
    int incy;
    double alpha;
    double beta;
    double *ab;
    double *x;
    double *y;
} g_dgbmvx_cblas_call;

static struct {
    int called;
    char trans;
    int m;
    int n;
    int kl;
    int ku;
    int ldab;
    int incx;
    int incy;
    fb_complex_float_t alpha;
    fb_complex_float_t beta;
    fb_complex_float_t ab[9];
} g_cgbmvx_fortran_call;

static struct {
    int called;
    fb_layout_t layout;
    char trans;
    int m;
    int n;
    int kl;
    int ku;
    int ldab;
    int incx;
    int incy;
    fb_complex_float_t alpha;
    fb_complex_float_t beta;
    fb_complex_float_t *ab;
    fb_complex_float_t *x;
    fb_complex_float_t *y;
} g_cgbmvx_cblas_call;

static struct {
    int called;
    char trans;
    int m;
    int n;
    int kl;
    int ku;
    int ldab;
    int incx;
    int incy;
    fb_complex_double_t alpha;
    fb_complex_double_t beta;
    fb_complex_double_t ab[9];
} g_zgbmvx_fortran_call;

static struct {
    int called;
    fb_layout_t layout;
    char trans;
    int m;
    int n;
    int kl;
    int ku;
    int ldab;
    int incx;
    int incy;
    fb_complex_double_t alpha;
    fb_complex_double_t beta;
    fb_complex_double_t *ab;
    fb_complex_double_t *x;
    fb_complex_double_t *y;
} g_zgbmvx_cblas_call;

static fb_complex_float_t make_cf32(float real_value, float imag_value)
{
    fb_complex_float_t value;
    __real__ value = real_value;
    __imag__ value = imag_value;
    return value;
}

static int cf32_eq(fb_complex_float_t lhs, fb_complex_float_t rhs)
{
    return __real__ lhs == __real__ rhs && __imag__ lhs == __imag__ rhs;
}

static fb_complex_double_t make_cd64(double real_value, double imag_value)
{
    fb_complex_double_t value;
    __real__ value = real_value;
    __imag__ value = imag_value;
    return value;
}

static int cd64_eq(fb_complex_double_t lhs, fb_complex_double_t rhs)
{
    return __real__ lhs == __real__ rhs && __imag__ lhs == __imag__ rhs;
}

static void stub_sgbmvx_fortran(char *trans, int *m, int *n, int *kl, int *ku,
                                float *alpha, float *ab, int *ldab, float *x,
                                int *incx, float *beta, float *y, int *incy)
{
    (void)x;
    g_sgbmvx_fortran_call.called += 1;
    g_sgbmvx_fortran_call.trans = *trans;
    g_sgbmvx_fortran_call.m = *m;
    g_sgbmvx_fortran_call.n = *n;
    g_sgbmvx_fortran_call.kl = *kl;
    g_sgbmvx_fortran_call.ku = *ku;
    g_sgbmvx_fortran_call.ldab = *ldab;
    g_sgbmvx_fortran_call.incx = *incx;
    g_sgbmvx_fortran_call.incy = *incy;
    g_sgbmvx_fortran_call.alpha = *alpha;
    g_sgbmvx_fortran_call.beta = *beta;
    memcpy(g_sgbmvx_fortran_call.ab, ab, sizeof(g_sgbmvx_fortran_call.ab));
    y[0] = 91.0f;
    y[1] = 92.0f;
    y[2] = 93.0f;
}

static void stub_sgbmvx_cblas(fb_layout_t layout, char trans, int m, int n,
                              int kl, int ku, float alpha, float *ab, int ldab,
                              float *x, int incx, float beta, float *y,
                              int incy)
{
    g_sgbmvx_cblas_call.called += 1;
    g_sgbmvx_cblas_call.layout = layout;
    g_sgbmvx_cblas_call.trans = trans;
    g_sgbmvx_cblas_call.m = m;
    g_sgbmvx_cblas_call.n = n;
    g_sgbmvx_cblas_call.kl = kl;
    g_sgbmvx_cblas_call.ku = ku;
    g_sgbmvx_cblas_call.ldab = ldab;
    g_sgbmvx_cblas_call.incx = incx;
    g_sgbmvx_cblas_call.incy = incy;
    g_sgbmvx_cblas_call.alpha = alpha;
    g_sgbmvx_cblas_call.beta = beta;
    g_sgbmvx_cblas_call.ab = ab;
    g_sgbmvx_cblas_call.x = x;
    g_sgbmvx_cblas_call.y = y;
    y[0] = 71.0f;
}

static void stub_dgbmvx_fortran(char *trans, int *m, int *n, int *kl, int *ku,
                                double *alpha, double *ab, int *ldab,
                                double *x, int *incx, double *beta, double *y,
                                int *incy)
{
    (void)x;
    g_dgbmvx_fortran_call.called += 1;
    g_dgbmvx_fortran_call.trans = *trans;
    g_dgbmvx_fortran_call.m = *m;
    g_dgbmvx_fortran_call.n = *n;
    g_dgbmvx_fortran_call.kl = *kl;
    g_dgbmvx_fortran_call.ku = *ku;
    g_dgbmvx_fortran_call.ldab = *ldab;
    g_dgbmvx_fortran_call.incx = *incx;
    g_dgbmvx_fortran_call.incy = *incy;
    g_dgbmvx_fortran_call.alpha = *alpha;
    g_dgbmvx_fortran_call.beta = *beta;
    memcpy(g_dgbmvx_fortran_call.ab, ab, sizeof(g_dgbmvx_fortran_call.ab));
    y[0] = 191.0;
    y[1] = 192.0;
    y[2] = 193.0;
}

static void stub_dgbmvx_cblas(fb_layout_t layout, char trans, int m, int n,
                              int kl, int ku, double alpha, double *ab,
                              int ldab, double *x, int incx, double beta,
                              double *y, int incy)
{
    g_dgbmvx_cblas_call.called += 1;
    g_dgbmvx_cblas_call.layout = layout;
    g_dgbmvx_cblas_call.trans = trans;
    g_dgbmvx_cblas_call.m = m;
    g_dgbmvx_cblas_call.n = n;
    g_dgbmvx_cblas_call.kl = kl;
    g_dgbmvx_cblas_call.ku = ku;
    g_dgbmvx_cblas_call.ldab = ldab;
    g_dgbmvx_cblas_call.incx = incx;
    g_dgbmvx_cblas_call.incy = incy;
    g_dgbmvx_cblas_call.alpha = alpha;
    g_dgbmvx_cblas_call.beta = beta;
    g_dgbmvx_cblas_call.ab = ab;
    g_dgbmvx_cblas_call.x = x;
    g_dgbmvx_cblas_call.y = y;
    y[0] = 171.0;
}

static void stub_cgbmvx_fortran(char *trans, int *m, int *n, int *kl, int *ku,
                                fb_complex_float_t *alpha,
                                fb_complex_float_t *ab, int *ldab,
                                fb_complex_float_t *x, int *incx,
                                fb_complex_float_t *beta,
                                fb_complex_float_t *y, int *incy)
{
    (void)x;
    g_cgbmvx_fortran_call.called += 1;
    g_cgbmvx_fortran_call.trans = *trans;
    g_cgbmvx_fortran_call.m = *m;
    g_cgbmvx_fortran_call.n = *n;
    g_cgbmvx_fortran_call.kl = *kl;
    g_cgbmvx_fortran_call.ku = *ku;
    g_cgbmvx_fortran_call.ldab = *ldab;
    g_cgbmvx_fortran_call.incx = *incx;
    g_cgbmvx_fortran_call.incy = *incy;
    g_cgbmvx_fortran_call.alpha = *alpha;
    g_cgbmvx_fortran_call.beta = *beta;
    memcpy(g_cgbmvx_fortran_call.ab, ab, sizeof(g_cgbmvx_fortran_call.ab));
    y[0] = make_cf32(31.0f, -2.0f);
    y[1] = make_cf32(32.0f, -3.0f);
    y[2] = make_cf32(33.0f, -4.0f);
}

static void stub_cgbmvx_cblas(fb_layout_t layout, char trans, int m, int n,
                              int kl, int ku, fb_complex_float_t alpha,
                              fb_complex_float_t *ab, int ldab,
                              fb_complex_float_t *x, int incx,
                              fb_complex_float_t beta,
                              fb_complex_float_t *y, int incy)
{
    g_cgbmvx_cblas_call.called += 1;
    g_cgbmvx_cblas_call.layout = layout;
    g_cgbmvx_cblas_call.trans = trans;
    g_cgbmvx_cblas_call.m = m;
    g_cgbmvx_cblas_call.n = n;
    g_cgbmvx_cblas_call.kl = kl;
    g_cgbmvx_cblas_call.ku = ku;
    g_cgbmvx_cblas_call.ldab = ldab;
    g_cgbmvx_cblas_call.incx = incx;
    g_cgbmvx_cblas_call.incy = incy;
    g_cgbmvx_cblas_call.alpha = alpha;
    g_cgbmvx_cblas_call.beta = beta;
    g_cgbmvx_cblas_call.ab = ab;
    g_cgbmvx_cblas_call.x = x;
    g_cgbmvx_cblas_call.y = y;
    y[0] = make_cf32(41.0f, 5.0f);
}

static void stub_zgbmvx_fortran(char *trans, int *m, int *n, int *kl, int *ku,
                                fb_complex_double_t *alpha,
                                fb_complex_double_t *ab, int *ldab,
                                fb_complex_double_t *x, int *incx,
                                fb_complex_double_t *beta,
                                fb_complex_double_t *y, int *incy)
{
    (void)x;
    g_zgbmvx_fortran_call.called += 1;
    g_zgbmvx_fortran_call.trans = *trans;
    g_zgbmvx_fortran_call.m = *m;
    g_zgbmvx_fortran_call.n = *n;
    g_zgbmvx_fortran_call.kl = *kl;
    g_zgbmvx_fortran_call.ku = *ku;
    g_zgbmvx_fortran_call.ldab = *ldab;
    g_zgbmvx_fortran_call.incx = *incx;
    g_zgbmvx_fortran_call.incy = *incy;
    g_zgbmvx_fortran_call.alpha = *alpha;
    g_zgbmvx_fortran_call.beta = *beta;
    memcpy(g_zgbmvx_fortran_call.ab, ab, sizeof(g_zgbmvx_fortran_call.ab));
    y[0] = make_cd64(131.0, -12.0);
    y[1] = make_cd64(132.0, -13.0);
    y[2] = make_cd64(133.0, -14.0);
}

static void stub_zgbmvx_cblas(fb_layout_t layout, char trans, int m, int n,
                              int kl, int ku, fb_complex_double_t alpha,
                              fb_complex_double_t *ab, int ldab,
                              fb_complex_double_t *x, int incx,
                              fb_complex_double_t beta,
                              fb_complex_double_t *y, int incy)
{
    g_zgbmvx_cblas_call.called += 1;
    g_zgbmvx_cblas_call.layout = layout;
    g_zgbmvx_cblas_call.trans = trans;
    g_zgbmvx_cblas_call.m = m;
    g_zgbmvx_cblas_call.n = n;
    g_zgbmvx_cblas_call.kl = kl;
    g_zgbmvx_cblas_call.ku = ku;
    g_zgbmvx_cblas_call.ldab = ldab;
    g_zgbmvx_cblas_call.incx = incx;
    g_zgbmvx_cblas_call.incy = incy;
    g_zgbmvx_cblas_call.alpha = alpha;
    g_zgbmvx_cblas_call.beta = beta;
    g_zgbmvx_cblas_call.ab = ab;
    g_zgbmvx_cblas_call.x = x;
    g_zgbmvx_cblas_call.y = y;
    y[0] = make_cd64(141.0, 15.0);
}

static int check_sgbmvx_fortran_to_cblas(void)
{
    fb_backend_vtable_t vtable;
    test_sgbmvx_cblas_fn thunk = NULL;
    float ab_row[9] = {
        0.0f, 11.0f, 22.0f,
        10.0f, 21.0f, 32.0f,
        20.0f, 31.0f, 0.0f
    };
    float expected_ab_col[9] = {
        0.0f, 10.0f, 20.0f,
        11.0f, 21.0f, 31.0f,
        22.0f, 32.0f, 0.0f
    };
    float x[3] = { 1.0f, 2.0f, 3.0f };
    float y[3] = { 4.0f, 5.0f, 6.0f };

    memset(&vtable, 0, sizeof(vtable));
    memset(&g_sgbmvx_fortran_call, 0, sizeof(g_sgbmvx_fortran_call));

    vtable.ext_ops[FB_OP_SGBMVX][FB_CONV_FORTRAN] =
        (fb_generic_fn)(void (*)(void))stub_sgbmvx_fortran;
    fb_install_conv_thunks(&vtable, FB_OP_SGBMVX);

    thunk = (test_sgbmvx_cblas_fn)vtable.ext_ops[FB_OP_SGBMVX][FB_CONV_CBLAS];
    if (!thunk) {
        fprintf(stderr, "[FAIL] SGBMVX Fortran->CBLAS thunk was not installed\n");
        return 1;
    }

    thunk(FB_LAYOUT_ROW_MAJOR, 'T', 3, 3, 1, 1, 2.0f, ab_row, 3, x, 1, 0.5f,
          y, 1);
    if (g_sgbmvx_fortran_call.called != 1 || g_sgbmvx_fortran_call.trans != 'T' ||
        g_sgbmvx_fortran_call.m != 3 || g_sgbmvx_fortran_call.n != 3 ||
        g_sgbmvx_fortran_call.kl != 1 || g_sgbmvx_fortran_call.ku != 1 ||
        g_sgbmvx_fortran_call.ldab != 3 || g_sgbmvx_fortran_call.incx != 1 ||
        g_sgbmvx_fortran_call.incy != 1 ||
        g_sgbmvx_fortran_call.alpha != 2.0f ||
        g_sgbmvx_fortran_call.beta != 0.5f ||
        memcmp(g_sgbmvx_fortran_call.ab, expected_ab_col,
               sizeof(expected_ab_col)) != 0 ||
        y[0] != 91.0f || y[1] != 92.0f || y[2] != 93.0f) {
        fprintf(stderr, "[FAIL] SGBMVX Fortran->CBLAS thunk did not translate row-major band storage correctly\n");
        return 1;
    }

    printf("[PASS] SGBMVX Fortran->CBLAS thunk translates row-major band storage and forwards transpose metadata\n");
    return 0;
}

static int check_sgbmvx_cblas_to_fortran(void)
{
    fb_backend_vtable_t vtable;
    fb_sgbmvx_fortran_slot_fn thunk = NULL;
    char trans = 'N';
    int m = 3;
    int n = 3;
    int kl = 1;
    int ku = 1;
    float alpha = 1.5f;
    float ab[9] = { 0.0f };
    int ldab = 3;
    float x[3] = { 1.0f, -1.0f, 2.0f };
    int incx = 1;
    float beta = 0.25f;
    float y[3] = { 0.0f, 0.0f, 0.0f };
    int incy = 1;

    memset(&vtable, 0, sizeof(vtable));
    memset(&g_sgbmvx_cblas_call, 0, sizeof(g_sgbmvx_cblas_call));

    vtable.ext_ops[FB_OP_SGBMVX][FB_CONV_CBLAS] =
        (fb_generic_fn)(void (*)(void))stub_sgbmvx_cblas;
    fb_install_conv_thunks(&vtable, FB_OP_SGBMVX);

    thunk = (fb_sgbmvx_fortran_slot_fn)vtable.ext_ops[FB_OP_SGBMVX][FB_CONV_FORTRAN];
    if (!thunk) {
        fprintf(stderr, "[FAIL] SGBMVX CBLAS->Fortran thunk was not installed\n");
        return 1;
    }

    thunk(&trans, &m, &n, &kl, &ku, &alpha, ab, &ldab, x, &incx, &beta, y,
          &incy);
    if (g_sgbmvx_cblas_call.called != 1 ||
        g_sgbmvx_cblas_call.layout != FB_LAYOUT_COL_MAJOR ||
        g_sgbmvx_cblas_call.trans != 'N' || g_sgbmvx_cblas_call.m != 3 ||
        g_sgbmvx_cblas_call.n != 3 || g_sgbmvx_cblas_call.kl != 1 ||
        g_sgbmvx_cblas_call.ku != 1 || g_sgbmvx_cblas_call.ldab != 3 ||
        g_sgbmvx_cblas_call.incx != 1 || g_sgbmvx_cblas_call.incy != 1 ||
        g_sgbmvx_cblas_call.alpha != 1.5f ||
        g_sgbmvx_cblas_call.beta != 0.25f ||
        g_sgbmvx_cblas_call.ab != ab || g_sgbmvx_cblas_call.x != x ||
        g_sgbmvx_cblas_call.y != y || y[0] != 71.0f) {
        fprintf(stderr, "[FAIL] SGBMVX CBLAS->Fortran thunk delegated incorrectly\n");
        return 1;
    }

    printf("[PASS] SGBMVX CBLAS->Fortran thunk forwards band metadata through the column-major C bridge\n");
    return 0;
}

static int check_dgbmvx_fortran_to_cblas(void)
{
    fb_backend_vtable_t vtable;
    test_dgbmvx_cblas_fn thunk = NULL;
    double ab_row[9] = {
        0.0, 11.0, 22.0,
        10.0, 21.0, 32.0,
        20.0, 31.0, 0.0
    };
    double expected_ab_col[9] = {
        0.0, 10.0, 20.0,
        11.0, 21.0, 31.0,
        22.0, 32.0, 0.0
    };
    double x[3] = { 1.0, 2.0, 3.0 };
    double y[3] = { 4.0, 5.0, 6.0 };

    memset(&vtable, 0, sizeof(vtable));
    memset(&g_dgbmvx_fortran_call, 0, sizeof(g_dgbmvx_fortran_call));

    vtable.ext_ops[FB_OP_DGBMVX][FB_CONV_FORTRAN] =
        (fb_generic_fn)(void (*)(void))stub_dgbmvx_fortran;
    fb_install_conv_thunks(&vtable, FB_OP_DGBMVX);

    thunk = (test_dgbmvx_cblas_fn)vtable.ext_ops[FB_OP_DGBMVX][FB_CONV_CBLAS];
    if (!thunk) {
        fprintf(stderr, "[FAIL] DGBMVX Fortran->CBLAS thunk was not installed\n");
        return 1;
    }

    thunk(FB_LAYOUT_ROW_MAJOR, 'T', 3, 3, 1, 1, 2.0, ab_row, 3, x, 1, 0.5,
          y, 1);
    if (g_dgbmvx_fortran_call.called != 1 || g_dgbmvx_fortran_call.trans != 'T' ||
        g_dgbmvx_fortran_call.m != 3 || g_dgbmvx_fortran_call.n != 3 ||
        g_dgbmvx_fortran_call.kl != 1 || g_dgbmvx_fortran_call.ku != 1 ||
        g_dgbmvx_fortran_call.ldab != 3 || g_dgbmvx_fortran_call.incx != 1 ||
        g_dgbmvx_fortran_call.incy != 1 ||
        g_dgbmvx_fortran_call.alpha != 2.0 ||
        g_dgbmvx_fortran_call.beta != 0.5 ||
        memcmp(g_dgbmvx_fortran_call.ab, expected_ab_col,
               sizeof(expected_ab_col)) != 0 ||
        y[0] != 191.0 || y[1] != 192.0 || y[2] != 193.0) {
        fprintf(stderr, "[FAIL] DGBMVX Fortran->CBLAS thunk did not translate row-major band storage correctly\n");
        return 1;
    }

    printf("[PASS] DGBMVX Fortran->CBLAS thunk translates row-major band storage and forwards transpose metadata\n");
    return 0;
}

static int check_dgbmvx_cblas_to_fortran(void)
{
    fb_backend_vtable_t vtable;
    fb_dgbmvx_fortran_slot_fn thunk = NULL;
    char trans = 'N';
    int m = 3;
    int n = 3;
    int kl = 1;
    int ku = 1;
    double alpha = 1.5;
    double ab[9] = { 0.0 };
    int ldab = 3;
    double x[3] = { 1.0, -1.0, 2.0 };
    int incx = 1;
    double beta = 0.25;
    double y[3] = { 0.0, 0.0, 0.0 };
    int incy = 1;

    memset(&vtable, 0, sizeof(vtable));
    memset(&g_dgbmvx_cblas_call, 0, sizeof(g_dgbmvx_cblas_call));

    vtable.ext_ops[FB_OP_DGBMVX][FB_CONV_CBLAS] =
        (fb_generic_fn)(void (*)(void))stub_dgbmvx_cblas;
    fb_install_conv_thunks(&vtable, FB_OP_DGBMVX);

    thunk = (fb_dgbmvx_fortran_slot_fn)vtable.ext_ops[FB_OP_DGBMVX][FB_CONV_FORTRAN];
    if (!thunk) {
        fprintf(stderr, "[FAIL] DGBMVX CBLAS->Fortran thunk was not installed\n");
        return 1;
    }

    thunk(&trans, &m, &n, &kl, &ku, &alpha, ab, &ldab, x, &incx, &beta, y,
          &incy);
    if (g_dgbmvx_cblas_call.called != 1 ||
        g_dgbmvx_cblas_call.layout != FB_LAYOUT_COL_MAJOR ||
        g_dgbmvx_cblas_call.trans != 'N' || g_dgbmvx_cblas_call.m != 3 ||
        g_dgbmvx_cblas_call.n != 3 || g_dgbmvx_cblas_call.kl != 1 ||
        g_dgbmvx_cblas_call.ku != 1 || g_dgbmvx_cblas_call.ldab != 3 ||
        g_dgbmvx_cblas_call.incx != 1 || g_dgbmvx_cblas_call.incy != 1 ||
        g_dgbmvx_cblas_call.alpha != 1.5 ||
        g_dgbmvx_cblas_call.beta != 0.25 ||
        g_dgbmvx_cblas_call.ab != ab || g_dgbmvx_cblas_call.x != x ||
        g_dgbmvx_cblas_call.y != y || y[0] != 171.0) {
        fprintf(stderr, "[FAIL] DGBMVX CBLAS->Fortran thunk delegated incorrectly\n");
        return 1;
    }

    printf("[PASS] DGBMVX CBLAS->Fortran thunk forwards band metadata through the column-major C bridge\n");
    return 0;
}

static int check_cgbmvx_fortran_to_cblas(void)
{
    fb_backend_vtable_t vtable;
    test_cgbmvx_cblas_fn thunk = NULL;
    fb_complex_float_t ab_row[9] = {
        make_cf32(0.0f, 0.0f), make_cf32(11.0f, 1.0f), make_cf32(22.0f, 2.0f),
        make_cf32(10.0f, -1.0f), make_cf32(21.0f, -2.0f), make_cf32(32.0f, -3.0f),
        make_cf32(20.0f, 3.0f), make_cf32(31.0f, 4.0f), make_cf32(0.0f, 0.0f)
    };
    fb_complex_float_t expected_ab_col[9] = {
        make_cf32(0.0f, 0.0f), make_cf32(10.0f, -1.0f), make_cf32(20.0f, 3.0f),
        make_cf32(11.0f, 1.0f), make_cf32(21.0f, -2.0f), make_cf32(31.0f, 4.0f),
        make_cf32(22.0f, 2.0f), make_cf32(32.0f, -3.0f), make_cf32(0.0f, 0.0f)
    };
    fb_complex_float_t x[3] = {
        make_cf32(1.0f, 1.0f), make_cf32(2.0f, -1.0f), make_cf32(3.0f, 0.5f)
    };
    fb_complex_float_t y[3] = {
        make_cf32(4.0f, 0.0f), make_cf32(5.0f, 0.0f), make_cf32(6.0f, 0.0f)
    };

    memset(&vtable, 0, sizeof(vtable));
    memset(&g_cgbmvx_fortran_call, 0, sizeof(g_cgbmvx_fortran_call));

    vtable.ext_ops[FB_OP_CGBMVX][FB_CONV_FORTRAN] =
        (fb_generic_fn)(void (*)(void))stub_cgbmvx_fortran;
    fb_install_conv_thunks(&vtable, FB_OP_CGBMVX);

    thunk = (test_cgbmvx_cblas_fn)vtable.ext_ops[FB_OP_CGBMVX][FB_CONV_CBLAS];
    if (!thunk) {
        fprintf(stderr, "[FAIL] CGBMVX Fortran->CBLAS thunk was not installed\n");
        return 1;
    }

    thunk(FB_LAYOUT_ROW_MAJOR, 'C', 3, 3, 1, 1, make_cf32(2.0f, -1.0f),
          ab_row, 3, x, 1, make_cf32(0.5f, 0.25f), y, 1);
    if (g_cgbmvx_fortran_call.called != 1 || g_cgbmvx_fortran_call.trans != 'C' ||
        g_cgbmvx_fortran_call.m != 3 || g_cgbmvx_fortran_call.n != 3 ||
        g_cgbmvx_fortran_call.kl != 1 || g_cgbmvx_fortran_call.ku != 1 ||
        g_cgbmvx_fortran_call.ldab != 3 || g_cgbmvx_fortran_call.incx != 1 ||
        g_cgbmvx_fortran_call.incy != 1 ||
        !cf32_eq(g_cgbmvx_fortran_call.alpha, make_cf32(2.0f, -1.0f)) ||
        !cf32_eq(g_cgbmvx_fortran_call.beta, make_cf32(0.5f, 0.25f)) ||
        memcmp(g_cgbmvx_fortran_call.ab, expected_ab_col,
               sizeof(expected_ab_col)) != 0 ||
        !cf32_eq(y[0], make_cf32(31.0f, -2.0f)) ||
        !cf32_eq(y[1], make_cf32(32.0f, -3.0f)) ||
        !cf32_eq(y[2], make_cf32(33.0f, -4.0f))) {
        fprintf(stderr, "[FAIL] CGBMVX Fortran->CBLAS thunk did not translate complex row-major band storage correctly\n");
        return 1;
    }

    printf("[PASS] CGBMVX Fortran->CBLAS thunk translates complex row-major band storage and preserves conjugate-transpose metadata\n");
    return 0;
}

static int check_cgbmvx_cblas_to_fortran(void)
{
    fb_backend_vtable_t vtable;
    fb_cgbmvx_fortran_slot_fn thunk = NULL;
    char trans = 'T';
    int m = 3;
    int n = 3;
    int kl = 1;
    int ku = 1;
    fb_complex_float_t alpha = make_cf32(1.0f, 2.0f);
    fb_complex_float_t ab[9] = { 0 };
    int ldab = 3;
    fb_complex_float_t x[3] = {
        make_cf32(1.0f, 0.0f), make_cf32(2.0f, 1.0f), make_cf32(3.0f, -1.0f)
    };
    int incx = 1;
    fb_complex_float_t beta = make_cf32(-0.5f, 0.25f);
    fb_complex_float_t y[3] = { 0 };
    int incy = 1;

    memset(&vtable, 0, sizeof(vtable));
    memset(&g_cgbmvx_cblas_call, 0, sizeof(g_cgbmvx_cblas_call));

    vtable.ext_ops[FB_OP_CGBMVX][FB_CONV_CBLAS] =
        (fb_generic_fn)(void (*)(void))stub_cgbmvx_cblas;
    fb_install_conv_thunks(&vtable, FB_OP_CGBMVX);

    thunk = (fb_cgbmvx_fortran_slot_fn)vtable.ext_ops[FB_OP_CGBMVX][FB_CONV_FORTRAN];
    if (!thunk) {
        fprintf(stderr, "[FAIL] CGBMVX CBLAS->Fortran thunk was not installed\n");
        return 1;
    }

    thunk(&trans, &m, &n, &kl, &ku, &alpha, ab, &ldab, x, &incx, &beta, y,
          &incy);
    if (g_cgbmvx_cblas_call.called != 1 ||
        g_cgbmvx_cblas_call.layout != FB_LAYOUT_COL_MAJOR ||
        g_cgbmvx_cblas_call.trans != 'T' || g_cgbmvx_cblas_call.m != 3 ||
        g_cgbmvx_cblas_call.n != 3 || g_cgbmvx_cblas_call.kl != 1 ||
        g_cgbmvx_cblas_call.ku != 1 || g_cgbmvx_cblas_call.ldab != 3 ||
        g_cgbmvx_cblas_call.incx != 1 || g_cgbmvx_cblas_call.incy != 1 ||
        !cf32_eq(g_cgbmvx_cblas_call.alpha, make_cf32(1.0f, 2.0f)) ||
        !cf32_eq(g_cgbmvx_cblas_call.beta, make_cf32(-0.5f, 0.25f)) ||
        g_cgbmvx_cblas_call.ab != ab || g_cgbmvx_cblas_call.x != x ||
        g_cgbmvx_cblas_call.y != y ||
        !cf32_eq(y[0], make_cf32(41.0f, 5.0f))) {
        fprintf(stderr, "[FAIL] CGBMVX CBLAS->Fortran thunk delegated incorrectly\n");
        return 1;
    }

    printf("[PASS] CGBMVX CBLAS->Fortran thunk forwards complex band metadata through the column-major C bridge\n");
    return 0;
}

static int check_zgbmvx_fortran_to_cblas(void)
{
    fb_backend_vtable_t vtable;
    test_zgbmvx_cblas_fn thunk = NULL;
    fb_complex_double_t ab_row[9] = {
        make_cd64(0.0, 0.0), make_cd64(11.0, 1.0), make_cd64(22.0, 2.0),
        make_cd64(10.0, -1.0), make_cd64(21.0, -2.0), make_cd64(32.0, -3.0),
        make_cd64(20.0, 3.0), make_cd64(31.0, 4.0), make_cd64(0.0, 0.0)
    };
    fb_complex_double_t expected_ab_col[9] = {
        make_cd64(0.0, 0.0), make_cd64(10.0, -1.0), make_cd64(20.0, 3.0),
        make_cd64(11.0, 1.0), make_cd64(21.0, -2.0), make_cd64(31.0, 4.0),
        make_cd64(22.0, 2.0), make_cd64(32.0, -3.0), make_cd64(0.0, 0.0)
    };
    fb_complex_double_t x[3] = {
        make_cd64(1.0, 1.0), make_cd64(2.0, -1.0), make_cd64(3.0, 0.5)
    };
    fb_complex_double_t y[3] = {
        make_cd64(4.0, 0.0), make_cd64(5.0, 0.0), make_cd64(6.0, 0.0)
    };

    memset(&vtable, 0, sizeof(vtable));
    memset(&g_zgbmvx_fortran_call, 0, sizeof(g_zgbmvx_fortran_call));

    vtable.ext_ops[FB_OP_ZGBMVX][FB_CONV_FORTRAN] =
        (fb_generic_fn)(void (*)(void))stub_zgbmvx_fortran;
    fb_install_conv_thunks(&vtable, FB_OP_ZGBMVX);

    thunk = (test_zgbmvx_cblas_fn)vtable.ext_ops[FB_OP_ZGBMVX][FB_CONV_CBLAS];
    if (!thunk) {
        fprintf(stderr, "[FAIL] ZGBMVX Fortran->CBLAS thunk was not installed\n");
        return 1;
    }

    thunk(FB_LAYOUT_ROW_MAJOR, 'C', 3, 3, 1, 1, make_cd64(2.0, -1.0),
          ab_row, 3, x, 1, make_cd64(0.5, 0.25), y, 1);
    if (g_zgbmvx_fortran_call.called != 1 || g_zgbmvx_fortran_call.trans != 'C' ||
        g_zgbmvx_fortran_call.m != 3 || g_zgbmvx_fortran_call.n != 3 ||
        g_zgbmvx_fortran_call.kl != 1 || g_zgbmvx_fortran_call.ku != 1 ||
        g_zgbmvx_fortran_call.ldab != 3 || g_zgbmvx_fortran_call.incx != 1 ||
        g_zgbmvx_fortran_call.incy != 1 ||
        !cd64_eq(g_zgbmvx_fortran_call.alpha, make_cd64(2.0, -1.0)) ||
        !cd64_eq(g_zgbmvx_fortran_call.beta, make_cd64(0.5, 0.25)) ||
        memcmp(g_zgbmvx_fortran_call.ab, expected_ab_col,
               sizeof(expected_ab_col)) != 0 ||
        !cd64_eq(y[0], make_cd64(131.0, -12.0)) ||
        !cd64_eq(y[1], make_cd64(132.0, -13.0)) ||
        !cd64_eq(y[2], make_cd64(133.0, -14.0))) {
        fprintf(stderr, "[FAIL] ZGBMVX Fortran->CBLAS thunk did not translate complex row-major band storage correctly\n");
        return 1;
    }

    printf("[PASS] ZGBMVX Fortran->CBLAS thunk translates complex row-major band storage and preserves conjugate-transpose metadata\n");
    return 0;
}

static int check_zgbmvx_cblas_to_fortran(void)
{
    fb_backend_vtable_t vtable;
    fb_zgbmvx_fortran_slot_fn thunk = NULL;
    char trans = 'T';
    int m = 3;
    int n = 3;
    int kl = 1;
    int ku = 1;
    fb_complex_double_t alpha = make_cd64(1.0, 2.0);
    fb_complex_double_t ab[9] = { 0 };
    int ldab = 3;
    fb_complex_double_t x[3] = {
        make_cd64(1.0, 0.0), make_cd64(2.0, 1.0), make_cd64(3.0, -1.0)
    };
    int incx = 1;
    fb_complex_double_t beta = make_cd64(-0.5, 0.25);
    fb_complex_double_t y[3] = { 0 };
    int incy = 1;

    memset(&vtable, 0, sizeof(vtable));
    memset(&g_zgbmvx_cblas_call, 0, sizeof(g_zgbmvx_cblas_call));

    vtable.ext_ops[FB_OP_ZGBMVX][FB_CONV_CBLAS] =
        (fb_generic_fn)(void (*)(void))stub_zgbmvx_cblas;
    fb_install_conv_thunks(&vtable, FB_OP_ZGBMVX);

    thunk = (fb_zgbmvx_fortran_slot_fn)vtable.ext_ops[FB_OP_ZGBMVX][FB_CONV_FORTRAN];
    if (!thunk) {
        fprintf(stderr, "[FAIL] ZGBMVX CBLAS->Fortran thunk was not installed\n");
        return 1;
    }

    thunk(&trans, &m, &n, &kl, &ku, &alpha, ab, &ldab, x, &incx, &beta, y,
          &incy);
    if (g_zgbmvx_cblas_call.called != 1 ||
        g_zgbmvx_cblas_call.layout != FB_LAYOUT_COL_MAJOR ||
        g_zgbmvx_cblas_call.trans != 'T' || g_zgbmvx_cblas_call.m != 3 ||
        g_zgbmvx_cblas_call.n != 3 || g_zgbmvx_cblas_call.kl != 1 ||
        g_zgbmvx_cblas_call.ku != 1 || g_zgbmvx_cblas_call.ldab != 3 ||
        g_zgbmvx_cblas_call.incx != 1 || g_zgbmvx_cblas_call.incy != 1 ||
        !cd64_eq(g_zgbmvx_cblas_call.alpha, make_cd64(1.0, 2.0)) ||
        !cd64_eq(g_zgbmvx_cblas_call.beta, make_cd64(-0.5, 0.25)) ||
        g_zgbmvx_cblas_call.ab != ab || g_zgbmvx_cblas_call.x != x ||
        g_zgbmvx_cblas_call.y != y ||
        !cd64_eq(y[0], make_cd64(141.0, 15.0))) {
        fprintf(stderr, "[FAIL] ZGBMVX CBLAS->Fortran thunk delegated incorrectly\n");
        return 1;
    }

    printf("[PASS] ZGBMVX CBLAS->Fortran thunk forwards complex band metadata through the column-major C bridge\n");
    return 0;
}

int main(void)
{
    if (check_sgbmvx_fortran_to_cblas() != 0) {
        return 1;
    }
    if (check_sgbmvx_cblas_to_fortran() != 0) {
        return 1;
    }
    if (check_dgbmvx_fortran_to_cblas() != 0) {
        return 1;
    }
    if (check_dgbmvx_cblas_to_fortran() != 0) {
        return 1;
    }
    if (check_cgbmvx_fortran_to_cblas() != 0) {
        return 1;
    }
    if (check_cgbmvx_cblas_to_fortran() != 0) {
        return 1;
    }
    if (check_zgbmvx_fortran_to_cblas() != 0) {
        return 1;
    }
    if (check_zgbmvx_cblas_to_fortran() != 0) {
        return 1;
    }

    printf("Result: PASS\n");
    return 0;
}