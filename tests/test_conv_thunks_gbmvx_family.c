#include <stdio.h>
#include <string.h>

#include "../src/core/conv_thunks.h"

typedef void (*test_sgbmvx_cblas_fn)(fb_layout_t layout, char trans, int m,
                                     int n, int kl, int ku, float alpha,
                                     float *ab, int ldab, float *x, int incx,
                                     float beta, float *y, int incy);
typedef void (*test_cgbmvx_cblas_fn)(fb_layout_t layout, char trans, int m,
                                     int n, int kl, int ku,
                                     fb_complex_float_t alpha,
                                     fb_complex_float_t *ab, int ldab,
                                     fb_complex_float_t *x, int incx,
                                     fb_complex_float_t beta,
                                     fb_complex_float_t *y, int incy);

typedef void (*fb_sgbmvx_fortran_slot_fn)(char *trans, int *m, int *n,
                                          int *kl, int *ku, float *alpha,
                                          float *ab, int *ldab, float *x,
                                          int *incx, float *beta, float *y,
                                          int *incy);
typedef void (*fb_cgbmvx_fortran_slot_fn)(char *trans, int *m, int *n,
                                          int *kl, int *ku,
                                          fb_complex_float_t *alpha,
                                          fb_complex_float_t *ab, int *ldab,
                                          fb_complex_float_t *x, int *incx,
                                          fb_complex_float_t *beta,
                                          fb_complex_float_t *y, int *incy);

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

int main(void)
{
    if (check_sgbmvx_fortran_to_cblas() != 0) {
        return 1;
    }
    if (check_sgbmvx_cblas_to_fortran() != 0) {
        return 1;
    }
    if (check_cgbmvx_fortran_to_cblas() != 0) {
        return 1;
    }
    if (check_cgbmvx_cblas_to_fortran() != 0) {
        return 1;
    }

    printf("Result: PASS\n");
    return 0;
}