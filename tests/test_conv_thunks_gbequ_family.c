#include <stdio.h>
#include <string.h>

#include "../src/core/conv_thunks.h"

typedef int (*fb_sgbequ_fn)(fb_layout_t layout, int m, int n, int kl, int ku,
                            const float *ab, int ldab, float *r, float *c,
                            float *rowcnd, float *colcnd, float *amax);
typedef int (*fb_dgbequ_fn)(fb_layout_t layout, int m, int n, int kl, int ku,
                            const double *ab, int ldab, double *r, double *c,
                            double *rowcnd, double *colcnd, double *amax);
typedef int (*fb_cgbequ_fn)(fb_layout_t layout, int m, int n, int kl, int ku,
                            const fb_complex_float_t *ab, int ldab, float *r,
                            float *c, float *rowcnd, float *colcnd,
                            float *amax);
typedef int (*fb_zgbequ_fn)(fb_layout_t layout, int m, int n, int kl, int ku,
                            const fb_complex_double_t *ab, int ldab,
                            double *r, double *c, double *rowcnd,
                            double *colcnd, double *amax);

typedef void (*fb_sgbequ_fortran_slot_fn)(int *m, int *n, int *kl, int *ku,
                                          float *ab, int *ldab, float *r,
                                          float *c, float *rowcnd,
                                          float *colcnd, float *amax,
                                          int *info);
typedef void (*fb_dgbequ_fortran_slot_fn)(int *m, int *n, int *kl, int *ku,
                                          double *ab, int *ldab, double *r,
                                          double *c, double *rowcnd,
                                          double *colcnd, double *amax,
                                          int *info);
typedef void (*fb_cgbequ_fortran_slot_fn)(int *m, int *n, int *kl, int *ku,
                                          fb_complex_float_t *ab, int *ldab,
                                          float *r, float *c, float *rowcnd,
                                          float *colcnd, float *amax,
                                          int *info);
typedef void (*fb_zgbequ_fortran_slot_fn)(int *m, int *n, int *kl, int *ku,
                                          fb_complex_double_t *ab, int *ldab,
                                          double *r, double *c,
                                          double *rowcnd, double *colcnd,
                                          double *amax, int *info);

static struct {
    int calls;
    int m;
    int n;
    int kl;
    int ku;
    int ldab;
    float ab_snapshot[12];
} g_sgbequ_fortran_call;

static struct {
    int called;
    fb_layout_t layout;
    int m;
    int n;
    int kl;
    int ku;
    int ldab;
    const float *ab;
    float *r;
    float *c;
    float *rowcnd;
    float *colcnd;
    float *amax;
} g_sgbequ_cblas_call;

static struct {
    int calls;
    int m;
    int n;
    int kl;
    int ku;
    int ldab;
    double ab_snapshot[12];
} g_dgbequ_fortran_call;

static struct {
    int called;
    fb_layout_t layout;
    int m;
    int n;
    int kl;
    int ku;
    int ldab;
    const double *ab;
    double *r;
    double *c;
    double *rowcnd;
    double *colcnd;
    double *amax;
} g_dgbequ_cblas_call;

static struct {
    int calls;
    int m;
    int n;
    int kl;
    int ku;
    int ldab;
    float ab_real_snapshot[12];
} g_cgbequ_fortran_call;

static struct {
    int called;
    fb_layout_t layout;
    int m;
    int n;
    int kl;
    int ku;
    int ldab;
    const fb_complex_float_t *ab;
    float *r;
    float *c;
    float *rowcnd;
    float *colcnd;
    float *amax;
} g_cgbequ_cblas_call;

static struct {
    int calls;
    int m;
    int n;
    int kl;
    int ku;
    int ldab;
    double ab_real_snapshot[12];
} g_zgbequ_fortran_call;

static struct {
    int called;
    fb_layout_t layout;
    int m;
    int n;
    int kl;
    int ku;
    int ldab;
    const fb_complex_double_t *ab;
    double *r;
    double *c;
    double *rowcnd;
    double *colcnd;
    double *amax;
} g_zgbequ_cblas_call;

static fb_complex_float_t make_cfloat(float real_value)
{
    fb_complex_float_t value;

    __real__ value = real_value;
    __imag__ value = 0.0f;
    return value;
}

static float cfloat_real(fb_complex_float_t value)
{
    return __real__ value;
}

static fb_complex_double_t make_cdouble(double real_value)
{
    fb_complex_double_t value;

    __real__ value = real_value;
    __imag__ value = 0.0;
    return value;
}

static double cdouble_real(fb_complex_double_t value)
{
    return __real__ value;
}

static void stub_sgbequ_fortran(int *m, int *n, int *kl, int *ku, float *ab,
                                int *ldab, float *r, float *c,
                                float *rowcnd, float *colcnd, float *amax,
                                int *info)
{
    int index = 0;

    g_sgbequ_fortran_call.calls += 1;
    g_sgbequ_fortran_call.m = *m;
    g_sgbequ_fortran_call.n = *n;
    g_sgbequ_fortran_call.kl = *kl;
    g_sgbequ_fortran_call.ku = *ku;
    g_sgbequ_fortran_call.ldab = *ldab;
    for (index = 0; index < (*ldab * *n); ++index) {
        g_sgbequ_fortran_call.ab_snapshot[index] = ab[index];
    }
    for (index = 0; index < *m; ++index) {
        r[index] = (float)(index + 1);
    }
    for (index = 0; index < *n; ++index) {
        c[index] = (float)(10 + index);
    }
    *rowcnd = 0.25f;
    *colcnd = 0.5f;
    *amax = 23.0f;
    *info = 0;
}

static int stub_sgbequ_cblas(fb_layout_t layout, int m, int n, int kl, int ku,
                             const float *ab, int ldab, float *r, float *c,
                             float *rowcnd, float *colcnd, float *amax)
{
    int index = 0;

    g_sgbequ_cblas_call.called += 1;
    g_sgbequ_cblas_call.layout = layout;
    g_sgbequ_cblas_call.m = m;
    g_sgbequ_cblas_call.n = n;
    g_sgbequ_cblas_call.kl = kl;
    g_sgbequ_cblas_call.ku = ku;
    g_sgbequ_cblas_call.ldab = ldab;
    g_sgbequ_cblas_call.ab = ab;
    g_sgbequ_cblas_call.r = r;
    g_sgbequ_cblas_call.c = c;
    g_sgbequ_cblas_call.rowcnd = rowcnd;
    g_sgbequ_cblas_call.colcnd = colcnd;
    g_sgbequ_cblas_call.amax = amax;
    for (index = 0; index < m; ++index) {
        r[index] = (float)(20 + index);
    }
    for (index = 0; index < n; ++index) {
        c[index] = (float)(30 + index);
    }
    *rowcnd = 0.75f;
    *colcnd = 1.25f;
    *amax = 31.0f;
    return 181;
}

static void stub_dgbequ_fortran(int *m, int *n, int *kl, int *ku, double *ab,
                                int *ldab, double *r, double *c,
                                double *rowcnd, double *colcnd, double *amax,
                                int *info)
{
    int index = 0;

    g_dgbequ_fortran_call.calls += 1;
    g_dgbequ_fortran_call.m = *m;
    g_dgbequ_fortran_call.n = *n;
    g_dgbequ_fortran_call.kl = *kl;
    g_dgbequ_fortran_call.ku = *ku;
    g_dgbequ_fortran_call.ldab = *ldab;
    for (index = 0; index < (*ldab * *n); ++index) {
        g_dgbequ_fortran_call.ab_snapshot[index] = ab[index];
    }
    for (index = 0; index < *m; ++index) {
        r[index] = (double)(80 + index);
    }
    for (index = 0; index < *n; ++index) {
        c[index] = (double)(90 + index);
    }
    *rowcnd = 2.75;
    *colcnd = 3.0;
    *amax = 63.0;
    *info = 0;
}

static int stub_dgbequ_cblas(fb_layout_t layout, int m, int n, int kl, int ku,
                             const double *ab, int ldab, double *r, double *c,
                             double *rowcnd, double *colcnd, double *amax)
{
    int index = 0;

    g_dgbequ_cblas_call.called += 1;
    g_dgbequ_cblas_call.layout = layout;
    g_dgbequ_cblas_call.m = m;
    g_dgbequ_cblas_call.n = n;
    g_dgbequ_cblas_call.kl = kl;
    g_dgbequ_cblas_call.ku = ku;
    g_dgbequ_cblas_call.ldab = ldab;
    g_dgbequ_cblas_call.ab = ab;
    g_dgbequ_cblas_call.r = r;
    g_dgbequ_cblas_call.c = c;
    g_dgbequ_cblas_call.rowcnd = rowcnd;
    g_dgbequ_cblas_call.colcnd = colcnd;
    g_dgbequ_cblas_call.amax = amax;
    for (index = 0; index < m; ++index) {
        r[index] = (double)(100 + index);
    }
    for (index = 0; index < n; ++index) {
        c[index] = (double)(110 + index);
    }
    *rowcnd = 3.25;
    *colcnd = 3.5;
    *amax = 71.0;
    return 182;
}

static void stub_cgbequ_fortran(int *m, int *n, int *kl, int *ku,
                                fb_complex_float_t *ab, int *ldab, float *r,
                                float *c, float *rowcnd, float *colcnd,
                                float *amax, int *info)
{
    int index = 0;

    g_cgbequ_fortran_call.calls += 1;
    g_cgbequ_fortran_call.m = *m;
    g_cgbequ_fortran_call.n = *n;
    g_cgbequ_fortran_call.kl = *kl;
    g_cgbequ_fortran_call.ku = *ku;
    g_cgbequ_fortran_call.ldab = *ldab;
    for (index = 0; index < (*ldab * *n); ++index) {
        g_cgbequ_fortran_call.ab_real_snapshot[index] = cfloat_real(ab[index]);
    }
    for (index = 0; index < *m; ++index) {
        r[index] = (float)(40 + index);
    }
    for (index = 0; index < *n; ++index) {
        c[index] = (float)(50 + index);
    }
    *rowcnd = 1.5f;
    *colcnd = 1.75f;
    *amax = 43.0f;
    *info = 0;
}

static int stub_cgbequ_cblas(fb_layout_t layout, int m, int n, int kl, int ku,
                             const fb_complex_float_t *ab, int ldab, float *r,
                             float *c, float *rowcnd, float *colcnd,
                             float *amax)
{
    int index = 0;

    g_cgbequ_cblas_call.called += 1;
    g_cgbequ_cblas_call.layout = layout;
    g_cgbequ_cblas_call.m = m;
    g_cgbequ_cblas_call.n = n;
    g_cgbequ_cblas_call.kl = kl;
    g_cgbequ_cblas_call.ku = ku;
    g_cgbequ_cblas_call.ldab = ldab;
    g_cgbequ_cblas_call.ab = ab;
    g_cgbequ_cblas_call.r = r;
    g_cgbequ_cblas_call.c = c;
    g_cgbequ_cblas_call.rowcnd = rowcnd;
    g_cgbequ_cblas_call.colcnd = colcnd;
    g_cgbequ_cblas_call.amax = amax;
    for (index = 0; index < m; ++index) {
        r[index] = (float)(60 + index);
    }
    for (index = 0; index < n; ++index) {
        c[index] = (float)(70 + index);
    }
    *rowcnd = 2.25f;
    *colcnd = 2.5f;
    *amax = 53.0f;
    return 183;
}

static void stub_zgbequ_fortran(int *m, int *n, int *kl, int *ku,
                                fb_complex_double_t *ab, int *ldab,
                                double *r, double *c, double *rowcnd,
                                double *colcnd, double *amax, int *info)
{
    int index = 0;

    g_zgbequ_fortran_call.calls += 1;
    g_zgbequ_fortran_call.m = *m;
    g_zgbequ_fortran_call.n = *n;
    g_zgbequ_fortran_call.kl = *kl;
    g_zgbequ_fortran_call.ku = *ku;
    g_zgbequ_fortran_call.ldab = *ldab;
    for (index = 0; index < (*ldab * *n); ++index) {
        g_zgbequ_fortran_call.ab_real_snapshot[index] = cdouble_real(ab[index]);
    }
    for (index = 0; index < *m; ++index) {
        r[index] = (double)(120 + index);
    }
    for (index = 0; index < *n; ++index) {
        c[index] = (double)(130 + index);
    }
    *rowcnd = 4.25;
    *colcnd = 4.5;
    *amax = 83.0;
    *info = 0;
}

static int stub_zgbequ_cblas(fb_layout_t layout, int m, int n, int kl, int ku,
                             const fb_complex_double_t *ab, int ldab,
                             double *r, double *c, double *rowcnd,
                             double *colcnd, double *amax)
{
    int index = 0;

    g_zgbequ_cblas_call.called += 1;
    g_zgbequ_cblas_call.layout = layout;
    g_zgbequ_cblas_call.m = m;
    g_zgbequ_cblas_call.n = n;
    g_zgbequ_cblas_call.kl = kl;
    g_zgbequ_cblas_call.ku = ku;
    g_zgbequ_cblas_call.ldab = ldab;
    g_zgbequ_cblas_call.ab = ab;
    g_zgbequ_cblas_call.r = r;
    g_zgbequ_cblas_call.c = c;
    g_zgbequ_cblas_call.rowcnd = rowcnd;
    g_zgbequ_cblas_call.colcnd = colcnd;
    g_zgbequ_cblas_call.amax = amax;
    for (index = 0; index < m; ++index) {
        r[index] = (double)(140 + index);
    }
    for (index = 0; index < n; ++index) {
        c[index] = (double)(150 + index);
    }
    *rowcnd = 5.25;
    *colcnd = 5.5;
    *amax = 97.0;
    return 184;
}

static int check_sgbequ_fortran_to_cblas(void)
{
    fb_backend_vtable_t vtable;
    fb_sgbequ_fn thunk = NULL;
    float ab[12] = {
        10.0f, 11.0f, 12.0f, 13.0f,
        20.0f, 21.0f, 22.0f, 23.0f,
        30.0f, 31.0f, 32.0f, 33.0f
    };
    float r[3] = { 0.0f, 0.0f, 0.0f };
    float c[4] = { 0.0f, 0.0f, 0.0f, 0.0f };
    float rowcnd = 0.0f;
    float colcnd = 0.0f;
    float amax = 0.0f;
    float expected_snapshot[12] = {
        0.0f, 20.0f, 30.0f,
        11.0f, 21.0f, 31.0f,
        12.0f, 22.0f, 0.0f,
        13.0f, 0.0f, 0.0f
    };
    float expected_r[3] = { 1.0f, 2.0f, 3.0f };
    float expected_c[4] = { 10.0f, 11.0f, 12.0f, 13.0f };
    int info = 0;

    memset(&vtable, 0, sizeof(vtable));
    memset(&g_sgbequ_fortran_call, 0, sizeof(g_sgbequ_fortran_call));

    vtable.ext_ops[FB_OP_SGBEQU][FB_CONV_FORTRAN] =
        (fb_generic_fn)(void (*)(void))stub_sgbequ_fortran;
    fb_install_conv_thunks(&vtable, FB_OP_SGBEQU);

    thunk = (fb_sgbequ_fn)vtable.ext_ops[FB_OP_SGBEQU][FB_CONV_CBLAS];
    if (!thunk) {
        fprintf(stderr, "[FAIL] SGBEQU Fortran->CBLAS thunk was not installed\n");
        return 1;
    }

    info = thunk(FB_LAYOUT_ROW_MAJOR, 3, 4, 1, 1, ab, 4, r, c, &rowcnd,
                 &colcnd, &amax);
    if (info != 0 || g_sgbequ_fortran_call.calls != 1 ||
        g_sgbequ_fortran_call.m != 3 || g_sgbequ_fortran_call.n != 4 ||
        g_sgbequ_fortran_call.kl != 1 || g_sgbequ_fortran_call.ku != 1 ||
        g_sgbequ_fortran_call.ldab != 3 ||
        memcmp(g_sgbequ_fortran_call.ab_snapshot, expected_snapshot,
               sizeof(expected_snapshot)) != 0 ||
        memcmp(r, expected_r, sizeof(expected_r)) != 0 ||
        memcmp(c, expected_c, sizeof(expected_c)) != 0 ||
        rowcnd != 0.25f || colcnd != 0.5f || amax != 23.0f) {
        fprintf(stderr, "[FAIL] SGBEQU Fortran->CBLAS thunk did not preserve row-major general-band equilibration semantics\n");
        return 1;
    }

    printf("[PASS] SGBEQU Fortran->CBLAS thunk transposes row-major general band storage and preserves row/column scaling outputs\n");
    return 0;
}

static int check_sgbequ_cblas_to_fortran(void)
{
    fb_backend_vtable_t vtable;
    fb_sgbequ_fortran_slot_fn thunk = NULL;
    float ab[12] = {
        1.0f, 2.0f, 3.0f,
        4.0f, 5.0f, 6.0f,
        7.0f, 8.0f, 9.0f,
        10.0f, 11.0f, 12.0f
    };
    float r[3] = { 0.0f, 0.0f, 0.0f };
    float c[4] = { 0.0f, 0.0f, 0.0f, 0.0f };
    float rowcnd = 0.0f;
    float colcnd = 0.0f;
    float amax = 0.0f;
    int m = 3;
    int n = 4;
    int kl = 1;
    int ku = 1;
    int ldab = 3;
    int info = 0;

    memset(&vtable, 0, sizeof(vtable));
    memset(&g_sgbequ_cblas_call, 0, sizeof(g_sgbequ_cblas_call));

    vtable.ext_ops[FB_OP_SGBEQU][FB_CONV_CBLAS] =
        (fb_generic_fn)(void (*)(void))stub_sgbequ_cblas;
    fb_install_conv_thunks(&vtable, FB_OP_SGBEQU);

    thunk = (fb_sgbequ_fortran_slot_fn)vtable.ext_ops[FB_OP_SGBEQU][FB_CONV_FORTRAN];
    if (!thunk) {
        fprintf(stderr, "[FAIL] SGBEQU CBLAS->Fortran thunk was not installed\n");
        return 1;
    }

    thunk(&m, &n, &kl, &ku, ab, &ldab, r, c, &rowcnd, &colcnd, &amax, &info);
    if (info != 181 || g_sgbequ_cblas_call.called != 1 ||
        g_sgbequ_cblas_call.layout != FB_LAYOUT_COL_MAJOR ||
        g_sgbequ_cblas_call.m != 3 || g_sgbequ_cblas_call.n != 4 ||
        g_sgbequ_cblas_call.kl != 1 || g_sgbequ_cblas_call.ku != 1 ||
        g_sgbequ_cblas_call.ldab != 3 || g_sgbequ_cblas_call.ab != ab ||
        g_sgbequ_cblas_call.r != r || g_sgbequ_cblas_call.c != c ||
        g_sgbequ_cblas_call.rowcnd != &rowcnd ||
        g_sgbequ_cblas_call.colcnd != &colcnd ||
        g_sgbequ_cblas_call.amax != &amax ||
        r[0] != 20.0f || r[1] != 21.0f || r[2] != 22.0f ||
        c[0] != 30.0f || c[1] != 31.0f || c[2] != 32.0f || c[3] != 33.0f ||
        rowcnd != 0.75f || colcnd != 1.25f || amax != 31.0f) {
        fprintf(stderr, "[FAIL] SGBEQU CBLAS->Fortran thunk delegated incorrectly\n");
        return 1;
    }

    printf("[PASS] SGBEQU CBLAS->Fortran thunk maps the all-pointer ABI into the generic C band-equilibration entry\n");
    return 0;
}

static int check_dgbequ_fortran_to_cblas(void)
{
    fb_backend_vtable_t vtable;
    fb_dgbequ_fn thunk = NULL;
    double ab[12] = {
        10.0, 11.0, 12.0, 13.0,
        20.0, 21.0, 22.0, 23.0,
        30.0, 31.0, 32.0, 33.0
    };
    double r[3] = { 0.0, 0.0, 0.0 };
    double c[4] = { 0.0, 0.0, 0.0, 0.0 };
    double rowcnd = 0.0;
    double colcnd = 0.0;
    double amax = 0.0;
    double expected_snapshot[12] = {
        0.0, 20.0, 30.0,
        11.0, 21.0, 31.0,
        12.0, 22.0, 0.0,
        13.0, 0.0, 0.0
    };
    double expected_r[3] = { 80.0, 81.0, 82.0 };
    double expected_c[4] = { 90.0, 91.0, 92.0, 93.0 };
    int info = 0;

    memset(&vtable, 0, sizeof(vtable));
    memset(&g_dgbequ_fortran_call, 0, sizeof(g_dgbequ_fortran_call));

    vtable.ext_ops[FB_OP_DGBEQU][FB_CONV_FORTRAN] =
        (fb_generic_fn)(void (*)(void))stub_dgbequ_fortran;
    fb_install_conv_thunks(&vtable, FB_OP_DGBEQU);

    thunk = (fb_dgbequ_fn)vtable.ext_ops[FB_OP_DGBEQU][FB_CONV_CBLAS];
    if (!thunk) {
        fprintf(stderr, "[FAIL] DGBEQU Fortran->CBLAS thunk was not installed\n");
        return 1;
    }

    info = thunk(FB_LAYOUT_ROW_MAJOR, 3, 4, 1, 1, ab, 4, r, c, &rowcnd,
                 &colcnd, &amax);
    if (info != 0 || g_dgbequ_fortran_call.calls != 1 ||
        g_dgbequ_fortran_call.m != 3 || g_dgbequ_fortran_call.n != 4 ||
        g_dgbequ_fortran_call.kl != 1 || g_dgbequ_fortran_call.ku != 1 ||
        g_dgbequ_fortran_call.ldab != 3 ||
        memcmp(g_dgbequ_fortran_call.ab_snapshot, expected_snapshot,
               sizeof(expected_snapshot)) != 0 ||
        memcmp(r, expected_r, sizeof(expected_r)) != 0 ||
        memcmp(c, expected_c, sizeof(expected_c)) != 0 ||
        rowcnd != 2.75 || colcnd != 3.0 || amax != 63.0) {
        fprintf(stderr, "[FAIL] DGBEQU Fortran->CBLAS thunk did not preserve row-major double general-band equilibration semantics\n");
        return 1;
    }

    printf("[PASS] DGBEQU Fortran->CBLAS thunk transposes row-major general band storage and preserves row/column scaling outputs\n");
    return 0;
}

static int check_dgbequ_cblas_to_fortran(void)
{
    fb_backend_vtable_t vtable;
    fb_dgbequ_fortran_slot_fn thunk = NULL;
    double ab[12] = {
        1.0, 2.0, 3.0,
        4.0, 5.0, 6.0,
        7.0, 8.0, 9.0,
        10.0, 11.0, 12.0
    };
    double r[3] = { 0.0, 0.0, 0.0 };
    double c[4] = { 0.0, 0.0, 0.0, 0.0 };
    double rowcnd = 0.0;
    double colcnd = 0.0;
    double amax = 0.0;
    int m = 3;
    int n = 4;
    int kl = 1;
    int ku = 1;
    int ldab = 3;
    int info = 0;

    memset(&vtable, 0, sizeof(vtable));
    memset(&g_dgbequ_cblas_call, 0, sizeof(g_dgbequ_cblas_call));

    vtable.ext_ops[FB_OP_DGBEQU][FB_CONV_CBLAS] =
        (fb_generic_fn)(void (*)(void))stub_dgbequ_cblas;
    fb_install_conv_thunks(&vtable, FB_OP_DGBEQU);

    thunk = (fb_dgbequ_fortran_slot_fn)vtable.ext_ops[FB_OP_DGBEQU][FB_CONV_FORTRAN];
    if (!thunk) {
        fprintf(stderr, "[FAIL] DGBEQU CBLAS->Fortran thunk was not installed\n");
        return 1;
    }

    thunk(&m, &n, &kl, &ku, ab, &ldab, r, c, &rowcnd, &colcnd, &amax, &info);
    if (info != 182 || g_dgbequ_cblas_call.called != 1 ||
        g_dgbequ_cblas_call.layout != FB_LAYOUT_COL_MAJOR ||
        g_dgbequ_cblas_call.m != 3 || g_dgbequ_cblas_call.n != 4 ||
        g_dgbequ_cblas_call.kl != 1 || g_dgbequ_cblas_call.ku != 1 ||
        g_dgbequ_cblas_call.ldab != 3 || g_dgbequ_cblas_call.ab != ab ||
        g_dgbequ_cblas_call.r != r || g_dgbequ_cblas_call.c != c ||
        g_dgbequ_cblas_call.rowcnd != &rowcnd ||
        g_dgbequ_cblas_call.colcnd != &colcnd ||
        g_dgbequ_cblas_call.amax != &amax ||
        r[0] != 100.0 || r[1] != 101.0 || r[2] != 102.0 ||
        c[0] != 110.0 || c[1] != 111.0 || c[2] != 112.0 || c[3] != 113.0 ||
        rowcnd != 3.25 || colcnd != 3.5 || amax != 71.0) {
        fprintf(stderr, "[FAIL] DGBEQU CBLAS->Fortran thunk delegated incorrectly\n");
        return 1;
    }

    printf("[PASS] DGBEQU CBLAS->Fortran thunk maps the all-pointer ABI into the generic C double band-equilibration entry\n");
    return 0;
}

static int check_cgbequ_fortran_to_cblas(void)
{
    fb_backend_vtable_t vtable;
    fb_cgbequ_fn thunk = NULL;
    fb_complex_float_t ab[12] = {
        make_cfloat(30.0f), make_cfloat(31.0f), make_cfloat(32.0f), make_cfloat(33.0f),
        make_cfloat(40.0f), make_cfloat(41.0f), make_cfloat(42.0f), make_cfloat(43.0f),
        make_cfloat(50.0f), make_cfloat(51.0f), make_cfloat(52.0f), make_cfloat(53.0f)
    };
    float r[3] = { 0.0f, 0.0f, 0.0f };
    float c[4] = { 0.0f, 0.0f, 0.0f, 0.0f };
    float rowcnd = 0.0f;
    float colcnd = 0.0f;
    float amax = 0.0f;
    float expected_snapshot[12] = {
        0.0f, 40.0f, 50.0f,
        31.0f, 41.0f, 51.0f,
        32.0f, 42.0f, 0.0f,
        33.0f, 0.0f, 0.0f
    };
    float expected_r[3] = { 40.0f, 41.0f, 42.0f };
    float expected_c[4] = { 50.0f, 51.0f, 52.0f, 53.0f };
    int info = 0;

    memset(&vtable, 0, sizeof(vtable));
    memset(&g_cgbequ_fortran_call, 0, sizeof(g_cgbequ_fortran_call));

    vtable.ext_ops[FB_OP_CGBEQU][FB_CONV_FORTRAN] =
        (fb_generic_fn)(void (*)(void))stub_cgbequ_fortran;
    fb_install_conv_thunks(&vtable, FB_OP_CGBEQU);

    thunk = (fb_cgbequ_fn)vtable.ext_ops[FB_OP_CGBEQU][FB_CONV_CBLAS];
    if (!thunk) {
        fprintf(stderr, "[FAIL] CGBEQU Fortran->CBLAS thunk was not installed\n");
        return 1;
    }

    info = thunk(FB_LAYOUT_ROW_MAJOR, 3, 4, 1, 1, ab, 4, r, c, &rowcnd,
                 &colcnd, &amax);
    if (info != 0 || g_cgbequ_fortran_call.calls != 1 ||
        g_cgbequ_fortran_call.m != 3 || g_cgbequ_fortran_call.n != 4 ||
        g_cgbequ_fortran_call.kl != 1 || g_cgbequ_fortran_call.ku != 1 ||
        g_cgbequ_fortran_call.ldab != 3 ||
        memcmp(g_cgbequ_fortran_call.ab_real_snapshot, expected_snapshot,
               sizeof(expected_snapshot)) != 0 ||
        memcmp(r, expected_r, sizeof(expected_r)) != 0 ||
        memcmp(c, expected_c, sizeof(expected_c)) != 0 ||
        rowcnd != 1.5f || colcnd != 1.75f || amax != 43.0f) {
        fprintf(stderr, "[FAIL] CGBEQU Fortran->CBLAS thunk did not preserve complex row-major general-band equilibration semantics\n");
        return 1;
    }

    printf("[PASS] CGBEQU Fortran->CBLAS thunk transposes row-major general band storage and preserves real scaling outputs\n");
    return 0;
}

static int check_cgbequ_cblas_to_fortran(void)
{
    fb_backend_vtable_t vtable;
    fb_cgbequ_fortran_slot_fn thunk = NULL;
    fb_complex_float_t ab[12] = {
        make_cfloat(1.0f), make_cfloat(2.0f), make_cfloat(3.0f),
        make_cfloat(4.0f), make_cfloat(5.0f), make_cfloat(6.0f),
        make_cfloat(7.0f), make_cfloat(8.0f), make_cfloat(9.0f),
        make_cfloat(10.0f), make_cfloat(11.0f), make_cfloat(12.0f)
    };
    float r[3] = { 0.0f, 0.0f, 0.0f };
    float c[4] = { 0.0f, 0.0f, 0.0f, 0.0f };
    float rowcnd = 0.0f;
    float colcnd = 0.0f;
    float amax = 0.0f;
    int m = 3;
    int n = 4;
    int kl = 1;
    int ku = 1;
    int ldab = 3;
    int info = 0;

    memset(&vtable, 0, sizeof(vtable));
    memset(&g_cgbequ_cblas_call, 0, sizeof(g_cgbequ_cblas_call));

    vtable.ext_ops[FB_OP_CGBEQU][FB_CONV_CBLAS] =
        (fb_generic_fn)(void (*)(void))stub_cgbequ_cblas;
    fb_install_conv_thunks(&vtable, FB_OP_CGBEQU);

    thunk = (fb_cgbequ_fortran_slot_fn)vtable.ext_ops[FB_OP_CGBEQU][FB_CONV_FORTRAN];
    if (!thunk) {
        fprintf(stderr, "[FAIL] CGBEQU CBLAS->Fortran thunk was not installed\n");
        return 1;
    }

    thunk(&m, &n, &kl, &ku, ab, &ldab, r, c, &rowcnd, &colcnd, &amax, &info);
    if (info != 183 || g_cgbequ_cblas_call.called != 1 ||
        g_cgbequ_cblas_call.layout != FB_LAYOUT_COL_MAJOR ||
        g_cgbequ_cblas_call.m != 3 || g_cgbequ_cblas_call.n != 4 ||
        g_cgbequ_cblas_call.kl != 1 || g_cgbequ_cblas_call.ku != 1 ||
        g_cgbequ_cblas_call.ldab != 3 || g_cgbequ_cblas_call.ab != ab ||
        g_cgbequ_cblas_call.r != r || g_cgbequ_cblas_call.c != c ||
        g_cgbequ_cblas_call.rowcnd != &rowcnd ||
        g_cgbequ_cblas_call.colcnd != &colcnd ||
        g_cgbequ_cblas_call.amax != &amax ||
        r[0] != 60.0f || r[1] != 61.0f || r[2] != 62.0f ||
        c[0] != 70.0f || c[1] != 71.0f || c[2] != 72.0f || c[3] != 73.0f ||
        rowcnd != 2.25f || colcnd != 2.5f || amax != 53.0f) {
        fprintf(stderr, "[FAIL] CGBEQU CBLAS->Fortran thunk delegated incorrectly\n");
        return 1;
    }

    printf("[PASS] CGBEQU CBLAS->Fortran thunk maps the all-pointer ABI into the generic complex band-equilibration entry\n");
    return 0;
}

static int check_zgbequ_fortran_to_cblas(void)
{
    fb_backend_vtable_t vtable;
    fb_zgbequ_fn thunk = NULL;
    fb_complex_double_t ab[12] = {
        make_cdouble(30.0), make_cdouble(31.0), make_cdouble(32.0), make_cdouble(33.0),
        make_cdouble(40.0), make_cdouble(41.0), make_cdouble(42.0), make_cdouble(43.0),
        make_cdouble(50.0), make_cdouble(51.0), make_cdouble(52.0), make_cdouble(53.0)
    };
    double r[3] = { 0.0, 0.0, 0.0 };
    double c[4] = { 0.0, 0.0, 0.0, 0.0 };
    double rowcnd = 0.0;
    double colcnd = 0.0;
    double amax = 0.0;
    double expected_snapshot[12] = {
        0.0, 40.0, 50.0,
        31.0, 41.0, 51.0,
        32.0, 42.0, 0.0,
        33.0, 0.0, 0.0
    };
    double expected_r[3] = { 120.0, 121.0, 122.0 };
    double expected_c[4] = { 130.0, 131.0, 132.0, 133.0 };
    int info = 0;

    memset(&vtable, 0, sizeof(vtable));
    memset(&g_zgbequ_fortran_call, 0, sizeof(g_zgbequ_fortran_call));

    vtable.ext_ops[FB_OP_ZGBEQU][FB_CONV_FORTRAN] =
        (fb_generic_fn)(void (*)(void))stub_zgbequ_fortran;
    fb_install_conv_thunks(&vtable, FB_OP_ZGBEQU);

    thunk = (fb_zgbequ_fn)vtable.ext_ops[FB_OP_ZGBEQU][FB_CONV_CBLAS];
    if (!thunk) {
        fprintf(stderr, "[FAIL] ZGBEQU Fortran->CBLAS thunk was not installed\n");
        return 1;
    }

    info = thunk(FB_LAYOUT_ROW_MAJOR, 3, 4, 1, 1, ab, 4, r, c, &rowcnd,
                 &colcnd, &amax);
    if (info != 0 || g_zgbequ_fortran_call.calls != 1 ||
        g_zgbequ_fortran_call.m != 3 || g_zgbequ_fortran_call.n != 4 ||
        g_zgbequ_fortran_call.kl != 1 || g_zgbequ_fortran_call.ku != 1 ||
        g_zgbequ_fortran_call.ldab != 3 ||
        memcmp(g_zgbequ_fortran_call.ab_real_snapshot, expected_snapshot,
               sizeof(expected_snapshot)) != 0 ||
        memcmp(r, expected_r, sizeof(expected_r)) != 0 ||
        memcmp(c, expected_c, sizeof(expected_c)) != 0 ||
        rowcnd != 4.25 || colcnd != 4.5 || amax != 83.0) {
        fprintf(stderr, "[FAIL] ZGBEQU Fortran->CBLAS thunk did not preserve complex-double row-major general-band equilibration semantics\n");
        return 1;
    }

    printf("[PASS] ZGBEQU Fortran->CBLAS thunk transposes row-major general band storage and preserves real scaling outputs\n");
    return 0;
}

static int check_zgbequ_cblas_to_fortran(void)
{
    fb_backend_vtable_t vtable;
    fb_zgbequ_fortran_slot_fn thunk = NULL;
    fb_complex_double_t ab[12] = {
        make_cdouble(1.0), make_cdouble(2.0), make_cdouble(3.0),
        make_cdouble(4.0), make_cdouble(5.0), make_cdouble(6.0),
        make_cdouble(7.0), make_cdouble(8.0), make_cdouble(9.0),
        make_cdouble(10.0), make_cdouble(11.0), make_cdouble(12.0)
    };
    double r[3] = { 0.0, 0.0, 0.0 };
    double c[4] = { 0.0, 0.0, 0.0, 0.0 };
    double rowcnd = 0.0;
    double colcnd = 0.0;
    double amax = 0.0;
    int m = 3;
    int n = 4;
    int kl = 1;
    int ku = 1;
    int ldab = 3;
    int info = 0;

    memset(&vtable, 0, sizeof(vtable));
    memset(&g_zgbequ_cblas_call, 0, sizeof(g_zgbequ_cblas_call));

    vtable.ext_ops[FB_OP_ZGBEQU][FB_CONV_CBLAS] =
        (fb_generic_fn)(void (*)(void))stub_zgbequ_cblas;
    fb_install_conv_thunks(&vtable, FB_OP_ZGBEQU);

    thunk = (fb_zgbequ_fortran_slot_fn)vtable.ext_ops[FB_OP_ZGBEQU][FB_CONV_FORTRAN];
    if (!thunk) {
        fprintf(stderr, "[FAIL] ZGBEQU CBLAS->Fortran thunk was not installed\n");
        return 1;
    }

    thunk(&m, &n, &kl, &ku, ab, &ldab, r, c, &rowcnd, &colcnd, &amax, &info);
    if (info != 184 || g_zgbequ_cblas_call.called != 1 ||
        g_zgbequ_cblas_call.layout != FB_LAYOUT_COL_MAJOR ||
        g_zgbequ_cblas_call.m != 3 || g_zgbequ_cblas_call.n != 4 ||
        g_zgbequ_cblas_call.kl != 1 || g_zgbequ_cblas_call.ku != 1 ||
        g_zgbequ_cblas_call.ldab != 3 || g_zgbequ_cblas_call.ab != ab ||
        g_zgbequ_cblas_call.r != r || g_zgbequ_cblas_call.c != c ||
        g_zgbequ_cblas_call.rowcnd != &rowcnd ||
        g_zgbequ_cblas_call.colcnd != &colcnd ||
        g_zgbequ_cblas_call.amax != &amax ||
        r[0] != 140.0 || r[1] != 141.0 || r[2] != 142.0 ||
        c[0] != 150.0 || c[1] != 151.0 || c[2] != 152.0 || c[3] != 153.0 ||
        rowcnd != 5.25 || colcnd != 5.5 || amax != 97.0) {
        fprintf(stderr, "[FAIL] ZGBEQU CBLAS->Fortran thunk delegated incorrectly\n");
        return 1;
    }

    printf("[PASS] ZGBEQU CBLAS->Fortran thunk maps the all-pointer ABI into the generic complex-double band-equilibration entry\n");
    return 0;
}

int main(void)
{
    if (check_sgbequ_fortran_to_cblas() != 0) {
        return 1;
    }
    if (check_sgbequ_cblas_to_fortran() != 0) {
        return 1;
    }
    if (check_dgbequ_fortran_to_cblas() != 0) {
        return 1;
    }
    if (check_dgbequ_cblas_to_fortran() != 0) {
        return 1;
    }
    if (check_cgbequ_fortran_to_cblas() != 0) {
        return 1;
    }
    if (check_cgbequ_cblas_to_fortran() != 0) {
        return 1;
    }
    if (check_zgbequ_fortran_to_cblas() != 0) {
        return 1;
    }
    if (check_zgbequ_cblas_to_fortran() != 0) {
        return 1;
    }

    printf("Result: PASS\n");
    return 0;
}