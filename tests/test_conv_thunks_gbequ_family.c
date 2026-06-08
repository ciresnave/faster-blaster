#include <stdio.h>
#include <string.h>

#include "../src/core/conv_thunks.h"

typedef int (*fb_sgbequ_fn)(fb_layout_t layout, int m, int n, int kl, int ku,
                            const float *ab, int ldab, float *r, float *c,
                            float *rowcnd, float *colcnd, float *amax);
typedef int (*fb_cgbequ_fn)(fb_layout_t layout, int m, int n, int kl, int ku,
                            const fb_complex_float_t *ab, int ldab, float *r,
                            float *c, float *rowcnd, float *colcnd,
                            float *amax);

typedef void (*fb_sgbequ_fortran_slot_fn)(int *m, int *n, int *kl, int *ku,
                                          float *ab, int *ldab, float *r,
                                          float *c, float *rowcnd,
                                          float *colcnd, float *amax,
                                          int *info);
typedef void (*fb_cgbequ_fortran_slot_fn)(int *m, int *n, int *kl, int *ku,
                                          fb_complex_float_t *ab, int *ldab,
                                          float *r, float *c, float *rowcnd,
                                          float *colcnd, float *amax,
                                          int *info);

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

int main(void)
{
    if (check_sgbequ_fortran_to_cblas() != 0) {
        return 1;
    }
    if (check_sgbequ_cblas_to_fortran() != 0) {
        return 1;
    }
    if (check_cgbequ_fortran_to_cblas() != 0) {
        return 1;
    }
    if (check_cgbequ_cblas_to_fortran() != 0) {
        return 1;
    }

    printf("Result: PASS\n");
    return 0;
}