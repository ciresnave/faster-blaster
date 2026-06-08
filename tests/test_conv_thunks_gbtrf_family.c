#include <stdio.h>
#include <string.h>

#include "../src/core/conv_thunks.h"

typedef int (*fb_sgbtrf_fn)(fb_layout_t layout, int m, int n, int kl, int ku,
                            float *ab, int ldab, int *ipiv);
typedef int (*fb_cgbtrf_fn)(fb_layout_t layout, int m, int n, int kl, int ku,
                            fb_complex_float_t *ab, int ldab, int *ipiv);

typedef void (*fb_sgbtrf_fortran_slot_fn)(int *m, int *n, int *kl, int *ku,
                                          float *ab, int *ldab, int *ipiv,
                                          int *info);
typedef void (*fb_cgbtrf_fortran_slot_fn)(int *m, int *n, int *kl, int *ku,
                                          fb_complex_float_t *ab, int *ldab,
                                          int *ipiv, int *info);

static struct {
    int calls;
    int m;
    int n;
    int kl;
    int ku;
    int ldab;
    int ipiv_snapshot[4];
    float ab_snapshot[16];
} g_sgbtrf_fortran_call;

static struct {
    int called;
    fb_layout_t layout;
    int m;
    int n;
    int kl;
    int ku;
    int ldab;
    float *ab;
    int *ipiv;
} g_sgbtrf_cblas_call;

static struct {
    int calls;
    int m;
    int n;
    int kl;
    int ku;
    int ldab;
    int ipiv_snapshot[4];
    float ab_real_snapshot[16];
} g_cgbtrf_fortran_call;

static struct {
    int called;
    fb_layout_t layout;
    int m;
    int n;
    int kl;
    int ku;
    int ldab;
    fb_complex_float_t *ab;
    int *ipiv;
} g_cgbtrf_cblas_call;

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

static void stub_sgbtrf_fortran(int *m, int *n, int *kl, int *ku, float *ab,
                                int *ldab, int *ipiv, int *info)
{
    int index = 0;

    g_sgbtrf_fortran_call.calls += 1;
    g_sgbtrf_fortran_call.m = *m;
    g_sgbtrf_fortran_call.n = *n;
    g_sgbtrf_fortran_call.kl = *kl;
    g_sgbtrf_fortran_call.ku = *ku;
    g_sgbtrf_fortran_call.ldab = *ldab;
    memcpy(g_sgbtrf_fortran_call.ipiv_snapshot, ipiv,
           sizeof(g_sgbtrf_fortran_call.ipiv_snapshot));
    for (index = 0; index < (*ldab * *n); ++index) {
        g_sgbtrf_fortran_call.ab_snapshot[index] = ab[index];
        ab[index] = 1000.0f + (float)index;
    }
    ipiv[0] = 4;
    ipiv[1] = 3;
    ipiv[2] = 2;
    ipiv[3] = 1;
    *info = 0;
}

static int stub_sgbtrf_cblas(fb_layout_t layout, int m, int n, int kl, int ku,
                             float *ab, int ldab, int *ipiv)
{
    g_sgbtrf_cblas_call.called += 1;
    g_sgbtrf_cblas_call.layout = layout;
    g_sgbtrf_cblas_call.m = m;
    g_sgbtrf_cblas_call.n = n;
    g_sgbtrf_cblas_call.kl = kl;
    g_sgbtrf_cblas_call.ku = ku;
    g_sgbtrf_cblas_call.ldab = ldab;
    g_sgbtrf_cblas_call.ab = ab;
    g_sgbtrf_cblas_call.ipiv = ipiv;
    ab[0] = 911.0f;
    ab[5] = 922.0f;
    ipiv[0] = 3;
    ipiv[1] = 2;
    ipiv[2] = 1;
    return 191;
}

static void stub_cgbtrf_fortran(int *m, int *n, int *kl, int *ku,
                                fb_complex_float_t *ab, int *ldab, int *ipiv,
                                int *info)
{
    int index = 0;

    g_cgbtrf_fortran_call.calls += 1;
    g_cgbtrf_fortran_call.m = *m;
    g_cgbtrf_fortran_call.n = *n;
    g_cgbtrf_fortran_call.kl = *kl;
    g_cgbtrf_fortran_call.ku = *ku;
    g_cgbtrf_fortran_call.ldab = *ldab;
    memcpy(g_cgbtrf_fortran_call.ipiv_snapshot, ipiv,
           sizeof(g_cgbtrf_fortran_call.ipiv_snapshot));
    for (index = 0; index < (*ldab * *n); ++index) {
        g_cgbtrf_fortran_call.ab_real_snapshot[index] = cfloat_real(ab[index]);
        ab[index] = make_cfloat(2000.0f + (float)index);
    }
    ipiv[0] = 1;
    ipiv[1] = 4;
    ipiv[2] = 3;
    ipiv[3] = 2;
    *info = 0;
}

static int stub_cgbtrf_cblas(fb_layout_t layout, int m, int n, int kl, int ku,
                             fb_complex_float_t *ab, int ldab, int *ipiv)
{
    g_cgbtrf_cblas_call.called += 1;
    g_cgbtrf_cblas_call.layout = layout;
    g_cgbtrf_cblas_call.m = m;
    g_cgbtrf_cblas_call.n = n;
    g_cgbtrf_cblas_call.kl = kl;
    g_cgbtrf_cblas_call.ku = ku;
    g_cgbtrf_cblas_call.ldab = ldab;
    g_cgbtrf_cblas_call.ab = ab;
    g_cgbtrf_cblas_call.ipiv = ipiv;
    ab[0] = make_cfloat(811.0f);
    ab[5] = make_cfloat(822.0f);
    ipiv[0] = 2;
    ipiv[1] = 3;
    ipiv[2] = 1;
    return 193;
}

static int check_sgbtrf_fortran_to_cblas(void)
{
    fb_backend_vtable_t vtable;
    fb_sgbtrf_fn thunk = NULL;
    float ab[16] = {
        0.0f, 0.0f, 130.0f, 140.0f,
        0.0f, 220.0f, 230.0f, 240.0f,
        310.0f, 320.0f, 330.0f, 340.0f,
        410.0f, 420.0f, 430.0f, 0.0f
    };
    int ipiv[4] = { 9, 8, 7, 6 };
    static const float expected_snapshot[16] = {
        0.0f, 0.0f, 310.0f, 410.0f,
        0.0f, 220.0f, 320.0f, 420.0f,
        130.0f, 230.0f, 330.0f, 430.0f,
        140.0f, 240.0f, 340.0f, 0.0f
    };
    static const float expected_ab_after[16] = {
        0.0f, 0.0f, 1008.0f, 1012.0f,
        0.0f, 1005.0f, 1009.0f, 1013.0f,
        1002.0f, 1006.0f, 1010.0f, 1014.0f,
        1003.0f, 1007.0f, 1011.0f, 0.0f
    };
    static const int expected_ipiv_after[4] = { 4, 3, 2, 1 };
    int info = 0;

    memset(&vtable, 0, sizeof(vtable));
    memset(&g_sgbtrf_fortran_call, 0, sizeof(g_sgbtrf_fortran_call));

    vtable.ext_ops[FB_OP_SGBTRF][FB_CONV_FORTRAN] =
        (fb_generic_fn)(void (*)(void))stub_sgbtrf_fortran;
    fb_install_conv_thunks(&vtable, FB_OP_SGBTRF);

    thunk = (fb_sgbtrf_fn)vtable.ext_ops[FB_OP_SGBTRF][FB_CONV_CBLAS];
    if (!thunk) {
        fprintf(stderr, "[FAIL] SGBTRF Fortran->CBLAS thunk was not installed\n");
        return 1;
    }

    info = thunk(FB_LAYOUT_ROW_MAJOR, 4, 4, 1, 1, ab, 4, ipiv);
    if (info != 0 || g_sgbtrf_fortran_call.calls != 1 ||
        g_sgbtrf_fortran_call.m != 4 || g_sgbtrf_fortran_call.n != 4 ||
        g_sgbtrf_fortran_call.kl != 1 || g_sgbtrf_fortran_call.ku != 1 ||
        g_sgbtrf_fortran_call.ldab != 4 ||
        memcmp(g_sgbtrf_fortran_call.ab_snapshot, expected_snapshot,
               sizeof(expected_snapshot)) != 0 ||
        memcmp(ab, expected_ab_after, sizeof(expected_ab_after)) != 0 ||
        memcmp(ipiv, expected_ipiv_after, sizeof(expected_ipiv_after)) != 0) {
        fprintf(stderr, "[FAIL] SGBTRF Fortran->CBLAS thunk did not preserve expanded row-major factorization semantics\n");
        return 1;
    }

    printf("[PASS] SGBTRF Fortran->CBLAS thunk transposes expanded row-major band storage, preserves pivots, and copies factors back\n");
    return 0;
}

static int check_sgbtrf_cblas_to_fortran(void)
{
    fb_backend_vtable_t vtable;
    fb_sgbtrf_fortran_slot_fn thunk = NULL;
    float ab[12] = {
        1.0f, 2.0f, 3.0f,
        4.0f, 5.0f, 6.0f,
        7.0f, 8.0f, 9.0f,
        10.0f, 11.0f, 12.0f
    };
    int ipiv[3] = { 0, 0, 0 };
    int m = 3;
    int n = 3;
    int kl = 1;
    int ku = 1;
    int ldab = 4;
    int info = 0;

    memset(&vtable, 0, sizeof(vtable));
    memset(&g_sgbtrf_cblas_call, 0, sizeof(g_sgbtrf_cblas_call));

    vtable.ext_ops[FB_OP_SGBTRF][FB_CONV_CBLAS] =
        (fb_generic_fn)(void (*)(void))stub_sgbtrf_cblas;
    fb_install_conv_thunks(&vtable, FB_OP_SGBTRF);

    thunk = (fb_sgbtrf_fortran_slot_fn)vtable.ext_ops[FB_OP_SGBTRF][FB_CONV_FORTRAN];
    if (!thunk) {
        fprintf(stderr, "[FAIL] SGBTRF CBLAS->Fortran thunk was not installed\n");
        return 1;
    }

    thunk(&m, &n, &kl, &ku, ab, &ldab, ipiv, &info);
    if (info != 191 || g_sgbtrf_cblas_call.called != 1 ||
        g_sgbtrf_cblas_call.layout != FB_LAYOUT_COL_MAJOR ||
        g_sgbtrf_cblas_call.m != 3 || g_sgbtrf_cblas_call.n != 3 ||
        g_sgbtrf_cblas_call.kl != 1 || g_sgbtrf_cblas_call.ku != 1 ||
        g_sgbtrf_cblas_call.ldab != 4 || g_sgbtrf_cblas_call.ab != ab ||
        g_sgbtrf_cblas_call.ipiv != ipiv || ab[0] != 911.0f ||
        ab[5] != 922.0f || ipiv[0] != 3 || ipiv[1] != 2 || ipiv[2] != 1) {
        fprintf(stderr, "[FAIL] SGBTRF CBLAS->Fortran thunk delegated incorrectly\n");
        return 1;
    }

    printf("[PASS] SGBTRF CBLAS->Fortran thunk maps the all-pointer ABI into the generic C factorization entry\n");
    return 0;
}

static int check_cgbtrf_fortran_to_cblas(void)
{
    fb_backend_vtable_t vtable;
    fb_cgbtrf_fn thunk = NULL;
    fb_complex_float_t ab[16] = {
        make_cfloat(500.0f), make_cfloat(501.0f), make_cfloat(502.0f), make_cfloat(503.0f),
        make_cfloat(510.0f), make_cfloat(511.0f), make_cfloat(512.0f), make_cfloat(513.0f),
        make_cfloat(520.0f), make_cfloat(521.0f), make_cfloat(522.0f), make_cfloat(523.0f),
        make_cfloat(530.0f), make_cfloat(531.0f), make_cfloat(532.0f), make_cfloat(533.0f)
    };
    int ipiv[4] = { 4, 4, 4, 4 };
    static const float expected_snapshot[16] = {
        0.0f, 0.0f, 520.0f, 530.0f,
        0.0f, 511.0f, 521.0f, 531.0f,
        502.0f, 512.0f, 522.0f, 532.0f,
        503.0f, 513.0f, 523.0f, 0.0f
    };
    static const float expected_ab_after[16] = {
        0.0f, 0.0f, 2008.0f, 2012.0f,
        0.0f, 2005.0f, 2009.0f, 2013.0f,
        2002.0f, 2006.0f, 2010.0f, 2014.0f,
        2003.0f, 2007.0f, 2011.0f, 0.0f
    };
    static const int expected_ipiv_after[4] = { 1, 4, 3, 2 };
    float ab_real_after[16] = { 0.0f };
    int index = 0;
    int info = 0;

    memset(&vtable, 0, sizeof(vtable));
    memset(&g_cgbtrf_fortran_call, 0, sizeof(g_cgbtrf_fortran_call));

    vtable.ext_ops[FB_OP_CGBTRF][FB_CONV_FORTRAN] =
        (fb_generic_fn)(void (*)(void))stub_cgbtrf_fortran;
    fb_install_conv_thunks(&vtable, FB_OP_CGBTRF);

    thunk = (fb_cgbtrf_fn)vtable.ext_ops[FB_OP_CGBTRF][FB_CONV_CBLAS];
    if (!thunk) {
        fprintf(stderr, "[FAIL] CGBTRF Fortran->CBLAS thunk was not installed\n");
        return 1;
    }

    info = thunk(FB_LAYOUT_ROW_MAJOR, 4, 4, 1, 1, ab, 4, ipiv);
    for (index = 0; index < 16; ++index) {
        ab_real_after[index] = cfloat_real(ab[index]);
    }
    if (info != 0 || g_cgbtrf_fortran_call.calls != 1 ||
        g_cgbtrf_fortran_call.m != 4 || g_cgbtrf_fortran_call.n != 4 ||
        g_cgbtrf_fortran_call.kl != 1 || g_cgbtrf_fortran_call.ku != 1 ||
        g_cgbtrf_fortran_call.ldab != 4 ||
        memcmp(g_cgbtrf_fortran_call.ab_real_snapshot, expected_snapshot,
               sizeof(expected_snapshot)) != 0 ||
        memcmp(ab_real_after, expected_ab_after, sizeof(expected_ab_after)) != 0 ||
        memcmp(ipiv, expected_ipiv_after, sizeof(expected_ipiv_after)) != 0) {
        fprintf(stderr, "[FAIL] CGBTRF Fortran->CBLAS thunk did not preserve expanded complex row-major factorization semantics\n");
        return 1;
    }

    printf("[PASS] CGBTRF Fortran->CBLAS thunk transposes expanded row-major complex band storage, preserves pivots, and copies factors back\n");
    return 0;
}

static int check_cgbtrf_cblas_to_fortran(void)
{
    fb_backend_vtable_t vtable;
    fb_cgbtrf_fortran_slot_fn thunk = NULL;
    fb_complex_float_t ab[12] = {
        make_cfloat(1.0f), make_cfloat(2.0f), make_cfloat(3.0f),
        make_cfloat(4.0f), make_cfloat(5.0f), make_cfloat(6.0f),
        make_cfloat(7.0f), make_cfloat(8.0f), make_cfloat(9.0f),
        make_cfloat(10.0f), make_cfloat(11.0f), make_cfloat(12.0f)
    };
    int ipiv[3] = { 0, 0, 0 };
    int m = 3;
    int n = 3;
    int kl = 1;
    int ku = 1;
    int ldab = 4;
    int info = 0;

    memset(&vtable, 0, sizeof(vtable));
    memset(&g_cgbtrf_cblas_call, 0, sizeof(g_cgbtrf_cblas_call));

    vtable.ext_ops[FB_OP_CGBTRF][FB_CONV_CBLAS] =
        (fb_generic_fn)(void (*)(void))stub_cgbtrf_cblas;
    fb_install_conv_thunks(&vtable, FB_OP_CGBTRF);

    thunk = (fb_cgbtrf_fortran_slot_fn)vtable.ext_ops[FB_OP_CGBTRF][FB_CONV_FORTRAN];
    if (!thunk) {
        fprintf(stderr, "[FAIL] CGBTRF CBLAS->Fortran thunk was not installed\n");
        return 1;
    }

    thunk(&m, &n, &kl, &ku, ab, &ldab, ipiv, &info);
    if (info != 193 || g_cgbtrf_cblas_call.called != 1 ||
        g_cgbtrf_cblas_call.layout != FB_LAYOUT_COL_MAJOR ||
        g_cgbtrf_cblas_call.m != 3 || g_cgbtrf_cblas_call.n != 3 ||
        g_cgbtrf_cblas_call.kl != 1 || g_cgbtrf_cblas_call.ku != 1 ||
        g_cgbtrf_cblas_call.ldab != 4 || g_cgbtrf_cblas_call.ab != ab ||
        g_cgbtrf_cblas_call.ipiv != ipiv || cfloat_real(ab[0]) != 811.0f ||
        cfloat_real(ab[5]) != 822.0f || ipiv[0] != 2 || ipiv[1] != 3 ||
        ipiv[2] != 1) {
        fprintf(stderr, "[FAIL] CGBTRF CBLAS->Fortran thunk delegated incorrectly\n");
        return 1;
    }

    printf("[PASS] CGBTRF CBLAS->Fortran thunk maps the all-pointer ABI into the generic complex factorization entry\n");
    return 0;
}

int main(void)
{
    if (check_sgbtrf_fortran_to_cblas() != 0) {
        return 1;
    }
    if (check_sgbtrf_cblas_to_fortran() != 0) {
        return 1;
    }
    if (check_cgbtrf_fortran_to_cblas() != 0) {
        return 1;
    }
    if (check_cgbtrf_cblas_to_fortran() != 0) {
        return 1;
    }

    printf("Result: PASS\n");
    return 0;
}