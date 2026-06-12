#include <stdio.h>
#include <string.h>

#include "../src/core/conv_thunks.h"

typedef int (*fb_sgbsv_fn)(fb_layout_t layout, int n, int kl, int ku, int nrhs,
                           float *ab, int ldab, int *ipiv, float *b, int ldb);
typedef int (*fb_dgbsv_fn)(fb_layout_t layout, int n, int kl, int ku, int nrhs,
                           double *ab, int ldab, int *ipiv, double *b,
                           int ldb);
typedef int (*fb_cgbsv_fn)(fb_layout_t layout, int n, int kl, int ku, int nrhs,
                           fb_complex_float_t *ab, int ldab, int *ipiv,
                           fb_complex_float_t *b, int ldb);
typedef int (*fb_zgbsv_fn)(fb_layout_t layout, int n, int kl, int ku, int nrhs,
                           fb_complex_double_t *ab, int ldab, int *ipiv,
                           fb_complex_double_t *b, int ldb);

typedef void (*fb_sgbsv_fortran_slot_fn)(int *n, int *kl, int *ku, int *nrhs,
                                         float *ab, int *ldab, int *ipiv,
                                         float *b, int *ldb, int *info);
typedef void (*fb_dgbsv_fortran_slot_fn)(int *n, int *kl, int *ku, int *nrhs,
                                         double *ab, int *ldab, int *ipiv,
                                         double *b, int *ldb, int *info);
typedef void (*fb_cgbsv_fortran_slot_fn)(int *n, int *kl, int *ku, int *nrhs,
                                         fb_complex_float_t *ab, int *ldab,
                                         int *ipiv, fb_complex_float_t *b,
                                         int *ldb, int *info);
typedef void (*fb_zgbsv_fortran_slot_fn)(int *n, int *kl, int *ku, int *nrhs,
                                         fb_complex_double_t *ab, int *ldab,
                                         int *ipiv, fb_complex_double_t *b,
                                         int *ldb, int *info);

static struct {
    int calls;
    int n;
    int kl;
    int ku;
    int nrhs;
    int ldab;
    int ldb;
    int ipiv_snapshot[4];
    float ab_snapshot[16];
    float b_snapshot[8];
} g_sgbsv_fortran_call;

static struct {
    int called;
    fb_layout_t layout;
    int n;
    int kl;
    int ku;
    int nrhs;
    int ldab;
    int ldb;
    float *ab;
    int *ipiv;
    float *b;
} g_sgbsv_cblas_call;

static struct {
    int calls;
    int n;
    int kl;
    int ku;
    int nrhs;
    int ldab;
    int ldb;
    int ipiv_snapshot[4];
    double ab_snapshot[16];
    double b_snapshot[8];
} g_dgbsv_fortran_call;

static struct {
    int called;
    fb_layout_t layout;
    int n;
    int kl;
    int ku;
    int nrhs;
    int ldab;
    int ldb;
    double *ab;
    int *ipiv;
    double *b;
} g_dgbsv_cblas_call;

static struct {
    int calls;
    int n;
    int kl;
    int ku;
    int nrhs;
    int ldab;
    int ldb;
    int ipiv_snapshot[4];
    float ab_real_snapshot[16];
    float b_real_snapshot[8];
} g_cgbsv_fortran_call;

static struct {
    int called;
    fb_layout_t layout;
    int n;
    int kl;
    int ku;
    int nrhs;
    int ldab;
    int ldb;
    fb_complex_float_t *ab;
    int *ipiv;
    fb_complex_float_t *b;
} g_cgbsv_cblas_call;

static struct {
    int calls;
    int n;
    int kl;
    int ku;
    int nrhs;
    int ldab;
    int ldb;
    int ipiv_snapshot[4];
    double ab_real_snapshot[16];
    double b_real_snapshot[8];
} g_zgbsv_fortran_call;

static struct {
    int called;
    fb_layout_t layout;
    int n;
    int kl;
    int ku;
    int nrhs;
    int ldab;
    int ldb;
    fb_complex_double_t *ab;
    int *ipiv;
    fb_complex_double_t *b;
} g_zgbsv_cblas_call;

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

static void stub_sgbsv_fortran(int *n, int *kl, int *ku, int *nrhs, float *ab,
                               int *ldab, int *ipiv, float *b, int *ldb,
                               int *info)
{
    int index = 0;

    g_sgbsv_fortran_call.calls += 1;
    g_sgbsv_fortran_call.n = *n;
    g_sgbsv_fortran_call.kl = *kl;
    g_sgbsv_fortran_call.ku = *ku;
    g_sgbsv_fortran_call.nrhs = *nrhs;
    g_sgbsv_fortran_call.ldab = *ldab;
    g_sgbsv_fortran_call.ldb = *ldb;
    memcpy(g_sgbsv_fortran_call.ipiv_snapshot, ipiv,
           sizeof(g_sgbsv_fortran_call.ipiv_snapshot));
    for (index = 0; index < (*ldab * *n); ++index) {
        g_sgbsv_fortran_call.ab_snapshot[index] = ab[index];
        ab[index] = 1000.0f + (float)index;
    }
    for (index = 0; index < (*ldb * *nrhs); ++index) {
        g_sgbsv_fortran_call.b_snapshot[index] = b[index];
        b[index] = 600.0f + (float)index;
    }
    ipiv[0] = 4;
    ipiv[1] = 3;
    ipiv[2] = 2;
    ipiv[3] = 1;
    *info = 0;
}

static int stub_sgbsv_cblas(fb_layout_t layout, int n, int kl, int ku, int nrhs,
                            float *ab, int ldab, int *ipiv, float *b, int ldb)
{
    g_sgbsv_cblas_call.called += 1;
    g_sgbsv_cblas_call.layout = layout;
    g_sgbsv_cblas_call.n = n;
    g_sgbsv_cblas_call.kl = kl;
    g_sgbsv_cblas_call.ku = ku;
    g_sgbsv_cblas_call.nrhs = nrhs;
    g_sgbsv_cblas_call.ldab = ldab;
    g_sgbsv_cblas_call.ldb = ldb;
    g_sgbsv_cblas_call.ab = ab;
    g_sgbsv_cblas_call.ipiv = ipiv;
    g_sgbsv_cblas_call.b = b;
    ab[0] = 911.0f;
    ab[5] = 922.0f;
    b[0] = 933.0f;
    b[4] = 944.0f;
    ipiv[0] = 3;
    ipiv[1] = 2;
    ipiv[2] = 1;
    return 221;
}

static void stub_dgbsv_fortran(int *n, int *kl, int *ku, int *nrhs, double *ab,
                               int *ldab, int *ipiv, double *b, int *ldb,
                               int *info)
{
    int index = 0;

    g_dgbsv_fortran_call.calls += 1;
    g_dgbsv_fortran_call.n = *n;
    g_dgbsv_fortran_call.kl = *kl;
    g_dgbsv_fortran_call.ku = *ku;
    g_dgbsv_fortran_call.nrhs = *nrhs;
    g_dgbsv_fortran_call.ldab = *ldab;
    g_dgbsv_fortran_call.ldb = *ldb;
    memcpy(g_dgbsv_fortran_call.ipiv_snapshot, ipiv,
           sizeof(g_dgbsv_fortran_call.ipiv_snapshot));
    for (index = 0; index < (*ldab * *n); ++index) {
        g_dgbsv_fortran_call.ab_snapshot[index] = ab[index];
        ab[index] = 1100.0 + (double)index;
    }
    for (index = 0; index < (*ldb * *nrhs); ++index) {
        g_dgbsv_fortran_call.b_snapshot[index] = b[index];
        b[index] = 900.0 + (double)index;
    }
    ipiv[0] = 4;
    ipiv[1] = 3;
    ipiv[2] = 2;
    ipiv[3] = 1;
    *info = 0;
}

static int stub_dgbsv_cblas(fb_layout_t layout, int n, int kl, int ku, int nrhs,
                            double *ab, int ldab, int *ipiv, double *b,
                            int ldb)
{
    g_dgbsv_cblas_call.called += 1;
    g_dgbsv_cblas_call.layout = layout;
    g_dgbsv_cblas_call.n = n;
    g_dgbsv_cblas_call.kl = kl;
    g_dgbsv_cblas_call.ku = ku;
    g_dgbsv_cblas_call.nrhs = nrhs;
    g_dgbsv_cblas_call.ldab = ldab;
    g_dgbsv_cblas_call.ldb = ldb;
    g_dgbsv_cblas_call.ab = ab;
    g_dgbsv_cblas_call.ipiv = ipiv;
    g_dgbsv_cblas_call.b = b;
    ab[0] = 1011.0;
    ab[5] = 1022.0;
    b[0] = 1033.0;
    b[4] = 1044.0;
    ipiv[0] = 3;
    ipiv[1] = 2;
    ipiv[2] = 1;
    return 222;
}

static void stub_cgbsv_fortran(int *n, int *kl, int *ku, int *nrhs,
                               fb_complex_float_t *ab, int *ldab, int *ipiv,
                               fb_complex_float_t *b, int *ldb, int *info)
{
    int index = 0;

    g_cgbsv_fortran_call.calls += 1;
    g_cgbsv_fortran_call.n = *n;
    g_cgbsv_fortran_call.kl = *kl;
    g_cgbsv_fortran_call.ku = *ku;
    g_cgbsv_fortran_call.nrhs = *nrhs;
    g_cgbsv_fortran_call.ldab = *ldab;
    g_cgbsv_fortran_call.ldb = *ldb;
    memcpy(g_cgbsv_fortran_call.ipiv_snapshot, ipiv,
           sizeof(g_cgbsv_fortran_call.ipiv_snapshot));
    for (index = 0; index < (*ldab * *n); ++index) {
        g_cgbsv_fortran_call.ab_real_snapshot[index] = cfloat_real(ab[index]);
        ab[index] = make_cfloat(2000.0f + (float)index);
    }
    for (index = 0; index < (*ldb * *nrhs); ++index) {
        g_cgbsv_fortran_call.b_real_snapshot[index] = cfloat_real(b[index]);
        b[index] = make_cfloat(800.0f + (float)index);
    }
    ipiv[0] = 1;
    ipiv[1] = 4;
    ipiv[2] = 3;
    ipiv[3] = 2;
    *info = 0;
}

static int stub_cgbsv_cblas(fb_layout_t layout, int n, int kl, int ku, int nrhs,
                            fb_complex_float_t *ab, int ldab, int *ipiv,
                            fb_complex_float_t *b, int ldb)
{
    g_cgbsv_cblas_call.called += 1;
    g_cgbsv_cblas_call.layout = layout;
    g_cgbsv_cblas_call.n = n;
    g_cgbsv_cblas_call.kl = kl;
    g_cgbsv_cblas_call.ku = ku;
    g_cgbsv_cblas_call.nrhs = nrhs;
    g_cgbsv_cblas_call.ldab = ldab;
    g_cgbsv_cblas_call.ldb = ldb;
    g_cgbsv_cblas_call.ab = ab;
    g_cgbsv_cblas_call.ipiv = ipiv;
    g_cgbsv_cblas_call.b = b;
    ab[0] = make_cfloat(811.0f);
    ab[5] = make_cfloat(822.0f);
    b[0] = make_cfloat(833.0f);
    b[4] = make_cfloat(844.0f);
    ipiv[0] = 2;
    ipiv[1] = 3;
    ipiv[2] = 1;
    return 223;
}

static void stub_zgbsv_fortran(int *n, int *kl, int *ku, int *nrhs,
                               fb_complex_double_t *ab, int *ldab, int *ipiv,
                               fb_complex_double_t *b, int *ldb, int *info)
{
    int index = 0;

    g_zgbsv_fortran_call.calls += 1;
    g_zgbsv_fortran_call.n = *n;
    g_zgbsv_fortran_call.kl = *kl;
    g_zgbsv_fortran_call.ku = *ku;
    g_zgbsv_fortran_call.nrhs = *nrhs;
    g_zgbsv_fortran_call.ldab = *ldab;
    g_zgbsv_fortran_call.ldb = *ldb;
    memcpy(g_zgbsv_fortran_call.ipiv_snapshot, ipiv,
           sizeof(g_zgbsv_fortran_call.ipiv_snapshot));
    for (index = 0; index < (*ldab * *n); ++index) {
        g_zgbsv_fortran_call.ab_real_snapshot[index] = cdouble_real(ab[index]);
        ab[index] = make_cdouble(2100.0 + (double)index);
    }
    for (index = 0; index < (*ldb * *nrhs); ++index) {
        g_zgbsv_fortran_call.b_real_snapshot[index] = cdouble_real(b[index]);
        b[index] = make_cdouble(950.0 + (double)index);
    }
    ipiv[0] = 1;
    ipiv[1] = 4;
    ipiv[2] = 3;
    ipiv[3] = 2;
    *info = 0;
}

static int stub_zgbsv_cblas(fb_layout_t layout, int n, int kl, int ku, int nrhs,
                            fb_complex_double_t *ab, int ldab, int *ipiv,
                            fb_complex_double_t *b, int ldb)
{
    g_zgbsv_cblas_call.called += 1;
    g_zgbsv_cblas_call.layout = layout;
    g_zgbsv_cblas_call.n = n;
    g_zgbsv_cblas_call.kl = kl;
    g_zgbsv_cblas_call.ku = ku;
    g_zgbsv_cblas_call.nrhs = nrhs;
    g_zgbsv_cblas_call.ldab = ldab;
    g_zgbsv_cblas_call.ldb = ldb;
    g_zgbsv_cblas_call.ab = ab;
    g_zgbsv_cblas_call.ipiv = ipiv;
    g_zgbsv_cblas_call.b = b;
    ab[0] = make_cdouble(1811.0);
    ab[5] = make_cdouble(1822.0);
    b[0] = make_cdouble(1833.0);
    b[4] = make_cdouble(1844.0);
    ipiv[0] = 2;
    ipiv[1] = 3;
    ipiv[2] = 1;
    return 224;
}

static int check_sgbsv_fortran_to_cblas(void)
{
    fb_backend_vtable_t vtable;
    fb_sgbsv_fn thunk = NULL;
    float ab[16] = {
        0.0f, 0.0f, 130.0f, 140.0f,
        0.0f, 220.0f, 230.0f, 240.0f,
        310.0f, 320.0f, 330.0f, 340.0f,
        410.0f, 420.0f, 430.0f, 0.0f
    };
    int ipiv[4] = { 9, 8, 7, 6 };
    float b[8] = {
        10.0f, 11.0f,
        20.0f, 21.0f,
        30.0f, 31.0f,
        40.0f, 41.0f
    };
    static const float expected_ab_snapshot[16] = {
        0.0f, 0.0f, 310.0f, 410.0f,
        0.0f, 220.0f, 320.0f, 420.0f,
        130.0f, 230.0f, 330.0f, 430.0f,
        140.0f, 240.0f, 340.0f, 0.0f
    };
    static const float expected_b_snapshot[8] = {
        10.0f, 20.0f, 30.0f, 40.0f,
        11.0f, 21.0f, 31.0f, 41.0f
    };
    static const float expected_ab_after[16] = {
        0.0f, 0.0f, 1008.0f, 1012.0f,
        0.0f, 1005.0f, 1009.0f, 1013.0f,
        1002.0f, 1006.0f, 1010.0f, 1014.0f,
        1003.0f, 1007.0f, 1011.0f, 0.0f
    };
    static const float expected_b_after[8] = {
        600.0f, 604.0f,
        601.0f, 605.0f,
        602.0f, 606.0f,
        603.0f, 607.0f
    };
    static const int expected_ipiv_snapshot[4] = { 9, 8, 7, 6 };
    static const int expected_ipiv_after[4] = { 4, 3, 2, 1 };
    int info = 0;

    memset(&vtable, 0, sizeof(vtable));
    memset(&g_sgbsv_fortran_call, 0, sizeof(g_sgbsv_fortran_call));

    vtable.ext_ops[FB_OP_SGBSV][FB_CONV_FORTRAN] =
        (fb_generic_fn)(void (*)(void))stub_sgbsv_fortran;
    fb_install_conv_thunks(&vtable, FB_OP_SGBSV);

    thunk = (fb_sgbsv_fn)vtable.ext_ops[FB_OP_SGBSV][FB_CONV_CBLAS];
    if (!thunk) {
        fprintf(stderr, "[FAIL] SGBSV Fortran->CBLAS thunk was not installed\n");
        return 1;
    }

    info = thunk(FB_LAYOUT_ROW_MAJOR, 4, 1, 1, 2, ab, 4, ipiv, b, 2);
    if (info != 0 || g_sgbsv_fortran_call.calls != 1 ||
        g_sgbsv_fortran_call.n != 4 || g_sgbsv_fortran_call.kl != 1 ||
        g_sgbsv_fortran_call.ku != 1 || g_sgbsv_fortran_call.nrhs != 2 ||
        g_sgbsv_fortran_call.ldab != 4 || g_sgbsv_fortran_call.ldb != 4 ||
         memcmp(g_sgbsv_fortran_call.ipiv_snapshot, expected_ipiv_snapshot,
             sizeof(expected_ipiv_snapshot)) != 0 ||
        memcmp(g_sgbsv_fortran_call.ab_snapshot, expected_ab_snapshot,
               sizeof(expected_ab_snapshot)) != 0 ||
        memcmp(g_sgbsv_fortran_call.b_snapshot, expected_b_snapshot,
               sizeof(expected_b_snapshot)) != 0 ||
        memcmp(ab, expected_ab_after, sizeof(expected_ab_after)) != 0 ||
        memcmp(b, expected_b_after, sizeof(expected_b_after)) != 0 ||
        memcmp(ipiv, expected_ipiv_after, sizeof(expected_ipiv_after)) != 0) {
        fprintf(stderr, "[FAIL] SGBSV Fortran->CBLAS thunk did not preserve expanded row-major direct-solve semantics\n");
        return 1;
    }

    printf("[PASS] SGBSV Fortran->CBLAS thunk transposes expanded row-major band storage, preserves pivots, and copies AB and RHS back\n");
    return 0;
}

static int check_sgbsv_cblas_to_fortran(void)
{
    fb_backend_vtable_t vtable;
    fb_sgbsv_fortran_slot_fn thunk = NULL;
    float ab[12] = {
        0.0f, 0.0f, 13.0f,
        0.0f, 22.0f, 23.0f,
        31.0f, 32.0f, 33.0f,
        41.0f, 42.0f, 0.0f
    };
    int ipiv[3] = { 0, 0, 0 };
    float b[6] = {
        1.0f, 2.0f,
        3.0f, 4.0f,
        5.0f, 6.0f
    };
    int n = 3;
    int kl = 1;
    int ku = 1;
    int nrhs = 2;
    int ldab = 4;
    int ldb = 3;
    int info = 0;

    memset(&vtable, 0, sizeof(vtable));
    memset(&g_sgbsv_cblas_call, 0, sizeof(g_sgbsv_cblas_call));

    vtable.ext_ops[FB_OP_SGBSV][FB_CONV_CBLAS] =
        (fb_generic_fn)(void (*)(void))stub_sgbsv_cblas;
    fb_install_conv_thunks(&vtable, FB_OP_SGBSV);

    thunk = (fb_sgbsv_fortran_slot_fn)vtable.ext_ops[FB_OP_SGBSV][FB_CONV_FORTRAN];
    if (!thunk) {
        fprintf(stderr, "[FAIL] SGBSV CBLAS->Fortran thunk was not installed\n");
        return 1;
    }

    thunk(&n, &kl, &ku, &nrhs, ab, &ldab, ipiv, b, &ldb, &info);
    if (info != 221 || g_sgbsv_cblas_call.called != 1 ||
        g_sgbsv_cblas_call.layout != FB_LAYOUT_COL_MAJOR ||
        g_sgbsv_cblas_call.n != 3 || g_sgbsv_cblas_call.kl != 1 ||
        g_sgbsv_cblas_call.ku != 1 || g_sgbsv_cblas_call.nrhs != 2 ||
        g_sgbsv_cblas_call.ldab != 4 || g_sgbsv_cblas_call.ldb != 3 ||
        g_sgbsv_cblas_call.ab != ab || g_sgbsv_cblas_call.ipiv != ipiv ||
        g_sgbsv_cblas_call.b != b || ab[0] != 911.0f || ab[5] != 922.0f ||
        b[0] != 933.0f || b[4] != 944.0f || ipiv[0] != 3 || ipiv[1] != 2 ||
        ipiv[2] != 1) {
        fprintf(stderr, "[FAIL] SGBSV CBLAS->Fortran thunk delegated incorrectly\n");
        return 1;
    }

    printf("[PASS] SGBSV CBLAS->Fortran thunk maps the all-pointer ABI into the generic C direct-solve entry\n");
    return 0;
}

static int check_dgbsv_fortran_to_cblas(void)
{
    fb_backend_vtable_t vtable;
    fb_dgbsv_fn thunk = NULL;
    double ab[16] = {
        0.0, 0.0, 130.0, 140.0,
        0.0, 220.0, 230.0, 240.0,
        310.0, 320.0, 330.0, 340.0,
        410.0, 420.0, 430.0, 0.0
    };
    int ipiv[4] = { 9, 8, 7, 6 };
    double b[8] = {
        10.0, 11.0,
        20.0, 21.0,
        30.0, 31.0,
        40.0, 41.0
    };
    static const double expected_ab_snapshot[16] = {
        0.0, 0.0, 310.0, 410.0,
        0.0, 220.0, 320.0, 420.0,
        130.0, 230.0, 330.0, 430.0,
        140.0, 240.0, 340.0, 0.0
    };
    static const double expected_b_snapshot[8] = {
        10.0, 20.0, 30.0, 40.0,
        11.0, 21.0, 31.0, 41.0
    };
    static const double expected_ab_after[16] = {
        0.0, 0.0, 1108.0, 1112.0,
        0.0, 1105.0, 1109.0, 1113.0,
        1102.0, 1106.0, 1110.0, 1114.0,
        1103.0, 1107.0, 1111.0, 0.0
    };
    static const double expected_b_after[8] = {
        900.0, 904.0,
        901.0, 905.0,
        902.0, 906.0,
        903.0, 907.0
    };
    static const int expected_ipiv_snapshot[4] = { 9, 8, 7, 6 };
    static const int expected_ipiv_after[4] = { 4, 3, 2, 1 };
    int info = 0;

    memset(&vtable, 0, sizeof(vtable));
    memset(&g_dgbsv_fortran_call, 0, sizeof(g_dgbsv_fortran_call));

    vtable.ext_ops[FB_OP_DGBSV][FB_CONV_FORTRAN] =
        (fb_generic_fn)(void (*)(void))stub_dgbsv_fortran;
    fb_install_conv_thunks(&vtable, FB_OP_DGBSV);

    thunk = (fb_dgbsv_fn)vtable.ext_ops[FB_OP_DGBSV][FB_CONV_CBLAS];
    if (!thunk) {
        fprintf(stderr, "[FAIL] DGBSV Fortran->CBLAS thunk was not installed\n");
        return 1;
    }

    info = thunk(FB_LAYOUT_ROW_MAJOR, 4, 1, 1, 2, ab, 4, ipiv, b, 2);
    if (info != 0 || g_dgbsv_fortran_call.calls != 1 ||
        g_dgbsv_fortran_call.n != 4 || g_dgbsv_fortran_call.kl != 1 ||
        g_dgbsv_fortran_call.ku != 1 || g_dgbsv_fortran_call.nrhs != 2 ||
        g_dgbsv_fortran_call.ldab != 4 || g_dgbsv_fortran_call.ldb != 4 ||
        memcmp(g_dgbsv_fortran_call.ipiv_snapshot, expected_ipiv_snapshot,
               sizeof(expected_ipiv_snapshot)) != 0 ||
        memcmp(g_dgbsv_fortran_call.ab_snapshot, expected_ab_snapshot,
               sizeof(expected_ab_snapshot)) != 0 ||
        memcmp(g_dgbsv_fortran_call.b_snapshot, expected_b_snapshot,
               sizeof(expected_b_snapshot)) != 0 ||
        memcmp(ab, expected_ab_after, sizeof(expected_ab_after)) != 0 ||
        memcmp(b, expected_b_after, sizeof(expected_b_after)) != 0 ||
        memcmp(ipiv, expected_ipiv_after, sizeof(expected_ipiv_after)) != 0) {
        fprintf(stderr, "[FAIL] DGBSV Fortran->CBLAS thunk did not preserve expanded row-major direct-solve semantics\n");
        return 1;
    }

    printf("[PASS] DGBSV Fortran->CBLAS thunk transposes expanded row-major double band storage, preserves pivots, and copies AB and RHS back\n");
    return 0;
}

static int check_dgbsv_cblas_to_fortran(void)
{
    fb_backend_vtable_t vtable;
    fb_dgbsv_fortran_slot_fn thunk = NULL;
    double ab[12] = {
        0.0, 0.0, 13.0,
        0.0, 22.0, 23.0,
        31.0, 32.0, 33.0,
        41.0, 42.0, 0.0
    };
    int ipiv[3] = { 0, 0, 0 };
    double b[6] = {
        1.0, 2.0,
        3.0, 4.0,
        5.0, 6.0
    };
    int n = 3;
    int kl = 1;
    int ku = 1;
    int nrhs = 2;
    int ldab = 4;
    int ldb = 3;
    int info = 0;

    memset(&vtable, 0, sizeof(vtable));
    memset(&g_dgbsv_cblas_call, 0, sizeof(g_dgbsv_cblas_call));

    vtable.ext_ops[FB_OP_DGBSV][FB_CONV_CBLAS] =
        (fb_generic_fn)(void (*)(void))stub_dgbsv_cblas;
    fb_install_conv_thunks(&vtable, FB_OP_DGBSV);

    thunk = (fb_dgbsv_fortran_slot_fn)vtable.ext_ops[FB_OP_DGBSV][FB_CONV_FORTRAN];
    if (!thunk) {
        fprintf(stderr, "[FAIL] DGBSV CBLAS->Fortran thunk was not installed\n");
        return 1;
    }

    thunk(&n, &kl, &ku, &nrhs, ab, &ldab, ipiv, b, &ldb, &info);
    if (info != 222 || g_dgbsv_cblas_call.called != 1 ||
        g_dgbsv_cblas_call.layout != FB_LAYOUT_COL_MAJOR ||
        g_dgbsv_cblas_call.n != 3 || g_dgbsv_cblas_call.kl != 1 ||
        g_dgbsv_cblas_call.ku != 1 || g_dgbsv_cblas_call.nrhs != 2 ||
        g_dgbsv_cblas_call.ldab != 4 || g_dgbsv_cblas_call.ldb != 3 ||
        g_dgbsv_cblas_call.ab != ab || g_dgbsv_cblas_call.ipiv != ipiv ||
        g_dgbsv_cblas_call.b != b || ab[0] != 1011.0 || ab[5] != 1022.0 ||
        b[0] != 1033.0 || b[4] != 1044.0 || ipiv[0] != 3 || ipiv[1] != 2 ||
        ipiv[2] != 1) {
        fprintf(stderr, "[FAIL] DGBSV CBLAS->Fortran thunk delegated incorrectly\n");
        return 1;
    }

    printf("[PASS] DGBSV CBLAS->Fortran thunk maps the all-pointer ABI into the generic C double direct-solve entry\n");
    return 0;
}

static int check_cgbsv_fortran_to_cblas(void)
{
    fb_backend_vtable_t vtable;
    fb_cgbsv_fn thunk = NULL;
    fb_complex_float_t ab[16] = {
        make_cfloat(0.0f), make_cfloat(0.0f), make_cfloat(130.0f), make_cfloat(140.0f),
        make_cfloat(0.0f), make_cfloat(220.0f), make_cfloat(230.0f), make_cfloat(240.0f),
        make_cfloat(310.0f), make_cfloat(320.0f), make_cfloat(330.0f), make_cfloat(340.0f),
        make_cfloat(410.0f), make_cfloat(420.0f), make_cfloat(430.0f), make_cfloat(0.0f)
    };
    int ipiv[4] = { 4, 4, 4, 4 };
    fb_complex_float_t b[8] = {
        make_cfloat(50.0f), make_cfloat(51.0f),
        make_cfloat(60.0f), make_cfloat(61.0f),
        make_cfloat(70.0f), make_cfloat(71.0f),
        make_cfloat(80.0f), make_cfloat(81.0f)
    };
    static const float expected_ab_snapshot[16] = {
        0.0f, 0.0f, 310.0f, 410.0f,
        0.0f, 220.0f, 320.0f, 420.0f,
        130.0f, 230.0f, 330.0f, 430.0f,
        140.0f, 240.0f, 340.0f, 0.0f
    };
    static const float expected_b_snapshot[8] = {
        50.0f, 60.0f, 70.0f, 80.0f,
        51.0f, 61.0f, 71.0f, 81.0f
    };
    static const float expected_ab_after[16] = {
        0.0f, 0.0f, 2008.0f, 2012.0f,
        0.0f, 2005.0f, 2009.0f, 2013.0f,
        2002.0f, 2006.0f, 2010.0f, 2014.0f,
        2003.0f, 2007.0f, 2011.0f, 0.0f
    };
    static const float expected_b_after[8] = {
        800.0f, 804.0f,
        801.0f, 805.0f,
        802.0f, 806.0f,
        803.0f, 807.0f
    };
    static const int expected_ipiv_snapshot[4] = { 4, 4, 4, 4 };
    static const int expected_ipiv_after[4] = { 1, 4, 3, 2 };
    float ab_real_after[16] = { 0.0f };
    float b_real_after[8] = { 0.0f };
    int index = 0;
    int info = 0;

    memset(&vtable, 0, sizeof(vtable));
    memset(&g_cgbsv_fortran_call, 0, sizeof(g_cgbsv_fortran_call));

    vtable.ext_ops[FB_OP_CGBSV][FB_CONV_FORTRAN] =
        (fb_generic_fn)(void (*)(void))stub_cgbsv_fortran;
    fb_install_conv_thunks(&vtable, FB_OP_CGBSV);

    thunk = (fb_cgbsv_fn)vtable.ext_ops[FB_OP_CGBSV][FB_CONV_CBLAS];
    if (!thunk) {
        fprintf(stderr, "[FAIL] CGBSV Fortran->CBLAS thunk was not installed\n");
        return 1;
    }

    info = thunk(FB_LAYOUT_ROW_MAJOR, 4, 1, 1, 2, ab, 4, ipiv, b, 2);
    for (index = 0; index < 16; ++index) {
        ab_real_after[index] = cfloat_real(ab[index]);
    }
    for (index = 0; index < 8; ++index) {
        b_real_after[index] = cfloat_real(b[index]);
    }
    if (info != 0 || g_cgbsv_fortran_call.calls != 1 ||
        g_cgbsv_fortran_call.n != 4 || g_cgbsv_fortran_call.kl != 1 ||
        g_cgbsv_fortran_call.ku != 1 || g_cgbsv_fortran_call.nrhs != 2 ||
        g_cgbsv_fortran_call.ldab != 4 || g_cgbsv_fortran_call.ldb != 4 ||
         memcmp(g_cgbsv_fortran_call.ipiv_snapshot, expected_ipiv_snapshot,
             sizeof(expected_ipiv_snapshot)) != 0 ||
        memcmp(g_cgbsv_fortran_call.ab_real_snapshot, expected_ab_snapshot,
               sizeof(expected_ab_snapshot)) != 0 ||
        memcmp(g_cgbsv_fortran_call.b_real_snapshot, expected_b_snapshot,
               sizeof(expected_b_snapshot)) != 0 ||
        memcmp(ab_real_after, expected_ab_after, sizeof(expected_ab_after)) != 0 ||
        memcmp(b_real_after, expected_b_after, sizeof(expected_b_after)) != 0 ||
        memcmp(ipiv, expected_ipiv_after, sizeof(expected_ipiv_after)) != 0) {
        fprintf(stderr, "[FAIL] CGBSV Fortran->CBLAS thunk did not preserve complex expanded row-major direct-solve semantics\n");
        return 1;
    }

    printf("[PASS] CGBSV Fortran->CBLAS thunk transposes expanded row-major complex band storage, preserves pivots, and copies AB and RHS back\n");
    return 0;
}

static int check_cgbsv_cblas_to_fortran(void)
{
    fb_backend_vtable_t vtable;
    fb_cgbsv_fortran_slot_fn thunk = NULL;
    fb_complex_float_t ab[12] = {
        make_cfloat(0.0f), make_cfloat(0.0f), make_cfloat(13.0f),
        make_cfloat(0.0f), make_cfloat(22.0f), make_cfloat(23.0f),
        make_cfloat(31.0f), make_cfloat(32.0f), make_cfloat(33.0f),
        make_cfloat(41.0f), make_cfloat(42.0f), make_cfloat(0.0f)
    };
    int ipiv[3] = { 0, 0, 0 };
    fb_complex_float_t b[6] = {
        make_cfloat(1.0f), make_cfloat(2.0f),
        make_cfloat(3.0f), make_cfloat(4.0f),
        make_cfloat(5.0f), make_cfloat(6.0f)
    };
    int n = 3;
    int kl = 1;
    int ku = 1;
    int nrhs = 2;
    int ldab = 4;
    int ldb = 3;
    int info = 0;

    memset(&vtable, 0, sizeof(vtable));
    memset(&g_cgbsv_cblas_call, 0, sizeof(g_cgbsv_cblas_call));

    vtable.ext_ops[FB_OP_CGBSV][FB_CONV_CBLAS] =
        (fb_generic_fn)(void (*)(void))stub_cgbsv_cblas;
    fb_install_conv_thunks(&vtable, FB_OP_CGBSV);

    thunk = (fb_cgbsv_fortran_slot_fn)vtable.ext_ops[FB_OP_CGBSV][FB_CONV_FORTRAN];
    if (!thunk) {
        fprintf(stderr, "[FAIL] CGBSV CBLAS->Fortran thunk was not installed\n");
        return 1;
    }

    thunk(&n, &kl, &ku, &nrhs, ab, &ldab, ipiv, b, &ldb, &info);
    if (info != 223 || g_cgbsv_cblas_call.called != 1 ||
        g_cgbsv_cblas_call.layout != FB_LAYOUT_COL_MAJOR ||
        g_cgbsv_cblas_call.n != 3 || g_cgbsv_cblas_call.kl != 1 ||
        g_cgbsv_cblas_call.ku != 1 || g_cgbsv_cblas_call.nrhs != 2 ||
        g_cgbsv_cblas_call.ldab != 4 || g_cgbsv_cblas_call.ldb != 3 ||
        g_cgbsv_cblas_call.ab != ab || g_cgbsv_cblas_call.ipiv != ipiv ||
        g_cgbsv_cblas_call.b != b || cfloat_real(ab[0]) != 811.0f ||
        cfloat_real(ab[5]) != 822.0f || cfloat_real(b[0]) != 833.0f ||
        cfloat_real(b[4]) != 844.0f || ipiv[0] != 2 || ipiv[1] != 3 ||
        ipiv[2] != 1) {
        fprintf(stderr, "[FAIL] CGBSV CBLAS->Fortran thunk delegated incorrectly\n");
        return 1;
    }

    printf("[PASS] CGBSV CBLAS->Fortran thunk maps the all-pointer ABI into the generic complex direct-solve entry\n");
    return 0;
}

static int check_zgbsv_fortran_to_cblas(void)
{
    fb_backend_vtable_t vtable;
    fb_zgbsv_fn thunk = NULL;
    fb_complex_double_t ab[16] = {
        make_cdouble(0.0), make_cdouble(0.0), make_cdouble(130.0), make_cdouble(140.0),
        make_cdouble(0.0), make_cdouble(220.0), make_cdouble(230.0), make_cdouble(240.0),
        make_cdouble(310.0), make_cdouble(320.0), make_cdouble(330.0), make_cdouble(340.0),
        make_cdouble(410.0), make_cdouble(420.0), make_cdouble(430.0), make_cdouble(0.0)
    };
    int ipiv[4] = { 4, 4, 4, 4 };
    fb_complex_double_t b[8] = {
        make_cdouble(50.0), make_cdouble(51.0),
        make_cdouble(60.0), make_cdouble(61.0),
        make_cdouble(70.0), make_cdouble(71.0),
        make_cdouble(80.0), make_cdouble(81.0)
    };
    static const double expected_ab_snapshot[16] = {
        0.0, 0.0, 310.0, 410.0,
        0.0, 220.0, 320.0, 420.0,
        130.0, 230.0, 330.0, 430.0,
        140.0, 240.0, 340.0, 0.0
    };
    static const double expected_b_snapshot[8] = {
        50.0, 60.0, 70.0, 80.0,
        51.0, 61.0, 71.0, 81.0
    };
    static const double expected_ab_after[16] = {
        0.0, 0.0, 2108.0, 2112.0,
        0.0, 2105.0, 2109.0, 2113.0,
        2102.0, 2106.0, 2110.0, 2114.0,
        2103.0, 2107.0, 2111.0, 0.0
    };
    static const double expected_b_after[8] = {
        950.0, 954.0,
        951.0, 955.0,
        952.0, 956.0,
        953.0, 957.0
    };
    static const int expected_ipiv_snapshot[4] = { 4, 4, 4, 4 };
    static const int expected_ipiv_after[4] = { 1, 4, 3, 2 };
    double ab_real_after[16] = { 0.0 };
    double b_real_after[8] = { 0.0 };
    int index = 0;
    int info = 0;

    memset(&vtable, 0, sizeof(vtable));
    memset(&g_zgbsv_fortran_call, 0, sizeof(g_zgbsv_fortran_call));

    vtable.ext_ops[FB_OP_ZGBSV][FB_CONV_FORTRAN] =
        (fb_generic_fn)(void (*)(void))stub_zgbsv_fortran;
    fb_install_conv_thunks(&vtable, FB_OP_ZGBSV);

    thunk = (fb_zgbsv_fn)vtable.ext_ops[FB_OP_ZGBSV][FB_CONV_CBLAS];
    if (!thunk) {
        fprintf(stderr, "[FAIL] ZGBSV Fortran->CBLAS thunk was not installed\n");
        return 1;
    }

    info = thunk(FB_LAYOUT_ROW_MAJOR, 4, 1, 1, 2, ab, 4, ipiv, b, 2);
    for (index = 0; index < 16; ++index) {
        ab_real_after[index] = cdouble_real(ab[index]);
    }
    for (index = 0; index < 8; ++index) {
        b_real_after[index] = cdouble_real(b[index]);
    }
    if (info != 0 || g_zgbsv_fortran_call.calls != 1 ||
        g_zgbsv_fortran_call.n != 4 || g_zgbsv_fortran_call.kl != 1 ||
        g_zgbsv_fortran_call.ku != 1 || g_zgbsv_fortran_call.nrhs != 2 ||
        g_zgbsv_fortran_call.ldab != 4 || g_zgbsv_fortran_call.ldb != 4 ||
        memcmp(g_zgbsv_fortran_call.ipiv_snapshot, expected_ipiv_snapshot,
               sizeof(expected_ipiv_snapshot)) != 0 ||
        memcmp(g_zgbsv_fortran_call.ab_real_snapshot, expected_ab_snapshot,
               sizeof(expected_ab_snapshot)) != 0 ||
        memcmp(g_zgbsv_fortran_call.b_real_snapshot, expected_b_snapshot,
               sizeof(expected_b_snapshot)) != 0 ||
        memcmp(ab_real_after, expected_ab_after, sizeof(expected_ab_after)) != 0 ||
        memcmp(b_real_after, expected_b_after, sizeof(expected_b_after)) != 0 ||
        memcmp(ipiv, expected_ipiv_after, sizeof(expected_ipiv_after)) != 0) {
        fprintf(stderr, "[FAIL] ZGBSV Fortran->CBLAS thunk did not preserve complex-double expanded row-major direct-solve semantics\n");
        return 1;
    }

    printf("[PASS] ZGBSV Fortran->CBLAS thunk transposes expanded row-major complex-double band storage, preserves pivots, and copies AB and RHS back\n");
    return 0;
}

static int check_zgbsv_cblas_to_fortran(void)
{
    fb_backend_vtable_t vtable;
    fb_zgbsv_fortran_slot_fn thunk = NULL;
    fb_complex_double_t ab[12] = {
        make_cdouble(0.0), make_cdouble(0.0), make_cdouble(13.0),
        make_cdouble(0.0), make_cdouble(22.0), make_cdouble(23.0),
        make_cdouble(31.0), make_cdouble(32.0), make_cdouble(33.0),
        make_cdouble(41.0), make_cdouble(42.0), make_cdouble(0.0)
    };
    int ipiv[3] = { 0, 0, 0 };
    fb_complex_double_t b[6] = {
        make_cdouble(1.0), make_cdouble(2.0),
        make_cdouble(3.0), make_cdouble(4.0),
        make_cdouble(5.0), make_cdouble(6.0)
    };
    int n = 3;
    int kl = 1;
    int ku = 1;
    int nrhs = 2;
    int ldab = 4;
    int ldb = 3;
    int info = 0;

    memset(&vtable, 0, sizeof(vtable));
    memset(&g_zgbsv_cblas_call, 0, sizeof(g_zgbsv_cblas_call));

    vtable.ext_ops[FB_OP_ZGBSV][FB_CONV_CBLAS] =
        (fb_generic_fn)(void (*)(void))stub_zgbsv_cblas;
    fb_install_conv_thunks(&vtable, FB_OP_ZGBSV);

    thunk = (fb_zgbsv_fortran_slot_fn)vtable.ext_ops[FB_OP_ZGBSV][FB_CONV_FORTRAN];
    if (!thunk) {
        fprintf(stderr, "[FAIL] ZGBSV CBLAS->Fortran thunk was not installed\n");
        return 1;
    }

    thunk(&n, &kl, &ku, &nrhs, ab, &ldab, ipiv, b, &ldb, &info);
    if (info != 224 || g_zgbsv_cblas_call.called != 1 ||
        g_zgbsv_cblas_call.layout != FB_LAYOUT_COL_MAJOR ||
        g_zgbsv_cblas_call.n != 3 || g_zgbsv_cblas_call.kl != 1 ||
        g_zgbsv_cblas_call.ku != 1 || g_zgbsv_cblas_call.nrhs != 2 ||
        g_zgbsv_cblas_call.ldab != 4 || g_zgbsv_cblas_call.ldb != 3 ||
        g_zgbsv_cblas_call.ab != ab || g_zgbsv_cblas_call.ipiv != ipiv ||
        g_zgbsv_cblas_call.b != b || cdouble_real(ab[0]) != 1811.0 ||
        cdouble_real(ab[5]) != 1822.0 || cdouble_real(b[0]) != 1833.0 ||
        cdouble_real(b[4]) != 1844.0 || ipiv[0] != 2 || ipiv[1] != 3 ||
        ipiv[2] != 1) {
        fprintf(stderr, "[FAIL] ZGBSV CBLAS->Fortran thunk delegated incorrectly\n");
        return 1;
    }

    printf("[PASS] ZGBSV CBLAS->Fortran thunk maps the all-pointer ABI into the generic complex-double direct-solve entry\n");
    return 0;
}

int main(void)
{
    if (check_sgbsv_fortran_to_cblas() != 0) {
        return 1;
    }
    if (check_sgbsv_cblas_to_fortran() != 0) {
        return 1;
    }
    if (check_dgbsv_fortran_to_cblas() != 0) {
        return 1;
    }
    if (check_dgbsv_cblas_to_fortran() != 0) {
        return 1;
    }
    if (check_cgbsv_fortran_to_cblas() != 0) {
        return 1;
    }
    if (check_cgbsv_cblas_to_fortran() != 0) {
        return 1;
    }
    if (check_zgbsv_fortran_to_cblas() != 0) {
        return 1;
    }
    if (check_zgbsv_cblas_to_fortran() != 0) {
        return 1;
    }

    printf("Result: PASS\n");
    return 0;
}