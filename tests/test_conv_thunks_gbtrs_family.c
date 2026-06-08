#include <stdio.h>
#include <string.h>

#include "../src/core/conv_thunks.h"

typedef int (*fb_sgbtrs_fn)(fb_layout_t layout, char trans, int n, int kl,
                            int ku, int nrhs, const float *ab, int ldab,
                            const int *ipiv, float *b, int ldb);
typedef int (*fb_cgbtrs_fn)(fb_layout_t layout, char trans, int n, int kl,
                            int ku, int nrhs, const fb_complex_float_t *ab,
                            int ldab, const int *ipiv,
                            fb_complex_float_t *b, int ldb);

typedef void (*fb_sgbtrs_fortran_slot_fn)(char *trans, int *n, int *kl,
                                          int *ku, int *nrhs, float *ab,
                                          int *ldab, int *ipiv, float *b,
                                          int *ldb, int *info);
typedef void (*fb_cgbtrs_fortran_slot_fn)(char *trans, int *n, int *kl,
                                          int *ku, int *nrhs,
                                          fb_complex_float_t *ab, int *ldab,
                                          int *ipiv, fb_complex_float_t *b,
                                          int *ldb, int *info);

static struct {
    int calls;
    char trans;
    int n;
    int kl;
    int ku;
    int nrhs;
    int ldab;
    int ldb;
    int ipiv_snapshot[4];
    float ab_snapshot[16];
    float b_snapshot[8];
} g_sgbtrs_fortran_call;

static struct {
    int called;
    fb_layout_t layout;
    char trans;
    int n;
    int kl;
    int ku;
    int nrhs;
    int ldab;
    int ldb;
    const float *ab;
    const int *ipiv;
    float *b;
} g_sgbtrs_cblas_call;

static struct {
    int calls;
    char trans;
    int n;
    int kl;
    int ku;
    int nrhs;
    int ldab;
    int ldb;
    int ipiv_snapshot[4];
    float ab_real_snapshot[16];
    float b_real_snapshot[8];
} g_cgbtrs_fortran_call;

static struct {
    int called;
    fb_layout_t layout;
    char trans;
    int n;
    int kl;
    int ku;
    int nrhs;
    int ldab;
    int ldb;
    const fb_complex_float_t *ab;
    const int *ipiv;
    fb_complex_float_t *b;
} g_cgbtrs_cblas_call;

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

static void stub_sgbtrs_fortran(char *trans, int *n, int *kl, int *ku,
                                int *nrhs, float *ab, int *ldab, int *ipiv,
                                float *b, int *ldb, int *info)
{
    int index = 0;

    g_sgbtrs_fortran_call.calls += 1;
    g_sgbtrs_fortran_call.trans = *trans;
    g_sgbtrs_fortran_call.n = *n;
    g_sgbtrs_fortran_call.kl = *kl;
    g_sgbtrs_fortran_call.ku = *ku;
    g_sgbtrs_fortran_call.nrhs = *nrhs;
    g_sgbtrs_fortran_call.ldab = *ldab;
    g_sgbtrs_fortran_call.ldb = *ldb;
    memcpy(g_sgbtrs_fortran_call.ipiv_snapshot, ipiv,
           sizeof(g_sgbtrs_fortran_call.ipiv_snapshot));
    for (index = 0; index < (*ldab * *n); ++index) {
        g_sgbtrs_fortran_call.ab_snapshot[index] = ab[index];
    }
    for (index = 0; index < (*ldb * *nrhs); ++index) {
        g_sgbtrs_fortran_call.b_snapshot[index] = b[index];
        b[index] = 600.0f + (float)index;
    }
    *info = 0;
}

static int stub_sgbtrs_cblas(fb_layout_t layout, char trans, int n, int kl,
                             int ku, int nrhs, const float *ab, int ldab,
                             const int *ipiv, float *b, int ldb)
{
    g_sgbtrs_cblas_call.called += 1;
    g_sgbtrs_cblas_call.layout = layout;
    g_sgbtrs_cblas_call.trans = trans;
    g_sgbtrs_cblas_call.n = n;
    g_sgbtrs_cblas_call.kl = kl;
    g_sgbtrs_cblas_call.ku = ku;
    g_sgbtrs_cblas_call.nrhs = nrhs;
    g_sgbtrs_cblas_call.ldab = ldab;
    g_sgbtrs_cblas_call.ldb = ldb;
    g_sgbtrs_cblas_call.ab = ab;
    g_sgbtrs_cblas_call.ipiv = ipiv;
    g_sgbtrs_cblas_call.b = b;
    b[0] = 711.0f;
    b[4] = 722.0f;
    return 211;
}

static void stub_cgbtrs_fortran(char *trans, int *n, int *kl, int *ku,
                                int *nrhs, fb_complex_float_t *ab, int *ldab,
                                int *ipiv, fb_complex_float_t *b, int *ldb,
                                int *info)
{
    int index = 0;

    g_cgbtrs_fortran_call.calls += 1;
    g_cgbtrs_fortran_call.trans = *trans;
    g_cgbtrs_fortran_call.n = *n;
    g_cgbtrs_fortran_call.kl = *kl;
    g_cgbtrs_fortran_call.ku = *ku;
    g_cgbtrs_fortran_call.nrhs = *nrhs;
    g_cgbtrs_fortran_call.ldab = *ldab;
    g_cgbtrs_fortran_call.ldb = *ldb;
    memcpy(g_cgbtrs_fortran_call.ipiv_snapshot, ipiv,
           sizeof(g_cgbtrs_fortran_call.ipiv_snapshot));
    for (index = 0; index < (*ldab * *n); ++index) {
        g_cgbtrs_fortran_call.ab_real_snapshot[index] = cfloat_real(ab[index]);
    }
    for (index = 0; index < (*ldb * *nrhs); ++index) {
        g_cgbtrs_fortran_call.b_real_snapshot[index] = cfloat_real(b[index]);
        b[index] = make_cfloat(800.0f + (float)index);
    }
    *info = 0;
}

static int stub_cgbtrs_cblas(fb_layout_t layout, char trans, int n, int kl,
                             int ku, int nrhs, const fb_complex_float_t *ab,
                             int ldab, const int *ipiv,
                             fb_complex_float_t *b, int ldb)
{
    g_cgbtrs_cblas_call.called += 1;
    g_cgbtrs_cblas_call.layout = layout;
    g_cgbtrs_cblas_call.trans = trans;
    g_cgbtrs_cblas_call.n = n;
    g_cgbtrs_cblas_call.kl = kl;
    g_cgbtrs_cblas_call.ku = ku;
    g_cgbtrs_cblas_call.nrhs = nrhs;
    g_cgbtrs_cblas_call.ldab = ldab;
    g_cgbtrs_cblas_call.ldb = ldb;
    g_cgbtrs_cblas_call.ab = ab;
    g_cgbtrs_cblas_call.ipiv = ipiv;
    g_cgbtrs_cblas_call.b = b;
    b[0] = make_cfloat(811.0f);
    b[4] = make_cfloat(822.0f);
    return 213;
}

static int check_sgbtrs_fortran_to_cblas(void)
{
    fb_backend_vtable_t vtable;
    fb_sgbtrs_fn thunk = NULL;
    float ab[16] = {
        0.0f, 0.0f, 130.0f, 140.0f,
        0.0f, 220.0f, 230.0f, 240.0f,
        310.0f, 320.0f, 330.0f, 340.0f,
        410.0f, 420.0f, 430.0f, 0.0f
    };
    int ipiv[4] = { 1, 4, 3, 2 };
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
    static const float expected_b_after[8] = {
        600.0f, 604.0f,
        601.0f, 605.0f,
        602.0f, 606.0f,
        603.0f, 607.0f
    };
    static const float expected_ab_after[16] = {
        0.0f, 0.0f, 130.0f, 140.0f,
        0.0f, 220.0f, 230.0f, 240.0f,
        310.0f, 320.0f, 330.0f, 340.0f,
        410.0f, 420.0f, 430.0f, 0.0f
    };
    int info = 0;

    memset(&vtable, 0, sizeof(vtable));
    memset(&g_sgbtrs_fortran_call, 0, sizeof(g_sgbtrs_fortran_call));

    vtable.ext_ops[FB_OP_SGBTRS][FB_CONV_FORTRAN] =
        (fb_generic_fn)(void (*)(void))stub_sgbtrs_fortran;
    fb_install_conv_thunks(&vtable, FB_OP_SGBTRS);

    thunk = (fb_sgbtrs_fn)vtable.ext_ops[FB_OP_SGBTRS][FB_CONV_CBLAS];
    if (!thunk) {
        fprintf(stderr, "[FAIL] SGBTRS Fortran->CBLAS thunk was not installed\n");
        return 1;
    }

    info = thunk(FB_LAYOUT_ROW_MAJOR, 'T', 4, 1, 1, 2, ab, 4, ipiv, b, 2);
    if (info != 0 || g_sgbtrs_fortran_call.calls != 1 ||
        g_sgbtrs_fortran_call.trans != 'T' || g_sgbtrs_fortran_call.n != 4 ||
        g_sgbtrs_fortran_call.kl != 1 || g_sgbtrs_fortran_call.ku != 1 ||
        g_sgbtrs_fortran_call.nrhs != 2 || g_sgbtrs_fortran_call.ldab != 4 ||
        g_sgbtrs_fortran_call.ldb != 4 ||
        memcmp(g_sgbtrs_fortran_call.ipiv_snapshot, ipiv,
               sizeof(ipiv)) != 0 ||
        memcmp(g_sgbtrs_fortran_call.ab_snapshot, expected_ab_snapshot,
               sizeof(expected_ab_snapshot)) != 0 ||
        memcmp(g_sgbtrs_fortran_call.b_snapshot, expected_b_snapshot,
               sizeof(expected_b_snapshot)) != 0 ||
        memcmp(ab, expected_ab_after, sizeof(expected_ab_after)) != 0 ||
        memcmp(b, expected_b_after, sizeof(expected_b_after)) != 0) {
        fprintf(stderr, "[FAIL] SGBTRS Fortran->CBLAS thunk did not preserve expanded factor-storage and RHS transpose semantics\n");
        return 1;
    }

    printf("[PASS] SGBTRS Fortran->CBLAS thunk transposes expanded row-major factor storage and round-trips the RHS matrix\n");
    return 0;
}

static int check_sgbtrs_cblas_to_fortran(void)
{
    fb_backend_vtable_t vtable;
    fb_sgbtrs_fortran_slot_fn thunk = NULL;
    float ab[12] = {
        0.0f, 0.0f, 13.0f,
        0.0f, 22.0f, 23.0f,
        31.0f, 32.0f, 33.0f,
        41.0f, 42.0f, 0.0f
    };
    int ipiv[3] = { 2, 1, 3 };
    float b[6] = {
        1.0f, 2.0f,
        3.0f, 4.0f,
        5.0f, 6.0f
    };
    char trans = 'N';
    int n = 3;
    int kl = 1;
    int ku = 1;
    int nrhs = 2;
    int ldab = 4;
    int ldb = 3;
    int info = 0;

    memset(&vtable, 0, sizeof(vtable));
    memset(&g_sgbtrs_cblas_call, 0, sizeof(g_sgbtrs_cblas_call));

    vtable.ext_ops[FB_OP_SGBTRS][FB_CONV_CBLAS] =
        (fb_generic_fn)(void (*)(void))stub_sgbtrs_cblas;
    fb_install_conv_thunks(&vtable, FB_OP_SGBTRS);

    thunk = (fb_sgbtrs_fortran_slot_fn)vtable.ext_ops[FB_OP_SGBTRS][FB_CONV_FORTRAN];
    if (!thunk) {
        fprintf(stderr, "[FAIL] SGBTRS CBLAS->Fortran thunk was not installed\n");
        return 1;
    }

    thunk(&trans, &n, &kl, &ku, &nrhs, ab, &ldab, ipiv, b, &ldb, &info);
    if (info != 211 || g_sgbtrs_cblas_call.called != 1 ||
        g_sgbtrs_cblas_call.layout != FB_LAYOUT_COL_MAJOR ||
        g_sgbtrs_cblas_call.trans != 'N' || g_sgbtrs_cblas_call.n != 3 ||
        g_sgbtrs_cblas_call.kl != 1 || g_sgbtrs_cblas_call.ku != 1 ||
        g_sgbtrs_cblas_call.nrhs != 2 || g_sgbtrs_cblas_call.ldab != 4 ||
        g_sgbtrs_cblas_call.ldb != 3 || g_sgbtrs_cblas_call.ab != ab ||
        g_sgbtrs_cblas_call.ipiv != ipiv || g_sgbtrs_cblas_call.b != b ||
        b[0] != 711.0f || b[4] != 722.0f) {
        fprintf(stderr, "[FAIL] SGBTRS CBLAS->Fortran thunk delegated incorrectly\n");
        return 1;
    }

    printf("[PASS] SGBTRS CBLAS->Fortran thunk maps the all-pointer ABI into the generic C factored-solve entry\n");
    return 0;
}

static int check_cgbtrs_fortran_to_cblas(void)
{
    fb_backend_vtable_t vtable;
    fb_cgbtrs_fn thunk = NULL;
    fb_complex_float_t ab[16] = {
        make_cfloat(0.0f), make_cfloat(0.0f), make_cfloat(130.0f), make_cfloat(140.0f),
        make_cfloat(0.0f), make_cfloat(220.0f), make_cfloat(230.0f), make_cfloat(240.0f),
        make_cfloat(310.0f), make_cfloat(320.0f), make_cfloat(330.0f), make_cfloat(340.0f),
        make_cfloat(410.0f), make_cfloat(420.0f), make_cfloat(430.0f), make_cfloat(0.0f)
    };
    int ipiv[4] = { 4, 3, 2, 1 };
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
    static const float expected_b_after[8] = {
        800.0f, 804.0f,
        801.0f, 805.0f,
        802.0f, 806.0f,
        803.0f, 807.0f
    };
    static const float expected_ab_after[16] = {
        0.0f, 0.0f, 130.0f, 140.0f,
        0.0f, 220.0f, 230.0f, 240.0f,
        310.0f, 320.0f, 330.0f, 340.0f,
        410.0f, 420.0f, 430.0f, 0.0f
    };
    float ab_real_after[16] = { 0.0f };
    float b_real_after[8] = { 0.0f };
    int index = 0;
    int info = 0;

    memset(&vtable, 0, sizeof(vtable));
    memset(&g_cgbtrs_fortran_call, 0, sizeof(g_cgbtrs_fortran_call));

    vtable.ext_ops[FB_OP_CGBTRS][FB_CONV_FORTRAN] =
        (fb_generic_fn)(void (*)(void))stub_cgbtrs_fortran;
    fb_install_conv_thunks(&vtable, FB_OP_CGBTRS);

    thunk = (fb_cgbtrs_fn)vtable.ext_ops[FB_OP_CGBTRS][FB_CONV_CBLAS];
    if (!thunk) {
        fprintf(stderr, "[FAIL] CGBTRS Fortran->CBLAS thunk was not installed\n");
        return 1;
    }

    info = thunk(FB_LAYOUT_ROW_MAJOR, 'C', 4, 1, 1, 2, ab, 4, ipiv, b, 2);
    for (index = 0; index < 16; ++index) {
        ab_real_after[index] = cfloat_real(ab[index]);
    }
    for (index = 0; index < 8; ++index) {
        b_real_after[index] = cfloat_real(b[index]);
    }
    if (info != 0 || g_cgbtrs_fortran_call.calls != 1 ||
        g_cgbtrs_fortran_call.trans != 'C' || g_cgbtrs_fortran_call.n != 4 ||
        g_cgbtrs_fortran_call.kl != 1 || g_cgbtrs_fortran_call.ku != 1 ||
        g_cgbtrs_fortran_call.nrhs != 2 || g_cgbtrs_fortran_call.ldab != 4 ||
        g_cgbtrs_fortran_call.ldb != 4 ||
        memcmp(g_cgbtrs_fortran_call.ipiv_snapshot, ipiv,
               sizeof(ipiv)) != 0 ||
        memcmp(g_cgbtrs_fortran_call.ab_real_snapshot, expected_ab_snapshot,
               sizeof(expected_ab_snapshot)) != 0 ||
        memcmp(g_cgbtrs_fortran_call.b_real_snapshot, expected_b_snapshot,
               sizeof(expected_b_snapshot)) != 0 ||
         memcmp(ab_real_after, expected_ab_after, sizeof(expected_ab_after)) != 0 ||
        memcmp(b_real_after, expected_b_after, sizeof(expected_b_after)) != 0) {
        fprintf(stderr, "[FAIL] CGBTRS Fortran->CBLAS thunk did not preserve complex expanded factor-storage and RHS transpose semantics\n");
        return 1;
    }

    printf("[PASS] CGBTRS Fortran->CBLAS thunk transposes expanded row-major complex factor storage and round-trips the RHS matrix\n");
    return 0;
}

static int check_cgbtrs_cblas_to_fortran(void)
{
    fb_backend_vtable_t vtable;
    fb_cgbtrs_fortran_slot_fn thunk = NULL;
    fb_complex_float_t ab[12] = {
        make_cfloat(0.0f), make_cfloat(0.0f), make_cfloat(13.0f),
        make_cfloat(0.0f), make_cfloat(22.0f), make_cfloat(23.0f),
        make_cfloat(31.0f), make_cfloat(32.0f), make_cfloat(33.0f),
        make_cfloat(41.0f), make_cfloat(42.0f), make_cfloat(0.0f)
    };
    int ipiv[3] = { 3, 1, 2 };
    fb_complex_float_t b[6] = {
        make_cfloat(1.0f), make_cfloat(2.0f),
        make_cfloat(3.0f), make_cfloat(4.0f),
        make_cfloat(5.0f), make_cfloat(6.0f)
    };
    char trans = 'T';
    int n = 3;
    int kl = 1;
    int ku = 1;
    int nrhs = 2;
    int ldab = 4;
    int ldb = 3;
    int info = 0;

    memset(&vtable, 0, sizeof(vtable));
    memset(&g_cgbtrs_cblas_call, 0, sizeof(g_cgbtrs_cblas_call));

    vtable.ext_ops[FB_OP_CGBTRS][FB_CONV_CBLAS] =
        (fb_generic_fn)(void (*)(void))stub_cgbtrs_cblas;
    fb_install_conv_thunks(&vtable, FB_OP_CGBTRS);

    thunk = (fb_cgbtrs_fortran_slot_fn)vtable.ext_ops[FB_OP_CGBTRS][FB_CONV_FORTRAN];
    if (!thunk) {
        fprintf(stderr, "[FAIL] CGBTRS CBLAS->Fortran thunk was not installed\n");
        return 1;
    }

    thunk(&trans, &n, &kl, &ku, &nrhs, ab, &ldab, ipiv, b, &ldb, &info);
    if (info != 213 || g_cgbtrs_cblas_call.called != 1 ||
        g_cgbtrs_cblas_call.layout != FB_LAYOUT_COL_MAJOR ||
        g_cgbtrs_cblas_call.trans != 'T' || g_cgbtrs_cblas_call.n != 3 ||
        g_cgbtrs_cblas_call.kl != 1 || g_cgbtrs_cblas_call.ku != 1 ||
        g_cgbtrs_cblas_call.nrhs != 2 || g_cgbtrs_cblas_call.ldab != 4 ||
        g_cgbtrs_cblas_call.ldb != 3 || g_cgbtrs_cblas_call.ab != ab ||
        g_cgbtrs_cblas_call.ipiv != ipiv || g_cgbtrs_cblas_call.b != b ||
        cfloat_real(b[0]) != 811.0f || cfloat_real(b[4]) != 822.0f) {
        fprintf(stderr, "[FAIL] CGBTRS CBLAS->Fortran thunk delegated incorrectly\n");
        return 1;
    }

    printf("[PASS] CGBTRS CBLAS->Fortran thunk maps the all-pointer ABI into the generic complex factored-solve entry\n");
    return 0;
}

int main(void)
{
    if (check_sgbtrs_fortran_to_cblas() != 0) {
        return 1;
    }
    if (check_sgbtrs_cblas_to_fortran() != 0) {
        return 1;
    }
    if (check_cgbtrs_fortran_to_cblas() != 0) {
        return 1;
    }
    if (check_cgbtrs_cblas_to_fortran() != 0) {
        return 1;
    }

    printf("Result: PASS\n");
    return 0;
}