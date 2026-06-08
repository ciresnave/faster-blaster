#include <stdio.h>
#include <string.h>

#include "../src/core/conv_thunks.h"

typedef int (*fb_spbtrf_fn)(fb_layout_t layout, fb_uplo_t uplo, int n, int kd,
                            float *ab, int ldab);
typedef int (*fb_cpbtrf_fn)(fb_layout_t layout, fb_uplo_t uplo, int n, int kd,
                            fb_complex_float_t *ab, int ldab);
typedef int (*fb_spbtrs_fn)(fb_layout_t layout, fb_uplo_t uplo, int n, int kd,
                            int nrhs, const float *ab, int ldab, float *b,
                            int ldb);
typedef int (*fb_cpbtrs_fn)(fb_layout_t layout, fb_uplo_t uplo, int n, int kd,
                            int nrhs, const fb_complex_float_t *ab, int ldab,
                            fb_complex_float_t *b, int ldb);

typedef void (*fb_spbtrf_fortran_slot_fn)(char *uplo, int *n, int *kd,
                                          float *ab, int *ldab, int *info);
typedef void (*fb_cpbtrf_fortran_slot_fn)(char *uplo, int *n, int *kd,
                                          fb_complex_float_t *ab, int *ldab,
                                          int *info);
typedef void (*fb_spbtrs_fortran_slot_fn)(char *uplo, int *n, int *kd,
                                          int *nrhs, float *ab, int *ldab,
                                          float *b, int *ldb, int *info);
typedef void (*fb_cpbtrs_fortran_slot_fn)(char *uplo, int *n, int *kd,
                                          int *nrhs, fb_complex_float_t *ab,
                                          int *ldab, fb_complex_float_t *b,
                                          int *ldb, int *info);

static struct {
    int calls;
    char uplo;
    int n;
    int kd;
    int ldab;
    float ab_snapshot[8];
} g_spbtrf_fortran_call;

static struct {
    int called;
    fb_layout_t layout;
    fb_uplo_t uplo;
    int n;
    int kd;
    int ldab;
    float *ab;
} g_spbtrf_cblas_call;

static struct {
    int calls;
    char uplo;
    int n;
    int kd;
    int ldab;
    float ab_real_snapshot[8];
} g_cpbtrf_fortran_call;

static struct {
    int called;
    fb_layout_t layout;
    fb_uplo_t uplo;
    int n;
    int kd;
    int ldab;
    fb_complex_float_t *ab;
} g_cpbtrf_cblas_call;

static struct {
    int calls;
    char uplo;
    int n;
    int kd;
    int nrhs;
    int ldab;
    int ldb;
    float ab_snapshot[8];
    float b_snapshot[8];
} g_spbtrs_fortran_call;

static struct {
    int called;
    fb_layout_t layout;
    fb_uplo_t uplo;
    int n;
    int kd;
    int nrhs;
    int ldab;
    int ldb;
    const float *ab;
    float *b;
} g_spbtrs_cblas_call;

static struct {
    int calls;
    char uplo;
    int n;
    int kd;
    int nrhs;
    int ldab;
    int ldb;
    float ab_real_snapshot[8];
    float b_real_snapshot[8];
} g_cpbtrs_fortran_call;

static struct {
    int called;
    fb_layout_t layout;
    fb_uplo_t uplo;
    int n;
    int kd;
    int nrhs;
    int ldab;
    int ldb;
    const fb_complex_float_t *ab;
    fb_complex_float_t *b;
} g_cpbtrs_cblas_call;

static fb_complex_float_t make_cfloat(float real_value)
{
    fb_complex_float_t value = (fb_complex_float_t)0;
    memcpy(&value, &real_value, sizeof(real_value));
    return value;
}

static float cfloat_real(fb_complex_float_t value)
{
    float real_value = 0.0f;
    memcpy(&real_value, &value, sizeof(real_value));
    return real_value;
}

static void stub_spbtrf_fortran(char *uplo, int *n, int *kd, float *ab,
                                int *ldab, int *info)
{
    static const float factor_ab_col[8] = { 0.0f, 320.0f, 311.0f, 321.0f,
                                            312.0f, 322.0f, 313.0f, 323.0f };
    int index = 0;

    g_spbtrf_fortran_call.calls += 1;
    g_spbtrf_fortran_call.uplo = *uplo;
    g_spbtrf_fortran_call.n = *n;
    g_spbtrf_fortran_call.kd = *kd;
    g_spbtrf_fortran_call.ldab = *ldab;
    for (index = 0; index < (*ldab * *n); ++index) {
        g_spbtrf_fortran_call.ab_snapshot[index] = ab[index];
    }
    memcpy(ab, factor_ab_col, sizeof(factor_ab_col));
    *info = 0;
}

static int stub_spbtrf_cblas(fb_layout_t layout, fb_uplo_t uplo, int n, int kd,
                             float *ab, int ldab)
{
    g_spbtrf_cblas_call.called += 1;
    g_spbtrf_cblas_call.layout = layout;
    g_spbtrf_cblas_call.uplo = uplo;
    g_spbtrf_cblas_call.n = n;
    g_spbtrf_cblas_call.kd = kd;
    g_spbtrf_cblas_call.ldab = ldab;
    g_spbtrf_cblas_call.ab = ab;
    ab[0] = 98.0f;
    return 181;
}

static void stub_cpbtrf_fortran(char *uplo, int *n, int *kd,
                                fb_complex_float_t *ab, int *ldab, int *info)
{
    static const float factor_ab_col_real[8] = { 330.0f, 430.0f, 331.0f, 431.0f,
                                                 332.0f, 432.0f, 333.0f, 0.0f };
    int index = 0;

    g_cpbtrf_fortran_call.calls += 1;
    g_cpbtrf_fortran_call.uplo = *uplo;
    g_cpbtrf_fortran_call.n = *n;
    g_cpbtrf_fortran_call.kd = *kd;
    g_cpbtrf_fortran_call.ldab = *ldab;
    for (index = 0; index < (*ldab * *n); ++index) {
        g_cpbtrf_fortran_call.ab_real_snapshot[index] = cfloat_real(ab[index]);
        ab[index] = make_cfloat(factor_ab_col_real[index]);
    }
    *info = 0;
}

static int stub_cpbtrf_cblas(fb_layout_t layout, fb_uplo_t uplo, int n, int kd,
                             fb_complex_float_t *ab, int ldab)
{
    g_cpbtrf_cblas_call.called += 1;
    g_cpbtrf_cblas_call.layout = layout;
    g_cpbtrf_cblas_call.uplo = uplo;
    g_cpbtrf_cblas_call.n = n;
    g_cpbtrf_cblas_call.kd = kd;
    g_cpbtrf_cblas_call.ldab = ldab;
    g_cpbtrf_cblas_call.ab = ab;
    ab[0] = make_cfloat(198.0f);
    return 183;
}

static void stub_spbtrs_fortran(char *uplo, int *n, int *kd, int *nrhs,
                                float *ab, int *ldab, float *b, int *ldb,
                                int *info)
{
    static const float solution_b_col[8] = { 410.0f, 430.0f, 450.0f, 470.0f,
                                             420.0f, 440.0f, 460.0f, 480.0f };
    int index = 0;

    g_spbtrs_fortran_call.calls += 1;
    g_spbtrs_fortran_call.uplo = *uplo;
    g_spbtrs_fortran_call.n = *n;
    g_spbtrs_fortran_call.kd = *kd;
    g_spbtrs_fortran_call.nrhs = *nrhs;
    g_spbtrs_fortran_call.ldab = *ldab;
    g_spbtrs_fortran_call.ldb = *ldb;
    for (index = 0; index < (*ldab * *n); ++index) {
        g_spbtrs_fortran_call.ab_snapshot[index] = ab[index];
    }
    for (index = 0; index < (*ldb * *nrhs); ++index) {
        g_spbtrs_fortran_call.b_snapshot[index] = b[index];
    }
    memcpy(b, solution_b_col, sizeof(solution_b_col));
    *info = 0;
}

static int stub_spbtrs_cblas(fb_layout_t layout, fb_uplo_t uplo, int n, int kd,
                             int nrhs, const float *ab, int ldab, float *b,
                             int ldb)
{
    g_spbtrs_cblas_call.called += 1;
    g_spbtrs_cblas_call.layout = layout;
    g_spbtrs_cblas_call.uplo = uplo;
    g_spbtrs_cblas_call.n = n;
    g_spbtrs_cblas_call.kd = kd;
    g_spbtrs_cblas_call.nrhs = nrhs;
    g_spbtrs_cblas_call.ldab = ldab;
    g_spbtrs_cblas_call.ldb = ldb;
    g_spbtrs_cblas_call.ab = ab;
    g_spbtrs_cblas_call.b = b;
    b[0] = 88.0f;
    return 185;
}

static void stub_cpbtrs_fortran(char *uplo, int *n, int *kd, int *nrhs,
                                fb_complex_float_t *ab, int *ldab,
                                fb_complex_float_t *b, int *ldb, int *info)
{
    static const float solution_b_col_real[8] = { 510.0f, 530.0f, 550.0f, 570.0f,
                                                  520.0f, 540.0f, 560.0f, 580.0f };
    int index = 0;

    g_cpbtrs_fortran_call.calls += 1;
    g_cpbtrs_fortran_call.uplo = *uplo;
    g_cpbtrs_fortran_call.n = *n;
    g_cpbtrs_fortran_call.kd = *kd;
    g_cpbtrs_fortran_call.nrhs = *nrhs;
    g_cpbtrs_fortran_call.ldab = *ldab;
    g_cpbtrs_fortran_call.ldb = *ldb;
    for (index = 0; index < (*ldab * *n); ++index) {
        g_cpbtrs_fortran_call.ab_real_snapshot[index] = cfloat_real(ab[index]);
    }
    for (index = 0; index < (*ldb * *nrhs); ++index) {
        g_cpbtrs_fortran_call.b_real_snapshot[index] = cfloat_real(b[index]);
        b[index] = make_cfloat(solution_b_col_real[index]);
    }
    *info = 0;
}

static int stub_cpbtrs_cblas(fb_layout_t layout, fb_uplo_t uplo, int n, int kd,
                             int nrhs, const fb_complex_float_t *ab, int ldab,
                             fb_complex_float_t *b, int ldb)
{
    g_cpbtrs_cblas_call.called += 1;
    g_cpbtrs_cblas_call.layout = layout;
    g_cpbtrs_cblas_call.uplo = uplo;
    g_cpbtrs_cblas_call.n = n;
    g_cpbtrs_cblas_call.kd = kd;
    g_cpbtrs_cblas_call.nrhs = nrhs;
    g_cpbtrs_cblas_call.ldab = ldab;
    g_cpbtrs_cblas_call.ldb = ldb;
    g_cpbtrs_cblas_call.ab = ab;
    g_cpbtrs_cblas_call.b = b;
    b[0] = make_cfloat(188.0f);
    return 187;
}

static int check_spbtrf_fortran_to_cblas(void)
{
    fb_backend_vtable_t vtable;
    fb_spbtrf_fn thunk = NULL;
    float ab[8] = { 10.0f, 11.0f, 12.0f, 13.0f, 20.0f, 21.0f, 22.0f, 23.0f };
    float expected_ab_in[8] = { 0.0f, 20.0f, 11.0f, 21.0f, 12.0f, 22.0f, 13.0f, 23.0f };
    float expected_ab_out[8] = { 0.0f, 311.0f, 312.0f, 313.0f, 320.0f, 321.0f, 322.0f, 323.0f };
    int info = 0;

    memset(&vtable, 0, sizeof(vtable));
    memset(&g_spbtrf_fortran_call, 0, sizeof(g_spbtrf_fortran_call));

    vtable.ext_ops[FB_OP_SPBTRF][FB_CONV_FORTRAN] =
        (fb_generic_fn)(void (*)(void))stub_spbtrf_fortran;
    fb_install_conv_thunks(&vtable, FB_OP_SPBTRF);

    thunk = (fb_spbtrf_fn)vtable.ext_ops[FB_OP_SPBTRF][FB_CONV_CBLAS];
    if (!thunk) {
        fprintf(stderr, "[FAIL] SPBTRF Fortran->CBLAS thunk was not installed\n");
        return 1;
    }

    info = thunk(FB_LAYOUT_ROW_MAJOR, FB_UPPER, 4, 1, ab, 4);
    if (info != 0 || g_spbtrf_fortran_call.calls != 1 ||
        g_spbtrf_fortran_call.uplo != 'U' || g_spbtrf_fortran_call.n != 4 ||
        g_spbtrf_fortran_call.kd != 1 || g_spbtrf_fortran_call.ldab != 2 ||
        memcmp(g_spbtrf_fortran_call.ab_snapshot, expected_ab_in,
               sizeof(expected_ab_in)) != 0 ||
        memcmp(ab, expected_ab_out, sizeof(expected_ab_out)) != 0) {
        fprintf(stderr, "[FAIL] SPBTRF Fortran->CBLAS thunk did not preserve band factor row-major translation and copy-back\n");
        return 1;
    }

    printf("[PASS] SPBTRF Fortran->CBLAS thunk transposes symmetric band storage and copies the factorized band matrix back to row-major\n");
    return 0;
}

static int check_spbtrf_cblas_to_fortran(void)
{
    fb_backend_vtable_t vtable;
    fb_spbtrf_fortran_slot_fn thunk = NULL;
    float ab[6] = { 1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f };
    char uplo = 'L';
    int n = 3;
    int kd = 1;
    int ldab = 2;
    int info = 0;

    memset(&vtable, 0, sizeof(vtable));
    memset(&g_spbtrf_cblas_call, 0, sizeof(g_spbtrf_cblas_call));

    vtable.ext_ops[FB_OP_SPBTRF][FB_CONV_CBLAS] =
        (fb_generic_fn)(void (*)(void))stub_spbtrf_cblas;
    fb_install_conv_thunks(&vtable, FB_OP_SPBTRF);

    thunk = (fb_spbtrf_fortran_slot_fn)vtable.ext_ops[FB_OP_SPBTRF][FB_CONV_FORTRAN];
    if (!thunk) {
        fprintf(stderr, "[FAIL] SPBTRF CBLAS->Fortran thunk was not installed\n");
        return 1;
    }

    thunk(&uplo, &n, &kd, ab, &ldab, &info);
    if (info != 181 || g_spbtrf_cblas_call.called != 1 ||
        g_spbtrf_cblas_call.layout != FB_LAYOUT_COL_MAJOR ||
        g_spbtrf_cblas_call.uplo != FB_LOWER ||
        g_spbtrf_cblas_call.n != 3 || g_spbtrf_cblas_call.kd != 1 ||
        g_spbtrf_cblas_call.ldab != 2 || g_spbtrf_cblas_call.ab != ab ||
        ab[0] != 98.0f) {
        fprintf(stderr, "[FAIL] SPBTRF CBLAS->Fortran thunk delegated incorrectly\n");
        return 1;
    }

    printf("[PASS] SPBTRF CBLAS->Fortran thunk maps the all-pointer ABI into the generic band-factor entry\n");
    return 0;
}

static int check_cpbtrf_fortran_to_cblas(void)
{
    fb_backend_vtable_t vtable;
    fb_cpbtrf_fn thunk = NULL;
    fb_complex_float_t ab[8] = {
        make_cfloat(30.0f), make_cfloat(31.0f), make_cfloat(32.0f), make_cfloat(33.0f),
        make_cfloat(40.0f), make_cfloat(41.0f), make_cfloat(42.0f), make_cfloat(43.0f)
    };
    float expected_ab_in[8] = { 30.0f, 40.0f, 31.0f, 41.0f, 32.0f, 42.0f, 33.0f, 0.0f };
    float expected_ab_out[8] = { 330.0f, 331.0f, 332.0f, 333.0f, 430.0f, 431.0f, 432.0f, 0.0f };
    float actual_ab_out[8] = { 0.0f };
    int index = 0;
    int info = 0;

    memset(&vtable, 0, sizeof(vtable));
    memset(&g_cpbtrf_fortran_call, 0, sizeof(g_cpbtrf_fortran_call));

    vtable.ext_ops[FB_OP_CPBTRF][FB_CONV_FORTRAN] =
        (fb_generic_fn)(void (*)(void))stub_cpbtrf_fortran;
    fb_install_conv_thunks(&vtable, FB_OP_CPBTRF);

    thunk = (fb_cpbtrf_fn)vtable.ext_ops[FB_OP_CPBTRF][FB_CONV_CBLAS];
    if (!thunk) {
        fprintf(stderr, "[FAIL] CPBTRF Fortran->CBLAS thunk was not installed\n");
        return 1;
    }

    info = thunk(FB_LAYOUT_ROW_MAJOR, FB_LOWER, 4, 1, ab, 4);
    for (index = 0; index < 8; ++index) {
        actual_ab_out[index] = cfloat_real(ab[index]);
    }
    if (info != 0 || g_cpbtrf_fortran_call.calls != 1 ||
        g_cpbtrf_fortran_call.uplo != 'L' || g_cpbtrf_fortran_call.n != 4 ||
        g_cpbtrf_fortran_call.kd != 1 || g_cpbtrf_fortran_call.ldab != 2 ||
        memcmp(g_cpbtrf_fortran_call.ab_real_snapshot, expected_ab_in,
               sizeof(expected_ab_in)) != 0 ||
        memcmp(actual_ab_out, expected_ab_out, sizeof(expected_ab_out)) != 0) {
        fprintf(stderr, "[FAIL] CPBTRF Fortran->CBLAS thunk did not preserve complex band factor row-major translation and copy-back\n");
        return 1;
    }

    printf("[PASS] CPBTRF Fortran->CBLAS thunk transposes Hermitian band storage and copies the factorized complex band matrix back to row-major\n");
    return 0;
}

static int check_cpbtrf_cblas_to_fortran(void)
{
    fb_backend_vtable_t vtable;
    fb_cpbtrf_fortran_slot_fn thunk = NULL;
    fb_complex_float_t ab[6];
    char uplo = 'U';
    int n = 3;
    int kd = 1;
    int ldab = 2;
    int info = 0;

    memset(&vtable, 0, sizeof(vtable));
    memset(&g_cpbtrf_cblas_call, 0, sizeof(g_cpbtrf_cblas_call));
    memset(ab, 0, sizeof(ab));

    vtable.ext_ops[FB_OP_CPBTRF][FB_CONV_CBLAS] =
        (fb_generic_fn)(void (*)(void))stub_cpbtrf_cblas;
    fb_install_conv_thunks(&vtable, FB_OP_CPBTRF);

    thunk = (fb_cpbtrf_fortran_slot_fn)vtable.ext_ops[FB_OP_CPBTRF][FB_CONV_FORTRAN];
    if (!thunk) {
        fprintf(stderr, "[FAIL] CPBTRF CBLAS->Fortran thunk was not installed\n");
        return 1;
    }

    thunk(&uplo, &n, &kd, ab, &ldab, &info);
    if (info != 183 || g_cpbtrf_cblas_call.called != 1 ||
        g_cpbtrf_cblas_call.layout != FB_LAYOUT_COL_MAJOR ||
        g_cpbtrf_cblas_call.uplo != FB_UPPER ||
        g_cpbtrf_cblas_call.n != 3 || g_cpbtrf_cblas_call.kd != 1 ||
        g_cpbtrf_cblas_call.ldab != 2 || g_cpbtrf_cblas_call.ab != ab ||
        cfloat_real(ab[0]) != 198.0f) {
        fprintf(stderr, "[FAIL] CPBTRF CBLAS->Fortran thunk delegated incorrectly\n");
        return 1;
    }

    printf("[PASS] CPBTRF CBLAS->Fortran thunk maps the all-pointer ABI into the generic complex band-factor entry\n");
    return 0;
}

static int check_spbtrs_fortran_to_cblas(void)
{
    fb_backend_vtable_t vtable;
    fb_spbtrs_fn thunk = NULL;
    float ab[8] = { 10.0f, 11.0f, 12.0f, 13.0f, 20.0f, 21.0f, 22.0f, 23.0f };
    float b[8] = { 1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f, 7.0f, 8.0f };
    float original_ab[8] = { 10.0f, 11.0f, 12.0f, 13.0f, 20.0f, 21.0f, 22.0f, 23.0f };
    float expected_ab_in[8] = { 0.0f, 20.0f, 11.0f, 21.0f, 12.0f, 22.0f, 13.0f, 23.0f };
    float expected_b_in[8] = { 1.0f, 3.0f, 5.0f, 7.0f, 2.0f, 4.0f, 6.0f, 8.0f };
    float expected_b_out[8] = { 410.0f, 420.0f, 430.0f, 440.0f, 450.0f, 460.0f, 470.0f, 480.0f };
    int info = 0;

    memset(&vtable, 0, sizeof(vtable));
    memset(&g_spbtrs_fortran_call, 0, sizeof(g_spbtrs_fortran_call));

    vtable.ext_ops[FB_OP_SPBTRS][FB_CONV_FORTRAN] =
        (fb_generic_fn)(void (*)(void))stub_spbtrs_fortran;
    fb_install_conv_thunks(&vtable, FB_OP_SPBTRS);

    thunk = (fb_spbtrs_fn)vtable.ext_ops[FB_OP_SPBTRS][FB_CONV_CBLAS];
    if (!thunk) {
        fprintf(stderr, "[FAIL] SPBTRS Fortran->CBLAS thunk was not installed\n");
        return 1;
    }

    info = thunk(FB_LAYOUT_ROW_MAJOR, FB_UPPER, 4, 1, 2, ab, 4, b, 2);
    if (info != 0 || g_spbtrs_fortran_call.calls != 1 ||
        g_spbtrs_fortran_call.uplo != 'U' || g_spbtrs_fortran_call.n != 4 ||
        g_spbtrs_fortran_call.kd != 1 || g_spbtrs_fortran_call.nrhs != 2 ||
        g_spbtrs_fortran_call.ldab != 2 || g_spbtrs_fortran_call.ldb != 4 ||
        memcmp(g_spbtrs_fortran_call.ab_snapshot, expected_ab_in,
               sizeof(expected_ab_in)) != 0 ||
        memcmp(g_spbtrs_fortran_call.b_snapshot, expected_b_in,
               sizeof(expected_b_in)) != 0 ||
        memcmp(ab, original_ab, sizeof(original_ab)) != 0 ||
        memcmp(b, expected_b_out, sizeof(expected_b_out)) != 0) {
        fprintf(stderr, "[FAIL] SPBTRS Fortran->CBLAS thunk did not preserve band factored-solve row-major translation and selective copy-back\n");
        return 1;
    }

    printf("[PASS] SPBTRS Fortran->CBLAS thunk transposes band storage and RHS data, then copies back only the solved RHS\n");
    return 0;
}

static int check_spbtrs_cblas_to_fortran(void)
{
    fb_backend_vtable_t vtable;
    fb_spbtrs_fortran_slot_fn thunk = NULL;
    float ab[6] = { 1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f };
    float b[6] = { 10.0f, 20.0f, 30.0f, 40.0f, 50.0f, 60.0f };
    char uplo = 'L';
    int n = 3;
    int kd = 1;
    int nrhs = 2;
    int ldab = 2;
    int ldb = 3;
    int info = 0;

    memset(&vtable, 0, sizeof(vtable));
    memset(&g_spbtrs_cblas_call, 0, sizeof(g_spbtrs_cblas_call));

    vtable.ext_ops[FB_OP_SPBTRS][FB_CONV_CBLAS] =
        (fb_generic_fn)(void (*)(void))stub_spbtrs_cblas;
    fb_install_conv_thunks(&vtable, FB_OP_SPBTRS);

    thunk = (fb_spbtrs_fortran_slot_fn)vtable.ext_ops[FB_OP_SPBTRS][FB_CONV_FORTRAN];
    if (!thunk) {
        fprintf(stderr, "[FAIL] SPBTRS CBLAS->Fortran thunk was not installed\n");
        return 1;
    }

    thunk(&uplo, &n, &kd, &nrhs, ab, &ldab, b, &ldb, &info);
    if (info != 185 || g_spbtrs_cblas_call.called != 1 ||
        g_spbtrs_cblas_call.layout != FB_LAYOUT_COL_MAJOR ||
        g_spbtrs_cblas_call.uplo != FB_LOWER ||
        g_spbtrs_cblas_call.n != 3 || g_spbtrs_cblas_call.kd != 1 ||
        g_spbtrs_cblas_call.nrhs != 2 || g_spbtrs_cblas_call.ldab != 2 ||
        g_spbtrs_cblas_call.ldb != 3 || g_spbtrs_cblas_call.ab != ab ||
        g_spbtrs_cblas_call.b != b || b[0] != 88.0f) {
        fprintf(stderr, "[FAIL] SPBTRS CBLAS->Fortran thunk delegated incorrectly\n");
        return 1;
    }

    printf("[PASS] SPBTRS CBLAS->Fortran thunk maps the all-pointer ABI into the generic band factored-solve entry\n");
    return 0;
}

static int check_cpbtrs_fortran_to_cblas(void)
{
    fb_backend_vtable_t vtable;
    fb_cpbtrs_fn thunk = NULL;
    fb_complex_float_t ab[8] = {
        make_cfloat(30.0f), make_cfloat(31.0f), make_cfloat(32.0f), make_cfloat(33.0f),
        make_cfloat(40.0f), make_cfloat(41.0f), make_cfloat(42.0f), make_cfloat(43.0f)
    };
    fb_complex_float_t b[8] = {
        make_cfloat(1.0f), make_cfloat(2.0f), make_cfloat(3.0f), make_cfloat(4.0f),
        make_cfloat(5.0f), make_cfloat(6.0f), make_cfloat(7.0f), make_cfloat(8.0f)
    };
    float original_ab[8] = { 30.0f, 31.0f, 32.0f, 33.0f, 40.0f, 41.0f, 42.0f, 43.0f };
    float expected_ab_in[8] = { 30.0f, 40.0f, 31.0f, 41.0f, 32.0f, 42.0f, 33.0f, 0.0f };
    float expected_b_in[8] = { 1.0f, 3.0f, 5.0f, 7.0f, 2.0f, 4.0f, 6.0f, 8.0f };
    float expected_b_out[8] = { 510.0f, 520.0f, 530.0f, 540.0f, 550.0f, 560.0f, 570.0f, 580.0f };
    float actual_ab_out[8] = { 0.0f };
    float actual_b_out[8] = { 0.0f };
    int index = 0;
    int info = 0;

    memset(&vtable, 0, sizeof(vtable));
    memset(&g_cpbtrs_fortran_call, 0, sizeof(g_cpbtrs_fortran_call));

    vtable.ext_ops[FB_OP_CPBTRS][FB_CONV_FORTRAN] =
        (fb_generic_fn)(void (*)(void))stub_cpbtrs_fortran;
    fb_install_conv_thunks(&vtable, FB_OP_CPBTRS);

    thunk = (fb_cpbtrs_fn)vtable.ext_ops[FB_OP_CPBTRS][FB_CONV_CBLAS];
    if (!thunk) {
        fprintf(stderr, "[FAIL] CPBTRS Fortran->CBLAS thunk was not installed\n");
        return 1;
    }

    info = thunk(FB_LAYOUT_ROW_MAJOR, FB_LOWER, 4, 1, 2, ab, 4, b, 2);
    for (index = 0; index < 8; ++index) {
        actual_ab_out[index] = cfloat_real(ab[index]);
        actual_b_out[index] = cfloat_real(b[index]);
    }
    if (info != 0 || g_cpbtrs_fortran_call.calls != 1 ||
        g_cpbtrs_fortran_call.uplo != 'L' || g_cpbtrs_fortran_call.n != 4 ||
        g_cpbtrs_fortran_call.kd != 1 || g_cpbtrs_fortran_call.nrhs != 2 ||
        g_cpbtrs_fortran_call.ldab != 2 || g_cpbtrs_fortran_call.ldb != 4 ||
        memcmp(g_cpbtrs_fortran_call.ab_real_snapshot, expected_ab_in,
               sizeof(expected_ab_in)) != 0 ||
        memcmp(g_cpbtrs_fortran_call.b_real_snapshot, expected_b_in,
               sizeof(expected_b_in)) != 0 ||
        memcmp(actual_ab_out, original_ab, sizeof(original_ab)) != 0 ||
        memcmp(actual_b_out, expected_b_out, sizeof(expected_b_out)) != 0) {
        fprintf(stderr, "[FAIL] CPBTRS Fortran->CBLAS thunk did not preserve complex band factored-solve row-major translation and selective copy-back\n");
        return 1;
    }

    printf("[PASS] CPBTRS Fortran->CBLAS thunk transposes complex band storage and RHS data, then copies back only the solved complex RHS\n");
    return 0;
}

static int check_cpbtrs_cblas_to_fortran(void)
{
    fb_backend_vtable_t vtable;
    fb_cpbtrs_fortran_slot_fn thunk = NULL;
    fb_complex_float_t ab[6];
    fb_complex_float_t b[6];
    char uplo = 'U';
    int n = 3;
    int kd = 1;
    int nrhs = 2;
    int ldab = 2;
    int ldb = 3;
    int info = 0;

    memset(&vtable, 0, sizeof(vtable));
    memset(&g_cpbtrs_cblas_call, 0, sizeof(g_cpbtrs_cblas_call));
    memset(ab, 0, sizeof(ab));
    memset(b, 0, sizeof(b));

    vtable.ext_ops[FB_OP_CPBTRS][FB_CONV_CBLAS] =
        (fb_generic_fn)(void (*)(void))stub_cpbtrs_cblas;
    fb_install_conv_thunks(&vtable, FB_OP_CPBTRS);

    thunk = (fb_cpbtrs_fortran_slot_fn)vtable.ext_ops[FB_OP_CPBTRS][FB_CONV_FORTRAN];
    if (!thunk) {
        fprintf(stderr, "[FAIL] CPBTRS CBLAS->Fortran thunk was not installed\n");
        return 1;
    }

    thunk(&uplo, &n, &kd, &nrhs, ab, &ldab, b, &ldb, &info);
    if (info != 187 || g_cpbtrs_cblas_call.called != 1 ||
        g_cpbtrs_cblas_call.layout != FB_LAYOUT_COL_MAJOR ||
        g_cpbtrs_cblas_call.uplo != FB_UPPER ||
        g_cpbtrs_cblas_call.n != 3 || g_cpbtrs_cblas_call.kd != 1 ||
        g_cpbtrs_cblas_call.nrhs != 2 || g_cpbtrs_cblas_call.ldab != 2 ||
        g_cpbtrs_cblas_call.ldb != 3 || g_cpbtrs_cblas_call.ab != ab ||
        g_cpbtrs_cblas_call.b != b || cfloat_real(b[0]) != 188.0f) {
        fprintf(stderr, "[FAIL] CPBTRS CBLAS->Fortran thunk delegated incorrectly\n");
        return 1;
    }

    printf("[PASS] CPBTRS CBLAS->Fortran thunk maps the all-pointer ABI into the generic complex band factored-solve entry\n");
    return 0;
}

int main(void)
{
    if (check_spbtrf_fortran_to_cblas() != 0) {
        return 1;
    }
    if (check_spbtrf_cblas_to_fortran() != 0) {
        return 1;
    }
    if (check_cpbtrf_fortran_to_cblas() != 0) {
        return 1;
    }
    if (check_cpbtrf_cblas_to_fortran() != 0) {
        return 1;
    }
    if (check_spbtrs_fortran_to_cblas() != 0) {
        return 1;
    }
    if (check_spbtrs_cblas_to_fortran() != 0) {
        return 1;
    }
    if (check_cpbtrs_fortran_to_cblas() != 0) {
        return 1;
    }
    if (check_cpbtrs_cblas_to_fortran() != 0) {
        return 1;
    }

    printf("Result: PASS\n");
    return 0;
}