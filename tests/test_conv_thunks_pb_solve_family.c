#include <stdio.h>
#include <string.h>

#include "../src/core/conv_thunks.h"

typedef int (*fb_spbsv_fn)(fb_layout_t layout, fb_uplo_t uplo, int n, int kd,
                           int nrhs, float *ab, int ldab, float *b, int ldb);
typedef int (*fb_cpbsv_fn)(fb_layout_t layout, fb_uplo_t uplo, int n, int kd,
                           int nrhs, fb_complex_float_t *ab, int ldab,
                           fb_complex_float_t *b, int ldb);

typedef void (*fb_spbsv_fortran_slot_fn)(char *uplo, int *n, int *kd, int *nrhs,
                                         float *ab, int *ldab, float *b,
                                         int *ldb, int *info);
typedef void (*fb_cpbsv_fortran_slot_fn)(char *uplo, int *n, int *kd, int *nrhs,
                                         fb_complex_float_t *ab, int *ldab,
                                         fb_complex_float_t *b, int *ldb,
                                         int *info);

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
} g_spbsv_fortran_call;

static struct {
    int called;
    fb_layout_t layout;
    fb_uplo_t uplo;
    int n;
    int kd;
    int nrhs;
    int ldab;
    int ldb;
    float *ab;
    float *b;
} g_spbsv_cblas_call;

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
} g_cpbsv_fortran_call;

static struct {
    int called;
    fb_layout_t layout;
    fb_uplo_t uplo;
    int n;
    int kd;
    int nrhs;
    int ldab;
    int ldb;
    fb_complex_float_t *ab;
    fb_complex_float_t *b;
} g_cpbsv_cblas_call;

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

static void stub_spbsv_fortran(char *uplo, int *n, int *kd, int *nrhs,
                               float *ab, int *ldab, float *b, int *ldb,
                               int *info)
{
    static const float factor_ab_col[8] = { 0.0f, 120.0f, 111.0f, 121.0f,
                                            112.0f, 122.0f, 113.0f, 123.0f };
    static const float solution_b_col[8] = { 10.0f, 30.0f, 50.0f, 70.0f,
                                             20.0f, 40.0f, 60.0f, 80.0f };
    int index = 0;

    g_spbsv_fortran_call.calls += 1;
    g_spbsv_fortran_call.uplo = *uplo;
    g_spbsv_fortran_call.n = *n;
    g_spbsv_fortran_call.kd = *kd;
    g_spbsv_fortran_call.nrhs = *nrhs;
    g_spbsv_fortran_call.ldab = *ldab;
    g_spbsv_fortran_call.ldb = *ldb;
    for (index = 0; index < (*ldab * *n); ++index) {
        g_spbsv_fortran_call.ab_snapshot[index] = ab[index];
    }
    for (index = 0; index < (*ldb * *nrhs); ++index) {
        g_spbsv_fortran_call.b_snapshot[index] = b[index];
    }
    memcpy(ab, factor_ab_col, sizeof(factor_ab_col));
    memcpy(b, solution_b_col, sizeof(solution_b_col));
    *info = 0;
}

static int stub_spbsv_cblas(fb_layout_t layout, fb_uplo_t uplo, int n, int kd,
                            int nrhs, float *ab, int ldab, float *b, int ldb)
{
    g_spbsv_cblas_call.called += 1;
    g_spbsv_cblas_call.layout = layout;
    g_spbsv_cblas_call.uplo = uplo;
    g_spbsv_cblas_call.n = n;
    g_spbsv_cblas_call.kd = kd;
    g_spbsv_cblas_call.nrhs = nrhs;
    g_spbsv_cblas_call.ldab = ldab;
    g_spbsv_cblas_call.ldb = ldb;
    g_spbsv_cblas_call.ab = ab;
    g_spbsv_cblas_call.b = b;
    b[0] = 99.0f;
    return 161;
}

static void stub_cpbsv_fortran(char *uplo, int *n, int *kd, int *nrhs,
                               fb_complex_float_t *ab, int *ldab,
                               fb_complex_float_t *b, int *ldb, int *info)
{
    static const float factor_ab_col_real[8] = { 130.0f, 230.0f, 131.0f, 231.0f,
                                                 132.0f, 232.0f, 133.0f, 0.0f };
    static const float solution_b_col_real[8] = { 210.0f, 230.0f, 250.0f, 270.0f,
                                                  220.0f, 240.0f, 260.0f, 280.0f };
    int index = 0;

    g_cpbsv_fortran_call.calls += 1;
    g_cpbsv_fortran_call.uplo = *uplo;
    g_cpbsv_fortran_call.n = *n;
    g_cpbsv_fortran_call.kd = *kd;
    g_cpbsv_fortran_call.nrhs = *nrhs;
    g_cpbsv_fortran_call.ldab = *ldab;
    g_cpbsv_fortran_call.ldb = *ldb;
    for (index = 0; index < (*ldab * *n); ++index) {
        g_cpbsv_fortran_call.ab_real_snapshot[index] = cfloat_real(ab[index]);
    }
    for (index = 0; index < (*ldb * *nrhs); ++index) {
        g_cpbsv_fortran_call.b_real_snapshot[index] = cfloat_real(b[index]);
    }
    for (index = 0; index < 8; ++index) {
        ab[index] = make_cfloat(factor_ab_col_real[index]);
        b[index] = make_cfloat(solution_b_col_real[index]);
    }
    *info = 0;
}

static int stub_cpbsv_cblas(fb_layout_t layout, fb_uplo_t uplo, int n, int kd,
                            int nrhs, fb_complex_float_t *ab, int ldab,
                            fb_complex_float_t *b, int ldb)
{
    g_cpbsv_cblas_call.called += 1;
    g_cpbsv_cblas_call.layout = layout;
    g_cpbsv_cblas_call.uplo = uplo;
    g_cpbsv_cblas_call.n = n;
    g_cpbsv_cblas_call.kd = kd;
    g_cpbsv_cblas_call.nrhs = nrhs;
    g_cpbsv_cblas_call.ldab = ldab;
    g_cpbsv_cblas_call.ldb = ldb;
    g_cpbsv_cblas_call.ab = ab;
    g_cpbsv_cblas_call.b = b;
    b[0] = make_cfloat(299.0f);
    return 163;
}

static int check_spbsv_fortran_to_cblas(void)
{
    fb_backend_vtable_t vtable;
    fb_spbsv_fn thunk = NULL;
    float ab[8] = { 10.0f, 11.0f, 12.0f, 13.0f, 20.0f, 21.0f, 22.0f, 23.0f };
    float b[8] = { 1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f, 7.0f, 8.0f };
    float expected_ab_in[8] = { 0.0f, 20.0f, 11.0f, 21.0f, 12.0f, 22.0f, 13.0f, 23.0f };
    float expected_b_in[8] = { 1.0f, 3.0f, 5.0f, 7.0f, 2.0f, 4.0f, 6.0f, 8.0f };
    float expected_ab_out[8] = { 0.0f, 111.0f, 112.0f, 113.0f, 120.0f, 121.0f, 122.0f, 123.0f };
    float expected_b_out[8] = { 10.0f, 20.0f, 30.0f, 40.0f, 50.0f, 60.0f, 70.0f, 80.0f };
    int info = 0;

    memset(&vtable, 0, sizeof(vtable));
    memset(&g_spbsv_fortran_call, 0, sizeof(g_spbsv_fortran_call));

    vtable.ext_ops[FB_OP_SPBSV][FB_CONV_FORTRAN] =
        (fb_generic_fn)(void (*)(void))stub_spbsv_fortran;
    fb_install_conv_thunks(&vtable, FB_OP_SPBSV);

    thunk = (fb_spbsv_fn)vtable.ext_ops[FB_OP_SPBSV][FB_CONV_CBLAS];
    if (!thunk) {
        fprintf(stderr, "[FAIL] SPBSV Fortran->CBLAS thunk was not installed\n");
        return 1;
    }

    info = thunk(FB_LAYOUT_ROW_MAJOR, FB_UPPER, 4, 1, 2, ab, 4, b, 2);
    if (info != 0 || g_spbsv_fortran_call.calls != 1 ||
        g_spbsv_fortran_call.uplo != 'U' || g_spbsv_fortran_call.n != 4 ||
        g_spbsv_fortran_call.kd != 1 || g_spbsv_fortran_call.nrhs != 2 ||
        g_spbsv_fortran_call.ldab != 2 || g_spbsv_fortran_call.ldb != 4 ||
        memcmp(g_spbsv_fortran_call.ab_snapshot, expected_ab_in, sizeof(expected_ab_in)) != 0 ||
        memcmp(g_spbsv_fortran_call.b_snapshot, expected_b_in, sizeof(expected_b_in)) != 0 ||
        memcmp(ab, expected_ab_out, sizeof(expected_ab_out)) != 0 ||
        memcmp(b, expected_b_out, sizeof(expected_b_out)) != 0) {
        fprintf(stderr, "[FAIL] SPBSV Fortran->CBLAS thunk did not preserve band solve row-major translation and copy-back\n");
        return 1;
    }

    printf("[PASS] SPBSV Fortran->CBLAS thunk transposes band storage and RHS data, then copies both outputs back to row-major\n");
    return 0;
}

static int check_spbsv_cblas_to_fortran(void)
{
    fb_backend_vtable_t vtable;
    fb_spbsv_fortran_slot_fn thunk = NULL;
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
    memset(&g_spbsv_cblas_call, 0, sizeof(g_spbsv_cblas_call));

    vtable.ext_ops[FB_OP_SPBSV][FB_CONV_CBLAS] =
        (fb_generic_fn)(void (*)(void))stub_spbsv_cblas;
    fb_install_conv_thunks(&vtable, FB_OP_SPBSV);

    thunk = (fb_spbsv_fortran_slot_fn)vtable.ext_ops[FB_OP_SPBSV][FB_CONV_FORTRAN];
    if (!thunk) {
        fprintf(stderr, "[FAIL] SPBSV CBLAS->Fortran thunk was not installed\n");
        return 1;
    }

    thunk(&uplo, &n, &kd, &nrhs, ab, &ldab, b, &ldb, &info);
    if (info != 161 || g_spbsv_cblas_call.called != 1 ||
        g_spbsv_cblas_call.layout != FB_LAYOUT_COL_MAJOR ||
        g_spbsv_cblas_call.uplo != FB_LOWER ||
        g_spbsv_cblas_call.n != 3 || g_spbsv_cblas_call.kd != 1 ||
        g_spbsv_cblas_call.nrhs != 2 || g_spbsv_cblas_call.ldab != 2 ||
        g_spbsv_cblas_call.ldb != 3 || g_spbsv_cblas_call.ab != ab ||
        g_spbsv_cblas_call.b != b || b[0] != 99.0f) {
        fprintf(stderr, "[FAIL] SPBSV CBLAS->Fortran thunk delegated incorrectly\n");
        return 1;
    }

    printf("[PASS] SPBSV CBLAS->Fortran thunk maps the all-pointer ABI into the generic band-solve entry\n");
    return 0;
}

static int check_cpbsv_fortran_to_cblas(void)
{
    fb_backend_vtable_t vtable;
    fb_cpbsv_fn thunk = NULL;
    fb_complex_float_t ab[8] = {
        make_cfloat(30.0f), make_cfloat(31.0f), make_cfloat(32.0f), make_cfloat(33.0f),
        make_cfloat(40.0f), make_cfloat(41.0f), make_cfloat(42.0f), make_cfloat(43.0f)
    };
    fb_complex_float_t b[8] = {
        make_cfloat(1.0f), make_cfloat(2.0f), make_cfloat(3.0f), make_cfloat(4.0f),
        make_cfloat(5.0f), make_cfloat(6.0f), make_cfloat(7.0f), make_cfloat(8.0f)
    };
    float expected_ab_in[8] = { 30.0f, 40.0f, 31.0f, 41.0f, 32.0f, 42.0f, 33.0f, 0.0f };
    float expected_b_in[8] = { 1.0f, 3.0f, 5.0f, 7.0f, 2.0f, 4.0f, 6.0f, 8.0f };
    float expected_ab_out[8] = { 130.0f, 131.0f, 132.0f, 133.0f, 230.0f, 231.0f, 232.0f, 0.0f };
    float expected_b_out[8] = { 210.0f, 220.0f, 230.0f, 240.0f, 250.0f, 260.0f, 270.0f, 280.0f };
    float actual_ab_out[8] = { 0.0f };
    float actual_b_out[8] = { 0.0f };
    int index = 0;
    int info = 0;

    memset(&vtable, 0, sizeof(vtable));
    memset(&g_cpbsv_fortran_call, 0, sizeof(g_cpbsv_fortran_call));

    vtable.ext_ops[FB_OP_CPBSV][FB_CONV_FORTRAN] =
        (fb_generic_fn)(void (*)(void))stub_cpbsv_fortran;
    fb_install_conv_thunks(&vtable, FB_OP_CPBSV);

    thunk = (fb_cpbsv_fn)vtable.ext_ops[FB_OP_CPBSV][FB_CONV_CBLAS];
    if (!thunk) {
        fprintf(stderr, "[FAIL] CPBSV Fortran->CBLAS thunk was not installed\n");
        return 1;
    }

    info = thunk(FB_LAYOUT_ROW_MAJOR, FB_LOWER, 4, 1, 2, ab, 4, b, 2);
    for (index = 0; index < 8; ++index) {
        actual_ab_out[index] = cfloat_real(ab[index]);
        actual_b_out[index] = cfloat_real(b[index]);
    }
    if (info != 0 || g_cpbsv_fortran_call.calls != 1 ||
        g_cpbsv_fortran_call.uplo != 'L' || g_cpbsv_fortran_call.n != 4 ||
        g_cpbsv_fortran_call.kd != 1 || g_cpbsv_fortran_call.nrhs != 2 ||
        g_cpbsv_fortran_call.ldab != 2 || g_cpbsv_fortran_call.ldb != 4 ||
        memcmp(g_cpbsv_fortran_call.ab_real_snapshot, expected_ab_in, sizeof(expected_ab_in)) != 0 ||
        memcmp(g_cpbsv_fortran_call.b_real_snapshot, expected_b_in, sizeof(expected_b_in)) != 0 ||
        memcmp(actual_ab_out, expected_ab_out, sizeof(expected_ab_out)) != 0 ||
        memcmp(actual_b_out, expected_b_out, sizeof(expected_b_out)) != 0) {
        fprintf(stderr, "[FAIL] CPBSV Fortran->CBLAS thunk did not preserve complex band solve row-major translation and copy-back\n");
        return 1;
    }

    printf("[PASS] CPBSV Fortran->CBLAS thunk transposes Hermitian band storage and RHS data, then copies both outputs back to row-major\n");
    return 0;
}

static int check_cpbsv_cblas_to_fortran(void)
{
    fb_backend_vtable_t vtable;
    fb_cpbsv_fortran_slot_fn thunk = NULL;
    fb_complex_float_t ab[6] = {
        make_cfloat(1.0f), make_cfloat(2.0f), make_cfloat(3.0f),
        make_cfloat(4.0f), make_cfloat(5.0f), make_cfloat(6.0f)
    };
    fb_complex_float_t b[6] = {
        make_cfloat(10.0f), make_cfloat(20.0f), make_cfloat(30.0f),
        make_cfloat(40.0f), make_cfloat(50.0f), make_cfloat(60.0f)
    };
    char uplo = 'U';
    int n = 3;
    int kd = 1;
    int nrhs = 2;
    int ldab = 2;
    int ldb = 3;
    int info = 0;

    memset(&vtable, 0, sizeof(vtable));
    memset(&g_cpbsv_cblas_call, 0, sizeof(g_cpbsv_cblas_call));

    vtable.ext_ops[FB_OP_CPBSV][FB_CONV_CBLAS] =
        (fb_generic_fn)(void (*)(void))stub_cpbsv_cblas;
    fb_install_conv_thunks(&vtable, FB_OP_CPBSV);

    thunk = (fb_cpbsv_fortran_slot_fn)vtable.ext_ops[FB_OP_CPBSV][FB_CONV_FORTRAN];
    if (!thunk) {
        fprintf(stderr, "[FAIL] CPBSV CBLAS->Fortran thunk was not installed\n");
        return 1;
    }

    thunk(&uplo, &n, &kd, &nrhs, ab, &ldab, b, &ldb, &info);
    if (info != 163 || g_cpbsv_cblas_call.called != 1 ||
        g_cpbsv_cblas_call.layout != FB_LAYOUT_COL_MAJOR ||
        g_cpbsv_cblas_call.uplo != FB_UPPER ||
        g_cpbsv_cblas_call.n != 3 || g_cpbsv_cblas_call.kd != 1 ||
        g_cpbsv_cblas_call.nrhs != 2 || g_cpbsv_cblas_call.ldab != 2 ||
        g_cpbsv_cblas_call.ldb != 3 || g_cpbsv_cblas_call.ab != ab ||
        g_cpbsv_cblas_call.b != b || cfloat_real(b[0]) != 299.0f) {
        fprintf(stderr, "[FAIL] CPBSV CBLAS->Fortran thunk delegated incorrectly\n");
        return 1;
    }

    printf("[PASS] CPBSV CBLAS->Fortran thunk maps the all-pointer ABI into the generic complex band-solve entry\n");
    return 0;
}

int main(void)
{
    if (check_spbsv_fortran_to_cblas() != 0) {
        return 1;
    }
    if (check_spbsv_cblas_to_fortran() != 0) {
        return 1;
    }
    if (check_cpbsv_fortran_to_cblas() != 0) {
        return 1;
    }
    if (check_cpbsv_cblas_to_fortran() != 0) {
        return 1;
    }

    printf("Result: PASS\n");
    return 0;
}