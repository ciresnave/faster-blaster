#include <stdio.h>
#include <string.h>

#include "../src/core/conv_thunks.h"

typedef int (*fb_spbrfs_fn)(fb_layout_t layout, fb_uplo_t uplo, int n, int kd,
                            int nrhs, const float *ab, int ldab,
                            const float *afb, int ldafb, const float *b,
                            int ldb, float *x, int ldx, float *ferr,
                            float *berr);
typedef int (*fb_cpbrfs_fn)(fb_layout_t layout, fb_uplo_t uplo, int n, int kd,
                            int nrhs, const fb_complex_float_t *ab, int ldab,
                            const fb_complex_float_t *afb, int ldafb,
                            const fb_complex_float_t *b, int ldb,
                            fb_complex_float_t *x, int ldx, float *ferr,
                            float *berr);

typedef void (*fb_spbrfs_fortran_slot_fn)(char *uplo, int *n, int *kd, int *nrhs,
                                          float *ab, int *ldab, float *afb,
                                          int *ldafb, float *b, int *ldb,
                                          float *x, int *ldx, float *ferr,
                                          float *berr, float *work, int *iwork,
                                          int *info);
typedef void (*fb_cpbrfs_fortran_slot_fn)(char *uplo, int *n, int *kd, int *nrhs,
                                          fb_complex_float_t *ab, int *ldab,
                                          fb_complex_float_t *afb, int *ldafb,
                                          fb_complex_float_t *b, int *ldb,
                                          fb_complex_float_t *x, int *ldx,
                                          float *ferr, float *berr,
                                          fb_complex_float_t *work,
                                          float *rwork, int *info);

static struct {
    int calls;
    char uplo;
    int n;
    int kd;
    int nrhs;
    int ldab;
    int ldafb;
    int ldb;
    int ldx;
    int work_seen;
    int aux_seen;
    float ab_snapshot[8];
    float afb_snapshot[8];
    float b_snapshot[8];
    float x_snapshot[8];
} g_spbrfs_fortran_call;

static struct {
    int called;
    fb_layout_t layout;
    fb_uplo_t uplo;
    int n;
    int kd;
    int nrhs;
    int ldab;
    int ldafb;
    int ldb;
    int ldx;
    const float *ab;
    const float *afb;
    const float *b;
    float *x;
    float *ferr;
    float *berr;
} g_spbrfs_cblas_call;

static struct {
    int calls;
    char uplo;
    int n;
    int kd;
    int nrhs;
    int ldab;
    int ldafb;
    int ldb;
    int ldx;
    int work_seen;
    int aux_seen;
    float ab_real_snapshot[8];
    float afb_real_snapshot[8];
    float b_real_snapshot[8];
    float x_real_snapshot[8];
} g_cpbrfs_fortran_call;

static struct {
    int called;
    fb_layout_t layout;
    fb_uplo_t uplo;
    int n;
    int kd;
    int nrhs;
    int ldab;
    int ldafb;
    int ldb;
    int ldx;
    const fb_complex_float_t *ab;
    const fb_complex_float_t *afb;
    const fb_complex_float_t *b;
    fb_complex_float_t *x;
    float *ferr;
    float *berr;
} g_cpbrfs_cblas_call;

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

static void stub_spbrfs_fortran(char *uplo, int *n, int *kd, int *nrhs,
                                float *ab, int *ldab, float *afb, int *ldafb,
                                float *b, int *ldb, float *x, int *ldx,
                                float *ferr, float *berr, float *work,
                                int *iwork, int *info)
{
    static const float solution_x_col[8] = { 101.0f, 103.0f, 105.0f, 107.0f,
                                             102.0f, 104.0f, 106.0f, 108.0f };
    int index = 0;

    g_spbrfs_fortran_call.calls += 1;
    g_spbrfs_fortran_call.uplo = *uplo;
    g_spbrfs_fortran_call.n = *n;
    g_spbrfs_fortran_call.kd = *kd;
    g_spbrfs_fortran_call.nrhs = *nrhs;
    g_spbrfs_fortran_call.ldab = *ldab;
    g_spbrfs_fortran_call.ldafb = *ldafb;
    g_spbrfs_fortran_call.ldb = *ldb;
    g_spbrfs_fortran_call.ldx = *ldx;
    g_spbrfs_fortran_call.work_seen = (work != NULL);
    g_spbrfs_fortran_call.aux_seen = (iwork != NULL);
    for (index = 0; index < (*ldab * *n); ++index) {
        g_spbrfs_fortran_call.ab_snapshot[index] = ab[index];
        g_spbrfs_fortran_call.afb_snapshot[index] = afb[index];
    }
    for (index = 0; index < (*ldb * *nrhs); ++index) {
        g_spbrfs_fortran_call.b_snapshot[index] = b[index];
        g_spbrfs_fortran_call.x_snapshot[index] = x[index];
    }
    memcpy(x, solution_x_col, sizeof(solution_x_col));
    ferr[0] = 0.1f;
    ferr[1] = 0.2f;
    berr[0] = 0.01f;
    berr[1] = 0.02f;
    *info = 0;
}

static int stub_spbrfs_cblas(fb_layout_t layout, fb_uplo_t uplo, int n, int kd,
                             int nrhs, const float *ab, int ldab,
                             const float *afb, int ldafb, const float *b,
                             int ldb, float *x, int ldx, float *ferr,
                             float *berr)
{
    g_spbrfs_cblas_call.called += 1;
    g_spbrfs_cblas_call.layout = layout;
    g_spbrfs_cblas_call.uplo = uplo;
    g_spbrfs_cblas_call.n = n;
    g_spbrfs_cblas_call.kd = kd;
    g_spbrfs_cblas_call.nrhs = nrhs;
    g_spbrfs_cblas_call.ldab = ldab;
    g_spbrfs_cblas_call.ldafb = ldafb;
    g_spbrfs_cblas_call.ldb = ldb;
    g_spbrfs_cblas_call.ldx = ldx;
    g_spbrfs_cblas_call.ab = ab;
    g_spbrfs_cblas_call.afb = afb;
    g_spbrfs_cblas_call.b = b;
    g_spbrfs_cblas_call.x = x;
    g_spbrfs_cblas_call.ferr = ferr;
    g_spbrfs_cblas_call.berr = berr;
    x[0] = 299.0f;
    ferr[0] = 1.5f;
    berr[0] = 0.5f;
    return 171;
}

static void stub_cpbrfs_fortran(char *uplo, int *n, int *kd, int *nrhs,
                                fb_complex_float_t *ab, int *ldab,
                                fb_complex_float_t *afb, int *ldafb,
                                fb_complex_float_t *b, int *ldb,
                                fb_complex_float_t *x, int *ldx, float *ferr,
                                float *berr, fb_complex_float_t *work,
                                float *rwork, int *info)
{
    static const float solution_x_col_real[8] = { 301.0f, 303.0f, 305.0f, 307.0f,
                                                  302.0f, 304.0f, 306.0f, 308.0f };
    int index = 0;

    g_cpbrfs_fortran_call.calls += 1;
    g_cpbrfs_fortran_call.uplo = *uplo;
    g_cpbrfs_fortran_call.n = *n;
    g_cpbrfs_fortran_call.kd = *kd;
    g_cpbrfs_fortran_call.nrhs = *nrhs;
    g_cpbrfs_fortran_call.ldab = *ldab;
    g_cpbrfs_fortran_call.ldafb = *ldafb;
    g_cpbrfs_fortran_call.ldb = *ldb;
    g_cpbrfs_fortran_call.ldx = *ldx;
    g_cpbrfs_fortran_call.work_seen = (work != NULL);
    g_cpbrfs_fortran_call.aux_seen = (rwork != NULL);
    for (index = 0; index < (*ldab * *n); ++index) {
        g_cpbrfs_fortran_call.ab_real_snapshot[index] = cfloat_real(ab[index]);
        g_cpbrfs_fortran_call.afb_real_snapshot[index] = cfloat_real(afb[index]);
    }
    for (index = 0; index < (*ldb * *nrhs); ++index) {
        g_cpbrfs_fortran_call.b_real_snapshot[index] = cfloat_real(b[index]);
        g_cpbrfs_fortran_call.x_real_snapshot[index] = cfloat_real(x[index]);
    }
    for (index = 0; index < 8; ++index) {
        x[index] = make_cfloat(solution_x_col_real[index]);
    }
    ferr[0] = 0.3f;
    ferr[1] = 0.4f;
    berr[0] = 0.03f;
    berr[1] = 0.04f;
    *info = 0;
}

static int stub_cpbrfs_cblas(fb_layout_t layout, fb_uplo_t uplo, int n, int kd,
                             int nrhs, const fb_complex_float_t *ab, int ldab,
                             const fb_complex_float_t *afb, int ldafb,
                             const fb_complex_float_t *b, int ldb,
                             fb_complex_float_t *x, int ldx, float *ferr,
                             float *berr)
{
    g_cpbrfs_cblas_call.called += 1;
    g_cpbrfs_cblas_call.layout = layout;
    g_cpbrfs_cblas_call.uplo = uplo;
    g_cpbrfs_cblas_call.n = n;
    g_cpbrfs_cblas_call.kd = kd;
    g_cpbrfs_cblas_call.nrhs = nrhs;
    g_cpbrfs_cblas_call.ldab = ldab;
    g_cpbrfs_cblas_call.ldafb = ldafb;
    g_cpbrfs_cblas_call.ldb = ldb;
    g_cpbrfs_cblas_call.ldx = ldx;
    g_cpbrfs_cblas_call.ab = ab;
    g_cpbrfs_cblas_call.afb = afb;
    g_cpbrfs_cblas_call.b = b;
    g_cpbrfs_cblas_call.x = x;
    g_cpbrfs_cblas_call.ferr = ferr;
    g_cpbrfs_cblas_call.berr = berr;
    x[0] = make_cfloat(399.0f);
    ferr[0] = 2.5f;
    berr[0] = 0.75f;
    return 173;
}

static int check_spbrfs_fortran_to_cblas(void)
{
    fb_backend_vtable_t vtable;
    fb_spbrfs_fn thunk = NULL;
    float ab[8] = { 10.0f, 11.0f, 12.0f, 13.0f, 20.0f, 21.0f, 22.0f, 23.0f };
    float afb[8] = { 30.0f, 31.0f, 32.0f, 33.0f, 40.0f, 41.0f, 42.0f, 43.0f };
    float b[8] = { 1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f, 7.0f, 8.0f };
    float x[8] = { 9.0f, 10.0f, 11.0f, 12.0f, 13.0f, 14.0f, 15.0f, 16.0f };
    float ferr[2] = { 0.0f, 0.0f };
    float berr[2] = { 0.0f, 0.0f };
    float expected_ab_in[8] = { 0.0f, 20.0f, 11.0f, 21.0f, 12.0f, 22.0f, 13.0f, 23.0f };
    float expected_afb_in[8] = { 0.0f, 40.0f, 31.0f, 41.0f, 32.0f, 42.0f, 33.0f, 43.0f };
    float expected_b_in[8] = { 1.0f, 3.0f, 5.0f, 7.0f, 2.0f, 4.0f, 6.0f, 8.0f };
    float expected_x_in[8] = { 9.0f, 11.0f, 13.0f, 15.0f, 10.0f, 12.0f, 14.0f, 16.0f };
    float expected_x_out[8] = { 101.0f, 102.0f, 103.0f, 104.0f, 105.0f, 106.0f, 107.0f, 108.0f };
    int info = 0;

    memset(&vtable, 0, sizeof(vtable));
    memset(&g_spbrfs_fortran_call, 0, sizeof(g_spbrfs_fortran_call));

    vtable.ext_ops[FB_OP_SPBRFS][FB_CONV_FORTRAN] =
        (fb_generic_fn)(void (*)(void))stub_spbrfs_fortran;
    fb_install_conv_thunks(&vtable, FB_OP_SPBRFS);

    thunk = (fb_spbrfs_fn)vtable.ext_ops[FB_OP_SPBRFS][FB_CONV_CBLAS];
    if (!thunk) {
        fprintf(stderr, "[FAIL] SPBRFS Fortran->CBLAS thunk was not installed\n");
        return 1;
    }

    info = thunk(FB_LAYOUT_ROW_MAJOR, FB_UPPER, 4, 1, 2,
                 ab, 4, afb, 4, b, 2, x, 2, ferr, berr);
    if (info != 0 || g_spbrfs_fortran_call.calls != 1 ||
        g_spbrfs_fortran_call.uplo != 'U' || g_spbrfs_fortran_call.n != 4 ||
        g_spbrfs_fortran_call.kd != 1 || g_spbrfs_fortran_call.nrhs != 2 ||
        g_spbrfs_fortran_call.ldab != 2 || g_spbrfs_fortran_call.ldafb != 2 ||
        g_spbrfs_fortran_call.ldb != 4 || g_spbrfs_fortran_call.ldx != 4 ||
        !g_spbrfs_fortran_call.work_seen || !g_spbrfs_fortran_call.aux_seen ||
        memcmp(g_spbrfs_fortran_call.ab_snapshot, expected_ab_in, sizeof(expected_ab_in)) != 0 ||
        memcmp(g_spbrfs_fortran_call.afb_snapshot, expected_afb_in, sizeof(expected_afb_in)) != 0 ||
        memcmp(g_spbrfs_fortran_call.b_snapshot, expected_b_in, sizeof(expected_b_in)) != 0 ||
        memcmp(g_spbrfs_fortran_call.x_snapshot, expected_x_in, sizeof(expected_x_in)) != 0 ||
        memcmp(x, expected_x_out, sizeof(expected_x_out)) != 0 ||
        ferr[0] != 0.1f || ferr[1] != 0.2f || berr[0] != 0.01f || berr[1] != 0.02f) {
        fprintf(stderr, "[FAIL] SPBRFS Fortran->CBLAS thunk did not preserve row-major band refinement semantics\n");
        return 1;
    }

    printf("[PASS] SPBRFS Fortran->CBLAS thunk transposes banded inputs, allocates fixed real scratch, and copies back only X\n");
    return 0;
}

static int check_spbrfs_cblas_to_fortran(void)
{
    fb_backend_vtable_t vtable;
    fb_spbrfs_fortran_slot_fn thunk = NULL;
    float ab[6] = { 1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f };
    float afb[6] = { 7.0f, 8.0f, 9.0f, 10.0f, 11.0f, 12.0f };
    float b[6] = { 13.0f, 14.0f, 15.0f, 16.0f, 17.0f, 18.0f };
    float x[6] = { 19.0f, 20.0f, 21.0f, 22.0f, 23.0f, 24.0f };
    float ferr[2] = { 0.0f, 0.0f };
    float berr[2] = { 0.0f, 0.0f };
    float work[9] = { 0.0f };
    int iwork[3] = { 0, 0, 0 };
    char uplo = 'L';
    int n = 3;
    int kd = 1;
    int nrhs = 2;
    int ldab = 2;
    int ldafb = 2;
    int ldb = 3;
    int ldx = 3;
    int info = 0;

    memset(&vtable, 0, sizeof(vtable));
    memset(&g_spbrfs_cblas_call, 0, sizeof(g_spbrfs_cblas_call));

    vtable.ext_ops[FB_OP_SPBRFS][FB_CONV_CBLAS] =
        (fb_generic_fn)(void (*)(void))stub_spbrfs_cblas;
    fb_install_conv_thunks(&vtable, FB_OP_SPBRFS);

    thunk = (fb_spbrfs_fortran_slot_fn)vtable.ext_ops[FB_OP_SPBRFS][FB_CONV_FORTRAN];
    if (!thunk) {
        fprintf(stderr, "[FAIL] SPBRFS CBLAS->Fortran thunk was not installed\n");
        return 1;
    }

    thunk(&uplo, &n, &kd, &nrhs, ab, &ldab, afb, &ldafb, b, &ldb, x, &ldx,
          ferr, berr, work, iwork, &info);
    if (info != 171 || g_spbrfs_cblas_call.called != 1 ||
        g_spbrfs_cblas_call.layout != FB_LAYOUT_COL_MAJOR ||
        g_spbrfs_cblas_call.uplo != FB_LOWER || g_spbrfs_cblas_call.n != 3 ||
        g_spbrfs_cblas_call.kd != 1 || g_spbrfs_cblas_call.nrhs != 2 ||
        g_spbrfs_cblas_call.ldab != 2 || g_spbrfs_cblas_call.ldafb != 2 ||
        g_spbrfs_cblas_call.ldb != 3 || g_spbrfs_cblas_call.ldx != 3 ||
        g_spbrfs_cblas_call.ab != ab || g_spbrfs_cblas_call.afb != afb ||
        g_spbrfs_cblas_call.b != b || g_spbrfs_cblas_call.x != x ||
        g_spbrfs_cblas_call.ferr != ferr || g_spbrfs_cblas_call.berr != berr ||
        x[0] != 299.0f || ferr[0] != 1.5f || berr[0] != 0.5f) {
        fprintf(stderr, "[FAIL] SPBRFS CBLAS->Fortran thunk delegated incorrectly\n");
        return 1;
    }

    printf("[PASS] SPBRFS CBLAS->Fortran thunk maps the all-pointer ABI into the generic band-refinement entry\n");
    return 0;
}

static int check_cpbrfs_fortran_to_cblas(void)
{
    fb_backend_vtable_t vtable;
    fb_cpbrfs_fn thunk = NULL;
    fb_complex_float_t ab[8] = {
        make_cfloat(130.0f), make_cfloat(131.0f), make_cfloat(132.0f), make_cfloat(133.0f),
        make_cfloat(230.0f), make_cfloat(231.0f), make_cfloat(232.0f), make_cfloat(233.0f)
    };
    fb_complex_float_t afb[8] = {
        make_cfloat(330.0f), make_cfloat(331.0f), make_cfloat(332.0f), make_cfloat(333.0f),
        make_cfloat(430.0f), make_cfloat(431.0f), make_cfloat(432.0f), make_cfloat(433.0f)
    };
    fb_complex_float_t b[8] = {
        make_cfloat(1.0f), make_cfloat(2.0f), make_cfloat(3.0f), make_cfloat(4.0f),
        make_cfloat(5.0f), make_cfloat(6.0f), make_cfloat(7.0f), make_cfloat(8.0f)
    };
    fb_complex_float_t x[8] = {
        make_cfloat(9.0f), make_cfloat(10.0f), make_cfloat(11.0f), make_cfloat(12.0f),
        make_cfloat(13.0f), make_cfloat(14.0f), make_cfloat(15.0f), make_cfloat(16.0f)
    };
    float ferr[2] = { 0.0f, 0.0f };
    float berr[2] = { 0.0f, 0.0f };
    float expected_ab_in[8] = { 130.0f, 230.0f, 131.0f, 231.0f, 132.0f, 232.0f, 133.0f, 0.0f };
    float expected_afb_in[8] = { 330.0f, 430.0f, 331.0f, 431.0f, 332.0f, 432.0f, 333.0f, 0.0f };
    float expected_b_in[8] = { 1.0f, 3.0f, 5.0f, 7.0f, 2.0f, 4.0f, 6.0f, 8.0f };
    float expected_x_in[8] = { 9.0f, 11.0f, 13.0f, 15.0f, 10.0f, 12.0f, 14.0f, 16.0f };
    float expected_x_out[8] = { 301.0f, 302.0f, 303.0f, 304.0f, 305.0f, 306.0f, 307.0f, 308.0f };
    float actual_x_out[8] = { 0.0f };
    int index = 0;
    int info = 0;

    memset(&vtable, 0, sizeof(vtable));
    memset(&g_cpbrfs_fortran_call, 0, sizeof(g_cpbrfs_fortran_call));

    vtable.ext_ops[FB_OP_CPBRFS][FB_CONV_FORTRAN] =
        (fb_generic_fn)(void (*)(void))stub_cpbrfs_fortran;
    fb_install_conv_thunks(&vtable, FB_OP_CPBRFS);

    thunk = (fb_cpbrfs_fn)vtable.ext_ops[FB_OP_CPBRFS][FB_CONV_CBLAS];
    if (!thunk) {
        fprintf(stderr, "[FAIL] CPBRFS Fortran->CBLAS thunk was not installed\n");
        return 1;
    }

    info = thunk(FB_LAYOUT_ROW_MAJOR, FB_LOWER, 4, 1, 2,
                 ab, 4, afb, 4, b, 2, x, 2, ferr, berr);
    for (index = 0; index < 8; ++index) {
        actual_x_out[index] = cfloat_real(x[index]);
    }
    if (info != 0 || g_cpbrfs_fortran_call.calls != 1 ||
        g_cpbrfs_fortran_call.uplo != 'L' || g_cpbrfs_fortran_call.n != 4 ||
        g_cpbrfs_fortran_call.kd != 1 || g_cpbrfs_fortran_call.nrhs != 2 ||
        g_cpbrfs_fortran_call.ldab != 2 || g_cpbrfs_fortran_call.ldafb != 2 ||
        g_cpbrfs_fortran_call.ldb != 4 || g_cpbrfs_fortran_call.ldx != 4 ||
        !g_cpbrfs_fortran_call.work_seen || !g_cpbrfs_fortran_call.aux_seen ||
        memcmp(g_cpbrfs_fortran_call.ab_real_snapshot, expected_ab_in, sizeof(expected_ab_in)) != 0 ||
        memcmp(g_cpbrfs_fortran_call.afb_real_snapshot, expected_afb_in, sizeof(expected_afb_in)) != 0 ||
        memcmp(g_cpbrfs_fortran_call.b_real_snapshot, expected_b_in, sizeof(expected_b_in)) != 0 ||
        memcmp(g_cpbrfs_fortran_call.x_real_snapshot, expected_x_in, sizeof(expected_x_in)) != 0 ||
        memcmp(actual_x_out, expected_x_out, sizeof(expected_x_out)) != 0 ||
        ferr[0] != 0.3f || ferr[1] != 0.4f || berr[0] != 0.03f || berr[1] != 0.04f) {
        fprintf(stderr, "[FAIL] CPBRFS Fortran->CBLAS thunk did not preserve complex row-major band refinement semantics\n");
        return 1;
    }

    printf("[PASS] CPBRFS Fortran->CBLAS thunk transposes Hermitian banded inputs, allocates fixed complex scratch, and copies back only X\n");
    return 0;
}

static int check_cpbrfs_cblas_to_fortran(void)
{
    fb_backend_vtable_t vtable;
    fb_cpbrfs_fortran_slot_fn thunk = NULL;
    fb_complex_float_t ab[6] = {
        make_cfloat(1.0f), make_cfloat(2.0f), make_cfloat(3.0f),
        make_cfloat(4.0f), make_cfloat(5.0f), make_cfloat(6.0f)
    };
    fb_complex_float_t afb[6] = {
        make_cfloat(7.0f), make_cfloat(8.0f), make_cfloat(9.0f),
        make_cfloat(10.0f), make_cfloat(11.0f), make_cfloat(12.0f)
    };
    fb_complex_float_t b[6] = {
        make_cfloat(13.0f), make_cfloat(14.0f), make_cfloat(15.0f),
        make_cfloat(16.0f), make_cfloat(17.0f), make_cfloat(18.0f)
    };
    fb_complex_float_t x[6] = {
        make_cfloat(19.0f), make_cfloat(20.0f), make_cfloat(21.0f),
        make_cfloat(22.0f), make_cfloat(23.0f), make_cfloat(24.0f)
    };
    float ferr[2] = { 0.0f, 0.0f };
    float berr[2] = { 0.0f, 0.0f };
    fb_complex_float_t work[6];
    float rwork[3] = { 0.0f, 0.0f, 0.0f };
    char uplo = 'U';
    int n = 3;
    int kd = 1;
    int nrhs = 2;
    int ldab = 2;
    int ldafb = 2;
    int ldb = 3;
    int ldx = 3;
    int info = 0;

    memset(&vtable, 0, sizeof(vtable));
    memset(&g_cpbrfs_cblas_call, 0, sizeof(g_cpbrfs_cblas_call));
    memset(work, 0, sizeof(work));

    vtable.ext_ops[FB_OP_CPBRFS][FB_CONV_CBLAS] =
        (fb_generic_fn)(void (*)(void))stub_cpbrfs_cblas;
    fb_install_conv_thunks(&vtable, FB_OP_CPBRFS);

    thunk = (fb_cpbrfs_fortran_slot_fn)vtable.ext_ops[FB_OP_CPBRFS][FB_CONV_FORTRAN];
    if (!thunk) {
        fprintf(stderr, "[FAIL] CPBRFS CBLAS->Fortran thunk was not installed\n");
        return 1;
    }

    thunk(&uplo, &n, &kd, &nrhs, ab, &ldab, afb, &ldafb, b, &ldb, x, &ldx,
          ferr, berr, work, rwork, &info);
    if (info != 173 || g_cpbrfs_cblas_call.called != 1 ||
        g_cpbrfs_cblas_call.layout != FB_LAYOUT_COL_MAJOR ||
        g_cpbrfs_cblas_call.uplo != FB_UPPER || g_cpbrfs_cblas_call.n != 3 ||
        g_cpbrfs_cblas_call.kd != 1 || g_cpbrfs_cblas_call.nrhs != 2 ||
        g_cpbrfs_cblas_call.ldab != 2 || g_cpbrfs_cblas_call.ldafb != 2 ||
        g_cpbrfs_cblas_call.ldb != 3 || g_cpbrfs_cblas_call.ldx != 3 ||
        g_cpbrfs_cblas_call.ab != ab || g_cpbrfs_cblas_call.afb != afb ||
        g_cpbrfs_cblas_call.b != b || g_cpbrfs_cblas_call.x != x ||
        g_cpbrfs_cblas_call.ferr != ferr || g_cpbrfs_cblas_call.berr != berr ||
        cfloat_real(x[0]) != 399.0f || ferr[0] != 2.5f || berr[0] != 0.75f) {
        fprintf(stderr, "[FAIL] CPBRFS CBLAS->Fortran thunk delegated incorrectly\n");
        return 1;
    }

    printf("[PASS] CPBRFS CBLAS->Fortran thunk maps the all-pointer ABI into the generic complex band-refinement entry\n");
    return 0;
}

int main(void)
{
    if (check_spbrfs_fortran_to_cblas() != 0) {
        return 1;
    }
    if (check_spbrfs_cblas_to_fortran() != 0) {
        return 1;
    }
    if (check_cpbrfs_fortran_to_cblas() != 0) {
        return 1;
    }
    if (check_cpbrfs_cblas_to_fortran() != 0) {
        return 1;
    }

    printf("Result: PASS\n");
    return 0;
}